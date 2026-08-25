import sys
import argparse
import csv
import copy
import json
import math
import os
import random
import time
import torch
import numpy as np
import datetime

from torch.optim import AdamW
from config import cfg
from dataset.load_dataset import create_dataset
from models.networking_head import NetworkingHead
from utils.console_logger import ConsoleLogger
from utils.plms_utils import load_plm
from utils.normalize import normalize_data, denormalize_data
from utils.result_notebook import ResultNotebook
from torch.utils.data import DataLoader
from models.pipeline import Pipeline
from models.patch_selection import PatchSelectionModule
from models.selectable_pipeline import LlamaSelectablePipeline
from models.selectors import RecentKSelector
from models.speculative_pipeline import LlamaSpeculativeBlockVerifyPipeline
from models.low_rank import peft_model, print_trainable_parameters
from models.eva_initializer import validate_eva_state
from models.rank_allocator import RankAllocationConstraintError
from models.nbs_compaction import (
    compact_nbs_model_for_inference,
    extract_nbs_compaction_specs,
    load_nbs_compact_checkpoint,
    save_nbs_compact_checkpoint,
    validate_nbs_compaction_factors,
    write_nbs_compaction_validation,
)
from utils.latency_utils import write_latency_artifacts
from utils.seed_utils import (
    isolated_seed,
    make_data_generator,
    resolve_experiment_seeds,
    seed_data_worker,
)


NASH_DIAGNOSTIC_FIELDS = [
    'optimizer_step', 'phase', 'event', 'layer_name',
    'transformer_layer_index', 'module_type', 'rank', 'sensitivity',
    'alpha', 'spectral_energy_total', 'utility', 'next_utility_increment',
    'next_marginal_utility_gain', 'next_marginal_gain',
    'min_rank', 'max_rank', 'at_min_rank', 'at_max_rank', 'rank_delta',
    'total_rank', 'rank_budget', 'budget_mode', 'relative_lambda',
    'reference_gain', 'stopping_threshold', 'last_allocated_gain',
    'next_rejected_gain', 'effective_rank_budget', 'rank_budget_cap',
    'adaptive_min_budget', 'stopping_reason',
    'validation_event', 'validation_loss',
    'teacher_forcing_validation_loss',
    'allocation_error', 'allocation_error_message', 'requested_rank',
    'allocation_error_action',
    'validation_position_kind', 'validation_position', 'is_best_validation',
    'nbs_allocation_started', 'is_best_post_nbs_validation',
]


def _initialize_nash_diagnostics(path, append=False):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    if append and os.path.exists(path) and os.path.getsize(path) > 0:
        with open(path, newline='', encoding='utf-8') as handle:
            reader = csv.DictReader(handle)
            existing_fields = reader.fieldnames or []
            if existing_fields == NASH_DIAGNOSTIC_FIELDS:
                return
            if not set(existing_fields).issubset(NASH_DIAGNOSTIC_FIELDS):
                raise ValueError(
                    f'Unsupported NBS diagnostics schema in {path}: {existing_fields}'
                )
            existing_rows = list(reader)
        migrated_path = path + '.schema_upgrade'
        with open(migrated_path, 'w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=NASH_DIAGNOSTIC_FIELDS)
            writer.writeheader()
            writer.writerows(existing_rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(migrated_path, path)
        return
    mode = 'a' if append and os.path.exists(path) else 'w'
    with open(path, mode, newline='', encoding='utf-8') as handle:
        if mode == 'w' or handle.tell() == 0:
            csv.DictWriter(handle, fieldnames=NASH_DIAGNOSTIC_FIELDS).writeheader()
        handle.flush()
        os.fsync(handle.fileno())


def _append_nash_diagnostics(path, rows):
    """Durably append one diagnostic event so interruptions preserve history."""
    if not rows:
        return
    with open(path, 'a', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(
            handle, fieldnames=NASH_DIAGNOSTIC_FIELDS, extrasaction='ignore'
        )
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())


def _last_validation_event(path):
    """Return the last durable validation index, including after resume."""
    if path is None or not os.path.exists(path):
        return 0
    last_event = 0
    with open(path, newline='', encoding='utf-8') as handle:
        for row in csv.DictReader(handle):
            value = row.get('validation_event')
            if value:
                last_event = max(last_event, int(value))
    return last_event


def _save_allocator_snapshot(path, allocator, diagnostics, metadata,
                             snapshot_state=None):
    """Atomically save a small, model-free allocator state snapshot."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    snapshot = allocator.state_dict() if snapshot_state is None else snapshot_state
    snapshot['snapshot_metadata'] = dict(metadata)
    snapshot['diagnostics'] = copy.deepcopy(diagnostics)
    temporary_path = path + '.tmp'
    try:
        torch.save(snapshot, temporary_path)
        os.replace(temporary_path, path)
    finally:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)


def _write_json_atomic(path, payload):
    temporary_path = path + '.tmp'
    try:
        with open(temporary_path, 'w', encoding='utf-8') as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)


def _seed_metadata(args):
    return {
        'seed': int(args.seed),
        'lora_seed': int(args.lora_seed),
        'data_seed': int(args.data_seed),
    }


def _seed_path_fragment(args):
    """Keep historical paths unless component seeds are explicitly different."""
    fragment = f'seed_{args.seed}'
    if args.lora_seed != args.seed or args.data_seed != args.seed:
        fragment += f'_lora_seed_{args.lora_seed}_data_seed_{args.data_seed}'
    return fragment


def _warn_checkpoint_seed_mismatch(args, model_dir):
    """Warn about reproducibility metadata without rejecting legacy checkpoints."""
    metadata_path = os.path.join(model_dir, 'checkpoint_metadata.json')
    if not os.path.isfile(metadata_path):
        return
    with open(metadata_path, 'r', encoding='utf-8') as handle:
        metadata = json.load(handle)
    expected = _seed_metadata(args)
    mismatches = []
    for key, current in expected.items():
        saved = metadata.get(key)
        if saved is not None and int(saved) != current:
            mismatches.append(f'{key}: checkpoint={saved}, current={current}')
    if mismatches:
        print(
            '\033[33mWarning:\033[0m checkpoint seed metadata differs from the '
            'current invocation (' + '; '.join(mismatches) + '). Loading continues.'
        )


def _write_inference_trace_artifacts(path, records, inference_tag):
    """Persist selector/speculative behavior without including it in latency."""
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    numeric_fields = (
        'initial_token_count', 'selected_token_count', 'target_forward_count',
        'accepted_drafts', 'proposed_drafts',
    )
    summary = {
        'inference_tag': inference_tag,
        'sample_count': len(records),
    }
    for field in numeric_fields:
        values = [float(row[field]) for row in records if row.get(field) is not None]
        summary[f'mean_{field}'] = sum(values) / len(values) if values else None
    initial = summary.get('mean_initial_token_count')
    selected = summary.get('mean_selected_token_count')
    summary['mean_token_reduction_percent'] = (
        (initial - selected) / initial * 100.0
        if initial not in (None, 0) and selected is not None else None
    )
    accepted = sum(float(row.get('accepted_drafts') or 0) for row in records)
    proposed = sum(float(row.get('proposed_drafts') or 0) for row in records)
    summary['draft_acceptance_rate'] = accepted / proposed if proposed > 0 else None
    _write_json_atomic(path, summary)

    detail_path = os.path.splitext(path)[0] + '_per_sample.csv'
    fields = [
        'test_step', 'video', 'user', 'timestep', 'selector_enabled',
        'initial_token_count', 'selected_token_count', 'token_reduction_percent',
        'target_forward_count', 'accepted_drafts', 'proposed_drafts',
        'draft_acceptance_rate',
    ]
    temporary_path = detail_path + '.tmp'
    with open(temporary_path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary_path, detail_path)
    return summary, detail_path


CHECKPOINT_PAYLOAD_FILES = (
    'adapter_model.bin',
    'adapter_model.safetensors',
    'adapter_config.json',
    'modules_except_plm.bin',
    'nash_rank_allocator.pt',
    'eva_state.pt',
    'model.bin',
    'README.md',
    'checkpoint_alias.json',
    'checkpoint_metadata.json',
)


def _resolve_checkpoint_alias(model_dir):
    """Resolve a role directory to its physical checkpoint without cycles."""
    current = os.path.abspath(model_dir)
    visited = set()
    while True:
        if current in visited:
            raise ValueError(f'Checkpoint alias cycle detected at {current}')
        visited.add(current)
        alias_path = os.path.join(current, 'checkpoint_alias.json')
        if not os.path.exists(alias_path):
            return current
        with open(alias_path, 'r', encoding='utf-8') as handle:
            alias = json.load(handle)
        target = alias.get('alias_of')
        if not target:
            raise ValueError(f'Checkpoint alias has no alias_of target: {alias_path}')
        current = (
            os.path.abspath(target)
            if os.path.isabs(target)
            else os.path.abspath(os.path.join(current, target))
        )


def save_checkpoint_alias(alias_dir, canonical_dir, metadata):
    """Store one logical checkpoint role without duplicating model tensors."""
    alias_absolute = os.path.abspath(alias_dir)
    canonical_absolute = _resolve_checkpoint_alias(canonical_dir)
    if alias_absolute == canonical_absolute:
        raise ValueError('Checkpoint alias cannot target itself')
    if not os.path.isdir(canonical_absolute):
        raise FileNotFoundError(
            f'Canonical checkpoint does not exist: {canonical_absolute}'
        )
    os.makedirs(alias_dir, exist_ok=True)
    for filename in CHECKPOINT_PAYLOAD_FILES:
        path = os.path.join(alias_dir, filename)
        if os.path.isfile(path):
            os.remove(path)
    relative_target = os.path.relpath(
        canonical_absolute, alias_absolute
    )
    alias_metadata = dict(metadata)
    alias_metadata.update({'is_alias': True, 'alias_of': relative_target})
    _write_json_atomic(
        os.path.join(alias_dir, 'checkpoint_alias.json'), alias_metadata
    )
    _write_json_atomic(
        os.path.join(alias_dir, 'checkpoint_metadata.json'), alias_metadata
    )


def save_model(args, model, save_dir, metadata=None):
    """
    save fune-tune model
    """
    os.makedirs(save_dir, exist_ok=True)
    alias_path = os.path.join(save_dir, 'checkpoint_alias.json')
    if os.path.isfile(alias_path):
        os.remove(alias_path)
    if args.rank != -1:
        # save low rank matrices
        model.plm.save_pretrained(save_dir)
        # save other modules except plm
        torch.save(model.modules_except_plm.state_dict(), os.path.join(save_dir, 'modules_except_plm.bin'))
        allocator = getattr(model.plm, 'nash_rank_allocator', None)
        if allocator is not None:
            torch.save(allocator.state_dict(), os.path.join(save_dir, 'nash_rank_allocator.pt'))
        eva_state = getattr(model.plm, 'eva_state', None)
        if eva_state is not None:
            torch.save(eva_state, os.path.join(save_dir, 'eva_state.pt'))
    else:
        # low rank matrices are disabled, save whole model
        torch.save(model.state_dict(), os.path.join(save_dir, 'model.bin'))
    checkpoint_metadata = _seed_metadata(args)
    if metadata is not None:
        checkpoint_metadata.update(metadata)
    _write_json_atomic(
        os.path.join(save_dir, 'checkpoint_metadata.json'), checkpoint_metadata
    )


def _resize_adalora_to_ckpt(plm, adapter_bin, adapter_name='default'):
    """
    AdaLoRA prunes each target layer to its own final rank during training, so a
    trained checkpoint's lora_A/lora_B/lora_E tensors are NOT uniformly sized at
    init_r like a freshly peft-wrapped model expects. Resize this model's adapter
    tensors to match the checkpoint's per-layer ranks before load_adapter(), or
    the load fails on a shape mismatch.
    """
    sd = torch.load(adapter_bin, map_location='cpu')
    name2mod = dict(plm.named_modules())
    for k, v in sd.items():
        if 'lora_E' not in k:
            continue
        mod_key = k[:k.find('.lora_E')]
        tgt = name2mod.get(mod_key)
        if tgt is None:
            continue
        r = v.shape[0]  # this layer's final (post-pruning) rank
        in_f = tgt.lora_A[adapter_name].shape[1]
        out_f = tgt.lora_B[adapter_name].shape[0]
        dev, dt = tgt.lora_A[adapter_name].device, tgt.lora_A[adapter_name].dtype
        tgt.lora_A[adapter_name] = torch.nn.Parameter(torch.zeros(r, in_f, device=dev, dtype=dt))
        tgt.lora_B[adapter_name] = torch.nn.Parameter(torch.zeros(out_f, r, device=dev, dtype=dt))
        tgt.lora_E[adapter_name] = torch.nn.Parameter(torch.zeros(r, 1, device=dev, dtype=dt))


def load_model(args, model, model_dir):
    """
    load fune-tune model

    :return: the pretrained model corresponding to using model_dir
    """
    model_dir = _resolve_checkpoint_alias(model_dir)
    _warn_checkpoint_seed_mismatch(args, model_dir)
    if args.rank != -1:
        if args.use_adalora:
            _resize_adalora_to_ckpt(model.plm, os.path.join(model_dir, 'adapter_model.bin'))
        # load low rank matrices
        model.plm.load_adapter(model_dir, adapter_name='default')
        # load other modules except plm
        model.modules_except_plm.load_state_dict(torch.load(os.path.join(model_dir, 'modules_except_plm.bin')))
        allocator_state_path = os.path.join(model_dir, 'nash_rank_allocator.pt')
        allocator = getattr(model.plm, 'nash_rank_allocator', None)
        if allocator is not None and os.path.exists(allocator_state_path):
            allocator.load_state_dict(torch.load(allocator_state_path, map_location='cpu'))
    else:
        # low rank matrices are disabled, load whole model
        model.load_state_dict(torch.load(os.path.join(model_dir, 'model.bin')))
    return model


def load_compact_nbs_model(model, model_dir):
    """Load an inference-only compact NBS derivative into a fresh pipeline."""
    model_dir = _resolve_checkpoint_alias(model_dir)
    report = load_nbs_compact_checkpoint(model.plm, model_dir)
    modules_path = os.path.join(model_dir, 'modules_except_plm.bin')
    if not os.path.isfile(modules_path):
        raise FileNotFoundError(
            f'compact NBS checkpoint is missing modules_except_plm.bin: {model_dir}'
        )
    model.modules_except_plm.load_state_dict(
        torch.load(modules_path, map_location='cpu')
    )
    equivalence_path = os.path.join(model_dir, 'equivalence_report.json')
    if not os.path.isfile(equivalence_path):
        raise FileNotFoundError(
            f'compact NBS checkpoint has no equivalence report: {model_dir}'
        )
    with open(equivalence_path, 'r', encoding='utf-8') as handle:
        equivalence = json.load(handle)
    if not equivalence.get('passed'):
        raise RuntimeError(
            f'compact NBS checkpoint did not pass equivalence validation: {model_dir}'
        )
    return model, report


def adapt(args, pipeline, dataloader_train, dataloader_valid, models_dir, grad_accum_steps):
    seed_path_fragment = _seed_path_fragment(args)
    file_prefix = f'his_{args.his_window}_fut_{args.fut_window}_ss_{args.sample_step}_epochs_{args.epochs}_bs_{args.bs * args.grad_accum_steps}_'\
                  f'lr_{args.lr}_{seed_path_fragment}_rank_{args.rank}_scheduled_sampling_{args.scheduled_sampling}'
    checkpoint_path = os.path.join(models_dir, file_prefix, 'checkpoint')
    if ((args.save_checkpoint_per_step is not None or
         args.save_checkpoint_per_epoch is not None) and
            not os.path.exists(checkpoint_path)):
        os.makedirs(checkpoint_path)
    checkpoint_root = os.path.join(models_dir, file_prefix)
    best_model_path = os.path.join(
        checkpoint_root, 'best_ar_model' if args.use_adalora else 'best_model'
    )
    best_post_nbs_model_path = os.path.join(checkpoint_root, 'best_post_nbs_model')
    final_nbs_model_path = os.path.join(checkpoint_root, 'final_nbs_model')
    final_shapley_model_path = os.path.join(
        checkpoint_root, 'final_shapley_model'
    )
    console_log = open(os.path.join(models_dir, file_prefix + '_console.log'), 'w')
    sys.stdout = ConsoleLogger(sys.__stdout__, console_log)

    if args.resume:
        pipeline = load_model(args, pipeline, args.resume_path)
        print('Resume weights for training from:', args.resume_path)

    allocator = None
    nash_diagnostics_path = None
    allocator_snapshot_dir = None
    validation_event_count = 0
    if (args.rank != -1 and args.use_adalora and
            args.adalora_allocator == 'nbs'):
        allocator = pipeline.plm.nash_rank_allocator
        nash_diagnostics_path = args.adalora_diagnostics_path or os.path.join(
            models_dir, file_prefix, 'nbs_rank_diagnostics.csv'
        )
        _initialize_nash_diagnostics(nash_diagnostics_path, append=args.resume)
        allocator_snapshot_dir = os.path.join(
            os.path.dirname(nash_diagnostics_path), 'rank_snapshots'
        )
        os.makedirs(allocator_snapshot_dir, exist_ok=True)
        pre_mask_snapshot_path = os.path.join(
            allocator_snapshot_dir, 'pre_mask_initial_spectrum.pt'
        )
        if not os.path.exists(pre_mask_snapshot_path):
            try:
                pre_mask_state = allocator.pre_mask_spectrum_state_dict()
            except RuntimeError as exc:
                print('Pre-mask initial spectrum snapshot unavailable:', exc)
            else:
                _save_allocator_snapshot(
                    pre_mask_snapshot_path,
                    allocator,
                    diagnostics=[],
                    metadata={
                        **_seed_metadata(args),
                        'snapshot_kind': 'pre_mask_initialization',
                        'optimizer_step': 0,
                        'validation_event': 0,
                        'validation_position_kind': 'initialization',
                        'validation_position': 0,
                        'nbs_allocation_started': False,
                    },
                    snapshot_state=pre_mask_state,
                )
                print(
                    'Pre-mask initial spectrum snapshot saved at',
                    pre_mask_snapshot_path,
                )
        validation_event_count = _last_validation_event(nash_diagnostics_path)
        initial_event = 'resume' if args.resume else 'initialization'
        _append_nash_diagnostics(
            nash_diagnostics_path,
            allocator.snapshot_diagnostics(step=0, event=initial_event),
        )
        print('NBS rank diagnostics saved at', nash_diagnostics_path)

    if (args.rank != -1 and args.use_adalora and
            args.adalora_allocator == 'shapley'):
        shapley_allocator = getattr(
            pipeline.plm, 'shapley_rank_allocator', None
        )
        if shapley_allocator is None:
            raise RuntimeError(
                'Shapley allocator was requested but was not attached to the PEFT model'
            )
        fixed_batches = []
        for batch_index, (history, future, video_user_info) in enumerate(
                dataloader_valid):
            if batch_index >= args.shapley_validation_batches:
                break
            fixed_batches.append((
                normalize_data(history.to(args.device), args.train_dataset),
                normalize_data(future.to(args.device), args.train_dataset),
                video_user_info,
            ))
        if not fixed_batches:
            raise RuntimeError('Shapley allocator could not capture a validation batch')

        def shapley_validation_loss():
            was_training = pipeline.training
            pipeline.eval()
            try:
                with torch.no_grad():
                    losses = [
                        pipeline(
                            history,
                            future,
                            video_user_info,
                            teacher_forcing=(
                                args.shapley_value_mode == 'teacher-forcing'
                            ),
                        ).float().item()
                        for history, future, video_user_info in fixed_batches
                    ]
                return sum(losses) / len(losses)
            finally:
                pipeline.train(was_training)

        shapley_allocator.set_loss_fn(shapley_validation_loss)
        print(
            '[Shapley AdaLoRA] fixed validation batches={}, value mode={}'.format(
                len(fixed_batches), args.shapley_value_mode
            )
        )

    if not args.freeze_plm:
        no_decay = ['bias', 'LayerNorm.weight']
        # multimodal_lr defaults to args.lr (multiplier=1.0), so this group's LR is
        # unchanged from before unless --multimodal-lr-multiplier is explicitly passed.
        multimodal_lr = args.lr * args.multimodal_lr_multiplier
        optimizer_grouped_parameters = [
            {'params': [p for n, p in pipeline.plm.named_parameters() if not any(nd in n for nd in no_decay)],
            'weight_decay': args.weight_decay, 'lr': args.lr},
            {'params': [p for n, p in pipeline.plm.named_parameters() if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0, 'lr': args.lr},
            {'params': pipeline.embed_vp.parameters(), 'weight_decay': args.weight_decay, 'lr': args.lr},
            {'params': pipeline.embed_ln.parameters(), 'weight_decay': args.weight_decay, 'lr': args.lr},
            # embed_multimodal and conv1d get their own group so --multimodal-lr-multiplier
            # can give them a different (typically higher) LR than the LoRA/networking_head/
            # embed_vp/embed_ln group above. conv1d was previously MISSING from this list
            # entirely (a real bug, not a design choice -- found 2026-08-14): it's the
            # trajectory encoder shared by every multimodal_mode, and leaving it out of the
            # optimizer meant it never trained, regardless of mode. It doesn't only feed the
            # multimodal path, but it's grouped with embed_multimodal here since both are the
            # "encoder-side" modules this experiment is testing a higher LR for.
            {'params': pipeline.embed_multimodal.parameters(), 'weight_decay': args.weight_decay, 'lr': multimodal_lr},
            {'params': pipeline.conv1d.parameters(), 'weight_decay': args.weight_decay, 'lr': multimodal_lr},
        ]
        optimizer = AdamW(optimizer_grouped_parameters)

    else:
        # tune everything except the plm itself: pipeline.modules_except_plm is exactly
        # embed_vp, embed_multimodal, embed_ln, conv1d, and plm.networking_head. All of
        # these must stay trainable even when the plm is frozen -- networking_head in
        # particular starts randomly initialized, so leaving it out of the optimizer (as
        # the previous version of this branch did) means it never learns anything and loss
        # can't meaningfully decrease regardless of how good the upstream features are.
        optimizer_grouped_parameters = [
            {'params': pipeline.modules_except_plm.parameters(), 'weight_decay': args.weight_decay, 'lr': args.lr},
        ]
        optimizer = AdamW(optimizer_grouped_parameters)

    # Clip exactly the parameters updated by the optimizer. This includes the
    # LoRA adapter and the trainable NetLLM heads/embeddings, not just the PLM.
    gradient_clip_parameters = [
        parameter
        for group in optimizer.param_groups
        for parameter in group['params']
        if parameter.requires_grad
    ]

    assert args.epochs_per_valid is None or args.steps_per_valid is None, "You can only specify args.epochs_per_valid or args.steps_per_valid."

    global_step = 0
    opt_step = 0  # AdaLoRA: counts actual optimizer update steps (post grad-accum), must match total_step
    report_loss_per_steps = args.report_loss_per_steps
    tot_loss = 0
    log_loss = 0
    best_loss = float('inf')
    best_post_nbs_loss = float('inf')
    best_epoch, best_step = 0, 0
    best_post_nbs_epoch, best_post_nbs_step = 0, 0
    saved_checkpoint_steps = {
        'best_ar': None,
        'best_post_nbs': None,
    }
    last_valid_loss = None
    last_teacher_forcing_valid_loss = None
    non_improving_validations = 0
    stop_training = False
    if args.early_stopping_patience is not None and args.early_stopping_patience <= 0:
        raise ValueError('--early-stopping-patience must be positive')
    if args.early_stopping_min_delta < 0:
        raise ValueError('--early-stopping-min-delta must be non-negative')

    def validate():
        pipeline.eval()
        with torch.no_grad():
            ps_history_start = len(pipeline.patch_selection_history) if args.multimodal_mode == 'patch-selection' else None
            autoregressive_losses = []
            teacher_forcing_losses = []
            for history, future, video_user_info in dataloader_valid:
                history, future = history.to(args.device), future.to(args.device)
                history = normalize_data(history, args.train_dataset)
                future = normalize_data(future, args.train_dataset)
                autoregressive_loss = pipeline(
                    history, future, video_user_info, teacher_forcing=False
                )
                teacher_forcing_loss = pipeline(
                    history, future, video_user_info, teacher_forcing=True
                )
                autoregressive_losses.append(autoregressive_loss.item())
                teacher_forcing_losses.append(teacher_forcing_loss.item())
            autoregressive_loss = sum(autoregressive_losses) / len(autoregressive_losses)
            teacher_forcing_loss = sum(teacher_forcing_losses) / len(teacher_forcing_losses)
            if ps_history_start is not None:
                counts = pipeline.patch_selection_history[ps_history_start:]
                if counts:
                    print(f'[patch-selection] valid selected-patch counts (n={len(counts)}): {counts}')
                    print(f'[patch-selection] valid selected-patch avg={sum(counts)/len(counts):.2f} '
                          f'min={min(counts)} max={max(counts)}')
            pipeline.train()
            return autoregressive_loss, teacher_forcing_loss

    def process_validation(valid_loss, teacher_forcing_valid_loss, position_kind, position):
        nonlocal best_loss, best_epoch, best_step, non_improving_validations
        nonlocal best_post_nbs_loss, best_post_nbs_epoch, best_post_nbs_step
        nonlocal last_valid_loss, last_teacher_forcing_valid_loss
        nonlocal validation_event_count
        validation_event_count += 1
        last_valid_loss = float(valid_loss)
        last_teacher_forcing_valid_loss = float(teacher_forcing_valid_loss)
        allocation_started = allocator is not None and allocator.last_step is not None
        improved = valid_loss < best_loss - args.early_stopping_min_delta
        if improved:
            best_loss = valid_loss
            if position_kind == 'step':
                best_step = position
            else:
                best_epoch = position
            non_improving_validations = 0
            save_model(args, pipeline, best_model_path, metadata={
                'checkpoint_role': 'best_ar',
                'optimizer_step': int(opt_step),
                'validation_event': int(validation_event_count),
                'validation_position_kind': position_kind,
                'validation_position': int(position),
                'validation_loss': float(valid_loss),
                'teacher_forcing_validation_loss': float(teacher_forcing_valid_loss),
                'nbs_allocation_started': bool(allocation_started),
            })
            saved_checkpoint_steps['best_ar'] = int(opt_step)
            print(
                f'Best model ({position_kind} {position}, average valid loss {best_loss}) '
                f'saved at', best_model_path
            )
        else:
            non_improving_validations += 1

        post_nbs_improved = (
            allocation_started
            and valid_loss < best_post_nbs_loss - args.early_stopping_min_delta
        )
        if post_nbs_improved:
            best_post_nbs_loss = float(valid_loss)
            if position_kind == 'step':
                best_post_nbs_step = position
            else:
                best_post_nbs_epoch = position
            post_metadata = {
                **_seed_metadata(args),
                'checkpoint_role': 'best_post_nbs',
                'optimizer_step': int(opt_step),
                'validation_event': int(validation_event_count),
                'validation_position_kind': position_kind,
                'validation_position': int(position),
                'validation_loss': float(valid_loss),
                'teacher_forcing_validation_loss': float(teacher_forcing_valid_loss),
                'nbs_allocation_started': True,
                'first_successful_allocation_step': int(allocator.last_step),
            }
            if saved_checkpoint_steps['best_ar'] == int(opt_step):
                save_checkpoint_alias(
                    best_post_nbs_model_path, best_model_path, post_metadata
                )
                storage_description = f'alias of {best_model_path}'
            else:
                save_model(
                    args, pipeline, best_post_nbs_model_path,
                    metadata=post_metadata,
                )
                storage_description = best_post_nbs_model_path
            saved_checkpoint_steps['best_post_nbs'] = int(opt_step)
            print(
                f'Best post-NBS model ({position_kind} {position}, average valid loss '
                f'{best_post_nbs_loss}) saved as', storage_description
            )

        if allocator is not None:
            validation_rows = allocator.snapshot_diagnostics(
                opt_step,
                event='best_validation' if improved else 'validation',
            )
            for row in validation_rows:
                row.update({
                    'validation_event': validation_event_count,
                    'validation_loss': float(valid_loss),
                    'teacher_forcing_validation_loss': float(teacher_forcing_valid_loss),
                    'validation_position_kind': position_kind,
                    'validation_position': position,
                    'is_best_validation': int(improved),
                    'nbs_allocation_started': int(allocation_started),
                    'is_best_post_nbs_validation': int(post_nbs_improved),
                })
            _append_nash_diagnostics(nash_diagnostics_path, validation_rows)
            snapshot_path = os.path.join(
                allocator_snapshot_dir,
                f'validation_{validation_event_count:04d}_{position_kind}_{position}.pt',
            )
            _save_allocator_snapshot(
                snapshot_path,
                allocator,
                validation_rows,
                {
                    'optimizer_step': int(opt_step),
                    'validation_event': int(validation_event_count),
                    'validation_position_kind': position_kind,
                    'validation_position': int(position),
                    'validation_loss': float(valid_loss),
                    'teacher_forcing_validation_loss': float(
                        teacher_forcing_valid_loss
                    ),
                    'is_best_validation': bool(improved),
                    'nbs_allocation_started': bool(allocation_started),
                    'is_best_post_nbs_validation': bool(post_nbs_improved),
                },
            )
            print('NBS allocator snapshot saved at', snapshot_path)

        best_position = best_step if position_kind == 'step' else best_epoch
        print('Teacher-forcing validation loss', teacher_forcing_valid_loss)
        print(
            'Valid loss', valid_loss, '-', 'Best loss', best_loss,
            'at', position_kind, best_position,
            '- non-improving validations', non_improving_validations,
        )
        should_stop = (
            args.early_stopping_patience is not None
            and non_improving_validations >= args.early_stopping_patience
        )
        if should_stop:
            print(
                'Early stopping triggered at', position_kind, position,
                f'(patience={args.early_stopping_patience}, '
                f'min_delta={args.early_stopping_min_delta}, best_loss={best_loss})'
            )
            if allocator is not None:
                allocator.enforce_masks()
                _append_nash_diagnostics(
                    nash_diagnostics_path,
                    allocator.snapshot_diagnostics(
                        opt_step, event='early_stopping'
                    ),
                )
        return should_stop
        
    print(
        f'Training on {args.train_dataset} - bs: {args.bs} - lr: {args.lr} '
        f'- seed: {args.seed} (LoRA={args.lora_seed}, data={args.data_seed})'
    )
    for epoch in range(args.epochs):
        pipeline.train()
        for step, (history, future, video_user_info) in enumerate(dataloader_train): 
            global_step += 1
            history, future = history.to(args.device), future.to(args.device)
            history = normalize_data(history, args.train_dataset)
            future = normalize_data(future, args.train_dataset)
            # using scheduled sampling
            if args.scheduled_sampling:
                if np.random.rand() > args.mix_rate:
                    loss = pipeline(history, future, video_user_info, teacher_forcing=True)
                else:
                    loss = pipeline.scheduled_sampling_loss(
                        history, future, video_user_info
                    )
            else:
                loss = pipeline(history, future, video_user_info, teacher_forcing=True)
            tot_loss += loss.item()
            loss = loss / grad_accum_steps
            loss.backward()

            # perform gradient accumulation update
            if ((step + 1) % grad_accum_steps == 0) or (step + 1 == len(dataloader_train)):
                if allocator is not None:
                    # Read the accumulated, unclipped A/B gradients first so
                    # sensitivity matches the raw gradient-norm definition.
                    allocator.update_sensitivity()
                # Clip once per effective batch, after NBS has measured the raw
                # accumulated A/B gradients and immediately before optimizer.step().
                torch.nn.utils.clip_grad_norm_(gradient_clip_parameters, 1.0)
                optimizer.step()
                if allocator is not None:
                    # Allocation/mask enforcement happens after optimizer.step
                    # and before zero_grad, while the next forward sees the
                    # selected rank mask.
                    current_opt_step = opt_step + 1
                    if allocator.should_allocate(current_opt_step):
                        try:
                            allocator.allocate(current_opt_step)
                        except RankAllocationConstraintError as exc:
                            # The candidate was rejected before any mask was
                            # changed. Preserve the latest spectral shadow from
                            # this optimizer update, then restore the previous
                            # valid mask and continue training.
                            allocator.enforce_masks()
                            violation_by_layer = {
                                violation['layer_name']: violation
                                for violation in exc.violations
                                if violation['layer_name'] is not None
                            }
                            error_rows = allocator.snapshot_diagnostics(
                                current_opt_step, event='allocation_error'
                            )
                            for row in error_rows:
                                violation = violation_by_layer.get(row['layer_name'])
                                row.update({
                                    'allocation_error': 1,
                                    'allocation_error_message': str(exc),
                                    'requested_rank': (
                                        violation['requested_rank']
                                        if violation is not None else ''
                                    ),
                                    'allocation_error_action': (
                                        'candidate rejected; previous valid mask preserved'
                                    ),
                                })
                            _append_nash_diagnostics(
                                nash_diagnostics_path, error_rows
                            )
                            print(
                                '[NBS ALLOCATION ERROR]',
                                f'optimizer_step={current_opt_step}', str(exc),
                                'Action: candidate rejected; previous valid rank mask preserved.',
                                flush=True,
                            )
                        else:
                            _append_nash_diagnostics(
                                nash_diagnostics_path, allocator.last_diagnostics
                            )
                    else:
                        # During warm-up, between allocation intervals, and
                        # throughout cooldown, preserve the existing topology.
                        # This also prevents a final-step allocation directly
                        # before validation.
                        allocator.enforce_masks()
                        if current_opt_step == allocator.warmup_steps:
                            _append_nash_diagnostics(
                                nash_diagnostics_path,
                                allocator.snapshot_diagnostics(
                                    current_opt_step, event='warmup_end'
                                ),
                            )
                        if current_opt_step == allocator.cooldown_start_step:
                            _append_nash_diagnostics(
                                nash_diagnostics_path,
                                allocator.snapshot_diagnostics(
                                    current_opt_step, event='cooldown_start'
                                ),
                            )
                    is_final_update = (
                        epoch == args.epochs - 1 and
                        step + 1 == len(dataloader_train)
                    )
                    if is_final_update:
                        _append_nash_diagnostics(
                            nash_diagnostics_path,
                            allocator.snapshot_diagnostics(
                                current_opt_step, event='training_end'
                            ),
                        )
                elif args.rank != -1 and args.use_adalora:
                    # Stock PEFT AdaLoRA baseline. This deliberately bypasses
                    # NashRankAllocator and delegates importance scoring and
                    # pruning to PEFT's own RankAllocator implementation.
                    current_opt_step = opt_step + 1
                    update_and_allocate = getattr(
                        pipeline.plm.base_model, 'update_and_allocate', None
                    )
                    if update_and_allocate is None:
                        raise RuntimeError(
                            'PEFT AdaLoRA model has no base_model.update_and_allocate()'
                        )
                    update_and_allocate(current_opt_step)
                optimizer.zero_grad()
                opt_step += 1

            # report training loss
            if global_step % report_loss_per_steps == 0:
                print("Epoch {}, global_step {}, average loss: {}".format(epoch, global_step, (tot_loss - log_loss) / report_loss_per_steps), flush=True)
                log_loss = tot_loss
            
            # for debug
            # if global_step >= 300:
            #     save_model(args, pipeline, best_model_path)
            #     break
            
            # validation by steps
            if args.steps_per_valid is not None and global_step % args.steps_per_valid == 0:
                valid_loss, teacher_forcing_valid_loss = validate()
                stop_training = process_validation(
                    valid_loss, teacher_forcing_valid_loss, 'step', global_step
                )
                if stop_training:
                    break
            
            # save checkpoint by save_checkpoint_per_step
            if args.save_checkpoint_per_step is not None and global_step % args.save_checkpoint_per_step == 0:
                save_checkpoint_path = os.path.join(checkpoint_path, str(global_step // args.save_checkpoint_per_step)) # save checkpoint
                if not os.path.exists(save_checkpoint_path):
                    os.makedirs(save_checkpoint_path)
                save_model(args, pipeline, save_checkpoint_path)
                print('save checkpoint at', save_checkpoint_path)

        if stop_training:
            break

        # validation by epochs
        if args.epochs_per_valid is not None and epoch % args.epochs_per_valid == 0:
            valid_loss, teacher_forcing_valid_loss = validate()
            stop_training = process_validation(
                valid_loss, teacher_forcing_valid_loss, 'epoch', epoch
            )
            if stop_training:
                break
        
        # save checkpoint by save_checkpoint_per_epoch
        if args.save_checkpoint_per_epoch is not None and epoch % args.save_checkpoint_per_epoch == 0 and epoch > 0:
            save_checkpoint_path = os.path.join(checkpoint_path, f'epoch{epoch}') # save checkpoint
            if not os.path.exists(save_checkpoint_path):
                os.makedirs(save_checkpoint_path)
            save_model(args, pipeline, save_checkpoint_path)
            print('save checkpoint at', save_checkpoint_path)

    if allocator is not None:
        allocator.enforce_masks()
        final_rows = allocator.snapshot_diagnostics(opt_step, event='final_nbs_model')
        _append_nash_diagnostics(nash_diagnostics_path, final_rows)
        post_training_snapshot_path = os.path.join(
            allocator_snapshot_dir, 'post_training_spectral_shadow.pt'
        )
        _save_allocator_snapshot(
            post_training_snapshot_path,
            allocator,
            diagnostics=final_rows,
            metadata={
                **_seed_metadata(args),
                'snapshot_kind': 'post_training',
                'optimizer_step': int(opt_step),
                'global_step': int(global_step),
                'validation_event': int(validation_event_count),
                'validation_loss': last_valid_loss,
                'teacher_forcing_validation_loss': last_teacher_forcing_valid_loss,
                'nbs_allocation_started': bool(allocator.last_step is not None),
            },
        )
        print(
            'Post-training spectral shadow snapshot saved at',
            post_training_snapshot_path,
        )
        final_metadata = {
            **_seed_metadata(args),
            'checkpoint_role': 'final_nbs',
            'optimizer_step': int(opt_step),
            'global_step': int(global_step),
            'validation_event': int(validation_event_count),
            'validation_loss': last_valid_loss,
            'teacher_forcing_validation_loss': last_teacher_forcing_valid_loss,
            'nbs_allocation_started': bool(allocator.last_step is not None),
            'last_successful_allocation_step': allocator.last_step,
            'stopped_early': bool(stop_training),
        }
        if saved_checkpoint_steps['best_ar'] == int(opt_step):
            save_checkpoint_alias(
                final_nbs_model_path, best_model_path, final_metadata
            )
            final_storage_description = f'alias of {best_model_path}'
        elif saved_checkpoint_steps['best_post_nbs'] == int(opt_step):
            save_checkpoint_alias(
                final_nbs_model_path, best_post_nbs_model_path, final_metadata
            )
            final_storage_description = f'alias of {best_post_nbs_model_path}'
        else:
            save_model(
                args, pipeline, final_nbs_model_path, metadata=final_metadata
            )
            final_storage_description = final_nbs_model_path
        print('Final NBS model saved as', final_storage_description)
    elif (args.rank != -1 and args.use_adalora and
          args.adalora_allocator == 'shapley'):
        save_model(
            args,
            pipeline,
            final_shapley_model_path,
            metadata={
                'checkpoint_role': 'final_shapley',
                'optimizer_step': int(opt_step),
                'validation_loss': last_valid_loss,
                'teacher_forcing_validation_loss': (
                    last_teacher_forcing_valid_loss
                ),
            },
        )
        print('Final Shapley AdaLoRA model saved at', final_shapley_model_path)

    print('Done adaptation, average training loss =', tot_loss / global_step)


def test(args, pipeline, dataloader_test, models_dir, results_dir):
    seed_path_fragment = _seed_path_fragment(args)
    file_prefix = f'his_{args.his_window}_fut_{args.fut_window}_axes_ss_{args.sample_step}_epochs_{args.epochs}_bs_{args.bs * args.grad_accum_steps}_'\
                  f'lr_{args.lr}_{seed_path_fragment}_rank_{args.rank}_scheduled_sampling_{args.scheduled_sampling}'
    default_model_name = 'best_ar_model' if args.use_adalora else 'best_model'
    best_model_path = os.path.join(models_dir, file_prefix, default_model_name)
    evaluation_suffix = (
        f'_checkpoint_{args.evaluation_tag}' if args.evaluation_tag is not None else ''
    )
    if args.inference_tag is not None:
        evaluation_suffix += f'_inference_{args.inference_tag}'
    if args.nbs_inference_mode == 'compact':
        evaluation_suffix += '_nbs_compact'
    result_path = os.path.join(
        results_dir, file_prefix + evaluation_suffix + '_results.csv'
    )
    partial_result_path = os.path.join(
        results_dir, file_prefix + evaluation_suffix + '_partial_results.csv'
    )
    progress_interval = getattr(args, 'save_test_progress_per_steps', None)
    if progress_interval is not None and progress_interval <= 0:
        raise ValueError('--save-test-progress-per-steps must be positive')
    notebook = ResultNotebook()
    measure_latency = bool(getattr(args, 'measure_inference_latency', False))
    latency_warmup_steps = int(getattr(args, 'latency_warmup_steps', 5))
    latency_deadline_s = float(getattr(args, 'latency_deadline_ms', 1000.0)) / 1000.0
    if latency_warmup_steps < 0:
        raise ValueError('--latency-warmup-steps must be non-negative')
    if latency_deadline_s <= 0:
        raise ValueError('--latency-deadline-ms must be positive')
    latency_path = getattr(args, 'latency_output_path', None) or result_path.replace(
        '_results.csv', '_latency.json'
    )
    partial_latency_path = latency_path.replace('.json', '_partial.json')
    latency_elapsed_s = []
    latency_records = []
    latency_peak_memory_mb = float('nan')
    inference_trace_records = []
    inference_trace_path = (
        args.inference_trace_output_path
        or result_path.replace('_results.csv', '_inference_trace.json')
    )

    model_path = args.model_path if args.model_path is not None else best_model_path
    compact_loaded_directly = False
    compact_output_dir = args.nbs_compact_output_dir
    if os.path.exists(model_path):
        compact_state_path = os.path.join(
            _resolve_checkpoint_alias(model_path), 'compact_adapter.pt'
        )
        if args.nbs_inference_mode == 'compact' and os.path.isfile(compact_state_path):
            pipeline, compact_report = load_compact_nbs_model(pipeline, model_path)
            compact_loaded_directly = True
            print('Load compact NBS weights from:', model_path)
            print(
                'Compact NBS ranks: physical total {} -> compact total {}'.format(
                    compact_report['physical_rank_total_before'],
                    compact_report['compact_rank_total'],
                )
            )
        else:
            pipeline = load_model(args, pipeline, model_path)
            print('Load weights from:', model_path)
    else:
        print('\033[33mWarning:\033[0m', model_path, 'not found, skip loading weights.')

    if args.nbs_inference_mode == 'compact':
        if (args.nbs_compaction_rtol < 0 or args.nbs_compaction_atol < 0 or
                args.nbs_compaction_output_rtol < 0 or
                args.nbs_compaction_output_atol < 0):
            raise ValueError('NBS compaction tolerances must be non-negative')
        if not args.use_adalora or args.adalora_allocator != 'nbs':
            raise ValueError(
                '--nbs-inference-mode compact requires NBS AdaLoRA '
                '(--use-adalora --adalora-allocator nbs)'
            )
        if args.adapt:
            raise ValueError(
                '--nbs-inference-mode compact is evaluation-only; run training '
                'and compact evaluation as separate commands'
            )
        if not compact_loaded_directly and not os.path.exists(model_path):
            raise FileNotFoundError(
                'compact inference requires an existing source NBS checkpoint'
            )
        if compact_output_dir is None and not compact_loaded_directly:
            compact_output_dir = _resolve_checkpoint_alias(model_path) + '_compact'

    print(
        f'Testing on {args.test_dataset} - seed: {args.seed} '
        f'(LoRA={args.lora_seed}, data={args.data_seed})'
    )
    # Real accuracy bug, found 2026-08-14: this was missing entirely, so a `--adapt --test`
    # invocation in one command evaluated with the pipeline still in train() mode from the
    # end of adapt() (dropout active everywhere -- patch_selection_module's 0.1, LoRA's
    # 0.05 -- and BatchNorm-like layers, if any, using batch stats instead of running
    # stats). Confirmed NOT the cause of the patch-selection MAE investigation's numbers
    # (those came from a separate script that already called pipeline.eval() correctly),
    # but a real correctness issue for anyone using --adapt --test together.
    pipeline.eval()
    if args.nbs_inference_mode == 'compact' and not compact_loaded_directly:
        # Validate on one real viewport example before any inference wrapper is
        # installed.  These two calls are setup work and are intentionally not
        # included in the reported latency samples.
        try:
            validation_batch = next(iter(dataloader_test))
        except StopIteration:
            raise ValueError('cannot validate NBS compaction on an empty test set')
        validation_history, validation_future, validation_info = validation_batch
        validation_history = validation_history.to(args.device)
        validation_future = validation_future.to(args.device)
        validation_history = normalize_data(validation_history, args.train_dataset)
        specs, _ = extract_nbs_compaction_specs(pipeline.plm)
        factor_validation = validate_nbs_compaction_factors(
            pipeline.plm,
            specs,
            rtol=args.nbs_compaction_rtol,
            atol=args.nbs_compaction_atol,
        )
        save_nbs_compact_checkpoint(
            pipeline.plm,
            compact_output_dir,
            modules_except_plm_state=pipeline.modules_except_plm.state_dict(),
            source_checkpoint=_resolve_checkpoint_alias(model_path),
            factor_validation=factor_validation,
        )
        with torch.no_grad():
            original_prediction, _ = pipeline.inference(
                validation_history, validation_future, validation_info
            )
        compact_report = compact_nbs_model_for_inference(pipeline.plm)
        pipeline.eval()
        with torch.no_grad():
            compact_prediction, _ = pipeline.inference(
                validation_history, validation_future, validation_info
            )
        difference = (compact_prediction.float() - original_prediction.float()).abs()
        denominator = original_prediction.float().abs().clamp_min(
            args.nbs_compaction_output_atol
        )
        full_output_validation = {
            'passed': False,
            'rtol': float(args.nbs_compaction_output_rtol),
            'atol': float(args.nbs_compaction_output_atol),
            'max_abs_error': float(difference.max().item()),
            'mean_abs_error': float(difference.mean().item()),
            'rmse': float(difference.square().mean().sqrt().item()),
            'max_rel_error': float((difference / denominator).max().item()),
            'output_shape': list(original_prediction.shape),
            'dataset': args.test_dataset,
        }
        try:
            torch.testing.assert_close(
                compact_prediction.float(), original_prediction.float(),
                rtol=args.nbs_compaction_output_rtol,
                atol=args.nbs_compaction_output_atol,
            )
        except AssertionError:
            write_nbs_compaction_validation(
                compact_output_dir, factor_validation, full_output_validation
            )
            raise
        full_output_validation['passed'] = True
        equivalence = write_nbs_compaction_validation(
            compact_output_dir, factor_validation, full_output_validation
        )
        print('Compact NBS checkpoint saved at:', compact_output_dir)
        print(
            'Compact NBS ranks: physical total {} -> compact total {}'.format(
                compact_report['physical_rank_total_before'],
                compact_report['compact_rank_total'],
            )
        )
        print(
            'NBS compaction equivalence passed: max abs error={:.3e}, '
            'max relative error={:.3e}'.format(
                equivalence['full_output_validation']['max_abs_error'],
                equivalence['full_output_validation']['max_rel_error'],
            )
        )
    if args.speculative_gamma is not None:
        selector = (
            RecentKSelector(k=args.selector_recent_k)
            if args.selector_recent_k is not None else None
        )
        pipeline = LlamaSpeculativeBlockVerifyPipeline(
            pipeline,
            selector=selector,
            gamma=args.speculative_gamma,
            acceptance_threshold=args.speculative_threshold,
        )
        pipeline.eval()
        print(
            '[inference] speculative decoding enabled: '
            f'gamma={args.speculative_gamma}, threshold={args.speculative_threshold}, '
            f'selector_recent_k={args.selector_recent_k}'
        )
    elif args.selector_recent_k is not None:
        pipeline = LlamaSelectablePipeline(
            pipeline, selector=RecentKSelector(k=args.selector_recent_k)
        )
        pipeline.eval()
        print(f'[inference] RecentK selector enabled: k={args.selector_recent_k}')
    test_step = 0
    with torch.no_grad():
        for test_step, (history, future, video_user_info) in enumerate(dataloader_test, start=1):
            history, future = history.to(args.device), future.to(args.device)
            history = normalize_data(history, args.train_dataset)
            timed_call = measure_latency and test_step > latency_warmup_steps
            if timed_call and history.is_cuda:
                torch.cuda.synchronize(history.device)
                if not latency_elapsed_s:
                    torch.cuda.reset_peak_memory_stats(history.device)
            started_at = time.perf_counter() if timed_call else None
            pred, gt = pipeline.inference(history, future, video_user_info)
            if args.inference_tag is not None:
                trace = getattr(pipeline, 'last_trace', {}) or {}
                initial_shape = trace.get('initial_sequence_shape_before_selection')
                initial_count = (
                    int(initial_shape[1])
                    if isinstance(initial_shape, (list, tuple)) and len(initial_shape) > 1
                    else trace.get('initial_token_count')
                )
                selected_count = trace.get('selected_length')
                accepted_per_iteration = trace.get('accepted_per_iteration') or []
                proposed_per_iteration = trace.get('proposed_per_iteration') or []
                accepted = int(sum(accepted_per_iteration))
                proposed = int(sum(proposed_per_iteration))
                inference_trace_records.append({
                    'test_step': int(test_step),
                    'video': int(video_user_info[0]),
                    'user': int(video_user_info[1]),
                    'timestep': int(video_user_info[2]),
                    'selector_enabled': int(bool(trace.get('selector_enabled'))),
                    'initial_token_count': initial_count,
                    'selected_token_count': selected_count,
                    'token_reduction_percent': (
                        (initial_count - selected_count) / initial_count * 100.0
                        if initial_count not in (None, 0) and selected_count is not None
                        else None
                    ),
                    'target_forward_count': trace.get(
                        'target_forward_count', trace.get('plm_forward_count')
                    ),
                    'accepted_drafts': accepted if proposed_per_iteration else None,
                    'proposed_drafts': proposed if proposed_per_iteration else None,
                    'draft_acceptance_rate': accepted / proposed if proposed > 0 else None,
                })
            if timed_call:
                if history.is_cuda:
                    torch.cuda.synchronize(history.device)
                elapsed_s = time.perf_counter() - started_at
                latency_elapsed_s.append(elapsed_s)
                batch_size = int(history.shape[0]) if history.ndim > 0 else 1
                latency_records.append({
                    'test_step': int(test_step),
                    'video': int(video_user_info[0]),
                    'user': int(video_user_info[1]),
                    'timestep': int(video_user_info[2]),
                    'batch_size': batch_size,
                    'latency_ms': elapsed_s * 1000.0,
                })
                if history.is_cuda:
                    latency_peak_memory_mb = (
                        torch.cuda.max_memory_allocated(history.device) / (1024 ** 2)
                    )
            pred = denormalize_data(pred, args.test_dataset)
            videos, users, timesteps = [], [], []
            videos.append(int(video_user_info[0]))
            users.append(int(video_user_info[1]))
            timesteps.append(int(video_user_info[2]))
            videos, users, timesteps = torch.IntTensor(videos), torch.IntTensor(users), torch.IntTensor(timesteps)
            notebook.record(pred, gt, videos, users, timesteps)
            if progress_interval is not None and test_step % progress_interval == 0:
                notebook.write(partial_result_path, write_predictions=False)
                print(f'Partial evaluation results saved after {test_step} batches at {partial_result_path}')
                if measure_latency:
                    write_latency_artifacts(
                        partial_latency_path,
                        latency_elapsed_s,
                        latency_records,
                        deadline_s=latency_deadline_s,
                        warmup_calls=min(test_step, latency_warmup_steps),
                        total_calls=test_step,
                        peak_memory_mb=latency_peak_memory_mb,
                    )
                    print(
                        f'Partial latency results saved after {test_step} batches at '
                        f'{partial_latency_path}'
                    )
                if args.inference_tag is not None:
                    partial_trace_path = inference_trace_path.replace(
                        '.json', '_partial.json'
                    )
                    _write_inference_trace_artifacts(
                        partial_trace_path, inference_trace_records,
                        args.inference_tag,
                    )
        notebook.write(result_path)
        print("show detail result:")
        detail_result_path = result_path.replace('_results.csv', '_per_sample_results.csv')
        notebook.write_detail(detail_result_path)
        if args.inference_tag is not None:
            trace_summary, trace_detail_path = _write_inference_trace_artifacts(
                inference_trace_path, inference_trace_records, args.inference_tag
            )
            print('Inference trace summary saved at', inference_trace_path)
            print('Per-sample inference trace saved at', trace_detail_path)
            print('Inference trace summary:', trace_summary)
        if measure_latency:
            latency_summary, latency_detail_path = write_latency_artifacts(
                latency_path,
                latency_elapsed_s,
                latency_records,
                deadline_s=latency_deadline_s,
                warmup_calls=min(test_step, latency_warmup_steps),
                total_calls=test_step,
                peak_memory_mb=latency_peak_memory_mb,
            )
            print('Latency summary saved at', latency_path)
            print('Per-sample latency saved at', latency_detail_path)
            if latency_summary['mean_s'] is not None:
                print(
                    'Inference latency: mean={:.3f} ms, median={:.3f} ms, '
                    'p95={:.3f} ms over {} calls'.format(
                        latency_summary['mean_s'] * 1000.0,
                        latency_summary['median_s'] * 1000.0,
                        latency_summary['p95_s'] * 1000.0,
                        latency_summary['measured_calls'],
                    )
                )


def run(args):
    args.seed, args.lora_seed, args.data_seed = resolve_experiment_seeds(
        args.seed,
        getattr(args, 'lora_seed', None),
        getattr(args, 'data_seed', None),
    )
    assert args.train_dataset in cfg.dataset_list
    assert args.test_dataset in cfg.dataset_list
    assert args.plm_type in cfg.plm_types
    assert args.plm_size in cfg.plm_sizes
    assert args.trim_head >= args.his_window and args.trim_tail >= args.fut_window

    # The master seed controls the rest of the model/training stochasticity.
    # Dataset ordering and LoRA initialization use isolated component seeds below.
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    print(
        f'Experiment seeds: master={args.seed}, LoRA={args.lora_seed}, '
        f'data={args.data_seed}'
    )
    # multimodal_mode is part of the checkpoint/result path so that baseline/all-patch/
    # patch-selection runs (otherwise identical hyperparameters) never collide on the same
    # directory -- without this, two modes trained with the same his/fut window, epochs,
    # bs, lr, seed, and rank would silently write over each other's checkpoints.
    multimodal_tag = f'multimodal_{args.multimodal_mode}'
    if args.rank != -1:
        # '_adalora' suffix so an AdaLoRA run never collides with a plain-LoRA run at the
        # same rank/hyperparams (same directory formula otherwise) -- same convention as
        # the multimodal_tag above, for the same reason.
        low_rank_tag = f'{args.plm_type}_{args.plm_size}_low_rank' + ('_adalora' if args.use_adalora else '')
        experiment_tag = getattr(args, 'experiment_tag', None)
        if experiment_tag is not None:
            low_rank_tag += f'_{experiment_tag}'
        models_dir = os.path.join(cfg.plms_finetuned_dir, low_rank_tag,
                              f'freeze_plm_{args.freeze_plm}', multimodal_tag, args.train_dataset, f'{args.dataset_frequency}Hz')
        results_dir = os.path.join(cfg.results_dir, low_rank_tag,
                               f'freeze_plm_{args.freeze_plm}', multimodal_tag, args.test_dataset, f'{args.dataset_frequency}Hz')
    else:
        models_dir = os.path.join(cfg.plms_finetuned_dir, f'{args.plm_type}_{args.plm_size}',
                              f'freeze_plm_{args.freeze_plm}', multimodal_tag, args.train_dataset, f'{args.dataset_frequency}Hz')
        results_dir = os.path.join(cfg.results_dir, f'{args.plm_type}_{args.plm_size}',
                               f'freeze_plm_{args.freeze_plm}', multimodal_tag, args.test_dataset, f'{args.dataset_frequency}Hz')
    if args.experiment_run_id is not None:
        # Long ablation runs must not reuse stale checkpoints/results from an
        # earlier invocation with the same variant and hyperparameter prefix.
        models_dir = os.path.join(models_dir, args.experiment_run_id)
        results_dir = os.path.join(results_dir, args.experiment_run_id)
    if args.results_output_dir is not None:
        results_dir = args.results_output_dir
    if not os.path.exists(models_dir): 
        os.makedirs(models_dir)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    # args.device_out and args.device_mid are used used for model parallelism (currently only necessary for llama) 
    # For data/modules near the input side, we use args.device.
    # For data/modules near the output side, we use args.device_out.
    # For data/modules lying in the middle, we use args.device_mid (it can be None). 
    # If args.device == args.device_out == args.device_mid (if not None), everything will be the same as using only one device.
    plm, tokenizer, _ = load_plm(args.plm_type, os.path.join(cfg.plms_dir, args.plm_type, args.plm_size), plm_size=args.plm_size,
                                     device_input_side=args.device, device_output_side=args.device_out, device_middle_side=args.device_mid,
                                     torch_dtype=torch.float16 if args.fp16 else None)
    if (args.plm_type == 'opt' or args.plm_type == 'gpt2') and args.plm_size!= 'large':  # other plm can simply be loaded on one device
        plm = plm.to(args.device)

    if args.rank == -1 and args.freeze_plm:
        # peft_model() (rank != -1 path) already does this as part of LoRA setup; when
        # LoRA is disabled we still want the plm fully frozen (no grad buffers) rather than
        # merely excluded from the optimizer, since those buffers double the plm's memory
        # footprint for no benefit.
        for p in plm.parameters():
            p.requires_grad_(False)

    if args.gradient_checkpointing:
        # opt-in memory/engineering trade-off only -- does not change what gets trained.
        plm.gradient_checkpointing_enable()
        plm.enable_input_require_grads()

    # AdaLoRA's rank-pruning schedule needs total_step (number of optimizer update
    # steps) at wrap time, so the train/valid dataloaders -- normally built just
    # before adapt() -- are built here instead when adapting. This is a pure
    # reordering (create_dataset()/DataLoader() don't depend on plm/pipeline), so
    # it changes nothing for the non-adapt or non-AdaLoRA paths.
    dataloader_train = dataloader_valid = None
    total_step = 1  # placeholder; only used by AdaLoRA, and only meaningful when args.adapt
    if args.adapt:
        with isolated_seed(args.data_seed, include_cuda=False):
            raw_dataset_train, raw_dataset_valid = create_dataset(
                args.train_dataset,
                his_window=args.his_window,
                fut_window=args.fut_window,
                trim_head=args.trim_head,
                trim_tail=args.trim_tail,
                include=['train', 'valid'],
                frequency=args.dataset_frequency,
                step=args.sample_step,
            )

        if args.limit_train_samples is not None:
            n = min(args.limit_train_samples, len(raw_dataset_train))
            raw_dataset_train = torch.utils.data.Subset(raw_dataset_train, range(n))
            print(f'\033[33mDebug:\033[0m truncated training set to {n} samples (--limit-train-samples).')
        if args.limit_valid_samples is not None:
            n = min(args.limit_valid_samples, len(raw_dataset_valid))
            raw_dataset_valid = torch.utils.data.Subset(raw_dataset_valid, range(n))
            print(f'\033[33mDebug:\033[0m truncated validation set to {n} samples (--limit-valid-samples).')

        dataloader_train = DataLoader(
            raw_dataset_train,
            batch_size=args.bs,
            shuffle=True,
            pin_memory=True,
            generator=make_data_generator(args.data_seed, offset=0),
            worker_init_fn=seed_data_worker,
        )
        dataloader_valid = DataLoader(
            raw_dataset_valid,
            batch_size=args.bs,
            shuffle=False,
            pin_memory=True,
            generator=make_data_generator(args.data_seed, offset=1),
            worker_init_fn=seed_data_worker,
        )
        steps_per_epoch = math.ceil(len(dataloader_train) / args.grad_accum_steps)
        total_step = steps_per_epoch * args.epochs
        if args.use_adalora:
            print(f'[AdaLoRA] total optimizer steps = {total_step} '
                  f'(steps_per_epoch={steps_per_epoch}, epochs={args.epochs})')

    adalora_rank_config = None
    if args.adalora_rank_config is not None:
        with open(args.adalora_rank_config, 'r', encoding='utf-8') as handle:
            adalora_rank_config = json.load(handle)

    eva_state = None
    if args.use_eva:
        if args.rank == -1:
            raise ValueError('--use-eva requires --rank to enable LoRA')
        if args.use_adalora:
            raise ValueError('--use-eva and --use-adalora are mutually exclusive')
        if args.lora_rank_config is not None:
            raise ValueError('--use-eva cannot be combined with --lora-rank-config')
        if not args.eva_state_path:
            raise ValueError('--use-eva requires --eva-state-path')
        eva_state_path = args.eva_state_path
        if os.path.isdir(eva_state_path):
            eva_state_path = os.path.join(eva_state_path, 'eva_state.pt')
        if not os.path.isfile(eva_state_path):
            raise FileNotFoundError(f'EVA state not found: {eva_state_path}')
        eva_state = torch.load(eva_state_path, map_location='cpu')
        validate_eva_state(eva_state)
        print(
            f'[EVA] loaded {eva_state_path}: '
            f'modules={len(eva_state["rank_pattern"])}, '
            f'positive modules={sum(int(rank) > 0 for rank in eva_state["rank_pattern"].values())}, '
            f'total active rank={eva_state["total_rank_budget"]}'
        )

    lora_rank_pattern = None
    if args.lora_rank_config is not None:
        if args.use_adalora:
            raise ValueError('--lora-rank-config is only supported by fixed plain LoRA')
        with open(args.lora_rank_config, 'r', encoding='utf-8') as handle:
            fixed_rank_config = json.load(handle)
        lora_rank_pattern = fixed_rank_config.get('rank_pattern')
        if not isinstance(lora_rank_pattern, dict) or not lora_rank_pattern:
            raise ValueError('LoRA rank config must contain a non-empty rank_pattern dictionary')
        configured_budget = sum(int(value) for value in lora_rank_pattern.values())
        expected_budget = fixed_rank_config.get('total_rank_budget')
        if expected_budget is not None and configured_budget != int(expected_budget):
            raise ValueError(
                f'LoRA rank pattern sums to {configured_budget}, expected {expected_budget}'
            )
        print(
            f'[LoRA] fixed rank pattern: modules={len(lora_rank_pattern)}, '
            f'total active rank={configured_budget}, '
            f'range=[{min(map(int, lora_rank_pattern.values()))}, '
            f'{max(map(int, lora_rank_pattern.values()))}]'
        )

    if args.rank != -1:
        with isolated_seed(args.lora_seed, include_cuda=True):
            plm = peft_model(
                plm, args.plm_type, args.rank,
                use_adalora=args.use_adalora,
                total_step=total_step,
                adalora_min_rank=args.adalora_min_rank,
                adalora_ema_beta=args.adalora_ema_beta,
                adalora_eps=args.adalora_eps,
                adalora_allocation_interval=args.adalora_allocation_interval,
                adalora_rank_budget=args.adalora_rank_budget,
                adalora_rank_config=adalora_rank_config,
                adalora_missing_grad_policy=args.adalora_missing_grad_policy,
                lora_rank_pattern=lora_rank_pattern,
                adalora_allocator=args.adalora_allocator,
                adalora_shadow_update_policy=args.adalora_shadow_update_policy,
                adalora_budget_mode=args.adalora_budget_mode,
                adalora_relative_lambda=args.adalora_relative_lambda,
                adalora_adaptive_min_budget=args.adalora_adaptive_min_budget,
                adalora_adaptive_max_budget=args.adalora_adaptive_max_budget,
                shapley_permutations=args.shapley_permutations,
                shapley_truncate_fraction=args.shapley_truncate_fraction,
                shapley_antithetic=args.shapley_antithetic,
                shapley_seed=args.lora_seed,
                eva_state=eva_state,
            )

    # set up networking head
    input_dim = plm.hidden_size
    out_dim = 3  # = the number of viewport coordinates
    if args.plm_type == 'opt' and args.plm_size == 'xxs':
        networking_head = NetworkingHead(input_dim=512, output_dim=out_dim, fut_window=args.fut_window).to(args.device_out)
    else:
        networking_head = NetworkingHead(input_dim=input_dim, output_dim=out_dim, fut_window=args.fut_window).to(args.device_out)
    plm.set_networking_head(networking_head)
    print('PLM model architecture:')
    print(plm)
    
    if args.plm_type == 'gpt2':
        embed_size = 1024
    if args.plm_type == 'llama':
        embed_size = 4096
    if args.plm_type == 'mistral':
        embed_size = 4096
    if args.plm_type == 'opt' and args.plm_size == 'xxs':
        embed_size = 512
    if args.plm_type == 'opt' and args.plm_size == 'xs':
        embed_size = 2048
    if args.plm_type == 'opt' and args.plm_size == 'small':
        embed_size = 2560
    if args.plm_type == 'opt' and args.plm_size == 'base':
        embed_size = 4096
    if args.plm_type == 'opt' and args.plm_size == 'large':
        embed_size = 5120
    if args.plm_type == 'llava':
        embed_size = 4096

    patch_selection_module = None
    if args.multimodal_mode == 'patch-selection':
        patch_selection_module = PatchSelectionModule(grid_rows=cfg.default_patch_grid[0], grid_cols=cfg.default_patch_grid[1]).to(args.device)
        if args.patch_selection_weights:
            patch_selection_module.load_state_dict(torch.load(args.patch_selection_weights, map_location=args.device))
            patch_selection_module.eval()
        else:
            print('\033[33mWarning:\033[0m --multimodal-mode patch-selection was set without --patch-selection-weights; '
                  'using a freshly-initialized (UNTRAINED) patch selection module.')

    pipeline = Pipeline(plm, fut_window=args.fut_window, device=args.device, embed_size=embed_size, frequency=args.dataset_frequency,
                         multimodal_mode=args.multimodal_mode, dataset=args.train_dataset,
                         patch_selection_module=patch_selection_module,
                         patch_top_k=args.patch_top_k, patch_threshold=args.patch_threshold)
    # print_trainable_parameters(pipeline)

    if args.compile:
        assert torch.__version__ >= '2.0.0', 'Compile model requires torch version >= 2.0.0, but current torch version is ' + torch.__version__
        print("\033[33mWarning:\033[0m There seems to be some bugs in torch.compile. If batch size is too large, it will raise errors (I don't know why this happens).")
        prompt_model = torch.compile(prompt_model).to(args.device)  # recommend to compile model when you are using PyTorch 2.0
    
    torch.set_float32_matmul_precision('high')

    if args.adapt:
        adapt(args, pipeline, dataloader_train, dataloader_valid, models_dir, args.grad_accum_steps)

    if args.test:
        test_video_split = copy.deepcopy(cfg.dataset_video_split[args.test_dataset])
        short_videos = set(cfg.dataset_short_frame_videos.get(args.test_dataset, []))
        present_short = [v for v in test_video_split['test'] if v in short_videos]
        if present_short:
            if args.exclude_short_videos_test:
                test_video_split['test'] = [v for v in test_video_split['test'] if v not in short_videos]
                print(f'\033[33mWarning:\033[0m excluded short-frame-count videos {present_short} from the '
                      f'test split (--exclude-short-videos-test was set).')
            else:
                print(f'\033[33mWarning:\033[0m test split includes short-frame-count videos {present_short}; '
                      f'their tail frames are clamp-repeated (see utils/frame_utils.py), which may distort MAE. '
                      f'Pass --exclude-short-videos-test to drop them from the test split instead.')

        with isolated_seed(args.data_seed, include_cuda=False):
            raw_dataset_test = create_dataset(
                args.test_dataset,
                dataset_video_split=test_video_split,
                his_window=args.his_window,
                fut_window=args.fut_window,
                trim_head=args.trim_head,
                trim_tail=args.trim_tail,
                include=['test'],
                frequency=args.dataset_frequency,
                step=args.sample_step,
            )[0]

        if args.limit_test_samples is not None:
            if args.limit_test_samples <= 0:
                raise ValueError('--limit-test-samples must be positive')
            n = min(args.limit_test_samples, len(raw_dataset_test))
            raw_dataset_test = torch.utils.data.Subset(raw_dataset_test, range(n))
            print(f'\033[33mDebug:\033[0m truncated test set to {n} samples (--limit-test-samples).')

        dataloader_test = DataLoader(
            raw_dataset_test,
            batch_size=args.bs,
            shuffle=True,
            pin_memory=True,
            generator=make_data_generator(args.data_seed, offset=2),
            worker_init_fn=seed_data_worker,
        )
        test(args, pipeline, dataloader_test, models_dir, results_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process the input parameters to train the network.')
    
    # ========== model/plm settings related arguments ==========
    parser.add_argument('--adapt', action="store_true", help='adapt llm.')
    parser.add_argument('--test', action="store_true", help='test llm.')
    parser.add_argument('--plm-type', action="store", dest='plm_type', help='type of plm.', default='t5-lm')
    parser.add_argument('--plm-size', action="store", dest='plm_size', help='size of plm.', default='base')
    parser.add_argument('--model-path', action="store", dest='model_path', type=str, help='(Optional) The directory of model weights to be loaded for testing.')
    parser.add_argument(
        '--evaluation-tag',
        choices=['best_ar', 'best_post_nbs', 'final_nbs', 'final_shapley'],
        default=None,
        help='Optional checkpoint role appended to evaluation result filenames.',
    )
    parser.add_argument(
        '--inference-tag',
        choices=['selector', 'speculative', 'full_stack'],
        default=None,
        help='Optional suffix isolating results produced by an inference wrapper.',
    )
    parser.add_argument(
        '--results-output-dir', type=str, default=None,
        help='Optional explicit result directory for evaluation-only ablations.',
    )
    parser.add_argument('--device', action='store', dest='device', help='the device (cuda or cpu) to run experiment.')
    parser.add_argument('--device-out', action='store', dest='device_out', help='the device (cuda or cpu) to place the split of model near the output.')
    parser.add_argument('--device-mid', action='store', dest='device_mid', help='the device (cuda or cpu) to place the split of model between the input and output.')
    parser.add_argument('--freeze-plm', action='store_true', dest='freeze_plm', help='freeze weights of plm during training')
    parser.add_argument('--fp16', action='store_true', dest='fp16', help='(Optional) Load the plm weights in fp16 (no quantization, no adapters). '
                                                                          'Intended for frozen-plm pilots where only the multimodal/patch-selection '
                                                                          'modules need gradients; see Pipeline for the fp32<->fp16 bridging at the plm boundary.')
    parser.add_argument('--gradient-checkpointing', action='store_true', dest='gradient_checkpointing',
                        help='(Optional) Enable gradient checkpointing on the plm to trade compute for activation memory. '
                             'Pure memory-saving trick (no effect on results); use only if you hit OOM.')
    parser.add_argument('--compile', action='store_true', dest='compile', help='(Optional) Compile model for speed up (available only for PyTorch 2.0).')
    parser.add_argument('--resume', action='store_true', dest='resume', help='(Optional) Resume model weights from checkpoint for training.')
    
    # ========== dataset settings related arguments ==========
    parser.add_argument('--train-dataset', action='store', dest='train_dataset', help='Dataset for training.')
    parser.add_argument('--test-dataset', action='store', dest='test_dataset', help='Dataset for testing.')

    # ========== dataset loading/processing settings related arguments ==========
    parser.add_argument('--his-window', action='store', dest='his_window',
                        help='(Optional) Historical window (default 10)', type=int)
    parser.add_argument('--fut-window', action='store', dest='fut_window',
                        help='(Optional) Future (prediction) window (default 10).', type=int)
    parser.add_argument('--trim-head', action='store', dest='trim_head',
                        help='(Optional) Trim some part of the viewport trajectory head (default 30).', type=int)
    parser.add_argument('--trim-tail', action='store', dest='trim_tail',
                        help='(Optional) Trim some part of the viewport trajectory tail (default 30).', type=int)
    parser.add_argument('--dataset-frequency', action='store', dest='dataset_frequency',
                        help='(Optional) The frequency version of the dataset (default 10).', type=int)
    parser.add_argument('--sample-step', action='store', dest='sample_step',
                        help='(Optional) The steps for sampling viewports (default 1).', type=int)

    # ========== training related settings ==========
    parser.add_argument('--epochs', action="store", dest='epochs', help='(Optional) Neural network learning epochs.', type=int)
    parser.add_argument('--epochs-per-valid', action='store', dest='epochs_per_valid', type=int,
                        help='(Optional) The number of epochs per validation (default 3).')
    parser.add_argument('--steps-per-valid', action='store', dest='steps_per_valid', type=int,
                        help='(Optional) The number of steps per validation (default 50).')
    parser.add_argument('--early-stopping-patience', type=int, default=None,
                        help='Stop after this many consecutive validation events without sufficient improvement.')
    parser.add_argument('--early-stopping-min-delta', type=float, default=0.0,
                        help='Minimum validation-loss decrease required to reset early-stopping patience.')
    parser.add_argument('--report-loss-per-steps', action='store', dest='report_loss_per_steps', type=int, default=100,
                        help='(Optional) The number of steps per validation (default 100).')
    parser.add_argument('--lr', action="store", dest='lr', help='(Optional) Neural network learning rate.', type=float)
    parser.add_argument('--multimodal-lr-multiplier', action='store', dest='multimodal_lr_multiplier', type=float, default=1.0,
                         help='(Optional) Multiplier applied to --lr for the embed_multimodal/conv1d optimizer '
                              'group only (LoRA/networking_head/embed_vp/embed_ln stay at --lr). Default 1.0 '
                              '(no change). E.g. 5.0 or 10.0 to let the multimodal encoder adapt faster within '
                              'the same epoch budget.')
    parser.add_argument('--weight-decay', action="store", dest='weight_decay', help='(Optional) Neural network weight decay.', type=float, default=1e-4)
    parser.add_argument('--bs', action="store", dest='bs', help='(Optional) Neural network batch size.', type=int)
    parser.add_argument('--grad-accum-steps', action="store", dest='grad_accum_steps', type=int, default=16)
    parser.add_argument('--seed', action="store", dest='seed', type=int, default=1, help='(Optional) Random seed (default to 1).')
    parser.add_argument(
        '--lora-seed', type=int, default=None,
        help='LoRA/AdaLoRA initialization seed. Defaults to --seed for compatibility.',
    )
    parser.add_argument(
        '--data-seed', type=int, default=None,
        help='Dataset/DataLoader ordering seed. Defaults to --seed for compatibility.',
    )
    parser.add_argument('--multimodal', action="store_true", dest='using_multimodal', help='(deprecated) using multimodal image features; equivalent to --multimodal-mode baseline.')
    parser.add_argument('--multimodal-mode', action='store', dest='multimodal_mode', choices=['baseline', 'all-patch', 'patch-selection'],
                        help="(Optional) Multimodal mode: 'baseline' (single cached ViT CLS-token feature per frame), "
                             "'all-patch' (all patch-grid patches through frozen ViT), or "
                             "'patch-selection' (patch_selection module picks a subset of patches). "
                             "Defaults to 'baseline' if --multimodal is set, otherwise no multimodal features are used.")
    parser.add_argument('--patch-top-k', action='store', dest='patch_top_k', type=int,
                        help='(Optional) For --multimodal-mode patch-selection: select exactly this many patches per frame.')
    parser.add_argument('--patch-threshold', action='store', dest='patch_threshold', type=float,
                        help='(Optional) For --multimodal-mode patch-selection: select patches with sigmoid(logit) above this threshold '
                             '(ignored if --patch-top-k is set; defaults to 0.5 if neither is set).')
    parser.add_argument('--patch-selection-weights', action='store', dest='patch_selection_weights', type=str,
                        help='(Optional) Path to a pretrained PatchSelectionModule state_dict for --multimodal-mode patch-selection. '
                             'If omitted, a freshly-initialized (UNTRAINED) module is used.')
    parser.add_argument('--exclude-short-videos-test', action='store_true', dest='exclude_short_videos_test',
                        help='(Optional) Exclude videos with fewer extracted frames than the nominal count '
                             '(see cfg.dataset_short_frame_videos) from the test split, to avoid MAE distortion '
                             'from clamp-repeated tail frames.')
    parser.add_argument('--save-checkpoint-per-epoch', action="store", dest='save_checkpoint_per_epoch', help='save checkpoint per epoch', type=int)
    parser.add_argument('--save-checkpoint-per-step', action="store", dest='save_checkpoint_per_step', help='save checkpoint per step', type=int)
    parser.add_argument('--save-test-progress-per-steps', type=int, default=None,
                        help='Write a partial evaluation summary CSV every N test batches. '
                             'Useful for preserving progress if a long evaluation is interrupted.')
    parser.add_argument('--measure-inference-latency', action='store_true',
                        help='Measure pipeline.inference latency during the normal test pass. '
                             'CUDA calls are synchronized; data loading and metric computation are excluded.')
    parser.add_argument('--latency-warmup-steps', type=int, default=5,
                        help='Number of initial evaluation calls excluded from latency statistics.')
    parser.add_argument('--latency-deadline-ms', type=float, default=1000.0,
                        help='Optional real-time deadline used to report latency margin (milliseconds).')
    parser.add_argument('--latency-output-path', type=str, default=None,
                        help='Optional JSON output path for latency summary; a per-sample CSV is written beside it.')
    parser.add_argument(
        '--nbs-inference-mode', choices=['original', 'compact'], default='original',
        help=(
            'NBS evaluation representation. original preserves masked PEFT '
            'AdaLoRA; compact opt-in converts/loads an equivalent inference-only '
            'fixed LoRA with physical inactive slots removed.'
        ),
    )
    parser.add_argument(
        '--nbs-compact-output-dir', type=str, default=None,
        help=(
            'Separate output directory for a newly derived compact NBS '
            'checkpoint. Defaults to <resolved source checkpoint>_compact.'
        ),
    )
    parser.add_argument(
        '--nbs-compaction-rtol', type=float, default=1e-4,
        help='Strict relative tolerance for layer-level compact-factor equivalence.',
    )
    parser.add_argument(
        '--nbs-compaction-atol', type=float, default=1e-5,
        help='Strict absolute tolerance for layer-level compact-factor equivalence.',
    )
    parser.add_argument(
        '--nbs-compaction-output-rtol', type=float, default=1e-3,
        help='Relative tolerance for end-to-end autoregressive output equivalence.',
    )
    parser.add_argument(
        '--nbs-compaction-output-atol', type=float, default=2e-3,
        help=(
            'Absolute tolerance for end-to-end autoregressive output equivalence; '
            'separate from strict factor checks to allow accumulated FP16 rounding.'
        ),
    )
    parser.add_argument('--selector-recent-k', type=int, default=None,
                        help='Keep the most recent K trajectory tokens before decoding.')
    parser.add_argument('--speculative-gamma', type=int, default=None,
                        help='Enable block-verified speculative decoding with this draft block size.')
    parser.add_argument('--speculative-threshold', type=float, default=0.3,
                        help='Acceptance threshold in normalized viewport-coordinate space.')
    parser.add_argument('--inference-trace-output-path', type=str, default=None,
                        help='Optional JSON output for selector/speculative trace statistics.')
    parser.add_argument('--rank', action="store", dest='rank', help='the rank of low rank matrices', type=int, default=-1)
    parser.add_argument('--use-adalora', action='store_true', dest='use_adalora',
                         help='Use AdaLoRA instead of plain LoRA when --rank != -1; '
                              'the allocator is selected by --adalora-allocator.')
    parser.add_argument('--use-eva', action='store_true',
                        help='Initialize fixed LoRA ranks and lora_A directions from a precomputed EVA state. '
                             'Mutually exclusive with AdaLoRA and fixed --lora-rank-config.')
    parser.add_argument('--eva-state-path', type=str, default=None,
                        help='Path to EVA eva_state.pt, or to its containing directory.')
    parser.add_argument('--adalora-min-rank', type=int, default=None,
                        help='Minimum rank per LoRA layer for Nash allocation (default: rank//2).')
    parser.add_argument('--adalora-ema-beta', type=float, default=0.9,
                        help='EMA coefficient for layer gradient sensitivity.')
    parser.add_argument('--adalora-eps', type=float, default=1e-8,
                        help='Epsilon used by normalized utility and Nash gain.')
    parser.add_argument('--adalora-allocation-interval', type=int, default=10,
                        help='Optimizer-step interval between custom rank allocations.')
    parser.add_argument('--adalora-rank-budget', type=int, default=None,
                        help=('Global active rank budget R in fixed mode; default adaptive '
                              'warm-up/cap budget in adaptive mode.'))
    parser.add_argument(
        '--adalora-budget-mode', choices=['fixed', 'adaptive'], default='fixed',
        help=(
            'Use the historical exact global rank budget or stop allocation '
            'when relative marginal Nash gain falls below a threshold.'
        ),
    )
    parser.add_argument(
        '--adalora-relative-lambda', type=float, default=0.15,
        help=(
            'Adaptive stopping ratio tau in lambda_t=tau*max_l Delta_l(r_l_min). '
            'Ignored in fixed mode.'
        ),
    )
    parser.add_argument(
        '--adalora-adaptive-min-budget', type=int, default=None,
        help=(
            'Optional adaptive total-rank floor; defaults to the sum of all '
            'layer minimum ranks.'
        ),
    )
    parser.add_argument(
        '--adalora-adaptive-max-budget', type=int, default=None,
        help=(
            'Optional adaptive total-rank cap and warm-up budget; defaults to '
            '--adalora-rank-budget (or target rank times layer count).'
        ),
    )
    parser.add_argument('--adalora-rank-config', type=str, default=None,
                        help='JSON file mapping LoRA module names to min_rank/max_rank overrides.')
    parser.add_argument('--lora-rank-config', type=str, default=None,
                        help='JSON file containing a fixed plain-LoRA rank_pattern dictionary.')
    parser.add_argument('--adalora-missing-grad-policy', choices=['zero', 'hold'], default='zero',
                        help='Sensitivity EMA behavior when a layer has no A/B gradient: decay with zero or hold.')
    parser.add_argument('--adalora-allocator', choices=['nbs', 'peft', 'shapley'], default='nbs',
                        help='Use the proposed NBS allocator, stock PEFT allocator, or '
                             'the optional validation-loss Shapley comparison allocator.')
    parser.add_argument('--shapley-permutations', type=int, default=3,
                        help='Monte Carlo module permutations per Shapley allocation.')
    parser.add_argument('--shapley-validation-batches', type=int, default=4,
                        help='Fixed validation batches used for each coalition value.')
    parser.add_argument('--shapley-truncate-fraction', type=float, default=0.05,
                        help='Early truncation tolerance as a fraction of full-minus-empty value.')
    parser.add_argument('--shapley-value-mode',
                        choices=['autoregressive', 'teacher-forcing'],
                        default='autoregressive',
                        help='Validation forward used as the Shapley coalition value.')
    parser.add_argument('--no-shapley-antithetic', action='store_false',
                        dest='shapley_antithetic', default=True,
                        help='Disable reversed-permutation antithetic sampling.')
    parser.add_argument(
        '--adalora-shadow-update-policy',
        choices=['legacy', 'active-only'],
        default='legacy',
        help=(
            'How NBS refreshes lora_E spectral_shadow: legacy updates every '
            'nonzero slot; active-only updates only currently active mask slots.'
        ),
    )
    parser.add_argument('--adalora-diagnostics-path', type=str, default=None,
                        help='CSV path for durable per-allocation NBS statistics and rank trajectory. '
                             'Defaults to the current training artifact directory.')
    parser.add_argument('--experiment-tag',
                        choices=['nbs_v2', 'nbs_v3', 'nbs_v4', 'nbs_v5',
                                 'nbs_v6', 'nbs_v7', 'nbs_v8', 'nbs_v9',
                                 'nbs_v10', 'nbs_v11', 'nbs_v12',
                                 'nbs_v12_repeat', 'nbs_v13',
                                 'nbs_v14', 'nbs_v15', 'nbs_v16', 'nbs_v17',
                                 'nbs_v18', 'nbs_v19', 'nbs_v20', 'nbs_v21',
                                 'nbs_v22', 'nbs_v23', 'nbs_v24', 'nbs_v25',
                                 'nbs_v27', 'nbs_v28', 'nbs_v29',
                                 'nbs_v19_data2', 'nbs_budget256_seed1',
                                 'nbs_adaptive_tau015',
                                 'uniform_r12', 'uniform_b736', 'adalora_peft_r12',
                                 'adalora_shapley', 'shapley_v19', 'eva'],
                        default=None,
                        help='Optional suffix that isolates model/result directories for an experiment variant.')
    parser.add_argument(
        '--experiment-run-id',
        default=None,
        help='Optional run-specific subdirectory preventing stale checkpoint/result reuse.',
    )
    parser.add_argument('--resume-path', action="store", dest='resume_path', help='using for resume')
    parser.add_argument('--scheduled-sampling', action="store_true", dest='scheduled_sampling', help='using scheduled sampling, a common method to reduce exposure bias to improve '\
                                                                                                     'sequence generation by mixing teacher-forcing generation and auto-regressive generation. '\
                                                                                                     'see: https://www.activeloop.ai/resources/glossary/scheduled-sampling/')
    parser.add_argument('--mix-rate', action="store", dest='mix_rate', help='the mixing rate when using scheduled sampling', type=float, default=0.04)
    parser.add_argument('--limit-train-samples', action='store', dest='limit_train_samples', type=int,
                        help='(Optional, debug) Truncate the training set to this many samples (first N, no shuffle applied to the truncation itself).')
    parser.add_argument('--limit-valid-samples', action='store', dest='limit_valid_samples', type=int,
                        help='(Optional, debug) Truncate the validation set to this many samples.')
    parser.add_argument('--limit-test-samples', action='store', dest='limit_test_samples', type=int,
                        help='(Optional, smoke test) Truncate the test set to the first N samples.')
    args = parser.parse_args()
    try:
        args.seed, args.lora_seed, args.data_seed = resolve_experiment_seeds(
            args.seed, args.lora_seed, args.data_seed
        )
    except ValueError as exc:
        parser.error(str(exc))
    if args.adalora_budget_mode == 'adaptive':
        if not args.use_adalora or args.adalora_allocator != 'nbs':
            parser.error(
                '--adalora-budget-mode adaptive requires --use-adalora '
                'and --adalora-allocator nbs'
            )
        if args.adalora_shadow_update_policy != 'active-only':
            parser.error(
                '--adalora-budget-mode adaptive requires '
                '--adalora-shadow-update-policy active-only'
            )
    if args.adalora_allocator == 'shapley':
        if not args.use_adalora or args.rank == -1:
            parser.error(
                '--adalora-allocator shapley requires --use-adalora and --rank'
            )
        if any(value is not None for value in (
                args.adalora_min_rank,
                args.adalora_rank_budget,
                args.adalora_rank_config,
        )):
            parser.error(
                'Shapley uses --rank as PEFT target_r; do not pass NBS '
                '--adalora-min-rank, --adalora-rank-budget, or '
                '--adalora-rank-config options'
            )
    if args.shapley_permutations <= 0:
        parser.error('--shapley-permutations must be positive')
    if args.shapley_validation_batches <= 0:
        parser.error('--shapley-validation-batches must be positive')
    if not 0.0 <= args.shapley_truncate_fraction <= 1.0:
        parser.error('--shapley-truncate-fraction must be in [0, 1]')

    # resolve the 3-way multimodal mode; --multimodal (legacy) maps to 'baseline' when
    # --multimodal-mode isn't explicitly given
    if args.multimodal_mode is None:
        args.multimodal_mode = 'baseline' if args.using_multimodal else 'none'

    # for debug
    # args.adapt = True
    # args.test = True
    # args.device = 'cuda:5'
    # args.train_dataset = 'Jin2022'
    # args.test_dataset = 'Jin2022'
    # args.dataset_frequency = 5
    # args.sample_step = 15
    # args.his_window = 10
    # args.fut_window = 20
    # args.plm_type = 'opt'
    # args.plm_size = 'xs'
    # args.epochs = 30
    # args.bs = 1
    # args.lr = 5e-4
    # args.scheduled_sampling = True
    # args.steps_per_valid = 500
    # args.rank = 32
    # args.seed = 1

    # handle defautl settings
    args.his_window = cfg.default_history_window if args.his_window is None else args.his_window
    args.fut_window = cfg.default_future_window if args.fut_window is None else args.fut_window
    args.trim_head = cfg.default_trim_head if args.trim_head is None else args.trim_head
    args.trim_tail = cfg.default_trim_tail if args.trim_tail is None else args.trim_tail
    args.dataset_frequency = cfg.default_dataset_frequency if args.dataset_frequency is None else args.dataset_frequency
    args.sample_step = cfg.default_sample_step if args.sample_step is None else args.sample_step
    args.epochs = cfg.default_epochs if args.epochs is None else args.epochs
    args.lr = cfg.default_lr if args.lr is None else args.lr
    args.weight_decay = cfg.default_weight_decay if args.weight_decay is None else args.weight_decay
    args.bs = cfg.default_bs if args.bs is None else args.bs
    args.grad_accum_steps = cfg.default_grad_accum_step if args.grad_accum_steps is None else args.grad_accum_steps
    args.steps_per_valid = cfg.default_steps_per_valid if args.steps_per_valid is None else args.steps_per_valid

    
    if args.device_out is None:  
        args.device_out = args.device

    if args.train_dataset is None:
        args.train_dataset = args.test_dataset
    if args.test_dataset is None:
        args.test_dataset = args.train_dataset

    print(args)
    run(args)
