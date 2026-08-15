import sys
import argparse
import copy
import json
import math
import os
import random
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
from models.low_rank import peft_model, print_trainable_parameters


def save_model(args, model, save_dir):
    """
    save fune-tune model
    """
    if args.rank != -1:
        # save low rank matrices
        model.plm.save_pretrained(save_dir)
        # save other modules except plm
        torch.save(model.modules_except_plm.state_dict(), os.path.join(save_dir, 'modules_except_plm.bin'))
        allocator = getattr(model.plm, 'nash_rank_allocator', None)
        if allocator is not None:
            torch.save(allocator.state_dict(), os.path.join(save_dir, 'nash_rank_allocator.pt'))
    else:
        # low rank matrices are disabled, save whole model
        torch.save(model.state_dict(), os.path.join(save_dir, 'model.bin'))


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


def adapt(args, pipeline, dataloader_train, dataloader_valid, models_dir, grad_accum_steps):
    file_prefix = f'his_{args.his_window}_fut_{args.fut_window}_ss_{args.sample_step}_epochs_{args.epochs}_bs_{args.bs * args.grad_accum_steps}_'\
                  f'lr_{args.lr}_seed_{args.seed}_rank_{args.rank}_scheduled_sampling_{args.scheduled_sampling}'
    checkpoint_path = os.path.join(models_dir, file_prefix, 'checkpoint')
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    best_model_path = os.path.join(models_dir, file_prefix, 'best_model')
    if not os.path.exists(best_model_path):
        os.makedirs(best_model_path)
    console_log = open(os.path.join(models_dir, file_prefix + '_console.log'), 'w')
    sys.stdout = ConsoleLogger(sys.__stdout__, console_log)

    if args.resume:
        pipeline = load_model(args, pipeline, args.resume_path)
        print('Resume weights for training from:', args.resume_path)

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

    assert args.epochs_per_valid is None or args.steps_per_valid is None, "You can only specify args.epochs_per_valid or args.steps_per_valid."

    global_step = 0
    opt_step = 0  # AdaLoRA: counts actual optimizer update steps (post grad-accum), must match total_step
    report_loss_per_steps = args.report_loss_per_steps
    tot_loss = 0
    log_loss = 0
    best_loss = float('inf')
    best_epoch, best_step = 0, 0

    def validate():
        pipeline.eval()
        with torch.no_grad():
            validata_checkpoint_path = os.path.join(checkpoint_path)
            if not os.path.exists(validata_checkpoint_path):
                os.makedirs(validata_checkpoint_path)
            save_model(args, pipeline, validata_checkpoint_path)
            print(f'Checkpoint saved at', checkpoint_path)
            ps_history_start = len(pipeline.patch_selection_history) if args.multimodal_mode == 'patch-selection' else None
            valid_loss = []
            for history, future, video_user_info in dataloader_valid:
                history, future = history.to(args.device), future.to(args.device)
                history = normalize_data(history, args.train_dataset)
                future = normalize_data(future, args.train_dataset)
                loss = pipeline(history, future, video_user_info, teacher_forcing=False)
                valid_loss.append(loss.item())
            valid_loss = sum(valid_loss) / len(valid_loss)
            if ps_history_start is not None:
                counts = pipeline.patch_selection_history[ps_history_start:]
                if counts:
                    print(f'[patch-selection] valid selected-patch counts (n={len(counts)}): {counts}')
                    print(f'[patch-selection] valid selected-patch avg={sum(counts)/len(counts):.2f} '
                          f'min={min(counts)} max={max(counts)}')
            pipeline.train()
            return valid_loss
        
    print(f'Training on {args.train_dataset} - bs: {args.bs} - lr: {args.lr} - seed: {args.seed}')
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
                    loss = pipeline(history, future, video_user_info, teacher_forcing=False)
            else:
                loss = pipeline(history, future, video_user_info, teacher_forcing=True)
            tot_loss += loss.item()
            loss = loss / grad_accum_steps
            loss.backward()
            if not (args.rank != -1 and args.use_adalora):
                # Preserve the existing non-AdaLoRA training behavior.
                torch.nn.utils.clip_grad_norm_(pipeline.plm.parameters(), 1.0)

            # perform gradient accumulation update
            if ((step + 1) % grad_accum_steps == 0) or (step + 1 == len(dataloader_train)):
                if args.rank != -1 and args.use_adalora:
                    # Read the accumulated, unclipped A/B gradients first so
                    # sensitivity matches the raw gradient-norm definition.
                    allocator = pipeline.plm.nash_rank_allocator
                    interval = pipeline.plm.nash_rank_allocation_interval
                    allocator.update_sensitivity()
                if args.rank != -1 and args.use_adalora:
                    # Clip once per effective batch, after sensitivity has been
                    # measured and immediately before the optimizer update.
                    torch.nn.utils.clip_grad_norm_(pipeline.plm.parameters(), 1.0)
                optimizer.step()
                if args.rank != -1 and args.use_adalora:
                    # Allocation/mask enforcement happens after optimizer.step
                    # and before zero_grad, while the next forward sees the
                    # selected rank mask.
                    is_final_update = (
                        epoch == args.epochs - 1 and
                        step + 1 == len(dataloader_train)
                    )
                    if (opt_step + 1) % interval == 0 or is_final_update:
                        allocator.allocate(opt_step + 1)
                    else:
                        allocator.enforce_masks()
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
                valid_loss = validate()
                if valid_loss < best_loss:
                    best_loss, best_step = valid_loss, global_step
                    save_model(args, pipeline, best_model_path)
                    print(f'Best model (step {best_step}, average valid loss {best_loss}) saved at', best_model_path)
                print('Valid loss', valid_loss, ' - ', 'Best loss', best_loss, 'at step', best_step)
            
            # save checkpoint by save_checkpoint_per_step
            if args.save_checkpoint_per_step is not None and global_step % args.save_checkpoint_per_step == 0:
                save_checkpoint_path = os.path.join(checkpoint_path, str(global_step // args.save_checkpoint_per_step)) # save checkpoint
                if not os.path.exists(save_checkpoint_path):
                    os.makedirs(save_checkpoint_path)
                save_model(args, pipeline, save_checkpoint_path)
                print('save checkpoint at', save_checkpoint_path)

        # validation by epochs
        if args.epochs_per_valid is not None and epoch % args.epochs_per_valid == 0:
            valid_loss = validate()
            if valid_loss < best_loss:
                best_loss, best_epoch = valid_loss, epoch
                save_model(args, pipeline, best_model_path)
                print(f'Best model (epoch {best_epoch}, average valid loss {best_loss}) saved at', best_model_path)
            print('Valid loss', valid_loss, ' - ', 'Best loss', best_loss, 'at epoch', best_epoch)
        
        # save checkpoint by save_checkpoint_per_epoch
        if args.save_checkpoint_per_epoch is not None and epoch % args.save_checkpoint_per_epoch == 0 and epoch > 0:
            save_checkpoint_path = os.path.join(checkpoint_path, f'epoch{epoch}') # save checkpoint
            if not os.path.exists(save_checkpoint_path):
                os.makedirs(save_checkpoint_path)
            save_model(args, pipeline, save_checkpoint_path)
            print('save checkpoint at', save_checkpoint_path)

    print('Done adaptation, average training loss =', tot_loss / global_step)


def test(args, pipeline, dataloader_test, models_dir, results_dir):
    file_prefix = f'his_{args.his_window}_fut_{args.fut_window}_axes_ss_{args.sample_step}_epochs_{args.epochs}_bs_{args.bs * args.grad_accum_steps}_'\
                  f'lr_{args.lr}_seed_{args.seed}_rank_{args.rank}_scheduled_sampling_{args.scheduled_sampling}'
    best_model_path = os.path.join(models_dir, file_prefix, 'best_model')
    result_path = os.path.join(results_dir, file_prefix + '_results.csv')
    partial_result_path = os.path.join(results_dir, file_prefix + '_partial_results.csv')
    progress_interval = getattr(args, 'save_test_progress_per_steps', None)
    if progress_interval is not None and progress_interval <= 0:
        raise ValueError('--save-test-progress-per-steps must be positive')
    notebook = ResultNotebook()

    model_path = args.model_path if args.model_path is not None else best_model_path
    if os.path.exists(model_path):
        pipeline = load_model(args, pipeline, model_path)
        print('Load weights from:', model_path)
    else:
        print('\033[33mWarning:\033[0m', model_path, 'not found, skip loading weights.')

    print(f'Testing on {args.test_dataset} - seed: {args.seed}')
    # Real accuracy bug, found 2026-08-14: this was missing entirely, so a `--adapt --test`
    # invocation in one command evaluated with the pipeline still in train() mode from the
    # end of adapt() (dropout active everywhere -- patch_selection_module's 0.1, LoRA's
    # 0.05 -- and BatchNorm-like layers, if any, using batch stats instead of running
    # stats). Confirmed NOT the cause of the patch-selection MAE investigation's numbers
    # (those came from a separate script that already called pipeline.eval() correctly),
    # but a real correctness issue for anyone using --adapt --test together.
    pipeline.eval()
    with torch.no_grad():
        for test_step, (history, future, video_user_info) in enumerate(dataloader_test, start=1):
            history, future = history.to(args.device), future.to(args.device)
            history = normalize_data(history, args.train_dataset)
            pred, gt = pipeline.inference(history, future, video_user_info)
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
        notebook.write(result_path)
        print("show detail result:")
        detail_result_path = result_path.replace('_results.csv', '_per_sample_results.csv')
        notebook.write_detail(detail_result_path)


def run(args):
    assert args.train_dataset in cfg.dataset_list 
    assert args.test_dataset in cfg.dataset_list
    assert args.plm_type in cfg.plm_types
    assert args.plm_size in cfg.plm_sizes
    assert args.trim_head >= args.his_window and args.trim_tail >= args.fut_window

    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
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
        models_dir = os.path.join(cfg.plms_finetuned_dir, low_rank_tag,
                              f'freeze_plm_{args.freeze_plm}', multimodal_tag, args.train_dataset, f'{args.dataset_frequency}Hz')
        results_dir = os.path.join(cfg.results_dir, low_rank_tag,
                               f'freeze_plm_{args.freeze_plm}', multimodal_tag, args.test_dataset, f'{args.dataset_frequency}Hz')
    else:
        models_dir = os.path.join(cfg.plms_finetuned_dir, f'{args.plm_type}_{args.plm_size}',
                              f'freeze_plm_{args.freeze_plm}', multimodal_tag, args.train_dataset, f'{args.dataset_frequency}Hz')
        results_dir = os.path.join(cfg.results_dir, f'{args.plm_type}_{args.plm_size}',
                               f'freeze_plm_{args.freeze_plm}', multimodal_tag, args.test_dataset, f'{args.dataset_frequency}Hz')
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
        raw_dataset_train, raw_dataset_valid = create_dataset(args.train_dataset, his_window=args.his_window,
                                                              fut_window=args.fut_window, trim_head=args.trim_head, trim_tail=args.trim_tail,
                                                              include=['train', 'valid'], frequency=args.dataset_frequency, step=args.sample_step)

        if args.limit_train_samples is not None:
            n = min(args.limit_train_samples, len(raw_dataset_train))
            raw_dataset_train = torch.utils.data.Subset(raw_dataset_train, range(n))
            print(f'\033[33mDebug:\033[0m truncated training set to {n} samples (--limit-train-samples).')
        if args.limit_valid_samples is not None:
            n = min(args.limit_valid_samples, len(raw_dataset_valid))
            raw_dataset_valid = torch.utils.data.Subset(raw_dataset_valid, range(n))
            print(f'\033[33mDebug:\033[0m truncated validation set to {n} samples (--limit-valid-samples).')

        dataloader_train = DataLoader(raw_dataset_train, batch_size=args.bs, shuffle=True, pin_memory=True)
        dataloader_valid = DataLoader(raw_dataset_valid, batch_size=args.bs, shuffle=False, pin_memory=True)
        steps_per_epoch = math.ceil(len(dataloader_train) / args.grad_accum_steps)
        total_step = steps_per_epoch * args.epochs
        if args.use_adalora:
            print(f'[AdaLoRA] total optimizer steps = {total_step} '
                  f'(steps_per_epoch={steps_per_epoch}, epochs={args.epochs})')

    adalora_rank_config = None
    if args.adalora_rank_config is not None:
        with open(args.adalora_rank_config, 'r', encoding='utf-8') as handle:
            adalora_rank_config = json.load(handle)

    if args.rank != -1:
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

        raw_dataset_test = create_dataset(args.test_dataset, dataset_video_split=test_video_split,
                                          his_window=args.his_window, fut_window=args.fut_window,
                                          trim_head=args.trim_head, trim_tail=args.trim_tail, include=['test'], frequency=args.dataset_frequency, step=args.sample_step)[0]

        dataloader_test = DataLoader(raw_dataset_test, batch_size=args.bs, shuffle=True, pin_memory=True)
        test(args, pipeline, dataloader_test, models_dir, results_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process the input parameters to train the network.')
    
    # ========== model/plm settings related arguments ==========
    parser.add_argument('--adapt', action="store_true", help='adapt llm.')
    parser.add_argument('--test', action="store_true", help='test llm.')
    parser.add_argument('--plm-type', action="store", dest='plm_type', help='type of plm.', default='t5-lm')
    parser.add_argument('--plm-size', action="store", dest='plm_size', help='size of plm.', default='base')
    parser.add_argument('--model-path', action="store", dest='model_path', type=str, help='(Optional) The directory of model weights to be loaded for testing.')
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
    parser.add_argument('--rank', action="store", dest='rank', help='the rank of low rank matrices', type=int, default=-1)
    parser.add_argument('--use-adalora', action='store_true', dest='use_adalora',
                         help='(Optional) Use AdaLoRA (rank-adaptive LoRA) instead of plain LoRA. '
                              'Only has an effect when --rank != -1. Uses gradient/spectral '
                              'Nash rank allocation over init_r=rank*2 slots.')
    parser.add_argument('--adalora-min-rank', type=int, default=None,
                        help='Minimum rank per LoRA layer for Nash allocation (default: rank//2).')
    parser.add_argument('--adalora-ema-beta', type=float, default=0.9,
                        help='EMA coefficient for layer gradient sensitivity.')
    parser.add_argument('--adalora-eps', type=float, default=1e-8,
                        help='Epsilon used by normalized utility and Nash gain.')
    parser.add_argument('--adalora-allocation-interval', type=int, default=10,
                        help='Optimizer-step interval between custom rank allocations.')
    parser.add_argument('--adalora-rank-budget', type=int, default=None,
                        help='Global active rank budget R (default: target rank multiplied by layer count).')
    parser.add_argument('--adalora-rank-config', type=str, default=None,
                        help='JSON file mapping LoRA module names to min_rank/max_rank overrides.')
    parser.add_argument('--adalora-missing-grad-policy', choices=['zero', 'hold'], default='zero',
                        help='Sensitivity EMA behavior when a layer has no A/B gradient: decay with zero or hold.')
    parser.add_argument('--resume-path', action="store", dest='resume_path', help='using for resume')
    parser.add_argument('--scheduled-sampling', action="store_true", dest='scheduled_sampling', help='using scheduled sampling, a common method to reduce exposure bias to improve '\
                                                                                                     'sequence generation by mixing teacher-forcing generation and auto-regressive generation. '\
                                                                                                     'see: https://www.activeloop.ai/resources/glossary/scheduled-sampling/')
    parser.add_argument('--mix-rate', action="store", dest='mix_rate', help='the mixing rate when using scheduled sampling', type=float, default=0.04)
    parser.add_argument('--limit-train-samples', action='store', dest='limit_train_samples', type=int,
                        help='(Optional, debug) Truncate the training set to this many samples (first N, no shuffle applied to the truncation itself).')
    parser.add_argument('--limit-valid-samples', action='store', dest='limit_valid_samples', type=int,
                        help='(Optional, debug) Truncate the validation set to this many samples.')
    args = parser.parse_args()

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
