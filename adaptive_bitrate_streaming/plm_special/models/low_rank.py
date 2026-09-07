import atexit
import json
import os
from pathlib import Path
import sys
from types import MethodType

import torch
import torch.nn as nn
from peft import AdaLoraConfig, LoraConfig, get_peft_model, TaskType


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from models.rank_allocator import NashRankAllocator
from models.eva_initializer import (
    eva_lora_spec,
    initialize_eva_lora_weights,
)


TARGET_MODULES = {
    'llama': ["q_proj", "v_proj"],
    'llava': ["q_proj", "v_proj"],
    'mistral': ["q_proj", "v_proj"],
    'opt': ["q_proj", "v_proj"],
    'gpt2': ["q_proj", "v_proj"],
    't5-lm': ["q", "v"]
}

LORA_METHODS = ('uniform', 'nbs', 'adalora', 'shapley', 'eva')


_RANGE_AUDIT_PATH = os.environ.get('ABR_LORA_RANGE_AUDIT_PATH')
_RANGE_AUDIT_THRESHOLD = float(
    os.environ.get('ABR_LORA_RANGE_AUDIT_THRESHOLD', '60000')
)
_RANGE_AUDIT = {
    'schema_version': 1,
    'mode': 'detect-only',
    'threshold': _RANGE_AUDIT_THRESHOLD,
    'modules': {},
}


def _record_range_audit(module, *, input_absmax, input_finite, base_absmax,
                        base_finite, delta_absmax, delta_finite,
                        precast_absmax, precast_finite, dtype_limit):
    """Record projection health without changing the forward result."""
    if not _RANGE_AUDIT_PATH:
        return
    name = getattr(module, '_nbs_module_name', module.__class__.__name__)
    stats = _RANGE_AUDIT['modules'].setdefault(name, {
        'calls': 0,
        'would_clamp_calls': 0,
        'input_nonfinite_calls': 0,
        'base_nonfinite_calls': 0,
        'delta_nonfinite_calls': 0,
        'precast_nonfinite_calls': 0,
        'max_input_absmax': 0.0,
        'max_base_absmax': 0.0,
        'max_delta_absmax': 0.0,
        'max_precast_absmax': 0.0,
        'output_dtype_limit': float(dtype_limit),
    })
    values = {
        'input': (input_absmax, input_finite),
        'base': (base_absmax, base_finite),
        'delta': (delta_absmax, delta_finite),
        'precast': (precast_absmax, precast_finite),
    }
    stats['calls'] += 1
    for label, (absmax, finite) in values.items():
        is_finite = bool(finite.detach().item())
        if not is_finite:
            stats[f'{label}_nonfinite_calls'] += 1
            continue
        value = float(absmax.detach().item())
        stats[f'max_{label}_absmax'] = max(
            stats[f'max_{label}_absmax'], value
        )
    if (
        bool(precast_finite.detach().item())
        and float(precast_absmax.detach().item()) > _RANGE_AUDIT_THRESHOLD
    ):
        stats['would_clamp_calls'] += 1


def _write_range_audit():
    if not _RANGE_AUDIT_PATH:
        return
    modules = _RANGE_AUDIT['modules']
    summary = {
        'module_count': len(modules),
        'projection_calls': sum(item['calls'] for item in modules.values()),
        'would_clamp_calls': sum(
            item['would_clamp_calls'] for item in modules.values()
        ),
        'input_nonfinite_calls': sum(
            item['input_nonfinite_calls'] for item in modules.values()
        ),
        'base_nonfinite_calls': sum(
            item['base_nonfinite_calls'] for item in modules.values()
        ),
        'delta_nonfinite_calls': sum(
            item['delta_nonfinite_calls'] for item in modules.values()
        ),
        'precast_nonfinite_calls': sum(
            item['precast_nonfinite_calls'] for item in modules.values()
        ),
        'max_precast_absmax': max(
            (item['max_precast_absmax'] for item in modules.values()),
            default=0.0,
        ),
    }
    payload = {**_RANGE_AUDIT, 'summary': summary}
    path = Path(_RANGE_AUDIT_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding='utf-8'
    )
    temporary.replace(path)


if _RANGE_AUDIT_PATH:
    atexit.register(_write_range_audit)


def _target_module_count(plm, plm_type):
    """Count concrete q/v projection modules before PEFT wraps them."""
    targets = set(TARGET_MODULES[plm_type])
    names = [
        name for name, module in plm.named_modules()
        if isinstance(module, nn.Linear) and name.rsplit('.', 1)[-1] in targets
    ]
    if not names:
        raise RuntimeError(
            f'No LoRA target modules found for {plm_type}: {sorted(targets)}'
        )
    return len(names)


def _adalora_target_rank(rank_budget, module_count, physical_rank):
    """Translate an exact global budget into PEFT's integer target_r."""
    if rank_budget is None:
        return int(physical_rank), int(physical_rank) * int(module_count)
    if rank_budget <= 0:
        raise ValueError('AdaLoRA rank budget must be positive')
    target_rank, remainder = divmod(int(rank_budget), int(module_count))
    if remainder:
        raise ValueError(
            'Stock/Shapley AdaLoRA requires a global budget divisible by the '
            f'{module_count} target modules; got {rank_budget}'
        )
    if target_rank <= 0 or target_rank > physical_rank:
        raise ValueError(
            f'AdaLoRA target rank {target_rank} must be in [1, {physical_rank}]'
        )
    return target_rank, int(rank_budget)


def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


def _mixed_precision_adalora_forward(self, x):
    """Accumulate the AdaLoRA residual in FP32 before the base-dtype cast.

    Detached health tensors are retained for one batched check in Trainer;
    this avoids synchronizing every adapter layer during a normal forward.
    """
    base_input = x.to(self.weight.dtype)
    if self.disable_adapters:
        if self.merged:
            self.unmerge()
        return self._linear(base_input)
    if self.merged:
        return self._linear(base_input)

    detached_input = base_input.detach().float()
    self._nbs_last_input_absmax = detached_input.abs().amax().detach()
    self._nbs_last_input_finite = torch.isfinite(detached_input).all().detach()

    base_result = self._linear(base_input)
    detached_base = base_result.detach().float()
    self._nbs_last_base_absmax = detached_base.abs().amax().detach()
    self._nbs_last_base_finite = torch.isfinite(detached_base).all().detach()
    result_fp32 = base_result.float()
    delta_absmax = result_fp32.new_zeros(())
    delta_finite = torch.ones((), dtype=torch.bool, device=result_fp32.device)
    for active_adapter in self.active_adapters:
        if active_adapter not in self.lora_A:
            continue
        lora_a = self.lora_A[active_adapter]
        lora_b = self.lora_B[active_adapter]
        lora_e = self.lora_E[active_adapter]
        adapter_input = self.lora_dropout[active_adapter](x.to(lora_a.dtype))
        ranknum = self.ranknum[active_adapter].to(lora_a.dtype) + 1e-5
        delta = (
            adapter_input @ (lora_a * lora_e).T @ lora_b.T
        ) * self.scaling[active_adapter] / ranknum
        detached_delta = delta.detach()
        delta_absmax = torch.maximum(
            delta_absmax, detached_delta.float().abs().amax()
        )
        delta_finite = delta_finite & torch.isfinite(detached_delta).all()
        result_fp32 = result_fp32 + delta.float()

    detached_result = result_fp32.detach()
    self._nbs_last_delta_absmax = delta_absmax.detach()
    self._nbs_last_delta_finite = delta_finite.detach()
    self._nbs_last_precast_absmax = detached_result.abs().amax().detach()
    self._nbs_last_precast_finite = torch.isfinite(detached_result).all().detach()
    audit_precast_absmax = self._nbs_last_precast_absmax
    audit_precast_finite = self._nbs_last_precast_finite
    self._nbs_output_dtype = base_result.dtype
    dtype_limit = torch.finfo(base_result.dtype).max
    clamp_enabled = bool(
        getattr(self, '_abr_fp16_selective_clamp', False)
        and base_result.dtype == torch.float16
    )
    clamp_threshold = min(
        float(getattr(self, '_abr_fp16_clamp_threshold', dtype_limit)),
        float(dtype_limit),
    )
    self._abr_last_selective_clamp_count = 0
    if clamp_enabled:
        # Clamp finite values only. NaN/Inf remains observable through the
        # health tensors and is rejected by RLPolicy/Trainer as before.
        clamp_mask = torch.isfinite(result_fp32) & (
            result_fp32.abs() > clamp_threshold
        )
        clamp_count = int(clamp_mask.sum().item())
        self._abr_last_selective_clamp_count = clamp_count
        if clamp_count:
            self._abr_total_selective_clamp_calls = (
                int(getattr(self, '_abr_total_selective_clamp_calls', 0)) + 1
            )
            self._abr_total_selective_clamped_elements = (
                int(getattr(
                    self, '_abr_total_selective_clamped_elements', 0
                )) + clamp_count
            )
            result_fp32 = torch.where(
                clamp_mask,
                result_fp32.clamp(-clamp_threshold, clamp_threshold),
                result_fp32,
            )
            detached_result = result_fp32.detach()
            self._nbs_last_precast_absmax = (
                detached_result.abs().amax().detach()
            )
            self._nbs_last_precast_finite = (
                torch.isfinite(detached_result).all().detach()
            )
    _record_range_audit(
        self,
        input_absmax=self._nbs_last_input_absmax,
        input_finite=self._nbs_last_input_finite,
        base_absmax=self._nbs_last_base_absmax,
        base_finite=self._nbs_last_base_finite,
        delta_absmax=self._nbs_last_delta_absmax,
        delta_finite=self._nbs_last_delta_finite,
        precast_absmax=audit_precast_absmax,
        precast_finite=audit_precast_finite,
        dtype_limit=dtype_limit,
    )
    # Do not let one rejected FP16 projection turn every downstream layer into
    # NaN.  The policy checks the health tensors after the PLM call and raises,
    # so this bounded value is never accepted for loss/inference.  It merely
    # contains the failure long enough to identify the first faulty modules.
    contained_result = torch.nan_to_num(
        result_fp32,
        nan=0.0,
        posinf=dtype_limit,
        neginf=-dtype_limit,
    ).clamp(min=-dtype_limit, max=dtype_limit)
    return contained_result.to(base_result.dtype)


def _patch_adalora_mixed_precision(
    model, fp16_selective_clamp=False, fp16_clamp_threshold=60000.0,
):
    patched = 0
    for module_name, module in model.named_modules():
        if module.__class__.__name__ != 'SVDLinear':
            continue
        if not all(hasattr(module, name) for name in (
            'lora_A', 'lora_B', 'lora_E', 'ranknum', '_linear'
        )):
            continue
        module.forward = MethodType(_mixed_precision_adalora_forward, module)
        module._nbs_module_name = module_name
        module._abr_fp16_selective_clamp = bool(fp16_selective_clamp)
        module._abr_fp16_clamp_threshold = float(fp16_clamp_threshold)
        module._abr_total_selective_clamp_calls = 0
        module._abr_total_selective_clamped_elements = 0
        patched += 1
    return patched


def peft_model(
    plm,
    plm_type,
    rank,
    print_trainable=False,
    task_type=TaskType.FEATURE_EXTRACTION,
    nbs_v19=False,
    total_step=None,
    nbs_rank_budget=512,
    nbs_ema_beta=0.9,
    nbs_allocation_interval=10,
    nbs_rank_config=None,
    lora_method=None,
    adalora_rank_budget=None,
    adalora_allocation_interval=10,
    shapley_permutations=1,
    shapley_truncate_fraction=0.05,
    shapley_seed=0,
    shapley_antithetic=True,
    eva_state=None,
    fp16_selective_clamp=False,
    fp16_clamp_threshold=60000.0,
):
    if lora_method is None:
        lora_method = 'nbs' if nbs_v19 else 'uniform'
    if lora_method not in LORA_METHODS:
        raise ValueError(
            f'lora_method must be one of {LORA_METHODS}; got {lora_method!r}'
        )
    if nbs_v19 and lora_method != 'nbs':
        raise ValueError('nbs_v19=True is only compatible with lora_method=nbs')
    nbs_v19 = lora_method == 'nbs'
    use_adalora = lora_method in ('nbs', 'adalora', 'shapley')
    if lora_method == 'eva' and eva_state is None:
        raise ValueError('lora_method=eva requires a precomputed EVA state')
    if lora_method != 'eva' and eva_state is not None:
        raise ValueError('eva_state is only compatible with lora_method=eva')
    if use_adalora and (total_step is None or total_step <= 0):
        raise ValueError(
            f'{lora_method} requires a positive total optimizer-step count'
        )
    if rank <= 0:
        raise ValueError('LoRA physical rank must be positive')
    if adalora_allocation_interval <= 0:
        raise ValueError('AdaLoRA allocation interval must be positive')
    if shapley_permutations <= 0:
        raise ValueError('Shapley permutations must be positive')
    if not 0.0 <= shapley_truncate_fraction <= 1.0:
        raise ValueError('Shapley truncate fraction must be in [0, 1]')

    for param in plm.parameters():
        param.requires_grad = False
        # Keep frozen normalization weights in the dtype selected by
        # from_pretrained(). Upcasting 1-D LlamaRMSNorm weights would promote
        # FP16 hidden states to FP32 and break the next FP16 projection.

    plm.gradient_checkpointing_enable()
    plm.enable_input_require_grads()

    class CastOutputToFloat(nn.Sequential):
        def forward(self, x):
            return super().forward(x).to(torch.float32)

    module_count = _target_module_count(plm, plm_type) if use_adalora else None
    target_rank = rank
    effective_rank_budget = None
    if use_adalora and not nbs_v19:
        target_rank, effective_rank_budget = _adalora_target_rank(
            adalora_rank_budget, module_count, rank
        )

    eva_spec = None
    if use_adalora:
        tinit = max(1, int(total_step * 0.1))
        tfinal = max(tinit + 1, int(total_step * 0.15))
        cooldown_start = max(tinit + 1, total_step - tfinal)
        config = AdaLoraConfig(
            # ``rank`` is the physical AdaLoRA slot capacity.  Keeping this
            # hard-coded at 32 made larger NBS max-rank configurations pass
            # CLI validation but fail when the allocator inspected the model.
            init_r=rank,
            target_r=target_rank,
            tinit=tinit,
            tfinal=tfinal,
            deltaT=(
                nbs_allocation_interval
                if nbs_v19 else adalora_allocation_interval
            ),
            lora_alpha=32,
            target_modules=TARGET_MODULES[plm_type],
            lora_dropout=0.05,
            bias='none',
            task_type=task_type,
            total_step=total_step,
        )
    else:
        rank_pattern = {}
        target_modules = TARGET_MODULES[plm_type]
        if lora_method == 'eva':
            eva_spec = eva_lora_spec(eva_state)
            rank_pattern = eva_spec['rank_pattern']
            target_modules = eva_spec['target_modules']
        config = LoraConfig(
            r=rank,
            lora_alpha=32,
            target_modules=target_modules,
            lora_dropout=0.05,
            bias="none",
            task_type=task_type,
            rank_pattern=rank_pattern,
        )

    model = get_peft_model(plm, config)
    if module_count is None:
        module_count = sum(
            1 for module in model.modules()
            if hasattr(module, 'lora_A') and 'default' in module.lora_A
        )
        effective_rank_budget = (
            eva_spec['total_rank_budget']
            if eva_spec is not None else rank * module_count
        )
    elif effective_rank_budget is None:
        effective_rank_budget = nbs_rank_budget
    model.abr_lora_method = lora_method
    model.abr_target_module_count = module_count
    model.abr_effective_rank_budget = effective_rank_budget
    if use_adalora:
        patched = _patch_adalora_mixed_precision(
            model,
            fp16_selective_clamp=fp16_selective_clamp,
            fp16_clamp_threshold=fp16_clamp_threshold,
        )
        if patched == 0:
            raise RuntimeError(
                f'{lora_method} found no AdaLoRA SVDLinear modules'
            )
    if nbs_v19:
        model.nash_rank_allocator = NashRankAllocator(
            model,
            target_rank=rank,
            ema_beta=nbs_ema_beta,
            rank_budget=nbs_rank_budget,
            rank_config=nbs_rank_config,
            missing_grad_policy='zero',
            warmup_steps=tinit,
            cooldown_start_step=cooldown_start,
            allocation_interval=nbs_allocation_interval,
            shadow_update_policy='legacy',
            budget_mode='fixed',
        )
        model.nbs_variant = 'nbs_v19'
        print(
            'NBS v19 enabled: physical_rank={} budget={} seed-controlled '
            'initialization, allocation interval={}, shadow_policy=legacy'.format(
                rank, nbs_rank_budget, nbs_allocation_interval
            )
        )
    elif lora_method == 'shapley':
        # The comparison allocator is shared with viewport prediction but is
        # imported only for an explicitly selected Shapley run.
        from models.shapley_allocator import ShapleyRankAllocator

        adalora_model = model.base_model
        adapter_name = adalora_model.trainable_adapter_name
        shapley_allocator = ShapleyRankAllocator(
            adalora_model.model,
            adalora_model.peft_config[adapter_name],
            adapter_name,
            n_permutations=shapley_permutations,
            truncate_fraction=shapley_truncate_fraction,
            seed=shapley_seed,
            antithetic=shapley_antithetic,
        )
        adalora_model.rankallocator = shapley_allocator
        model.shapley_rank_allocator = shapley_allocator
        print(
            'ABR Shapley AdaLoRA enabled: physical_rank={} target_rank={} '
            'budget={} permutations={} antithetic={}'.format(
                rank, target_rank, effective_rank_budget,
                shapley_permutations, shapley_antithetic,
            )
        )
    elif lora_method == 'adalora':
        print(
            'ABR stock PEFT AdaLoRA enabled: physical_rank={} target_rank={} '
            'budget={} interval={}'.format(
                rank, target_rank, effective_rank_budget,
                adalora_allocation_interval,
            )
        )
    elif lora_method == 'eva':
        summary = initialize_eva_lora_weights(model, eva_state)
        model.eva_initialization_summary = summary
        model.eva_state = eva_state
        print(
            'ABR EVA LoRA enabled: modules={} budget={} '
            'lora_A=PCA components, lora_B=0'.format(
                summary['initialized_modules'],
                summary['total_rank_budget'],
            )
        )
    if print_trainable:
        print_trainable_parameters(model)
    return model
