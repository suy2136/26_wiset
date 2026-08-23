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


TARGET_MODULES = {
    'llama': ["q_proj", "v_proj"],
    'llava': ["q_proj", "v_proj"],
    'mistral': ["q_proj", "v_proj"],
    'opt': ["q_proj", "v_proj"],
    'gpt2': ["q_proj", "v_proj"],
    't5-lm': ["q", "v"]
}


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
    """Run the frozen base branch in FP16 and AdaLoRA math in its own dtype."""
    base_input = x.to(self.weight.dtype)
    if self.disable_adapters:
        if self.merged:
            self.unmerge()
        return self._linear(base_input)
    if self.merged:
        return self._linear(base_input)

    result = self._linear(base_input)
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
        result = result + delta.to(result.dtype)
    return result


def _patch_adalora_mixed_precision(model):
    patched = 0
    for module in model.modules():
        if module.__class__.__name__ != 'SVDLinear':
            continue
        if not all(hasattr(module, name) for name in (
            'lora_A', 'lora_B', 'lora_E', 'ranknum', '_linear'
        )):
            continue
        module.forward = MethodType(_mixed_precision_adalora_forward, module)
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
):
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

    if nbs_v19:
        if total_step is None or total_step <= 0:
            raise ValueError('NBS v19 requires a positive total optimizer-step count')
        if rank != 32:
            raise ValueError('NBS v19 requires --rank 32')
        tinit = max(1, int(total_step * 0.1))
        tfinal = max(tinit + 1, int(total_step * 0.15))
        cooldown_start = max(tinit + 1, total_step - tfinal)
        config = AdaLoraConfig(
            init_r=32,
            target_r=rank,
            tinit=tinit,
            tfinal=tfinal,
            deltaT=nbs_allocation_interval,
            lora_alpha=32,
            target_modules=TARGET_MODULES[plm_type],
            lora_dropout=0.05,
            bias='none',
            task_type=task_type,
            total_step=total_step,
        )
    else:
        config = LoraConfig(
            r=rank,
            lora_alpha=32,
            target_modules=TARGET_MODULES[plm_type],
            lora_dropout=0.05,
            bias="none",
            task_type=task_type
        )

    model = get_peft_model(plm, config)
    if nbs_v19:
        patched = _patch_adalora_mixed_precision(model)
        if patched == 0:
            raise RuntimeError('NBS v19 found no AdaLoRA SVDLinear modules')
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
            'NBS v19 enabled: min=2 max=32 budget={} seed-controlled '
            'initialization, allocation interval={}'.format(
                nbs_rank_budget, nbs_allocation_interval
            )
        )
    if print_trainable:
        print_trainable_parameters(model)
    return model
