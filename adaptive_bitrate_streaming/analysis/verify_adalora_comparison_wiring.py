"""Tiny CPU checks for optional ABR stock/Shapley AdaLoRA wiring.

Run from ``adaptive_bitrate_streaming`` on the experiment server:

    python analysis/verify_adalora_comparison_wiring.py
"""

from pathlib import Path
import sys

ABR_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ABR_ROOT.parent
for path in (str(ABR_ROOT), str(PROJECT_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

import torch
import torch.nn as nn
from peft import TaskType

from plm_special.models.low_rank import peft_model


class TinyAttention(nn.Module):
    def __init__(self, width=8):
        super().__init__()
        self.q_proj = nn.Linear(width, width, bias=False)
        self.v_proj = nn.Linear(width, width, bias=False)


class TinyModel(nn.Module):
    def __init__(self, width=8, layers=2):
        super().__init__()
        self.layers = nn.ModuleList(
            [TinyAttention(width) for _ in range(layers)]
        )

    def gradient_checkpointing_enable(self):
        pass

    def enable_input_require_grads(self):
        pass

    def forward(self, input_ids=None, **kwargs):
        hidden = input_ids
        for layer in self.layers:
            hidden = layer.q_proj(hidden) + layer.v_proj(hidden)
        return hidden


def active_rank(model):
    return sum(
        int((parameter.detach().reshape(-1).abs() > 1e-12).sum().item())
        for name, parameter in model.named_parameters()
        if 'lora_E.default' in name
    )


def train(model, steps=6):
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=1e-2,
    )
    for step in range(1, steps + 1):
        model(input_ids=torch.randn(2, 8)).float().pow(2).mean().backward()
        optimizer.step()
        model.base_model.update_and_allocate(step)
        optimizer.zero_grad(set_to_none=True)


def main():
    torch.manual_seed(19)
    expected_budget = 8  # 4 q/v modules x target rank 2

    stock = peft_model(
        TinyModel(), 'llama', rank=4,
        task_type=TaskType.FEATURE_EXTRACTION,
        lora_method='adalora', total_step=6,
        adalora_rank_budget=expected_budget,
        adalora_allocation_interval=1,
    )
    train(stock)
    assert stock.abr_lora_method == 'adalora'
    assert not hasattr(stock, 'nash_rank_allocator')
    assert active_rank(stock) == expected_budget
    print('[PASS] ABR stock AdaLoRA exact global target budget')

    shapley = peft_model(
        TinyModel(), 'llama', rank=4,
        task_type=TaskType.FEATURE_EXTRACTION,
        lora_method='shapley', total_step=6,
        adalora_rank_budget=expected_budget,
        adalora_allocation_interval=1,
        shapley_permutations=1,
        shapley_truncate_fraction=0.0,
        shapley_seed=19,
    )
    fixed_input = torch.randn(2, 8)

    def fixed_loss():
        was_training = shapley.training
        shapley.eval()
        try:
            with torch.no_grad():
                return float(
                    shapley(input_ids=fixed_input).float().pow(2).mean().item()
                )
        finally:
            shapley.train(was_training)

    shapley.shapley_rank_allocator.set_loss_fn(fixed_loss)
    train(shapley)
    assert shapley.abr_lora_method == 'shapley'
    assert not hasattr(shapley, 'nash_rank_allocator')
    assert active_rank(shapley) == expected_budget
    assert shapley.shapley_rank_allocator.last_budget == expected_budget
    print('[PASS] ABR Shapley AdaLoRA exact global target budget')

    uniform = peft_model(
        TinyModel(), 'llama', rank=2,
        task_type=TaskType.FEATURE_EXTRACTION,
    )
    assert uniform.abr_lora_method == 'uniform'
    assert not hasattr(uniform, 'nash_rank_allocator')
    assert not hasattr(uniform, 'shapley_rank_allocator')
    print('[PASS] historical ABR Uniform LoRA remains isolated')

    module_names = [
        f'layers.{layer}.{projection}'
        for layer in range(2)
        for projection in ('q_proj', 'v_proj')
    ]
    components = {
        name: torch.eye(8, dtype=torch.float32)[:2].clone()
        for name in module_names
    }
    eva_state = {
        'version': 1,
        'method': 'eva',
        'rank_pattern': {name: 2 for name in module_names},
        'total_rank_budget': expected_budget,
        'metric': 'ratio',
        'max_components': 2,
        'components': components,
        'explained_variance': {},
        'representative_for': {name: name for name in module_names},
        'processed_batches': 2,
        'converged': True,
    }
    eva = peft_model(
        TinyModel(), 'llama', rank=4,
        task_type=TaskType.FEATURE_EXTRACTION,
        lora_method='eva', eva_state=eva_state,
    )
    assert eva.abr_lora_method == 'eva'
    assert eva.abr_effective_rank_budget == expected_budget
    assert not hasattr(eva, 'nash_rank_allocator')
    assert not hasattr(eva, 'shapley_rank_allocator')
    for name, module in eva.named_modules():
        if not hasattr(module, 'lora_B') or 'default' not in module.lora_B:
            continue
        assert torch.count_nonzero(module.lora_B['default'].weight) == 0
    print('[PASS] ABR EVA fixed ranks/PCA initialization remains isolated')


if __name__ == '__main__':
    main()
