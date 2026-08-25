from peft import LoraConfig, AdaLoraConfig, get_peft_model, TaskType
import re
import torch
import torch.nn as nn
from types import MethodType

from models.rank_allocator import NashRankAllocator
from models.eva_initializer import eva_lora_spec, initialize_eva_lora_weights


TARGET_MODULES = {
    'llama': ["q_proj", "v_proj"],
    'mistral': ["q_proj", "k_proj", "v_proj", "o_proj"],
    'opt': None,
    'gpt2': None,
    'llava': ["q_proj", "v_proj"]
}


def _adalora_physical_rank(rank, rank_config):
    """Choose the physical AdaLoRA width from configured layer ceilings.

    The custom allocator masks physical ``lora_E`` slots instead of resizing
    tensors during training.  Consequently the physical width must cover the
    largest configured per-layer maximum, but slots beyond that maximum only
    waste adapter/optimizer memory and introduce spectral candidates that the
    NBS formulation does not define.
    """
    configured_maxima = []
    if isinstance(rank_config, dict):
        for layer_config in rank_config.values():
            if isinstance(layer_config, dict) and layer_config.get("max_rank") is not None:
                configured_maxima.append(int(layer_config["max_rank"]))
    physical_rank = max(configured_maxima) if configured_maxima else int(rank) * 2
    if physical_rank <= 0:
        raise ValueError("AdaLoRA physical rank must be positive")
    if physical_rank < int(rank):
        raise ValueError(
            f"configured max_rank {physical_rank} is smaller than AdaLoRA target rank {rank}"
        )
    return physical_rank


def _mixed_precision_adalora_forward(self, x):
    """PEFT 0.6.x SVDLinear forward with explicit mixed-dtype bridges.

    PEFT 0.6.2 leaves AdaLoRA A/B/E in fp32 while an fp16-loaded base model
    keeps the frozen Linear weight in fp16.  Its stock SVDLinear.forward does
    not cast either branch, so both the base projection and adapter matmuls
    can fail with a float/half mismatch.  Keep adapter math in fp32 for
    training stability, then cast only the delta back to the base result.
    """
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
    """Patch only legacy PEFT SVDLinear modules that need dtype bridging."""
    patched = 0
    for module in model.modules():
        if module.__class__.__name__ != "SVDLinear":
            continue
        if not all(hasattr(module, name) for name in ("lora_A", "lora_B", "lora_E", "ranknum", "_linear")):
            continue
        module.forward = MethodType(_mixed_precision_adalora_forward, module)
        patched += 1
    return patched


def _verify_fixed_lora_rank_pattern(model, rank_pattern, adapter_name="default"):
    """Fail early if a PEFT rank pattern did not map one-to-one to LoRA modules."""
    named_modules = dict(model.named_modules())
    verified = {}
    for pattern, expected_rank in rank_pattern.items():
        matches = [
            (name, module)
            for name, module in named_modules.items()
            if re.match(rf".*\.{pattern}$", name)
            and hasattr(module, "lora_A")
            and adapter_name in module.lora_A
        ]
        if len(matches) != 1:
            raise ValueError(
                f"fixed LoRA rank pattern {pattern!r} matched {len(matches)} modules; expected 1"
            )
        module_name, module = matches[0]
        actual_rank = int(module.lora_A[adapter_name].weight.shape[0])
        if actual_rank != int(expected_rank):
            raise ValueError(
                f"fixed LoRA rank mismatch for {module_name}: "
                f"configured {expected_rank}, created {actual_rank}"
            )
        verified[module_name] = actual_rank
    print(
        "Verified fixed LoRA rank pattern: modules={}, total active rank={}".format(
            len(verified), sum(verified.values())
        )
    )


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


def peft_model(plm, plm_type, rank, task_type=TaskType.FEATURE_EXTRACTION,
                use_adalora=False, total_step=None, adalora_min_rank=None,
                adalora_ema_beta=0.9, adalora_eps=1e-8,
                adalora_allocation_interval=10, adalora_rank_budget=None,
                adalora_rank_config=None, adalora_missing_grad_policy="zero",
                lora_rank_pattern=None, adalora_allocator="nbs",
                adalora_shadow_update_policy="legacy",
                adalora_budget_mode="fixed",
                adalora_relative_lambda=0.15,
                adalora_adaptive_min_budget=None,
                adalora_adaptive_max_budget=None,
                shapley_permutations=3,
                shapley_truncate_fraction=0.05,
                shapley_antithetic=True,
                shapley_seed=0,
                eva_state=None):
    """
    :param use_adalora: if True, wrap with AdaLoraConfig instead of plain LoraConfig.
        Uses the largest configured layer max_rank as the physical init_r
        (falling back to rank*2 when no layer configuration is supplied).
        When use_adalora=True, the training loop calls the
        project's NashRankAllocator after optimizer.step() and before
        zero_grad(), using lora_A/lora_B gradients and lora_E energy.
    :param total_step: required when use_adalora=True. Total number of OPTIMIZER
        update steps (post grad-accumulation) over the whole run, used to derive
        the rank-pruning warmup/cooldown schedule.
    """
    if adalora_allocator not in ("nbs", "peft", "shapley"):
        raise ValueError("adalora_allocator must be 'nbs', 'peft', or 'shapley'")
    if adalora_allocator == "shapley" and not use_adalora:
        raise ValueError("the Shapley allocator requires use_adalora=True")
    if adalora_allocator == "shapley" and any(
        value is not None
        for value in (adalora_min_rank, adalora_rank_budget, adalora_rank_config)
    ):
        raise ValueError(
            "Shapley uses PEFT's target_r budget schedule; NBS min-rank, "
            "rank-budget, and layer-rank-config options are not supported"
        )
    if adalora_budget_mode not in ("fixed", "adaptive"):
        raise ValueError("adalora_budget_mode must be 'fixed' or 'adaptive'")
    if adalora_allocator != "nbs" and adalora_budget_mode != "fixed":
        raise ValueError("adaptive budget mode is only available with the NBS allocator")

    for param in plm.parameters():
        param.requires_grad = False
        # Keep frozen normalization weights in the dtype selected by
        # from_pretrained().  Upcasting every 1-D parameter to fp32 promotes
        # LlamaRMSNorm outputs to fp32, which then breaks the following
        # unadapted fp16 k/o projections outside autocast.

    plm.gradient_checkpointing_enable()
    plm.enable_input_require_grads()

    class CastOutputToFloat(nn.Sequential):
        def forward(self, x):
            return super().forward(x).to(torch.float32)

    if use_adalora:
        if eva_state is not None:
            raise ValueError("EVA initialization cannot be combined with AdaLoRA")
        if total_step is None:
            raise ValueError("use_adalora=True requires total_step (total optimizer update steps)")
        # Training-time schedule derived from optimizer updates.  The NBS
        # objective is unchanged; the schedule only controls when it may alter
        # the active lora_E mask.
        tinit = max(1, int(total_step * 0.1))
        tfinal = max(tinit + 1, int(total_step * 0.15))
        cooldown_start = max(tinit + 1, total_step - tfinal)
        deltaT = int(adalora_allocation_interval)
        physical_rank = (
            _adalora_physical_rank(rank, adalora_rank_config)
            if adalora_allocator == "nbs"
            else int(rank) * 2
        )
        config = AdaLoraConfig(
            init_r=physical_rank,
            target_r=rank,
            tinit=tinit,
            tfinal=tfinal,
            deltaT=deltaT,
            lora_alpha=32,
            target_modules=TARGET_MODULES[plm_type],
            lora_dropout=0.05,
            bias="none",
            task_type=task_type,
            total_step=total_step,
        )
    else:
        rank_pattern = {}
        target_modules = TARGET_MODULES[plm_type]
        if eva_state is not None:
            if lora_rank_pattern is not None:
                raise ValueError(
                    "EVA initialization cannot be combined with lora_rank_pattern"
                )
            eva_spec = eva_lora_spec(eva_state)
            rank_pattern = eva_spec["rank_pattern"]
            target_modules = eva_spec["target_modules"]
        elif lora_rank_pattern is not None:
            if not isinstance(lora_rank_pattern, dict):
                raise TypeError("lora_rank_pattern must be a dictionary")
            rank_pattern = {str(name): int(value) for name, value in lora_rank_pattern.items()}
            if any(value <= 0 for value in rank_pattern.values()):
                raise ValueError("all fixed LoRA ranks must be positive")
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
    if use_adalora:
        patched_layers = _patch_adalora_mixed_precision(model)
        if patched_layers == 0:
            raise RuntimeError("No PEFT AdaLoRA SVDLinear layers were patched for mixed precision")
        if adalora_allocation_interval <= 0:
            raise ValueError("adalora_allocation_interval must be positive")
        if adalora_allocator == "nbs":
            # The custom allocator keeps PEFT's physical init_r slots and applies
            # rank selection through lora_E masks, so adapter checkpoint format and
            # the existing AdaLoRA forward path remain unchanged.
            model.nash_rank_allocator = NashRankAllocator(
                model,
                target_rank=rank,
                min_rank=adalora_min_rank,
                ema_beta=adalora_ema_beta,
                eps=adalora_eps,
                rank_budget=adalora_rank_budget,
                rank_config=adalora_rank_config,
                missing_grad_policy=adalora_missing_grad_policy,
                warmup_steps=tinit,
                cooldown_start_step=cooldown_start,
                allocation_interval=adalora_allocation_interval,
                shadow_update_policy=adalora_shadow_update_policy,
                budget_mode=adalora_budget_mode,
                relative_lambda=adalora_relative_lambda,
                adaptive_min_budget=adalora_adaptive_min_budget,
                adaptive_max_budget=adalora_adaptive_max_budget,
            )
            model.nash_rank_allocation_interval = int(adalora_allocation_interval)
            model.nash_physical_rank = int(physical_rank)
            print(
                "NBS physical AdaLoRA rank: init_r={} (largest configured max_rank)".format(
                    physical_rank
                )
            )
            print(
                "NBS spectral-shadow update policy: {}".format(
                    adalora_shadow_update_policy
                )
            )
            allocator = model.nash_rank_allocator
            if allocator.budget_mode == "adaptive":
                print(
                    "NBS budget mode: adaptive (warm-up/cap={}, floor={}, "
                    "relative lambda={})".format(
                        allocator.adaptive_max_budget,
                        allocator.adaptive_min_budget,
                        allocator.relative_lambda,
                    )
                )
            else:
                print("NBS budget mode: fixed (rank budget={})".format(
                    allocator.rank_budget
                ))
            print(
                "NBS rank schedule: warm-up steps 1-{}, allocation window {}-{}, "
                "cooldown steps {}-{} (interval={})".format(
                    tinit,
                    tinit + 1,
                    max(tinit, cooldown_start - 1),
                    cooldown_start,
                    total_step,
                    adalora_allocation_interval,
                )
            )
        elif adalora_allocator == "shapley":
            # Lazy import keeps every historical LoRA/AdaLoRA/NBS path
            # independent from the optional comparison allocator.
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
                "Shapley AdaLoRA allocator: permutations={}, "
                "truncate_fraction={}, antithetic={}, seed={}".format(
                    shapley_permutations,
                    shapley_truncate_fraction,
                    shapley_antithetic,
                    shapley_seed,
                )
            )
            print(
                "PEFT AdaLoRA rank schedule: init_r={}, target_r={}, tinit={}, "
                "tfinal={}, deltaT={}, total_step={}".format(
                    physical_rank, rank, tinit, tfinal,
                    adalora_allocation_interval, total_step,
                )
            )
        else:
            print(
                "PEFT AdaLoRA rank schedule: init_r={}, target_r={}, tinit={}, "
                "tfinal={}, deltaT={}, total_step={}".format(
                    physical_rank, rank, tinit, tfinal,
                    adalora_allocation_interval, total_step,
                )
            )
    elif eva_state is not None:
        summary = initialize_eva_lora_weights(model, eva_state)
        _verify_fixed_lora_rank_pattern(model, rank_pattern)
        model.eva_initialization_summary = summary
        # Keep the small CPU EVA state with the adapter so a moved checkpoint
        # can reconstruct its exact rank pattern without the original run dir.
        model.eva_state = eva_state
        print(
            "EVA LoRA initialization: modules={}, total active rank={}, "
            "lora_A=PCA components, lora_B=0".format(
                summary["initialized_modules"], summary["total_rank_budget"]
            )
        )
    elif rank_pattern:
        _verify_fixed_lora_rank_pattern(model, rank_pattern)
    model.from_pretrained
    print_trainable_parameters(model)
    return model
