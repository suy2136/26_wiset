from peft import LoraConfig, AdaLoraConfig, get_peft_model, TaskType
import torch
import torch.nn as nn

from models.rank_allocator import NashRankAllocator


TARGET_MODULES = {
    'llama': ["q_proj", "v_proj"],
    'mistral': ["q_proj", "k_proj", "v_proj", "o_proj"],
    'opt': None,
    'gpt2': None,
    'llava': ["q_proj", "v_proj"]
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


def peft_model(plm, plm_type, rank, task_type=TaskType.FEATURE_EXTRACTION,
                use_adalora=False, total_step=None, adalora_min_rank=None,
                adalora_ema_beta=0.9, adalora_eps=1e-8,
                adalora_allocation_interval=10, adalora_rank_budget=None,
                adalora_rank_config=None, adalora_missing_grad_policy="zero"):
    """
    :param use_adalora: if True, wrap with AdaLoraConfig instead of plain LoraConfig.
        Starts at init_r=rank*2 and prunes the rank budget down to target_r=rank
        over training.  When use_adalora=True, the training loop calls the
        project's NashRankAllocator after optimizer.step() and before
        zero_grad(), using lora_A/lora_B gradients and lora_E energy.
    :param total_step: required when use_adalora=True. Total number of OPTIMIZER
        update steps (post grad-accumulation) over the whole run, used to derive
        the rank-pruning warmup/cooldown schedule.
    """
    for param in plm.parameters():
        param.requires_grad = False
        if param.ndim == 1:
            param.data = param.data.to(torch.float32)

    plm.gradient_checkpointing_enable()
    plm.enable_input_require_grads()

    class CastOutputToFloat(nn.Sequential):
        def forward(self, x):
            return super().forward(x).to(torch.float32)

    if use_adalora:
        if total_step is None:
            raise ValueError("use_adalora=True requires total_step (total optimizer update steps)")
        # pruning schedule derived from the total number of optimizer steps
        tinit = max(1, int(total_step * 0.1))            # warmup: no pruning before this
        tfinal = max(tinit + 1, int(total_step * 0.15))   # stop pruning after this
        deltaT = 10                                       # reallocate every deltaT steps
        config = AdaLoraConfig(
            init_r=rank * 2,        # start with a larger rank ...
            target_r=rank,          # ... and prune down to this average rank
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
        config = LoraConfig(
            r=rank,
            lora_alpha=32,
            target_modules=TARGET_MODULES[plm_type],
            lora_dropout=0.05,
            bias="none",
            task_type=task_type
        )

    model = get_peft_model(plm, config)
    if use_adalora:
        if adalora_allocation_interval <= 0:
            raise ValueError("adalora_allocation_interval must be positive")
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
        )
        model.nash_rank_allocation_interval = int(adalora_allocation_interval)
    model.from_pretrained
    print_trainable_parameters(model)
    return model
