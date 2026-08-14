"""
Step 7.1: re-run the threshold=0 equivalence gates (LlamaSelectablePipeline,
LlamaSpeculativeBlockVerifyPipeline) against the REAL Llama2-7B checkpoint
(try_llama2_7b, rank=32 standard LoRA, q_proj/v_proj) on real Jin2022 samples
-- forward passes only, no training, no weight updates. Confirms the tiny-CPU
(analysis/verify_selectable_pipeline_equivalence.py,
analysis/verify_speculative_pipeline.py) equivalence gates hold on the real
model/dtype (fp16, GPU) before spending any GPU time on the step 7 mini
training smoke tests.

fp16 tolerance follows the same convention already established in this
project's other verification scripts and in the Soyun_ModuleHead package's
own gates: atol=2e-3 on fp16/GPU (vs atol=1e-5 on fp32/CPU), because chained
KV-cache extension hops expose BLAS floating-point reassociation noise at
that scale, not because anything is actually wrong.

Loads the real checkpoint (trained standard LoRA, not the AdaLoRA this
project is adding) purely because it's already on this instance and is more
representative than random-init weights -- accuracy of predictions is
irrelevant here, only that all four code paths (direct Pipeline, wrapped
selector=None, wrapped IdentitySelector, wrapped speculative threshold=0)
agree with each other.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader

from config import cfg
from dataset.load_dataset import create_dataset
from utils.plms_utils import load_plm
from utils.normalize import normalize_data
from models.llama import LlamaNetworkingHeadModel
from models.networking_head import NetworkingHead
from models.low_rank import peft_model
from models.pipeline import Pipeline
from models.selectable_pipeline import LlamaSelectablePipeline
from models.selectors import IdentitySelector
from models.speculative_pipeline import LlamaSpeculativeBlockVerifyPipeline

DEVICE = 'cuda'
EMBED_SIZE = 4096
FUT_WINDOW = 20
CHECKPOINT_PATH = '/workspace/data/ft_plms/try_llama2_7b'
NUM_SAMPLES = 3
ATOL_FP16 = 2e-3


def build_pipeline(multimodal_mode):
    plm, tokenizer, _ = load_plm(
        'llama', os.path.join(cfg.plms_dir, 'llama', 'base'), plm_size='base',
        device_input_side=DEVICE, device_output_side=DEVICE, device_middle_side=None,
        torch_dtype=torch.float16,
    )
    plm = peft_model(plm, 'llama', rank=32)  # matches the real checkpoint's adapter_config.json (r=32, q_proj/v_proj)

    # Real bug, found by an actual crash here (not obvious from reading dtypes in
    # isolation -- get_input_embeddings().weight.dtype reports fp16 correctly either
    # way). peft_model() upcasts every 1-D param -- including the base LlamaModel's own
    # RMSNorm weights -- to fp32 "for training stability", regardless of the base
    # model's own dtype. transformers==4.34.1's LlamaRMSNorm.forward() does
    # `return self.weight * hidden_states.to(input_dtype)`; if self.weight is fp32
    # (from that upcast) and hidden_states is fp16 (from torch_dtype=torch.float16
    # loading), float32 * float16 PROMOTES to float32, which then crashes the next
    # fp16 q_proj Linear ("expected mat1 and mat2 to have the same dtype"). This is
    # exactly why this project's checkpoints are trained/evaluated in fp32 with no
    # --fp16 (see prior project notes) -- fp16 was apparently never exercised
    # end-to-end in this codebase before this check. Fix: recast just the base
    # model's own norm weights back to fp16 (matching the loaded base weights) --
    # leaves LoRA's lora_A/B fp32 masters and the (not yet attached) networking_head
    # untouched, unlike a blanket pipeline.half() would. Only doing this here, in
    # this forward-only equivalence check -- the step 7.2 AdaLoRA training smoke
    # test does not (and must not) touch this, since it needs the fp32 masters.
    for name, param in plm.named_parameters():
        if 'norm.weight' in name:
            param.data = param.data.to(torch.float16)

    networking_head = NetworkingHead(input_dim=EMBED_SIZE, output_dim=3, fut_window=FUT_WINDOW).to(DEVICE)
    plm.set_networking_head(networking_head)

    vit_model = None
    if multimodal_mode in ('all-patch', 'patch-selection'):
        import torchvision
        vit_model = torchvision.models.vit_b_16(pretrained=True).to(DEVICE)

    pipeline = Pipeline(
        plm, fut_window=FUT_WINDOW, device=DEVICE, embed_size=EMBED_SIZE, frequency=5,
        multimodal_mode=multimodal_mode, dataset='Jin2022', vit_model=vit_model,
    )

    model_path = CHECKPOINT_PATH
    pipeline.plm.load_adapter(model_path, adapter_name='default')
    pipeline.plm.set_adapter('default')
    state = torch.load(os.path.join(model_path, 'modules_except_plm.bin'), map_location=DEVICE)
    # This checkpoint predates this project's rename of the task head module from
    # "task_head" to "networking_head" -- its modules_except_plm.bin was saved with the
    # old key name (confirmed by an actual failed strict load: missing
    # '4.networking_head.0.{weight,bias}', unexpected '4.task_head.0.{weight,bias}').
    # Remap rather than loading strict=False + leaving the head randomly initialized,
    # so this check exercises the real trained weights, not an untrained head.
    state = {k.replace('task_head', 'networking_head'): v for k, v in state.items()}
    incompatible = pipeline.modules_except_plm.load_state_dict(state, strict=True)
    assert not incompatible.missing_keys and not incompatible.unexpected_keys, incompatible
    print(f'  [load] modules_except_plm loaded with 0 missing / 0 unexpected keys '
          f'(after task_head->networking_head rename fix)')
    pipeline.eval()
    return pipeline


def get_samples(n=NUM_SAMPLES):
    raw_test = create_dataset('Jin2022', include=['test'])[0]
    loader = DataLoader(raw_test, batch_size=1, shuffle=False)
    samples = []
    for i, (history, future, video_user_info) in enumerate(loader):
        if i >= n:
            break
        samples.append((history.to(DEVICE), future.to(DEVICE), video_user_info))
    return samples


def check_mode(multimodal_mode):
    print(f"=== multimodal_mode={multimodal_mode!r} (real Llama2-7B, fp16, GPU) ===")
    pipeline = build_pipeline(multimodal_mode)
    samples = get_samples()

    max_diff_identity_all = 0.0
    max_diff_speculative_all = 0.0
    with torch.no_grad():
        for idx, (history, future, video_user_info) in enumerate(samples):
            history_n = normalize_data(history, 'Jin2022')

            ref = pipeline.auto_regressive(history_n, future, video_user_info)

            out_none = LlamaSelectablePipeline(pipeline, selector=None).auto_regressive(
                history_n, future, video_user_info)
            assert torch.equal(ref, out_none), f"[{multimodal_mode}] sample {idx}: selector=None must match exactly"

            out_identity = LlamaSelectablePipeline(pipeline, selector=IdentitySelector()).auto_regressive(
                history_n, future, video_user_info)
            diff = (ref.float() - out_identity.float()).abs().max().item()
            max_diff_identity_all = max(max_diff_identity_all, diff)
            assert diff <= ATOL_FP16, f"[{multimodal_mode}] sample {idx}: IdentitySelector diff {diff} > {ATOL_FP16}"

            spec = LlamaSpeculativeBlockVerifyPipeline(pipeline, selector=None, gamma=4, acceptance_threshold=0.0)
            out_spec = spec.auto_regressive(history_n, video_user_info)
            diff_spec = (ref.float() - out_spec.float()).abs().max().item()
            max_diff_speculative_all = max(max_diff_speculative_all, diff_spec)
            assert diff_spec <= ATOL_FP16, f"[{multimodal_mode}] sample {idx}: speculative@0 diff {diff_spec} > {ATOL_FP16}"
            assert spec.target_forward_count == FUT_WINDOW, \
                f"[{multimodal_mode}] sample {idx}: expected {FUT_WINDOW} forwards at threshold=0, got {spec.target_forward_count}"

    print(f"  [PASS] selector=None matches Pipeline.auto_regressive() exactly on all {len(samples)} real samples")
    print(f"  [PASS] IdentitySelector max diff over samples = {max_diff_identity_all:.2e} (atol={ATOL_FP16})")
    print(f"  [PASS] Speculative@threshold=0 max diff over samples = {max_diff_speculative_all:.2e} "
          f"(atol={ATOL_FP16}), forward_count={FUT_WINDOW} every time")
    print(f"  peak GPU memory this mode: {torch.cuda.max_memory_allocated(DEVICE) / 1024**3:.2f} GiB")
    del pipeline
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(DEVICE)
    print()


if __name__ == "__main__":
    for mode in ("baseline", "all-patch", "patch-selection"):
        check_mode(mode)
    print("All real-7B equivalence checks passed.")
