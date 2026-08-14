"""
Patch selection module for the ViT patch-selection multimodal mode.

Given a historical viewport trajectory (roll, pitch, yaw per timestep), predicts which
grid patches of the *upcoming* frame are relevant enough to feed into the LLM as
image tokens. Trained separately from the VLM body: forward() produces one logit per
patch (independent binary classification, not softmax), and compute_loss() is the BCE
sum used for that pretraining.

Architecture:
    historical viewport (B, T, 3) -> linear + positional embedding -> N-layer
    transformer encoder -> memory (B, T, D)
    N learnable latent query vectors (grid_rows*grid_cols of them) cross-attend to
    memory -> per-query relevance logit (B, N)

Labeling for pretraining: reuse utils.patch_labeling.viewport_sequence_to_patch_labels
on the *future* viewport window to build the (B, N) binary target.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchSelectionModule(nn.Module):
    def __init__(self, grid_rows=4, grid_cols=4, d_model=128, nhead=8, num_encoder_layers=4,
                 dim_feedforward=256, history_input_dim=3, max_history_len=64, dropout=0.1):
        super().__init__()
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.num_patches = grid_rows * grid_cols
        self.d_model = d_model

        self.input_proj = nn.Linear(history_input_dim, d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_history_len, d_model))
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        self.patch_queries = nn.Parameter(torch.randn(self.num_patches, d_model) * 0.02)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.query_norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, 1)

    def forward(self, history_viewports):
        """
        :param history_viewports: (B, T, 3) tensor of (roll, pitch, yaw) history
        :return: (B, num_patches) relevance logits, one per patch (independent binary
            classification, not mutually exclusive)
        """
        B, T, _ = history_viewports.shape
        x = self.input_proj(history_viewports) + self.pos_embedding[:, :T, :]
        memory = self.encoder(x)  # (B, T, D)

        queries = self.patch_queries.unsqueeze(0).expand(B, -1, -1)  # (B, N, D)
        attn_out, _ = self.cross_attn(queries, memory, memory)  # (B, N, D)
        attn_out = self.query_norm(attn_out + queries)
        logits = self.classifier(attn_out).squeeze(-1)  # (B, N)
        return logits

    def compute_loss(self, logits, labels):
        """
        :param logits: (B, num_patches) raw logits from forward()
        :param labels: (B, num_patches) binary targets, e.g. from
            utils.patch_labeling.viewport_sequence_to_patch_labels
        :return: scalar loss = BCE summed over patches, averaged over the batch
        """
        per_patch = F.binary_cross_entropy_with_logits(logits, labels, reduction='none')  # (B, N)
        return per_patch.sum(dim=1).mean()

    def select_patches(self, logits, top_k=None, threshold=None):
        """
        :param logits: (B, num_patches) raw logits from forward()
        :param top_k: if set, select the top_k highest-logit patches per sample
        :param threshold: if set (and top_k is None), select patches with
            sigmoid(logit) > threshold
        :return: (B, num_patches) boolean mask of selected patches
        """
        if top_k is not None:
            B, N = logits.shape
            top_k = min(top_k, N)
            _, idx = torch.topk(logits, top_k, dim=1)
            mask = torch.zeros_like(logits, dtype=torch.bool)
            mask.scatter_(1, idx, True)
            return mask
        threshold = 0.5 if threshold is None else threshold
        return torch.sigmoid(logits) > threshold


def crop_patches(image, grid_rows, grid_cols):
    """
    Split a single image into a grid of non-overlapping patches.

    :param image: (C, H, W) tensor
    :return: (grid_rows * grid_cols, C, H // grid_rows, W // grid_cols) tensor, in
        row-major patch order matching utils.patch_labeling.grid_cell_to_patch_index
    """
    C, H, W = image.shape
    ph, pw = H // grid_rows, W // grid_cols
    patches = image[:, :ph * grid_rows, :pw * grid_cols]
    patches = patches.unfold(1, ph, ph).unfold(2, pw, pw)  # (C, grid_rows, grid_cols, ph, pw)
    patches = patches.permute(1, 2, 0, 3, 4).reshape(grid_rows * grid_cols, C, ph, pw)
    return patches


def crop_patches_at(image, grid_rows, grid_cols, indices):
    """
    Crop only the requested grid cells from an image, instead of the full grid (see
    crop_patches()). Lets callers that already know which patches they need (e.g.
    PatchSelectionModule.select_patches() picking k of num_patches) avoid slicing/copying
    the whole image when most of it will be discarded.

    :param image: (C, H, W) tensor
    :param indices: iterable of patch indices (row-major, matching crop_patches()/
        utils.patch_labeling.grid_cell_to_patch_index) to crop, in the given order
    :return: (len(indices), C, H // grid_rows, W // grid_cols) tensor
    """
    C, H, W = image.shape
    ph, pw = H // grid_rows, W // grid_cols
    out = []
    for idx in indices:
        r, c = divmod(idx, grid_cols)
        out.append(image[:, r * ph:(r + 1) * ph, c * pw:(c + 1) * pw])
    return torch.stack(out, dim=0)


def vit_features_for_patches(patches, patch_indices, feature_fn, device, patch_resize=224):
    """
    Run only the selected patches through a frozen feature extractor.

    :param patches: (num_patches, C, ph, pw) tensor from crop_patches()
    :param patch_indices: 1D iterable of patch indices to extract features for
    :param feature_fn: callable(img_batch, model=...) -> (k, feat_dim) tensor, e.g.
        dataset.extract_features.extract_vit_features (or a mock with the same signature
        for testing)
    :param device: device to run the feature extractor on
    :return: (len(patch_indices), feat_dim) tensor, gradient-free (frozen extractor)
    """
    selected = patches[list(patch_indices)].to(device)
    if selected.shape[-2:] != (patch_resize, patch_resize):
        selected = F.interpolate(selected, size=(patch_resize, patch_resize), mode='bilinear', align_corners=False)
    with torch.no_grad():
        features = feature_fn(selected)
    return features


if __name__ == '__main__':
    torch.manual_seed(0)
    grid_rows, grid_cols = 4, 4
    module = PatchSelectionModule(grid_rows=grid_rows, grid_cols=grid_cols, d_model=32, nhead=4,
                                   num_encoder_layers=4, dim_feedforward=64)

    B, T = 3, 10
    history = torch.randn(B, T, 3)
    logits = module(history)
    assert logits.shape == (B, grid_rows * grid_cols), f'unexpected logits shape {logits.shape}'

    labels = torch.randint(0, 2, (B, grid_rows * grid_cols)).float()
    loss = module.compute_loss(logits, labels)
    assert loss.dim() == 0, 'loss should be a scalar'
    assert torch.isfinite(loss), 'loss should be finite'

    module.zero_grad()
    loss.backward()
    assert module.classifier.weight.grad is not None and module.classifier.weight.grad.abs().sum() > 0, \
        'no gradient reached classifier head'
    assert module.patch_queries.grad is not None and module.patch_queries.grad.abs().sum() > 0, \
        'no gradient reached learnable patch queries'
    assert module.input_proj.weight.grad is not None and module.input_proj.weight.grad.abs().sum() > 0, \
        'no gradient reached the history encoder input projection'

    mask_topk = module.select_patches(logits, top_k=5)
    assert mask_topk.shape == (B, grid_rows * grid_cols)
    assert (mask_topk.sum(dim=1) == 5).all(), 'top_k selection should pick exactly k patches per sample'

    mask_thresh = module.select_patches(logits, threshold=0.5)
    assert mask_thresh.shape == (B, grid_rows * grid_cols)
    assert mask_thresh.dtype == torch.bool

    # patch cropping + selective feature extraction, with a mock ViT-shaped feature_fn
    # so this test doesn't depend on real pretrained ViT weights
    image = torch.randn(3, 224, 224)
    patches = crop_patches(image, grid_rows, grid_cols)
    assert patches.shape == (grid_rows * grid_cols, 3, 56, 56)

    def mock_vit(img_batch, model=None):
        return img_batch.mean(dim=(2, 3)).repeat(1, 768 // 3 + 1)[:, :768]

    selected_indices = [0, 5, 10]
    feats = vit_features_for_patches(patches, selected_indices, mock_vit, device='cpu')
    assert feats.shape == (3, 768), f'unexpected feature shape {feats.shape}'

    # crop_patches_at() must exactly match crop_patches()[indices] for any index subset/order
    at_all = crop_patches_at(image, grid_rows, grid_cols, list(range(grid_rows * grid_cols)))
    assert torch.equal(at_all, patches), 'crop_patches_at(all indices) must exactly match crop_patches()'
    at_subset = crop_patches_at(image, grid_rows, grid_cols, selected_indices)
    assert torch.equal(at_subset, patches[selected_indices]), \
        'crop_patches_at(subset) must exactly match crop_patches()[subset]'

    print('All patch_selection self-tests passed.')
