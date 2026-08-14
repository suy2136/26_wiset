import os
import torch
import torch.nn as nn
from typing import *
from PIL import Image
from torchvision import transforms
from transformers.utils.dummy_pt_objects import PreTrainedModel
from config import cfg
from dataset.extract_features import extract_vit_features
from models.patch_selection import PatchSelectionModule, crop_patches, crop_patches_at, vit_features_for_patches
from utils.frame_utils import FrameIndexClamper

MULTIMODAL_MODES = ('none', 'baseline', 'all-patch', 'patch-selection')


class Pipeline(nn.Module):
    '''
    Pipeline for viewport prediction.
    '''
    def __init__(self,
                plm: PreTrainedModel,
                loss_func = None,
                fut_window = None,
                device = 'cuda',
                embed_size = 1024,
                frequency = 5,
                using_multimodal = False,
                multimodal_mode = None,
                dataset = None,
                patch_selection_module: PatchSelectionModule = None,
                vit_model = None,
                patch_grid = None,
                patch_top_k = None,
                patch_threshold = None,
                ):
        """
        :param plm: the pretrained llm
        :param embed_size: the embed size of llm
        :param frequency: the frequency of dataset
        :param fut_window: future (prediction) window
        :param dataset: the dataset
        :param using_multimodal: (deprecated, kept for backward compatibility) equivalent to
            multimodal_mode='baseline' when True, multimodal_mode='none' when False. Ignored
            if multimodal_mode is explicitly given.
        :param multimodal_mode: one of 'none', 'baseline', 'all-patch', 'patch-selection'.
            - 'baseline': single offline-cached ViT CLS-token feature per frame (original behavior)
            - 'all-patch': frame split into patch_grid patches, ALL patches run through a frozen
              ViT, each patch fed as its own token (no pooling)
            - 'patch-selection': patch_selection_module picks a subset of patches from the
              historical viewport, only those patches run through the frozen ViT
        :param patch_selection_module: required (or lazily created, untrained) for
            multimodal_mode='patch-selection'
        :param vit_model: frozen ViT feature extractor for 'all-patch'/'patch-selection' modes;
            lazily loaded (ImageNet-pretrained vit_b_16) if not provided
        :param patch_grid: (rows, cols) grid for 'all-patch'/'patch-selection'; defaults to
            cfg.default_patch_grid
        :param patch_top_k: for 'patch-selection', select exactly this many patches (takes
            precedence over patch_threshold)
        :param patch_threshold: for 'patch-selection', select patches with sigmoid(logit) above
            this threshold; ignored if patch_top_k is set. Defaults to 0.5 if neither is set.
        :param device: cuda or cpu
        """
        super().__init__()
        self.plm = plm
        if multimodal_mode is None:
            multimodal_mode = 'baseline' if using_multimodal else 'none'
        assert multimodal_mode in MULTIMODAL_MODES, f'multimodal_mode must be one of {MULTIMODAL_MODES}, got {multimodal_mode}'
        self.multimodal_mode = multimodal_mode
        self.using_multimodal = multimodal_mode != 'none'
        self.dataset = dataset
        self.device = device
        self.frequency = frequency
        self.embed_size = embed_size
        self.fut_window_length = fut_window

        self.conv1d = nn.Sequential(nn.Conv1d(1, 256, 3), nn.LeakyReLU(), nn.Flatten()).to(device)
        self.embed_vp = nn.Linear(256, self.embed_size).to(device)
        self.embed_multimodal = nn.Linear(768, embed_size).to(device)  # 768 = ViT output feature size
        self.embed_ln = nn.LayerNorm(self.embed_size).to(device)

        self.loaded_tensor_cache = {}
        self.modules_except_plm = nn.ModuleList([  # used to save and load modules except plm
            self.embed_vp, self.embed_multimodal, self.embed_ln, self.conv1d, self.plm.networking_head
        ])

        self.patch_grid = patch_grid or cfg.default_patch_grid
        self.grid_rows, self.grid_cols = self.patch_grid
        self.patch_top_k = patch_top_k
        self.patch_threshold = patch_threshold
        self.patch_selection_module = patch_selection_module
        self.vit_model = vit_model
        self._patch_image_cache = {}
        # one entry appended per _get_multimodal_information_patch_selection() call (i.e.
        # per sample, in 'patch-selection' mode only) -- lets callers report how many
        # patches actually got selected (meaningful for threshold selection; constant for
        # top_k). Not touched by any other mode.
        self.patch_selection_history = []

        # frame-index clamping applies to every multimodal mode that indexes into per-video
        # frame data (baseline's offline cache included -- it only has entries up to the real
        # frame count for videos 9/18/27, so unclamped indices would KeyError/FileNotFoundError
        # on their tails), not just the raw-image patch modes.
        self.frame_clamper = FrameIndexClamper(self.dataset) if (self.using_multimodal and self.dataset == 'Jin2022') else None

        if self.multimodal_mode in ('all-patch', 'patch-selection'):
            if self.vit_model is None:
                import torchvision
                print('\033[33mWarning:\033[0m no vit_model passed to Pipeline; loading a fresh '
                      'ImageNet-pretrained vit_b_16. Pass one explicitly to avoid reloading it '
                      'per Pipeline instance.')
                self.vit_model = torchvision.models.vit_b_16(pretrained=True).to(device)
            self.vit_model.eval()
            for p in self.vit_model.parameters():
                p.requires_grad_(False)
            self.raw_image_transform = transforms.ToTensor()

        if self.multimodal_mode == 'patch-selection' and self.patch_selection_module is None:
            print('\033[33mWarning:\033[0m no patch_selection_module passed to Pipeline; '
                  'creating a fresh (UNTRAINED) one. Pretrain it separately before real runs.')
            self.patch_selection_module = PatchSelectionModule(
                grid_rows=self.grid_rows, grid_cols=self.grid_cols).to(device)

        if loss_func is None:
            loss_func = nn.MSELoss()
        self.loss_fct = loss_func
        self.fut_window = fut_window
    
    def forward(self, batch, future, video_user_position, teacher_forcing=True) -> torch.Tensor:
        """
        :param batch: history viewport trajectory
        :param future: future viewport trajectory
        :param video_user_position: details information for current trajectory
        :return: the loss value for training
        """
        if teacher_forcing:
            pred = self.teaching_forcing(batch, future, video_user_position)
        else:
            pred = self.auto_regressive(batch, future, video_user_position)
        gt = future.to(pred.device)
        loss = self.loss_fct(pred, gt)
        return loss
    
    def auto_regressive(self, x, future, video_user_position) -> torch.Tensor:
        """
        auto-regressive generation

        :return: the loss value for training
        """
        history_viewports = x  # raw (B, his_window, 3) history, before conv1d embedding
        seq_len = x.shape[1]
        batch_embeddings = []
        for i in range(seq_len):
            batch_embeddings.append(self.embed_vp(self.conv1d(x[:, i, :]).view(1,256)).unsqueeze(1))
        x = torch.cat(batch_embeddings, dim=1)

        if self.using_multimodal:  # we make using multimodal image features as an option, as not all datasets provide video information.
            mapped_tensor = self.get_multimodal_information(video_user_position, history_viewports)
            x = torch.cat([mapped_tensor, x], dim=1)

        x = self.embed_ln(x)

        outputlist = []
        # next(self.plm.parameters()).dtype is unreliable once a LoRA adapter is loaded:
        # get_peft_model()/load_adapter() leave lora_A/lora_B in fp32 (mixed-precision
        # master weights) while the frozen base weights stay fp16, and whichever happens
        # to be first in parameter-iteration order decides the (wrong, if fp32) result --
        # this silently produced fp32 inputs_embeds into a fp16 frozen linear and crashed
        # with a dtype mismatch during real (adapter-loaded) evaluation. embed_tokens is
        # never LoRA-adapted, so its weight dtype reliably reflects the frozen base dtype.
        plm_dtype = self.plm.get_input_embeddings().weight.dtype
        # KV-cache the growing sequence instead of re-feeding it from scratch every step:
        # causal attention means a token's hidden state only depends on itself and earlier
        # tokens, so incrementally feeding just the newest token (with past_key_values
        # covering everything before it) is mathematically equivalent to reprocessing the
        # whole sequence each time, but turns the fut_window-step loop from ~O(fut_window^2)
        # into ~O(fut_window) token-passes through the plm.
        past_key_values = None
        current_input = x
        total_len = x.shape[1]
        for _ in range(self.fut_window_length):
            attention_mask = torch.ones(x.shape[0], total_len, dtype=torch.long, device=self.device)
            outputs = self.plm(inputs_embeds=current_input.to(plm_dtype), attention_mask=attention_mask,
                                past_key_values=past_key_values, use_cache=True)
            logits = outputs.logits.float()  # bridge back to fp32 for conv1d/embed_vp when plm runs in fp16
            outputlist.append(logits)
            past_key_values = outputs.past_key_values
            current_input = self.embed_vp(self.conv1d(logits)).unsqueeze(1)  # only the new token goes in next
            total_len += 1

        pred = torch.cat(outputlist, dim=1)
        return pred

    def teaching_forcing(self, x, future, video_user_position) -> torch.Tensor:
        """
        teaching-forcing generation

        :param x: history viewport trajectory
        :param future: future viewport trajectory
        :param video_user_position: details information for current trajectory
        :return: the return value by llm
        """

        history_viewports = x  # raw (B, his_window, 3) history, before it's concatenated with future
        x = torch.cat((x, future), dim=1)
        seq_len = x.shape[1]
        batch_embeddings = []
        for i in range(seq_len):
            batch_embeddings.append(self.embed_vp(self.conv1d(x[:, i, :]).view(1,256)).unsqueeze(1))
        x = torch.cat(batch_embeddings, dim=1)

        if self.using_multimodal:
            mapped_tensor = self.get_multimodal_information(video_user_position, history_viewports)
            x = torch.cat([mapped_tensor, x], dim=1)
        
        x = self.embed_ln(x)

        plm_dtype = self.plm.get_input_embeddings().weight.dtype  # see auto_regressive() for why not next(self.plm.parameters()).dtype
        outputs = self.plm(inputs_embeds=x.to(plm_dtype), attention_mask = torch.ones(x.shape[0], x.shape[1], dtype=torch.long, device=self.device), teacher_forcing=True)
        return outputs.logits.float()
    
    def inference(self, batch, future, video_user_info) -> torch.Tensor:
        """
        Inference function. Use it for testing.
        """
        pred = self.auto_regressive(batch, future, video_user_info)
        gt = future.to(pred.device)
        return pred, gt
    
    def _resolve_frame_index(self, video_user_position):
        """
        :param video_user_position: details information for current trajectory
        :return: (video_index, image_index) for the current sample. image_index is clamped to
            the actual available frame count when a frame_clamper is configured (raw-image
            modes only; the offline CLS-token cache used by 'baseline' predates clamping and
            keeps its original, unclamped indexing to avoid changing existing cached results).
        """
        video_index = video_user_position[0].item()
        position_index = video_user_position[2].item()
        image_index = (position_index - 1) * (cfg.video_frame[self.dataset][video_index-1]//self.frequency)
        if getattr(self, 'frame_clamper', None) is not None:
            image_index = self.frame_clamper.clamp(video_index, image_index)
        return video_index, image_index

    def get_multimodal_information(self, video_user_position, history_viewports=None):
        """
        Get the image-derived tokens to prepend to the LLM input sequence. Dispatches on
        self.multimodal_mode:
        - 'baseline': single offline-cached ViT CLS-token feature per frame
        - 'all-patch': all patch_grid patches of the frame, each through a frozen ViT
        - 'patch-selection': only the patches picked by self.patch_selection_module from
          history_viewports, each through a frozen ViT

        :param video_user_position: details information for current trajectory
        :param history_viewports: (B, his_window, 3) raw historical viewport trajectory,
            required for 'patch-selection'
        :return: (1, num_tokens, embed_size) tensor of image tokens
        """
        if self.multimodal_mode == 'baseline':
            return self._get_multimodal_information_baseline(video_user_position)
        elif self.multimodal_mode == 'all-patch':
            return self._get_multimodal_information_all_patch(video_user_position)
        elif self.multimodal_mode == 'patch-selection':
            if history_viewports is None:
                raise ValueError("multimodal_mode='patch-selection' requires history_viewports")
            return self._get_multimodal_information_patch_selection(video_user_position, history_viewports)
        else:
            raise ValueError(f'unsupported multimodal_mode: {self.multimodal_mode}')

    def _get_multimodal_information_baseline(self, video_user_position):
        """
        Note that we use ViT to extract image features (the first output features of ViT that contains the overall information of the image).
        Since we use the frozen ViT for image feature extraction, we can actually use ViT to extract features first,
        then store all features into disk, and fetch them when needed.
        This way, we can avoid repeatedly using ViT to extract features for the same images.
        As a result, we can speed up the training process.
        """
        video_index, image_index = self._resolve_frame_index(video_user_position)
        # add cache_key
        if image_index % 100 == 0:
            cache_key = f'{video_index}_{image_index//100}'
        else:
            cache_key = f'{video_index}_{(image_index//100)+1}'
        if cache_key in self.loaded_tensor_cache:
            loaded_tensor_dict = self.loaded_tensor_cache[cache_key]
        else:
            if image_index % 100 == 0:
                loaded_tensor_dict = torch.load(os.path.join(cfg.dataset_image_features[self.dataset], f'video{video_index}_images/feature_dict{(image_index//100)}.pth'))
            else:
                loaded_tensor_dict = torch.load(os.path.join(cfg.dataset_image_features[self.dataset], f'video{video_index}_images/feature_dict{(image_index//100) + 1}.pth'))

        self.loaded_tensor_cache[cache_key] = loaded_tensor_dict  # add to loaded_tensor_dict
        load_tensor = loaded_tensor_dict[f'{image_index}'].to(self.device)
        mapped_tensor = self.embed_multimodal(load_tensor)
        mapped_tensor = mapped_tensor.unsqueeze(1)
        return mapped_tensor

    def _load_frame_patches(self, video_index, image_index, indices=None):
        """
        Load the raw frame (from disk, not the offline CLS-token cache) and split it into
        patches. The *decoded image* (not the cropped patches) is cached per
        (video_index, image_index) within this Pipeline instance, since consecutive samples
        in a trajectory often reuse nearby frames -- caching at this granularity means any
        caller can request any subset of patches for a frame and still hit the cache after
        the first decode, regardless of which indices previous callers asked for.

        :param indices: if None, crop and return the full patch_grid (all-patch mode). If
            given, crop and return only these patch indices, in the given order
            (patch-selection mode, where cropping the other patches would be wasted work
            since only the selected ones are ever fed to the ViT).
        """
        cache_key = (video_index, image_index)
        if cache_key in self._patch_image_cache:
            image_tensor = self._patch_image_cache[cache_key]
        else:
            ext = cfg.dataset_image_ext[self.dataset]
            image_path = os.path.join(cfg.dataset_images[self.dataset], f'video{video_index}_images', f'{image_index}.{ext}')
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.raw_image_transform(image)
            self._patch_image_cache[cache_key] = image_tensor

        if indices is None:
            return crop_patches(image_tensor, self.grid_rows, self.grid_cols)
        return crop_patches_at(image_tensor, self.grid_rows, self.grid_cols, indices)

    def _vit_feature_fn(self, img_batch):
        return extract_vit_features(img_batch, model=self.vit_model)

    def _get_multimodal_information_all_patch(self, video_user_position):
        video_index, image_index = self._resolve_frame_index(video_user_position)
        patches = self._load_frame_patches(video_index, image_index)
        indices = list(range(patches.shape[0]))
        features = vit_features_for_patches(patches, indices, self._vit_feature_fn, device=self.device)
        mapped_tensor = self.embed_multimodal(features)  # (num_patches, embed_size)
        return mapped_tensor.unsqueeze(0)  # (1, num_patches, embed_size)

    def _get_multimodal_information_patch_selection(self, video_user_position, history_viewports):
        video_index, image_index = self._resolve_frame_index(video_user_position)
        with torch.no_grad():
            logits = self.patch_selection_module(history_viewports.to(self.device))  # (1, num_patches)
        mask = self.patch_selection_module.select_patches(
            logits, top_k=self.patch_top_k, threshold=self.patch_threshold)
        indices = mask[0].nonzero(as_tuple=True)[0].tolist()
        if len(indices) == 0:  # always feed at least one patch
            indices = [logits[0].argmax().item()]
        # debug: eyeball whether selections look sane (stable-ish, not collapsing to
        # all/none or jumping to unrelated corners every call) -- capped to the first
        # few calls so it's cheap to leave in and doesn't spam real runs.
        if len(self.patch_selection_history) < int(os.environ.get('PS_DEBUG_PRINT_N', 0)):
            probs = torch.sigmoid(logits[0]).tolist()
            print(f'[ps-debug] call={len(self.patch_selection_history)} video={video_index} frame={image_index} '
                  f'n_selected={len(indices)} indices={sorted(indices)} probs={[round(p, 2) for p in probs]}')
        self.patch_selection_history.append(len(indices))
        # only crop the selected patches (not the full grid) -- crop_patches_at() already
        # returns them in `indices` order, so the patches array itself needs no further
        # indexing (identity range(len(indices)) below is just a pass-through).
        patches = self._load_frame_patches(video_index, image_index, indices=indices)
        features = vit_features_for_patches(patches, range(len(indices)), self._vit_feature_fn, device=self.device)
        mapped_tensor = self.embed_multimodal(features)  # (len(indices), embed_size)
        return mapped_tensor.unsqueeze(0)  # (1, len(indices), embed_size)