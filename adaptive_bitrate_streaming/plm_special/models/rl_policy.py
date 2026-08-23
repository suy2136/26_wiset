import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import deque

from plm_special.models.selectors import BaseSelector
from plm_special.models.selection_layout import aligned_context_window
from plm_special.speculative.acceptance import (
    buffer_deviation_seconds,
    build_acceptance_plan,
)


INF = 1e5
ABR_STATE_TOKEN_COUNT = 6
ABR_HISTORY_BLOCK_TOKENS = 2 + ABR_STATE_TOKEN_COUNT
ABR_CURRENT_BLOCK_TOKENS = 1 + ABR_STATE_TOKEN_COUNT


class OfflineRLPolicy(nn.Module):
    def __init__(
            self,
            state_feature_dim,
            bitrate_levels,
            state_encoder,
            plm,
            plm_embed_size,
            plm_context_length=None,
            max_length=None,
            max_ep_len=100,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            device_out = None,
            residual = False, 
            conv_size = 4,  
            which_layer = -1,  # for early stopping: specify which layer to stop
            token_selector=None,
            draft_generator=None,
            speculative_draft_steps=0,
            speculative_verification_mode='sample',
            speculative_buffer_tolerance=1.0,
            **kwargs
    ):
        super().__init__()
        
        if device_out is None:
            device_out = device

        self.bitrate_levels = bitrate_levels
        self.max_length = max_length

        self.plm = plm
        self.plm_embed_size = plm_embed_size
        self.plm_context_length = self._resolve_plm_context_length(
            plm, plm_context_length
        )

        # =========== multimodal encoder (start) ===========
        self.state_encoder = state_encoder
        self.state_feature_dim = state_feature_dim
        self.embed_timestep = nn.Embedding(max_ep_len + 1, plm_embed_size).to(device)
        self.embed_return = nn.Linear(1, plm_embed_size).to(device)
        self.embed_action = nn.Linear(1, plm_embed_size).to(device)
        self.embed_state1 = nn.Linear(state_feature_dim, plm_embed_size).to(device)
        self.embed_state2 = nn.Linear(state_feature_dim, plm_embed_size).to(device)    
        self.embed_state3 = nn.Linear(state_feature_dim * (6 - conv_size + 1), plm_embed_size).to(device)    
        self.embed_state4 = nn.Linear(state_feature_dim * (6 - conv_size + 1), plm_embed_size).to(device)    
        self.embed_state5 = nn.Linear(state_feature_dim, plm_embed_size).to(device)
        self.embed_state6 = nn.Linear(state_feature_dim, plm_embed_size).to(device)    

        self.embed_ln = nn.LayerNorm(plm_embed_size).to(device)
        # =========== multimodal encoder (end) ===========
    
        self.action_head = nn.Linear(plm_embed_size, bitrate_levels).to(device)  # the so-called networking head in our paper

        self.device = device
        self.device_out = device_out

        # the following are used for evaluation
        self.states_dq = deque([torch.zeros((1, 0, plm_embed_size), device=device)], maxlen=max_length)
        self.returns_dq = deque([torch.zeros((1, 0, plm_embed_size), device=device)], maxlen=max_length)
        self.actions_dq = deque([torch.zeros((1, 0, plm_embed_size), device=device)], maxlen=max_length)

        self.residual = residual
        self.which_layer = which_layer
        if token_selector is not None and not isinstance(token_selector, BaseSelector):
            raise TypeError("token_selector must be a BaseSelector instance or None")
        self.token_selector = token_selector
        if speculative_draft_steps < 0:
            raise ValueError("speculative_draft_steps must be non-negative")
        self.draft_generator = draft_generator
        self.speculative_draft_steps = speculative_draft_steps
        if speculative_verification_mode not in ('greedy', 'sample'):
            raise ValueError("speculative_verification_mode must be greedy or sample")
        if speculative_buffer_tolerance < 0:
            raise ValueError("speculative_buffer_tolerance must be non-negative")
        self.speculative_verification_mode = speculative_verification_mode
        self.speculative_buffer_tolerance = speculative_buffer_tolerance
        self._speculative_queue = deque()
        self.speculative_stats = {
            "target_plm_calls": 0,
            "draft_attempts": 0,
            "drafted_actions": 0,
            "accepted_actions": 0,
            "corrected_actions": 0,
            "executed_speculative_actions": 0,
            "queued_actions_served": 0,
            "fallback_calls": 0,
            "state_mismatch_fallbacks": 0,
            "draft_generation_failures": 0,
        }
        self.last_selection_output = None
        self.last_selection_trace = {}
        self.modules_except_plm = nn.ModuleList([  # used to save and load modules except plm
            self.state_encoder, self.embed_timestep, self.embed_return, self.embed_action, self.embed_ln, 
            self.embed_state1, self.embed_state2, self.embed_state3, self.embed_state4, self.embed_state5,
            self.embed_state6, self.action_head
        ])

    @staticmethod
    def _resolve_plm_context_length(plm, explicit_limit):
        if explicit_limit is not None:
            if (
                isinstance(explicit_limit, bool)
                or not isinstance(explicit_limit, int)
                or explicit_limit <= 0
            ):
                raise ValueError("plm_context_length must be a positive integer")
            return explicit_limit
        configs = [getattr(plm, "config", None)]
        base_model = getattr(plm, "base_model", None)
        configs.append(getattr(base_model, "config", None))
        get_base_model = getattr(plm, "get_base_model", None)
        if callable(get_base_model):
            configs.append(getattr(get_base_model(), "config", None))
        for config in configs:
            for name in ("max_position_embeddings", "n_positions", "n_ctx"):
                value = getattr(config, name, None) if config is not None else None
                if (
                    isinstance(value, int)
                    and not isinstance(value, bool)
                    and value > 0
                ):
                    return value
        return None

    def _truncate_inference_context(self, embeddings, protected_suffix_tokens):
        """Drop only complete oldest history blocks to fit the PLM context."""
        original_length = int(embeddings.shape[1])
        if self.plm_context_length is None:
            return embeddings, 0, original_length
        start = aligned_context_window(
            original_length=original_length,
            context_limit=self.plm_context_length,
            tokens_per_history_step=ABR_HISTORY_BLOCK_TOKENS,
            protected_suffix_tokens=protected_suffix_tokens,
        )
        return embeddings[:, start:, :], start, original_length

    def forward(self, states, actions, returns, timesteps, attention_mask=None):
        """
        Forward function, used for training.
        """
        assert actions.shape[0] == 1, 'batch size should be 1 to avoid CUDA memory exceed'

        # Step 1: process actions, returns and timesteps first as they are simple
        actions = actions.to(self.device)  # shape: (1, seq_len, 1)
        returns = returns.to(self.device)  # shape: (1, seq_len, 1)
        timesteps = timesteps.to(self.device)  # shape: (1, seq_len)

        # 1.1 embed action, return, timestep
        action_embeddings = self.embed_action(actions)  # shape: (1, seq_len, embed_size)
        returns_embeddings = self.embed_return(returns)  # shape: (1, seq_len, embed_size)
        time_embeddings = self.embed_timestep(timesteps)  # shape: (1, seq_len, embed_size)

        # 1.2 time embeddings are treated similar to positional embeddings
        action_embeddings = action_embeddings + time_embeddings
        returns_embeddings = returns_embeddings + time_embeddings

        # Step 2: process states, turn them into embeddings.
        states = states.to(self.device)  # shape: (1, seq_len, 6, 6)
        states_features = self.state_encoder(states)
        states_embeddings1 = self.embed_state1(states_features[0]) + time_embeddings
        states_embeddings2 = self.embed_state2(states_features[1]) + time_embeddings
        states_embeddings3 = self.embed_state3(states_features[2]) + time_embeddings
        states_embeddings4 = self.embed_state4(states_features[3]) + time_embeddings
        states_embeddings5 = self.embed_state5(states_features[4]) + time_embeddings
        states_embeddings6 = self.embed_state6(states_features[5]) + time_embeddings
        
        # Step 3: stack returns, states, actions embeddings.
        # this makes the sequence look like (R_1, s_1-1, s_1-2, ..., s_1-n, a_1, R_2, s_2-1, ..., s_2-m, a_2, ...)
        # which works nice in an autoregressive sense since states predict actions
        stacked_inputs = []
        action_embed_positions = np.zeros(returns_embeddings.shape[1])  # record the positions of action embeddings
        for i in range(returns_embeddings.shape[1]):
            stacked_input = torch.cat((returns_embeddings[0, i:i + 1], states_embeddings1[0, i:i + 1], states_embeddings2[0, i:i + 1], 
                                       states_embeddings3[0, i:i + 1], states_embeddings4[0, i:i + 1], states_embeddings5[0, i:i + 1], 
                                       states_embeddings6[0, i:i + 1], action_embeddings[0, i:i + 1]), dim=0)
            stacked_inputs.append(stacked_input)
            action_embed_positions[i] = (i + 1) * (2 + 6)
        stacked_inputs = torch.cat(stacked_inputs, dim=0).unsqueeze(0)
        if (
            self.plm_context_length is not None
            and stacked_inputs.shape[1] > self.plm_context_length
        ):
            raise ValueError(
                "training ABR sequence exceeds the PLM context limit; "
                "reduce --w so labels and complete 8-token blocks remain aligned"
            )
        stacked_inputs_ln = self.embed_ln(stacked_inputs)  # layer normalization
        
        # Step 4: feed stacked embeddings into the plm
        # 4.1 create attention mask
        if attention_mask is None:
            # 1 if can be attended to, 0 if not
            attention_mask = torch.ones((stacked_inputs_ln.shape[0], stacked_inputs_ln.shape[1]), dtype=torch.long, device=self.device)

        # we feed in the input embeddings (not word indices as in NLP) to the model
        transformer_outputs = self.plm(
            inputs_embeds=stacked_inputs_ln,
            attention_mask=attention_mask,
            output_hidden_states=True,
            stop_layer_idx=self.which_layer,
        )
        logits = transformer_outputs['last_hidden_state']
        if self.residual:
            logits = logits + stacked_inputs_ln  # residual add

        # Step 5: predict actions
        # we need to locate the logits corresponding to the state embeddings
        # simply using `action_embed_positions[i] - 2` will do.
        logits_used = logits[:, action_embed_positions - 2]
        action_pred = self.action_head(logits_used)

        return action_pred

    def _stack_previous_context(self):
        """Build complete ``[return, six states, action]`` history blocks."""
        previous_blocks = []
        for return_embeddings, state_embeddings, action_embeddings in zip(
            self.returns_dq, self.states_dq, self.actions_dq
        ):
            block = torch.cat(
                (return_embeddings, state_embeddings, action_embeddings), dim=1
            )
            if block.shape[1] not in (0, ABR_HISTORY_BLOCK_TOKENS):
                raise ValueError(
                    "invalid ABR history block length: "
                    f"expected 0 or {ABR_HISTORY_BLOCK_TOKENS}, got {block.shape[1]}"
                )
            previous_blocks.append(block)
        return torch.cat(previous_blocks, dim=1)

    def _build_current_step_embeddings(self, state, target_return, timestep):
        """Build the protected ``[return, six states]`` current ABR block."""
        target_return = torch.as_tensor(
            target_return, dtype=torch.float32, device=self.device
        ).reshape(1, 1, 1)
        timestep = torch.as_tensor(
            timestep, dtype=torch.int32, device=self.device
        ).reshape(1, 1)

        return_embeddings = self.embed_return(target_return)
        time_embeddings = self.embed_timestep(timestep)
        return_embeddings = return_embeddings + time_embeddings

        state = state.to(self.device)
        state_features = self.state_encoder(state)
        state_embeddings = torch.cat(
            [
                self.embed_state1(state_features[0]) + time_embeddings,
                self.embed_state2(state_features[1]) + time_embeddings,
                self.embed_state3(state_features[2]) + time_embeddings,
                self.embed_state4(state_features[3]) + time_embeddings,
                self.embed_state5(state_features[4]) + time_embeddings,
                self.embed_state6(state_features[5]) + time_embeddings,
            ],
            dim=1,
        )
        current_block = torch.cat((return_embeddings, state_embeddings), dim=1)
        if current_block.shape[1] != ABR_CURRENT_BLOCK_TOKENS:
            raise ValueError(
                "invalid ABR current block length: "
                f"expected {ABR_CURRENT_BLOCK_TOKENS}, got {current_block.shape[1]}"
            )
        return current_block, return_embeddings, state_embeddings, time_embeddings

    def _build_selected_inference_context(self, current_block, draft_blocks=None):
        """Prune real history, while protecting current and MPC draft blocks.

        ``draft_blocks`` is reserved for speculative verification.  It must be
        an embedded sequence of complete 8-token ``[return, state, action]``
        blocks.  Selection is applied only to the real-history prefix; the
        current 7-token block and every draft token remain untouched.
        """
        if draft_blocks is None:
            draft_blocks = current_block[:, :0, :]
        if draft_blocks.ndim != 3:
            raise ValueError("draft_blocks must have shape [B,L,E]")
        if (
            draft_blocks.shape[0] != current_block.shape[0]
            or draft_blocks.shape[2] != current_block.shape[2]
        ):
            raise ValueError("draft_blocks must match current_block batch/embed dims")
        if draft_blocks.shape[1] % ABR_HISTORY_BLOCK_TOKENS != 0:
            raise ValueError(
                "draft_blocks must contain complete ABR timestep blocks "
                f"({ABR_HISTORY_BLOCK_TOKENS} tokens each)"
            )

        previous_context = self._stack_previous_context()
        protected_suffix = torch.cat((current_block, draft_blocks), dim=1)
        stacked_inputs = torch.cat((previous_context, protected_suffix), dim=1)
        stacked_inputs, context_start, pre_truncation_length = (
            self._truncate_inference_context(
                stacked_inputs, int(protected_suffix.shape[1])
            )
        )
        stacked_inputs_ln = self.embed_ln(stacked_inputs)
        attention_mask = torch.ones(
            stacked_inputs_ln.shape[:2], dtype=torch.long, device=self.device
        )

        original_length = int(stacked_inputs_ln.shape[1])
        self.last_selection_output = None
        if self.token_selector is not None:
            selection = self.token_selector(
                stacked_inputs_ln,
                attention_mask,
                context={
                    "task": "adaptive_bitrate_streaming",
                    "stage": "online_action_selection",
                    "tokens_per_history_step": ABR_HISTORY_BLOCK_TOKENS,
                    "current_step_tokens": ABR_CURRENT_BLOCK_TOKENS,
                    "draft_tokens": int(draft_blocks.shape[1]),
                    "protected_suffix_tokens": int(protected_suffix.shape[1]),
                },
            )
            stacked_inputs_ln = selection.embeddings
            attention_mask = selection.attention_mask
            if attention_mask is None:
                attention_mask = torch.ones(
                    stacked_inputs_ln.shape[:2],
                    dtype=torch.long,
                    device=stacked_inputs_ln.device,
                )
            self.last_selection_output = selection

        self.last_selection_trace = {
            "target_model_called": True,
            "pre_truncation_length": pre_truncation_length,
            "context_truncated_tokens": context_start,
            "plm_context_length": self.plm_context_length,
            "selector_enabled": self.token_selector is not None,
            "selector_class": (
                None if self.token_selector is None
                else type(self.token_selector).__name__
            ),
            "original_length": original_length,
            "selected_length": int(stacked_inputs_ln.shape[1]),
            "history_block_tokens": ABR_HISTORY_BLOCK_TOKENS,
            "current_block_tokens": ABR_CURRENT_BLOCK_TOKENS,
            "draft_tokens": int(draft_blocks.shape[1]),
            "protected_suffix_tokens": int(protected_suffix.shape[1]),
        }
        return stacked_inputs_ln, attention_mask

    def build_speculative_verification_context(self, current_block, draft_blocks):
        """Build a selector-safe context for a future MPC draft verifier."""
        return self._build_selected_inference_context(current_block, draft_blocks)

    def _embed_mpc_draft_blocks(self, rollout):
        """Embed MPC decisions as ``[return, six states, action]`` blocks."""
        states = torch.as_tensor(
            rollout.states, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        actions = torch.as_tensor(
            rollout.actions, dtype=torch.float32, device=self.device
        ).reshape(1, -1, 1)
        actions = (actions + 1) / self.bitrate_levels
        returns = torch.as_tensor(
            rollout.returns, dtype=torch.float32, device=self.device
        ).reshape(1, -1, 1)
        timesteps = torch.as_tensor(
            rollout.timesteps, dtype=torch.long, device=self.device
        ).unsqueeze(0)

        time_embeddings = self.embed_timestep(timesteps)
        return_embeddings = self.embed_return(returns) + time_embeddings
        action_embeddings = self.embed_action(actions) + time_embeddings
        state_features = self.state_encoder(states)
        state_embeddings = [
            self.embed_state1(state_features[0]) + time_embeddings,
            self.embed_state2(state_features[1]) + time_embeddings,
            self.embed_state3(state_features[2]) + time_embeddings,
            self.embed_state4(state_features[3]) + time_embeddings,
            self.embed_state5(state_features[4]) + time_embeddings,
            self.embed_state6(state_features[5]) + time_embeddings,
        ]
        blocks = torch.stack(
            [return_embeddings, *state_embeddings, action_embeddings], dim=2
        )
        return blocks.reshape(1, rollout.length * ABR_HISTORY_BLOCK_TOKENS, -1)

    def _build_selected_mpc_verification_context(self, draft_blocks):
        """Select only real history and keep every MPC draft block protected."""
        if draft_blocks.ndim != 3:
            raise ValueError("draft_blocks must have shape [B,L,E]")
        draft_tokens = int(draft_blocks.shape[1])
        if draft_tokens == 0 or draft_tokens % ABR_HISTORY_BLOCK_TOKENS != 0:
            raise ValueError("draft_blocks must contain one or more complete ABR blocks")
        previous_context = self._stack_previous_context()
        stacked_inputs = torch.cat((previous_context, draft_blocks), dim=1)
        stacked_inputs, context_start, pre_truncation_length = (
            self._truncate_inference_context(stacked_inputs, draft_tokens)
        )
        stacked_inputs_ln = self.embed_ln(stacked_inputs)
        attention_mask = torch.ones(
            stacked_inputs_ln.shape[:2], dtype=torch.long, device=self.device
        )

        original_length = int(stacked_inputs_ln.shape[1])
        self.last_selection_output = None
        if self.token_selector is not None:
            selection = self.token_selector(
                stacked_inputs_ln,
                attention_mask,
                context={
                    "task": "adaptive_bitrate_streaming",
                    "stage": "mpc_speculative_verification",
                    "tokens_per_history_step": ABR_HISTORY_BLOCK_TOKENS,
                    "protected_suffix_tokens": draft_tokens,
                    "draft_tokens": draft_tokens,
                },
            )
            stacked_inputs_ln = selection.embeddings
            attention_mask = selection.attention_mask
            self.last_selection_output = selection

        self.last_selection_trace = {
            "target_model_called": True,
            "pre_truncation_length": pre_truncation_length,
            "context_truncated_tokens": context_start,
            "plm_context_length": self.plm_context_length,
            "selector_enabled": self.token_selector is not None,
            "selector_class": (
                None if self.token_selector is None
                else type(self.token_selector).__name__
            ),
            "stage": "mpc_speculative_verification",
            "original_length": original_length,
            "selected_length": int(stacked_inputs_ln.shape[1]),
            "history_block_tokens": ABR_HISTORY_BLOCK_TOKENS,
            "draft_tokens": draft_tokens,
            "protected_suffix_tokens": draft_tokens,
        }
        return stacked_inputs_ln, attention_mask

    def verify_mpc_draft(self, rollout):
        """Verify all MPC bitrate decisions with exactly one PLM forward call."""
        if rollout.length <= 0:
            raise ValueError("MPC rollout must contain at least one decision")
        draft_blocks = self._embed_mpc_draft_blocks(rollout)
        stacked_inputs_ln, attention_mask = (
            self._build_selected_mpc_verification_context(draft_blocks)
        )
        transformer_outputs = self.plm(
            inputs_embeds=stacked_inputs_ln,
            attention_mask=attention_mask,
            output_hidden_states=True,
            stop_layer_idx=self.which_layer,
        )
        self.speculative_stats["target_plm_calls"] += 1
        hidden = transformer_outputs['last_hidden_state']
        if self.residual:
            hidden = hidden + stacked_inputs_ln

        selected_history_tokens = hidden.shape[1] - draft_blocks.shape[1]
        state_positions = (
            selected_history_tokens
            + torch.arange(rollout.length, device=hidden.device)
            * ABR_HISTORY_BLOCK_TOKENS
            + ABR_STATE_TOKEN_COUNT
        )
        action_logits = self.action_head(hidden[:, state_positions, :])
        return {
            "draft_actions": torch.as_tensor(
                rollout.actions, dtype=torch.long, device=action_logits.device
            ),
            "action_logits": action_logits,
            "draft_blocks": draft_blocks,
            "selection_trace": dict(self.last_selection_trace),
            "rollout": rollout,
        }

    def draft_and_verify(
        self,
        state,
        last_bitrate,
        buffer_size,
        video_chunk_remain,
        target_return,
        timestep,
        reward_transform=None,
        horizon=None,
    ):
        """Generate an MPC rollout and verify it in one target-model call."""
        if self.draft_generator is None:
            raise RuntimeError("no MPC draft generator is configured")
        if horizon is None:
            horizon = self.speculative_draft_steps
        if horizon is None or horizon <= 0:
            raise ValueError("a positive speculative draft horizon is required")
        rollout = self.draft_generator.generate(
            state=state,
            last_bitrate=last_bitrate,
            buffer_size=buffer_size,
            video_chunk_remain=video_chunk_remain,
            target_return=target_return,
            timestep=timestep,
            horizon=horizon,
            reward_transform=reward_transform,
        )
        return self.verify_mpc_draft(rollout)

    def _actions_from_verification_logits(self, action_logits):
        if self.speculative_verification_mode == 'greedy':
            return action_logits.argmax(dim=-1).reshape(-1).tolist()
        return [self._sample(logits.reshape(-1))[0] for logits in action_logits[0]]

    def _append_observed_action(self, state, target_return, timestep, bitrate):
        """Commit only a real observed state/action pair to policy history."""
        _, return_embeddings, state_embeddings, time_embeddings = (
            self._build_current_step_embeddings(state, target_return, timestep)
        )
        self._append_embedded_action(
            return_embeddings, state_embeddings, time_embeddings, bitrate
        )

    def _append_embedded_action(
        self, return_embeddings, state_embeddings, time_embeddings, bitrate
    ):
        action_tensor = torch.zeros(
            1, 1, 1, dtype=torch.float32, device=self.device
        )
        action_tensor[..., 0] = (bitrate + 1) / self.bitrate_levels
        action_embeddings = self.embed_action(action_tensor) + time_embeddings
        self.returns_dq.append(return_embeddings)
        self.states_dq.append(state_embeddings)
        self.actions_dq.append(action_embeddings)

    def _fallback_sample(self, state, target_return, timestep):
        self.speculative_stats["fallback_calls"] += 1
        return self.sample(state, target_return, timestep)

    def sample_speculative(
        self,
        state,
        target_return,
        timestep,
        last_bitrate,
        buffer_size,
        video_chunk_remain,
        reward_transform=None,
    ):
        """Serve a verified draft action or create and verify a new MPC draft."""
        if self.draft_generator is None or self.speculative_draft_steps <= 0:
            return self.sample(state, target_return, timestep)

        if self._speculative_queue:
            queued = self._speculative_queue[0]
            deviation = buffer_deviation_seconds(state, queued["predicted_state"])
            if deviation <= self.speculative_buffer_tolerance:
                self._speculative_queue.popleft()
                bitrate = queued["action"]
                self._append_observed_action(state, target_return, timestep, bitrate)
                self.last_selection_trace = {
                    "target_model_called": False,
                    "stage": "serve_verified_queue",
                    "original_length": 0,
                    "selected_length": 0,
                }
                self.speculative_stats["queued_actions_served"] += 1
                self.speculative_stats["executed_speculative_actions"] += 1
                return bitrate
            self._speculative_queue.clear()
            self.speculative_stats["state_mismatch_fallbacks"] += 1
            return self._fallback_sample(state, target_return, timestep)

        self.speculative_stats["draft_attempts"] += 1
        try:
            verification = self.draft_and_verify(
                state=state,
                last_bitrate=last_bitrate,
                buffer_size=buffer_size,
                video_chunk_remain=video_chunk_remain,
                target_return=target_return,
                timestep=timestep,
                reward_transform=reward_transform,
            )
        except (ValueError, FloatingPointError, ZeroDivisionError):
            self.speculative_stats["draft_generation_failures"] += 1
            return self._fallback_sample(state, target_return, timestep)

        rollout = verification["rollout"]
        target_actions = self._actions_from_verification_logits(
            verification["action_logits"]
        )
        plan = build_acceptance_plan(rollout.actions, target_actions)
        self.speculative_stats["drafted_actions"] += rollout.length
        self.speculative_stats["accepted_actions"] += plan.accepted_count
        if not plan.fully_accepted:
            self.speculative_stats["corrected_actions"] += 1

        for index, action in enumerate(plan.actions):
            self._speculative_queue.append({
                "action": int(action),
                "predicted_state": rollout.states[index].copy(),
                "source": (
                    "accepted" if index < plan.accepted_count else "correction"
                ),
            })
        first = self._speculative_queue.popleft()
        self._append_observed_action(state, target_return, timestep, first["action"])
        self.speculative_stats["executed_speculative_actions"] += 1
        return first["action"]

    def get_speculative_metrics(self):
        metrics = dict(self.speculative_stats)
        drafted = metrics["drafted_actions"]
        metrics["acceptance_rate"] = (
            0.0 if drafted == 0 else metrics["accepted_actions"] / drafted
        )
        metrics["pending_actions"] = len(self._speculative_queue)
        return metrics

    def sample(self, state, target_return, timestep, **kwargs):
        """
        Sample action function, used for evaluation/testing.
        """
        current_block, return_embeddings, state_embeddings, time_embeddings = (
            self._build_current_step_embeddings(state, target_return, timestep)
        )
        stacked_inputs_ln, attention_mask = self._build_selected_inference_context(
            current_block
        )

        transformer_outputs = self.plm(
            inputs_embeds=stacked_inputs_ln,
            attention_mask=attention_mask,
            output_hidden_states=True,
            stop_layer_idx=self.which_layer,
        )
        self.speculative_stats["target_plm_calls"] += 1
        logits = transformer_outputs['last_hidden_state']
        if self.residual:
            logits = logits + stacked_inputs_ln  # residual add

        # Step 6: predict the bitrate for next chunk
        logits_used = logits[:, -1:]
        action_pred = self.action_head(logits_used)
        action_pred = action_pred.reshape(-1)
        bitrate, _ = self._sample(action_pred)

        self._append_embedded_action(
            return_embeddings, state_embeddings, time_embeddings, bitrate
        )

        return bitrate
    
    def clear_dq(self):
        self.states_dq.clear()
        self.actions_dq.clear()
        self.returns_dq.clear()
        
        self.states_dq.append(torch.zeros((1, 0, self.plm_embed_size), device=self.device))
        self.actions_dq.append(torch.zeros((1, 0, self.plm_embed_size), device=self.device))
        self.returns_dq.append(torch.zeros((1, 0, self.plm_embed_size), device=self.device))
        self._speculative_queue.clear()
        if self.draft_generator is not None:
            self.draft_generator.reset()

    def _sample(self, logits):
        pi = F.softmax(logits, 0).cpu().numpy()
        idx = random.choices(np.arange(pi.size), pi)[0]
        lgprob = np.log(pi[idx])
        return idx, lgprob
