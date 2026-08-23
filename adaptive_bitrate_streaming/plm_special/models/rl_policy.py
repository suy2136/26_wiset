import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import deque

from plm_special.models.selectors import BaseSelector


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
            max_length=None,
            max_ep_len=100,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            device_out = None,
            residual = False, 
            conv_size = 4,  
            which_layer = -1,  # for early stopping: specify which layer to stop
            token_selector=None,
            **kwargs
    ):
        super().__init__()
        
        if device_out is None:
            device_out = device

        self.bitrate_levels = bitrate_levels
        self.max_length = max_length

        self.plm = plm
        self.plm_embed_size = plm_embed_size

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
        self.last_selection_output = None
        self.last_selection_trace = {}
        self.modules_except_plm = nn.ModuleList([  # used to save and load modules except plm
            self.state_encoder, self.embed_timestep, self.embed_return, self.embed_action, self.embed_ln, 
            self.embed_state1, self.embed_state2, self.embed_state3, self.embed_state4, self.embed_state5,
            self.embed_state6, self.action_head
        ])

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
        stacked_inputs = stacked_inputs[:, -self.plm_embed_size:, :]  # truncate sequence length (should not exceed plm embed size)
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
        stacked_inputs = stacked_inputs[:, -self.plm_embed_size:, :]
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
        logits = transformer_outputs['last_hidden_state']
        if self.residual:
            logits = logits + stacked_inputs_ln  # residual add

        # Step 6: predict the bitrate for next chunk
        logits_used = logits[:, -1:]
        action_pred = self.action_head(logits_used)
        action_pred = action_pred.reshape(-1)
        bitrate, _ = self._sample(action_pred)

        # compute action embeddings 
        action_tensor = torch.zeros(1, 1, 1, dtype=torch.float32, device=self.device)
        action_tensor[..., 0] = (bitrate + 1) / self.bitrate_levels
        action_embeddings = self.embed_action(action_tensor) + time_embeddings
        
        # update deques
        self.returns_dq.append(return_embeddings)
        self.states_dq.append(state_embeddings) 
        self.actions_dq.append(action_embeddings)

        return bitrate
    
    def clear_dq(self):
        self.states_dq.clear()
        self.actions_dq.clear()
        self.returns_dq.clear()
        
        self.states_dq.append(torch.zeros((1, 0, self.plm_embed_size), device=self.device))
        self.actions_dq.append(torch.zeros((1, 0, self.plm_embed_size), device=self.device))
        self.returns_dq.append(torch.zeros((1, 0, self.plm_embed_size), device=self.device))

    def _sample(self, logits):
        pi = F.softmax(logits, 0).cpu().numpy()
        idx = random.choices(np.arange(pi.size), pi)[0]
        lgprob = np.log(pi[idx])
        return idx, lgprob
