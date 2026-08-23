import os
import sys
import unittest


ABR_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ABR_ROOT not in sys.path:
    sys.path.insert(0, ABR_ROOT)

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'PyTorch is not installed in this environment')
class SelectorTensorContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from plm_special.models.selectors import IdentitySelector, RecentTimestepSelector
        cls.IdentitySelector = IdentitySelector
        cls.RecentTimestepSelector = RecentTimestepSelector

    def test_identity_and_h20_are_exact_for_full_abr_window(self):
        embeddings = torch.arange(167 * 3, dtype=torch.float32).reshape(1, 167, 3)
        mask = torch.ones((1, 167), dtype=torch.long)
        identity = self.IdentitySelector()(embeddings, mask)
        h20 = self.RecentTimestepSelector(20)(embeddings, mask)
        self.assertTrue(torch.equal(identity.embeddings, h20.embeddings))
        self.assertTrue(torch.equal(identity.attention_mask, h20.attention_mask))
        self.assertTrue(torch.equal(identity.selected_indices, h20.selected_indices))

    def test_draft_suffix_override_is_kept_exactly(self):
        embeddings = torch.arange(183, dtype=torch.float32).reshape(1, 183, 1)
        output = self.RecentTimestepSelector(10)(
            embeddings, context={'protected_suffix_tokens': 23}
        )
        self.assertEqual(output.selected_length, 103)
        self.assertTrue(torch.equal(output.embeddings[:, -23:], embeddings[:, -23:]))
        self.assertEqual(output.metadata['protected_suffix_tokens'], 23)


@unittest.skipIf(torch is None, 'PyTorch is not installed in this environment')
class MPCVerificationContractTest(unittest.TestCase):
    def test_draft_and_verify_uses_one_plm_call(self):
        import torch.nn as nn

        from plm_special.models.rl_policy import OfflineRLPolicy
        from plm_special.models.selectors import RecentTimestepSelector
        from plm_special.speculative.mpc_draft import RobustMPCDraftGenerator

        class DummyEncoder(nn.Module):
            def forward(self, states):
                batch, steps = states.shape[:2]
                device = states.device
                return [
                    torch.zeros((batch, steps, 2), device=device),
                    torch.zeros((batch, steps, 2), device=device),
                    torch.zeros((batch, steps, 6), device=device),
                    torch.zeros((batch, steps, 6), device=device),
                    torch.zeros((batch, steps, 2), device=device),
                    torch.zeros((batch, steps, 2), device=device),
                ]

        class DummyPLM(nn.Module):
            def __init__(self):
                super().__init__()
                self.calls = 0
                self.input_dtypes = []
                self.embed_tokens = nn.Embedding(2, 32).to(torch.float16)

            def get_input_embeddings(self):
                return self.embed_tokens

            def forward(self, inputs_embeds, **kwargs):
                self.calls += 1
                self.input_dtypes.append(inputs_embeds.dtype)
                return {'last_hidden_state': inputs_embeds}

        plm = DummyPLM()
        generator = RobustMPCDraftGenerator(
            np.full((6, 48), 1_000_000, dtype=np.float64), max_horizon=3
        )
        policy = OfflineRLPolicy(
            state_feature_dim=2,
            bitrate_levels=6,
            state_encoder=DummyEncoder(),
            plm=plm,
            plm_embed_size=32,
            max_length=20,
            max_ep_len=48,
            device='cpu',
            token_selector=RecentTimestepSelector(5),
            draft_generator=generator,
            speculative_draft_steps=3,
        )
        state = torch.zeros((1, 1, 6, 6), dtype=torch.float32)
        state[..., 1, -1] = 1.0
        state[..., 2, :] = 10.0
        state[..., 5, -1] = 10.0 / 48.0
        result = policy.draft_and_verify(
            state=state,
            last_bitrate=0,
            buffer_size=10.0,
            video_chunk_remain=10,
            target_return=5.0,
            timestep=0,
        )
        self.assertEqual(plm.calls, 1)
        self.assertEqual(tuple(result['action_logits'].shape), (1, 3, 6))
        self.assertEqual(result['action_logits'].dtype, torch.float32)
        self.assertEqual(plm.input_dtypes, [torch.float16])
        self.assertEqual(tuple(result['draft_blocks'].shape), (1, 24, 32))
        self.assertEqual(result['selection_trace']['protected_suffix_tokens'], 24)

        # A fully accepted draft executes one real action now and queues two;
        # predicted states themselves must never be committed to history.
        policy.clear_dq()
        plm.calls = 0
        policy._actions_from_verification_logits = lambda logits: [1, 2, 3]
        with torch.no_grad():
            first = policy.sample_speculative(
                state=state,
                target_return=5.0,
                timestep=0,
                last_bitrate=0,
                buffer_size=10.0,
                video_chunk_remain=10,
            )
        self.assertEqual(first, 1)
        self.assertEqual(plm.calls, 1)
        self.assertEqual(len(policy.states_dq), 2)
        self.assertEqual(len(policy._speculative_queue), 2)

        predicted = policy._speculative_queue[0]['predicted_state']
        predicted_return = policy._speculative_queue[0]['predicted_return']
        observed = torch.as_tensor(predicted).reshape(1, 1, 6, 6)
        with torch.no_grad():
            second = policy.sample_speculative(
                state=observed,
                target_return=predicted_return,
                timestep=1,
                last_bitrate=1,
                buffer_size=float(predicted[1, -1] * 10),
                video_chunk_remain=9,
            )
        self.assertEqual(second, 2)
        self.assertEqual(plm.calls, 1)
        self.assertEqual(len(policy.states_dq), 3)
        self.assertEqual(
            policy.get_speculative_metrics()['throughput_predictor_updates'], 2
        )

        bad_observation = torch.as_tensor(
            policy._speculative_queue[0]['predicted_state']
        ).reshape(1, 1, 6, 6).clone()
        bad_observation[..., 1, -1] += 1.0
        with torch.no_grad():
            policy.sample_speculative(
                state=bad_observation,
                target_return=policy._speculative_queue[0]['predicted_return'],
                timestep=2,
                last_bitrate=2,
                buffer_size=float(bad_observation[..., 1, -1] * 10),
                video_chunk_remain=8,
            )
        self.assertEqual(plm.calls, 2)
        self.assertEqual(policy.get_speculative_metrics()['state_mismatch_fallbacks'], 1)
        self.assertTrue(plm.input_dtypes)
        self.assertEqual(set(plm.input_dtypes), {torch.float16})


if __name__ == '__main__':
    unittest.main()
