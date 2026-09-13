from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class FP16AttentionScoreSafetyTest(unittest.TestCase):
    def test_qk_operands_are_cast_before_matmul(self):
        source = (
            ROOT / "plm_special" / "models" / "low_rank.py"
        ).read_text(encoding="utf-8")
        self.assertIn(
            "query_states.float(), key_states.transpose(2, 3).float()",
            source,
        )
        self.assertIn(
            "attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32)",
            source,
        )
        self.assertIn(
            "attention_probabilities = attn_weights.to(value_states.dtype)",
            source,
        )

    def test_cli_flag_is_opt_in(self):
        source = (ROOT / "run_plm.py").read_text(encoding="utf-8")
        self.assertIn("'--fp16-attention-fp32-scores'", source)
        self.assertIn("action='store_true'", source)


if __name__ == "__main__":
    unittest.main()
