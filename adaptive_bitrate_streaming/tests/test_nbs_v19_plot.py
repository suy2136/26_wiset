import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image


ABR_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ABR_ROOT / 'analysis' / 'plot_nbs_v19_inference_results.py'
SPEC = importlib.util.spec_from_file_location('nbs_v19_plot', SCRIPT)
plot = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(plot)


class NBSV19PlotTest(unittest.TestCase):
    def test_official_netllm_metrics_are_added_to_svg(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            metrics_path = root / 'metrics.json'
            output_path = root / 'chart.png'
            metrics_path.write_text(json.dumps({
                'mean_reward': 0.75,
                'inference_latency_mean_ms': 60.0,
            }), encoding='utf-8')
            rows = [{
                'experiment': 'nbs_only',
                'mean_reward': 0.8,
                'inference_latency_mean_ms': 55.0,
            }]
            rows = plot.append_netllm_metrics(rows, metrics_path)
            plot.overview_chart(rows, output_path)
            with Image.open(output_path) as image:
                self.assertEqual(image.format, 'PNG')
                self.assertGreater(image.width, 1000)


if __name__ == '__main__':
    unittest.main()
