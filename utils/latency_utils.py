"""
Latency/memory measurement for a single pipeline.inference() call, used to compare the
baseline / all-patch / patch-selection multimodal modes against the 1-second viewport
prediction deadline.
"""
import csv
import json
import os
import tempfile
import time
import torch


def summarize_latency_samples(elapsed_s, deadline_s=1.0, warmup_calls=0,
                              total_calls=None, peak_memory_mb=float('nan')):
    """Summarize already-measured inference calls without repeating inference."""
    values = torch.as_tensor(elapsed_s, dtype=torch.float64)
    measured_calls = int(values.numel())
    total_calls = measured_calls + int(warmup_calls) if total_calls is None else int(total_calls)
    if measured_calls == 0:
        return {
            'mean_s': None, 'std_s': None, 'min_s': None, 'max_s': None,
            'median_s': None, 'p95_s': None, 'deadline_s': float(deadline_s),
            'margin_pct': None, 'peak_memory_mb': float(peak_memory_mb),
            'warmup_calls': int(warmup_calls), 'measured_calls': 0,
            'total_calls': total_calls,
        }
    mean_s = values.mean().item()
    return {
        'mean_s': mean_s,
        'std_s': values.std(unbiased=False).item() if measured_calls > 1 else 0.0,
        'min_s': values.min().item(),
        'max_s': values.max().item(),
        'median_s': torch.quantile(values, 0.5).item(),
        'p95_s': torch.quantile(values, 0.95).item(),
        'deadline_s': float(deadline_s),
        'margin_pct': (float(deadline_s) - mean_s) / float(deadline_s) * 100.0,
        'peak_memory_mb': float(peak_memory_mb),
        'warmup_calls': int(warmup_calls),
        'measured_calls': measured_calls,
        'total_calls': total_calls,
    }


def write_latency_artifacts(summary_path, elapsed_s, records, deadline_s=1.0,
                            warmup_calls=0, total_calls=None,
                            peak_memory_mb=float('nan'), measurement_scope=None):
    """Atomically write latency summary JSON and one row per timed inference."""
    summary = summarize_latency_samples(
        elapsed_s,
        deadline_s=deadline_s,
        warmup_calls=warmup_calls,
        total_calls=total_calls,
        peak_memory_mb=peak_memory_mb,
    )
    summary['measurement_scope'] = measurement_scope or {
        'operation': 'pipeline.inference',
        'includes_data_loading': False,
        'includes_metric_computation': False,
    }
    directory = os.path.dirname(summary_path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    temporary_path = summary_path + '.tmp'
    with open(temporary_path, 'w', encoding='utf-8') as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary_path, summary_path)

    detail_path = os.path.splitext(summary_path)[0] + '_per_sample.csv'
    temporary_detail_path = detail_path + '.tmp'
    fields = ['test_step', 'video', 'user', 'timestep', 'batch_size', 'latency_ms']
    with open(temporary_detail_path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary_detail_path, detail_path)
    return summary, detail_path


def measure_inference_latency(pipeline, batch, future, video_user_info, deadline_s=1.0, warmup=3, iters=10):
    """
    Times pipeline.inference() over `iters` repetitions (after `warmup` untimed calls) and
    reports GPU memory usage. Each timed call is bracketed with torch.cuda.synchronize() so
    the measurement includes actual GPU compute time, not just kernel-launch overhead.

    :param pipeline: a models.pipeline.Pipeline instance
    :param batch: history viewport trajectory, as passed to pipeline.inference()
    :param future: future viewport trajectory (ground truth), as passed to pipeline.inference()
    :param video_user_info: details information for current trajectory, as passed to pipeline.inference()
    :param deadline_s: the real-time deadline in seconds to compare against (default 1s)
    :param warmup: number of untimed warmup calls (lets CUDA kernels/cudnn autotune settle)
    :param iters: number of timed repetitions
    :return: dict with mean_s, std_s, min_s, max_s, margin_pct (positive = under deadline,
        negative = over), deadline_s, peak_memory_mb, iters
    """
    is_cuda = pipeline.device.startswith('cuda') if isinstance(pipeline.device, str) else torch.device(pipeline.device).type == 'cuda'

    for _ in range(warmup):
        with torch.no_grad():
            pipeline.inference(batch, future, video_user_info)

    if is_cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    elapsed = []
    for _ in range(iters):
        if is_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            pipeline.inference(batch, future, video_user_info)
        if is_cuda:
            torch.cuda.synchronize()
        elapsed.append(time.perf_counter() - t0)

    peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2) if is_cuda else float('nan')
    stats = summarize_latency_samples(
        elapsed, deadline_s=deadline_s, warmup_calls=warmup,
        total_calls=warmup + iters, peak_memory_mb=peak_memory_mb,
    )
    # Preserve the historical key consumed by existing benchmark scripts.
    stats['iters'] = iters
    return stats


def format_latency_report(label, stats):
    return (f'[{label}] mean={stats["mean_s"]*1000:.1f}ms std={stats["std_s"]*1000:.1f}ms '
            f'min={stats["min_s"]*1000:.1f}ms max={stats["max_s"]*1000:.1f}ms '
            f'margin={stats["margin_pct"]:+.1f}% vs {stats["deadline_s"]*1000:.0f}ms deadline '
            f'peak_mem={stats["peak_memory_mb"]:.1f}MB (n={stats["iters"]})')


if __name__ == '__main__':
    # self-test with a dummy pipeline-like object (no GPU/torch model needed)
    class DummyPipeline:
        device = 'cpu'

        def inference(self, batch, future, video_user_info):
            time.sleep(0.001)
            return None, None

    stats = measure_inference_latency(DummyPipeline(), None, None, None, deadline_s=1.0, warmup=2, iters=5)
    assert stats['iters'] == 5
    assert stats['mean_s'] > 0
    assert 0 < stats['margin_pct'] < 100  # ~1ms call against a 1s deadline should have a huge, but not >100%, margin
    with tempfile.TemporaryDirectory() as directory:
        summary_path = os.path.join(directory, 'latency.json')
        records = [
            {
                'test_step': index + 1, 'video': 0, 'user': 0,
                'timestep': index, 'batch_size': 1, 'latency_ms': value * 1000.0,
            }
            for index, value in enumerate((0.001, 0.002, 0.003))
        ]
        summary, detail_path = write_latency_artifacts(
            summary_path, (0.001, 0.002, 0.003), records,
            warmup_calls=2, total_calls=5,
        )
        assert summary['measured_calls'] == 3
        assert abs(summary['median_s'] - 0.002) < 1e-12
        assert os.path.exists(summary_path) and os.path.exists(detail_path)
    print(format_latency_report('dummy', stats))
    print('All latency_utils self-tests passed.')
