"""
Latency/memory measurement for a single pipeline.inference() call, used to compare the
baseline / all-patch / patch-selection multimodal modes against the 1-second viewport
prediction deadline.
"""
import time
import torch


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

    elapsed_t = torch.tensor(elapsed)
    mean_s = elapsed_t.mean().item()
    peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2) if is_cuda else float('nan')

    return {
        'mean_s': mean_s,
        'std_s': elapsed_t.std(unbiased=False).item() if iters > 1 else 0.0,
        'min_s': elapsed_t.min().item(),
        'max_s': elapsed_t.max().item(),
        'deadline_s': deadline_s,
        'margin_pct': (deadline_s - mean_s) / deadline_s * 100.0,
        'peak_memory_mb': peak_memory_mb,
        'iters': iters,
    }


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
    print(format_latency_report('dummy', stats))
    print('All latency_utils self-tests passed.')
