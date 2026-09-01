"""Server 1: run C-matched stock AdaLoRA followed by EVA."""

try:
    from adaptive_bitrate_streaming.analysis.run_abr_c_matched_allocators import (
        EXPERIMENTS,
        run_experiments,
    )
except ModuleNotFoundError:
    from run_abr_c_matched_allocators import EXPERIMENTS, run_experiments


def main(argv=None):
    run_experiments(
        argv,
        (EXPERIMENTS[0], EXPERIMENTS[1]),
        "abr_c_matched_server1_adalora_eva",
    )


if __name__ == "__main__":
    main()
