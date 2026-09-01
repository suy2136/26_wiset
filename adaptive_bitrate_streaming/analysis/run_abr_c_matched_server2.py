"""Server 2: run the compute-heavy C-matched Shapley AdaLoRA experiment."""

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
        (EXPERIMENTS[2],),
        "abr_c_matched_server2_shapley",
    )


if __name__ == "__main__":
    main()
