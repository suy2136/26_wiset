"""Server 2: EVA and Shapley AdaLoRA with data seed 3."""

try:
    from adaptive_bitrate_streaming.analysis.run_abr_c1536_data3_safeguarded_allocators import (
        EXPERIMENTS,
        run_experiments,
    )
except ModuleNotFoundError:
    from run_abr_c1536_data3_safeguarded_allocators import (
        EXPERIMENTS,
        run_experiments,
    )


SERVER_EXPERIMENTS = EXPERIMENTS[3:]


def main(argv=None):
    run_experiments(
        argv, SERVER_EXPERIMENTS, "abr_c1536_data3_server2_eva_shapley"
    )


if __name__ == "__main__":
    main()
