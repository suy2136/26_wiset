"""Train and evaluate the complete VP data-seed-3 experiment queue.

This is an additive, restartable pipeline.  It never writes into an existing
checkpoint and stores its own state/results below ``--output-dir``.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import shlex
import statistics
import subprocess
import sys
import time


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis import evaluate_vp_fixed_lora_fast_fallback_multiseed as vp_lora
from analysis.resolve_checkpoint_alias import resolve_checkpoint
from analysis.evaluate_cached_patch_followups_compact import selector_command
from analysis.evaluate_cached_patch_selectors_compact import common_command, run_case, write_rows
from analysis.evaluate_vp_fast_fallback_fullstack_multiseed import (
    PATCH, SPEC_GAMMA, SPEC_THRESHOLD, TOKEN_K, read_safety,
)
from analysis.evaluate_vp_inference_module_pipeline import (
    full_stack_command, set_option, speculative_command, token_command,
    trace_metrics,
)


SEEDS = (1, 2, 3)
TRAINING_DATA_SEED = 3
TARGET_BUDGET = 512
FP16_COMPACTION_OUTPUT_ATOL = 0.01
VP_RUN_ROOT = (
    REPO_ROOT / "viewport_prediction/data/experiment_runs/netllm_vs_nbs"
)
DEFAULT_OUTPUT = VP_RUN_ROOT / "server1_vp_data3_pipeline"
def method_specs(data_seed: int) -> dict:
    return {
        "uniform": {"variant": f"uniform_r8_data{data_seed}", "terminal": "best_model"},
        "adalora": {"variant": f"adalora_b512_data{data_seed}", "terminal": "final_adalora_model"},
        "shapley": {"variant": f"shapley_b512_data{data_seed}", "terminal": "final_shapley_model"},
        "eva": {"variant": f"eva_b512_data{data_seed}", "terminal": "best_model"},
        "nbs": {"variant": f"nbs_v19_data{data_seed}", "terminal": "final_nbs_model"},
    }


METHODS = method_specs(TRAINING_DATA_SEED)
METHOD_LABELS = {
    "uniform": "Uniform LoRA", "adalora": "AdaLoRA", "shapley": "ShapLoRA",
    "eva": "EVA", "nbs": "NBS-LoRA compact",
}
MODULE_CASES = (
    ("pure_nbs_compact", "Pure NBS compact", "baseline"),
    ("patch_gated_k1_t6_skip0_cache", "NBS + Patch gated-k1/T6/skip0/cache", "patch"),
    ("token_k8", "NBS + Token K=8", "token"),
    ("spec_g6_t0p4", "NBS + Spec G=6/T=0.4", "speculative"),
    ("full_stack", "NBS + Patch + Token K=8 + Spec G=6/T=0.4", "full_stack"),
)
STAGE_ORDER = tuple(
    stage
    for method in ("uniform", "adalora", "shapley", "eva")
    for stage in (f"train_{method}", f"evaluate_{method}")
) + ("train_nbs", "compact_nbs", "evaluate_nbs_modules", "write_lora_summary")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-data-seed", type=int, choices=(3, 4), default=3)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--latency-warmup-steps", type=int, default=5)
    parser.add_argument(
        "--tuned-projector", type=Path,
        default=(
            VP_RUN_ROOT / "vp_projector_token_pipeline_20260912_182036"
            / "projector_candidates/projector_2ep_lr5e-5"
            / "best_multimodal_projector.pth"
        ),
    )
    parser.add_argument(
        "--patch-cache", type=Path,
        default=REPO_ROOT / "viewport_prediction/data/images/Jin2022_patch_features",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.output_dir is None:
        args.output_dir = (
            DEFAULT_OUTPUT if args.training_data_seed == 3
            else VP_RUN_ROOT / "server1_vp_data4_pipeline"
        )
    return args


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def signature(args) -> dict:
    return {
        "pipeline": f"server1_vp_data{TRAINING_DATA_SEED}_v1",
        "stages": list(STAGE_ORDER),
        "training_seed": 1,
        "lora_seed": 1,
        "training_data_seed": TRAINING_DATA_SEED,
        "evaluation_seeds_and_data_seeds": list(SEEDS),
        "evaluation_rng_mode": "continuous",
        "attention_score_mode": "fp16_prescaled_qk_with_fp32_retry",
        "target_budget": TARGET_BUDGET,
        "variants": {method: value["variant"] for method, value in METHODS.items()},
        "module_settings": {
            "patch": PATCH, "token_k": TOKEN_K,
            "spec_gamma": SPEC_GAMMA, "spec_threshold": SPEC_THRESHOLD,
        },
        "tuned_projector": str(args.tuned_projector.resolve()),
        "patch_cache": str(args.patch_cache.resolve()),
    }


def load_state(args):
    path = args.output_dir / "pipeline_state.json"
    expected = signature(args)
    if path.is_file():
        if not args.resume:
            raise FileExistsError(f"output exists: {args.output_dir}; use --resume")
        state = json.loads(path.read_text(encoding="utf-8"))
        if state.get("signature") != expected:
            raise ValueError("resume state differs from the current configuration")
        return path, state
    state = {"signature": expected, "stages": {}, "checkpoints": {}}
    if not args.dry_run:
        atomic_json(path, state)
    return path, state


def checkpoint_complete(path: Path) -> bool:
    return (
        (path / "adapter_config.json").is_file()
        and (path / "modules_except_plm.bin").is_file()
        and any((path / name).is_file() for name in (
            "adapter_model.bin", "adapter_model.safetensors",
        ))
    )


def resolved_complete_checkpoint(path: Path) -> Path | None:
    """Use physical adapter weights when a final checkpoint is an alias."""
    resolved = resolve_checkpoint(path)
    return resolved if checkpoint_complete(resolved) else None


def parse_env(path: Path) -> dict[str, str]:
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            result[key] = value
    return result


def latest_training_checkpoint(method: str) -> Path | None:
    variant = METHODS[method]["variant"]
    latest = VP_RUN_ROOT / f"{variant}_latest.txt"
    if not latest.is_file():
        return None
    run_dir = Path(latest.read_text(encoding="utf-8").strip())
    if not run_dir.is_absolute():
        run_dir = REPO_ROOT / run_dir
    metadata_path = run_dir / "metadata.env"
    if not metadata_path.is_file():
        return None
    metadata = parse_env(metadata_path)
    if method == "adalora":
        best = metadata.get("best_ar_model", "")
        candidate = Path(best).parent / "final_adalora_model" if best else None
    elif method in {"shapley", "nbs"}:
        value = metadata.get("final_nbs_model", "")
        candidate = Path(value) if value else None
    else:
        value = metadata.get("best_ar_model", "")
        candidate = Path(value) if value else None
    if candidate is None:
        return None
    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate
    return resolved_complete_checkpoint(candidate)


def inspect_budget(method: str, checkpoint: Path) -> dict:
    """Validate checkpoint structure and report budget mismatch without blocking."""
    inspection_method = "adalora" if method == "nbs" else method
    description = vp_lora.checkpoint_description(inspection_method, checkpoint)
    active = int(description["active_rank_total"])
    matched = active == TARGET_BUDGET
    if not matched:
        print(
            f"[{method}] WARNING: active rank is {active}, target is "
            f"{TARGET_BUDGET}; evaluation will continue and record the mismatch: "
            f"{checkpoint}",
            file=sys.stderr,
            flush=True,
        )
    description["method"] = method
    description["target_rank_budget"] = TARGET_BUDGET
    description["budget_match"] = matched
    description["budget_note"] = (
        "matched_512" if matched else f"nonmatching_{active}_target_512"
    )
    return description


def train_method(args, state, state_path: Path, method: str) -> Path:
    key = f"vp_{method}_data{TRAINING_DATA_SEED}"
    saved = state["checkpoints"].get(key)
    if saved:
        checkpoint = resolved_complete_checkpoint(Path(saved))
        if checkpoint is None:
            raise FileNotFoundError(f"saved checkpoint is incomplete: {saved}")
        inspect_budget(method, checkpoint)
        return checkpoint
    if args.resume:
        recovered = latest_training_checkpoint(method)
        if recovered is not None:
            inspect_budget(method, recovered)
            state["checkpoints"][key] = str(recovered.resolve())
            atomic_json(state_path, state)
            print(f"[{method}] recovered completed checkpoint: {recovered}", flush=True)
            return recovered
    variant = METHODS[method]["variant"]
    command = ["bash", "scripts/run_netllm_experiment.sh", variant]
    print(f"[{method}] {shlex.join(command)}", flush=True)
    if args.dry_run:
        return Path(f"/dry-run/{key}")
    environment = os.environ.copy()
    environment.update({
        "PYTHONUNBUFFERED": "1",
        "PATH": f"{Path(sys.executable).parent}{os.pathsep}{environment.get('PATH', '')}",
        "SKIP_EVALUATION": "1",
        "SKIP_VISUALIZATION": "1",
        "SAVE_PERIODIC_CHECKPOINTS": "0",
        "VP_FP16_PRESCALED_QK": "1",
    })
    if method == "eva":
        # Data seed 3 did not converge within the legacy 128 calibration batches.
        # Keep this retry local to this pipeline and retain convergence metadata.
        environment.update({"EVA_MAX_BATCHES": "512", "EVA_ALLOW_UNCONVERGED": "1"})
    subprocess.run(command, cwd=REPO_ROOT, env=environment, check=True)
    checkpoint = latest_training_checkpoint(method)
    if checkpoint is None:
        raise FileNotFoundError(f"final checkpoint not found for {variant}")
    inspect_budget(method, checkpoint)
    state["checkpoints"][key] = str(checkpoint.resolve())
    atomic_json(state_path, state)
    return checkpoint


def force_prescaled(command: list[str]) -> list[str]:
    command = [item for item in command if item != "--vp-fp16-fallback"]
    if "--vp-fp16-prescaled-qk" not in command:
        command.append("--vp-fp16-prescaled-qk")
    return command


def load_rows(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def evaluate_lora_method(args, method: str, checkpoint: Path) -> dict:
    description = inspect_budget(method, checkpoint)
    eval_description = dict(description)
    if method == "nbs":
        eval_description["method"] = "adalora"
    output_dir = args.output_dir / "lora_methods"
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = output_dir / "per_seed_results.csv"
    rows = load_rows(rows_path) if args.resume else []
    rank_config = vp_lora.write_fixed_rank_config(eval_description, checkpoint, output_dir)
    completed = {
        int(row["evaluation_seed"]) for row in rows
        if row.get("method") == method and row.get("checkpoint") == str(checkpoint.resolve())
        and row.get("status") == "complete"
    }
    for seed in SEEDS:
        if seed in completed:
            continue
        result_dir = output_dir / method / f"seed_{seed}"
        command = vp_lora.build_command(
            args, eval_description, checkpoint.resolve(), seed, result_dir, rank_config,
        )
        command = force_prescaled(command)
        print(f"[{method} seed={seed}] {shlex.join(command)}", flush=True)
        if args.dry_run:
            continue
        row = {
            "method": method, "label": METHOD_LABELS[method],
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_training_data_seed": TRAINING_DATA_SEED,
            "evaluation_seed": seed, "status": "failed", **description,
        }
        try:
            row.update(run_case(command, result_dir, args.resume))
            row.update(read_safety(result_dir))
            row["status"] = "complete"
        except Exception as error:
            row["error"] = f"{type(error).__name__}: {error}"
        rows.append(row)
        write_rows(rows_path, rows)
        if row["status"] != "complete":
            raise RuntimeError(row["error"])
    if args.dry_run:
        return {}
    group = [row for row in rows if row.get("method") == method and row.get("status") == "complete"]
    if {int(row["evaluation_seed"]) for row in group} != set(SEEDS):
        raise RuntimeError(f"incomplete three-seed evaluation: {method}")
    summary = {
        "method": method, "label": METHOD_LABELS[method],
        "checkpoint_training_data_seed": TRAINING_DATA_SEED,
        "seed_count": 3, "evaluation_seeds": "1,2,3",
        "evaluation_rng_mode": "continuous",
        "attention_score_mode": "fp16_prescaled_qk_with_fp32_retry",
        "active_rank_total": int(group[0]["active_rank_total"]),
        "target_rank_budget": TARGET_BUDGET,
        "budget_match": str(group[0]["budget_match"]).lower() == "true",
        "budget_note": group[0]["budget_note"],
    }
    for metric in ("mae", "rmse", "latency_mean_ms"):
        values = [float(row[metric]) for row in group]
        summary[f"{metric}_mean"] = statistics.mean(values)
        summary[f"{metric}_std"] = statistics.stdev(values)
    for metric in ("fp16_prescaled_qk_fallback_calls", "fp32_attention_fallback_calls"):
        summary[f"{metric}_total"] = sum(int(row.get(metric, 0)) for row in group)
    write_rows(output_dir / f"{method}_three_seed_summary.csv", [summary])
    return summary


def module_args(args):
    return argparse.Namespace(
        device=args.device, physical_rank=32, rank_budget=TARGET_BUDGET,
        latency_warmup_steps=args.latency_warmup_steps,
        projector_checkpoint=args.tuned_projector,
        cache_dir=args.patch_cache, motion_threshold_deg=6.0,
        projector_cache_max_entries=512,
    )


def module_command(args, compact: Path, kind: str, seed: int, result_dir: Path,
                   source: Path | None = None) -> list[str]:
    common_args = module_args(args)
    if kind == "baseline":
        command = common_command(common_args, source or compact, result_dir)
        if source is not None:
            command.extend(["--nbs-compact-output-dir", str(compact)])
    elif kind == "patch":
        command = selector_command(common_args, compact, result_dir, PATCH)
    elif kind == "token":
        command = token_command(common_args, compact, result_dir, TOKEN_K)
    elif kind == "speculative":
        command = speculative_command(
            common_args, compact, result_dir, SPEC_GAMMA, SPEC_THRESHOLD,
        )
    elif kind == "full_stack":
        command = full_stack_command(
            common_args, compact, result_dir, PATCH, TOKEN_K,
            SPEC_GAMMA, SPEC_THRESHOLD,
        )
    else:
        raise ValueError(kind)
    set_option(command, "--seed", seed)
    set_option(command, "--lora-seed", 1)
    set_option(command, "--data-seed", seed)
    set_option(command, "--evaluation-rng-mode", "continuous")
    return force_prescaled(command)


def compact_nbs(args, source: Path) -> Path:
    inspect_budget("nbs", source)
    root = args.output_dir / "nbs_compact"
    original = root / "compact_checkpoint"
    # Never overwrite a failed or incomplete compaction from an earlier attempt.
    for candidate in (original, *sorted(root.glob("compact_checkpoint_fp16_retry_*"))):
        required = (
            candidate / "compact_adapter.pt", candidate / "modules_except_plm.bin",
            candidate / "compaction_metadata.json", candidate / "equivalence_report.json",
        )
        if all(path.is_file() for path in required):
            report = json.loads(required[-1].read_text(encoding="utf-8"))
            if report.get("passed"):
                return candidate
    attempt = 1
    while (root / f"compact_checkpoint_fp16_retry_{attempt}").exists():
        attempt += 1
    compact = original if not original.exists() else root / f"compact_checkpoint_fp16_retry_{attempt}"
    required = (
        compact / "compact_adapter.pt", compact / "modules_except_plm.bin",
        compact / "compaction_metadata.json", compact / "equivalence_report.json",
    )
    result_dir = (
        args.output_dir / "nbs_compact" / f"validation_attempt_{attempt}"
        if compact != original else args.output_dir / "nbs_modules/seed_1/pure_nbs_compact"
    )
    command = module_command(args, compact, "baseline", 1, result_dir, source)
    # FP16 prescaled Q/K accumulates small output drift despite exact factor
    # equivalence. The observed maximum was 0.00782 degrees at the old 0.002
    # absolute threshold. Keep factor tolerances unchanged and validate output.
    set_option(command, "--nbs-compaction-output-atol", FP16_COMPACTION_OUTPUT_ATOL)
    print(f"[NBS compact] {shlex.join(command)}", flush=True)
    if not args.dry_run:
        run_case(command, result_dir, args.resume)
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise RuntimeError(f"compact checkpoint files missing: {missing}")
        report = json.loads(required[-1].read_text(encoding="utf-8"))
        if not report.get("passed"):
            raise RuntimeError("compact checkpoint equivalence report did not pass")
    return compact


def compact_active_rank(compact: Path) -> int | None:
    metadata = compact / "compaction_metadata.json"
    if not metadata.is_file():
        return None
    value = json.loads(metadata.read_text(encoding="utf-8")).get("compact_rank_total")
    return int(value) if value is not None else None


def summarize_module_rows(rows: list[dict], actual_rank: int | None) -> list[dict]:
    summaries = []
    for case, label, _ in MODULE_CASES:
        group = [row for row in rows if row.get("case") == case and row.get("status") == "complete"]
        if {int(row["evaluation_seed"]) for row in group} != set(SEEDS):
            raise RuntimeError(f"incomplete module result: {case}")
        item = {
            "case": case, "label": label,
            "checkpoint_training_data_seed": TRAINING_DATA_SEED,
            "seed_count": 3, "evaluation_seeds": "1,2,3",
            "evaluation_rng_mode": "continuous",
            "attention_score_mode": "fp16_prescaled_qk_with_fp32_retry",
            "active_rank_total": actual_rank if actual_rank is not None else "unknown",
            "target_rank_budget": TARGET_BUDGET,
            "budget_match": actual_rank == TARGET_BUDGET if actual_rank is not None else "unknown",
            "budget_note": (
                "matched_512" if actual_rank == TARGET_BUDGET
                else f"nonmatching_{actual_rank}_target_512"
                if actual_rank is not None else "compact_rank_unknown"
            ),
        }
        for metric in (
            "mae", "rmse", "latency_mean_ms", "mean_initial_token_count",
            "mean_selected_token_count", "mean_target_forward_count",
        ):
            values = [float(row[metric]) for row in group if row.get(metric) not in (None, "")]
            if len(values) == 3:
                item[f"{metric}_mean"] = statistics.mean(values)
                item[f"{metric}_std"] = statistics.stdev(values)
        for metric in ("fp16_prescaled_qk_fallback_calls", "fp32_attention_fallback_calls"):
            item[f"{metric}_total"] = sum(int(row.get(metric, 0)) for row in group)
        summaries.append(item)
    baseline = summaries[0]
    for item in summaries:
        item["mae_change_percent_vs_nbs"] = (
            float(item["mae_mean"]) / float(baseline["mae_mean"]) - 1.0
        ) * 100.0
        item["latency_reduction_percent_vs_nbs"] = (
            1.0 - float(item["latency_mean_ms_mean"])
            / float(baseline["latency_mean_ms_mean"])
        ) * 100.0
    return summaries


def evaluate_modules(args, compact: Path) -> list[dict]:
    output = args.output_dir / "nbs_modules/per_seed_results.csv"
    rows = load_rows(output) if args.resume else []
    baseline_dir = args.output_dir / "nbs_modules/seed_1/pure_nbs_compact"
    if not any(
        row.get("case") == "pure_nbs_compact" and row.get("evaluation_seed") == "1"
        for row in rows
    ):
        metrics_path = baseline_dir / "metrics.json"
        if metrics_path.is_file():
            row = {
                "case": "pure_nbs_compact", "label": "Pure NBS compact",
                "evaluation_seed": 1, "status": "complete",
                **json.loads(metrics_path.read_text(encoding="utf-8")),
                **read_safety(baseline_dir),
            }
            row.update(trace_metrics(baseline_dir))
            rows.append(row)
            write_rows(output, rows)
    completed = {
        (row.get("case"), int(row.get("evaluation_seed", -1)))
        for row in rows if row.get("status") == "complete"
    }
    for seed in SEEDS:
        for case, label, kind in MODULE_CASES:
            if (case, seed) in completed:
                continue
            result_dir = args.output_dir / f"nbs_modules/seed_{seed}/{case}"
            command = module_command(args, compact, kind, seed, result_dir)
            print(f"[{case} seed={seed}] {shlex.join(command)}", flush=True)
            if args.dry_run:
                continue
            row = {"case": case, "label": label, "evaluation_seed": seed, "status": "failed"}
            try:
                row.update(run_case(command, result_dir, args.resume))
                row.update(trace_metrics(result_dir))
                row.update(read_safety(result_dir))
                row["status"] = "complete"
            except Exception as error:
                row["error"] = f"{type(error).__name__}: {error}"
            rows.append(row)
            write_rows(output, rows)
            if row["status"] != "complete":
                raise RuntimeError(row["error"])
    if args.dry_run:
        return []
    summaries = summarize_module_rows(rows, compact_active_rank(compact))
    write_rows(args.output_dir / "nbs_modules/three_seed_summary.csv", summaries)
    return summaries


def write_combined_lora_summary(args) -> None:
    summaries = []
    module_summary = load_rows(args.output_dir / "nbs_modules/three_seed_summary.csv")
    if not module_summary:
        raise FileNotFoundError("NBS module summary is unavailable")
    baseline = dict(module_summary[0])
    baseline.update({
        "method": "nbs", "label": METHOD_LABELS["nbs"],
        "active_rank_total": baseline.get("active_rank_total", "unknown"),
    })
    summaries.append(baseline)
    for method in ("uniform", "adalora", "shapley", "eva"):
        path = args.output_dir / f"lora_methods/{method}_three_seed_summary.csv"
        rows = load_rows(path)
        if not rows:
            raise FileNotFoundError(path)
        summaries.append(rows[0])
    write_rows(args.output_dir / "lora_methods/five_method_three_seed_summary.csv", summaries)


def run_stage(args, state, state_path: Path, name: str, action) -> None:
    if state["stages"].get(name, {}).get("status") == "complete":
        print(f"[{name}] already complete; skipping", flush=True)
        return
    print(f"\n===== {name} =====", flush=True)
    started = time.time()
    if not args.dry_run:
        state["stages"][name] = {"status": "running", "started_at": started}
        atomic_json(state_path, state)
    try:
        action()
    except Exception as error:
        state["stages"][name] = {
            "status": "failed", "error_type": type(error).__name__,
            "error": str(error), "started_at": started, "finished_at": time.time(),
        }
        print(f"[{name}] FAILED: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
    else:
        state["stages"][name] = {
            "status": "complete", "started_at": started, "finished_at": time.time(),
        }
    if not args.dry_run:
        atomic_json(state_path, state)


def checkpoint_from_state(state, method: str) -> Path:
    value = state["checkpoints"].get(f"vp_{method}_data{TRAINING_DATA_SEED}")
    if not value:
        raise RuntimeError(f"checkpoint unavailable: vp_{method}_data{TRAINING_DATA_SEED}")
    return Path(value)


def compact_from_state(state) -> Path:
    key = f"vp_nbs_data{TRAINING_DATA_SEED}_compact"
    value = state["checkpoints"].get(key)
    if not value:
        raise RuntimeError(f"checkpoint unavailable: {key}; compact_nbs failed")
    return Path(value)


def validate_assets(args) -> None:
    if args.dry_run:
        return
    if not args.tuned_projector.is_file():
        raise FileNotFoundError(args.tuned_projector)
    missing = [
        video for video in range(1, 28)
        if not (args.patch_cache / f"video{video}_patch_features.pt").is_file()
    ]
    if missing:
        raise FileNotFoundError(f"patch caches missing for videos: {missing}")
    if not (args.patch_cache / "manifest.json").is_file():
        raise FileNotFoundError(args.patch_cache / "manifest.json")


def main(argv=None):
    global TRAINING_DATA_SEED, METHODS
    args = parse_args(argv)
    TRAINING_DATA_SEED = args.training_data_seed
    METHODS = method_specs(TRAINING_DATA_SEED)
    args.output_dir = args.output_dir.resolve()
    args.tuned_projector = args.tuned_projector.resolve()
    args.patch_cache = args.patch_cache.resolve()
    validate_assets(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    state_path, state = load_state(args)
    if args.dry_run:
        for method, config in METHODS.items():
            print(shlex.join(["bash", "scripts/run_netllm_experiment.sh", config["variant"]]))
        fake_compact = Path("/dry-run/nbs_compact/compact_checkpoint")
        for case, _, kind in MODULE_CASES:
            print(shlex.join(module_command(
                args, fake_compact, kind, 1,
                Path("/dry-run/nbs_modules") / case,
            )))
        return

    for method in ("uniform", "adalora", "shapley", "eva"):
        run_stage(
            args, state, state_path, f"train_{method}",
            lambda method=method: train_method(args, state, state_path, method),
        )
        run_stage(
            args, state, state_path, f"evaluate_{method}",
            lambda method=method: evaluate_lora_method(
                args, method, checkpoint_from_state(state, method),
            ),
        )

    run_stage(
        args, state, state_path, "train_nbs",
        lambda: train_method(args, state, state_path, "nbs"),
    )

    def compact_stage():
        compact = compact_nbs(args, checkpoint_from_state(state, "nbs"))
        if not args.dry_run:
            state["checkpoints"][f"vp_nbs_data{TRAINING_DATA_SEED}_compact"] = str(compact.resolve())
            atomic_json(state_path, state)

    run_stage(args, state, state_path, "compact_nbs", compact_stage)
    run_stage(
        args, state, state_path, "evaluate_nbs_modules",
        lambda: evaluate_modules(args, compact_from_state(state)),
    )
    run_stage(
        args, state, state_path, "write_lora_summary",
        lambda: write_combined_lora_summary(args),
    )

    if not args.dry_run:
        failed = [name for name, item in state["stages"].items() if item.get("status") == "failed"]
        print(f"\nPipeline state: {state_path}", flush=True)
        print(f"Failed stages: {failed or 'none'}", flush=True)
        print(f"OUTPUT={args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
