"""Run server-1 VP/ABR training and fixed-checkpoint evaluations safely.

All newly trained checkpoints and all evaluation artifacts are written below a
new output directory. Existing checkpoints are opened read-only. The pipeline
is restartable and records each stage independently so an unrelated later stage
can still run after an earlier failure.
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
from analysis.evaluate_cached_patch_followups_compact import selector_command
from analysis.evaluate_cached_patch_selectors_compact import (
    common_command, run_case, write_rows,
)
from analysis.evaluate_vp_fast_fallback_fullstack_multiseed import (
    PATCH, SPEC_GAMMA, SPEC_THRESHOLD, TOKEN_K, read_safety,
)
from analysis.evaluate_vp_inference_module_pipeline import (
    full_stack_command, set_option, speculative_command, token_command,
    trace_metrics,
)
from adaptive_bitrate_streaming.analysis import (
    run_abr_lora_methods_fp16_prescaled_multiseed as abr_lora,
)


VP_MODEL_ROOT = REPO_ROOT / "viewport_prediction" / "data" / "ft_plms"
ABR_ROOT = REPO_ROOT / "adaptive_bitrate_streaming"
ABR_MODEL_ROOT = ABR_ROOT / "data" / "ft_plms"
DEFAULT_OUTPUT = (
    REPO_ROOT / "viewport_prediction" / "data" / "experiment_runs"
    / "netllm_vs_nbs" / "server1_overnight_pipeline"
)
SEEDS = (1, 2, 3)
STAGE_ORDER = (
    "train_vp_adalora_data1",
    "evaluate_vp_adalora_data1",
    "train_vp_shapley_data1",
    "evaluate_vp_shapley_data1",
    "validate_abr_data3",
    "evaluate_abr_data3_eva_shapley",
    "validate_vp_data2",
    "compact_vp_nbs_data2",
    "evaluate_vp_data2_lora_methods",
    "evaluate_vp_data2_nbs_modules",
)
VP_DATA2_FRAGMENTS = {
    "uniform": "llama_base_low_rank_uniform_r8_data2",
    "adalora": "llama_base_low_rank_adalora_adalora_b512_data2",
    "shapley": "llama_base_low_rank_adalora_shapley_b512_data2",
    "eva": "llama_base_low_rank_eva_b512_data2",
    "nbs": "llama_base_low_rank_adalora_nbs_v19_data2",
}
VP_TERMINALS = {
    "uniform": ("best_model",),
    "adalora": ("best_ar_model",),
    "shapley": ("final_shapley_model", "best_ar_model"),
    "eva": ("best_model",),
    "nbs": ("best_ar_model",),
}
ABR_DATA3_FRAGMENTS = {
    "eva": "rank_32_eva_budget1536_run_fp16safe_data3_v1",
    "shapley": "rank_32_shapley_budget1536_run_fp16safe_data3_v1",
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--latency-warmup-steps", type=int, default=5)
    parser.add_argument("--base-model-dir", type=Path,
                        default=REPO_ROOT / "downloaded_plms/llama/base")
    parser.add_argument("--exp-pool-path", type=Path,
                        default=ABR_ROOT / "artifacts/exp_pools/exp_pool.pkl")
    parser.add_argument(
        "--tuned-projector", type=Path,
        default=(
            REPO_ROOT / "viewport_prediction/data/experiment_runs/netllm_vs_nbs"
            / "vp_projector_token_pipeline_20260912_182036/projector_candidates"
            / "projector_2ep_lr5e-5/best_multimodal_projector.pth"
        ),
    )
    parser.add_argument(
        "--patch-cache", type=Path,
        default=REPO_ROOT / "viewport_prediction/data/images/Jin2022_patch_features",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def signature(args):
    return {
        "pipeline": "server1_overnight_v1",
        "stage_order": list(STAGE_ORDER),
        "vp_training": ["adalora_b512_data1", "shapley_b512_data1"],
        "vp_evaluation_seeds": list(SEEDS),
        "vp_evaluation_rng_mode": "continuous",
        "abr_evaluation_seeds": list(SEEDS),
        "abr_evaluation_rng_mode": "per-episode",
        "attention_score_mode": "fp16_prescaled_qk_with_fp32_retry",
        "vp_data2_methods": list(VP_DATA2_FRAGMENTS),
        "vp_module_settings": {
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
            raise ValueError("resume state differs from current configuration")
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


def find_checkpoint(root: Path, fragment: str, terminals: tuple[str, ...]) -> Path:
    candidates = []
    for config in root.rglob("adapter_config.json"):
        path = config.parent
        if fragment in str(path) and path.name in terminals and checkpoint_complete(path):
            candidates.append(path)
    if not candidates:
        raise FileNotFoundError(f"checkpoint not found: {fragment} / {terminals}")
    terminal_priority = {name: len(terminals) - index for index, name in enumerate(terminals)}
    return max(candidates, key=lambda path: (terminal_priority[path.name], path.stat().st_mtime))


def parse_env(path: Path) -> dict[str, str]:
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key] = value
    return values


def validate_exact_vp_budget(method: str, checkpoint: Path) -> dict:
    description = vp_lora.checkpoint_description(method, checkpoint)
    if description["active_rank_total"] != 512:
        raise ValueError(
            f"{method} active rank is {description['active_rank_total']}, expected 512"
        )
    return description


def train_vp(args, state, state_path, method: str) -> Path:
    key = f"vp_{method}_data1"
    saved = state["checkpoints"].get(key)
    if saved:
        checkpoint = Path(saved)
        validate_exact_vp_budget(method, checkpoint)
        return checkpoint
    variant = "adalora_b512_data1" if method == "adalora" else "shapley_b512_data1"
    command = ["bash", "scripts/run_netllm_experiment.sh", variant]
    print(f"[{key}] {shlex.join(command)}", flush=True)
    if args.dry_run:
        return Path(f"/dry-run/{key}")
    environment = os.environ.copy()
    environment.update({
        "PYTHONUNBUFFERED": "1", "SKIP_EVALUATION": "1",
        "SKIP_VISUALIZATION": "1", "SAVE_PERIODIC_CHECKPOINTS": "0",
    })
    subprocess.run(command, cwd=REPO_ROOT, env=environment, check=True)
    latest_file = (
        REPO_ROOT / "viewport_prediction/data/experiment_runs/netllm_vs_nbs"
        / f"{variant}_latest.txt"
    )
    run_dir = REPO_ROOT / latest_file.read_text(encoding="utf-8").strip()
    metadata = parse_env(run_dir / "metadata.env")
    if method == "adalora":
        checkpoint = Path(metadata["best_ar_model"]).parent / "final_adalora_model"
    else:
        checkpoint = Path(metadata["final_nbs_model"])
    checkpoint = checkpoint if checkpoint.is_absolute() else REPO_ROOT / checkpoint
    validate_exact_vp_budget(method, checkpoint)
    state["checkpoints"][key] = str(checkpoint.resolve())
    atomic_json(state_path, state)
    return checkpoint


def force_vp_prescaled(command: list[str]) -> list[str]:
    command = [item for item in command if item != "--vp-fp16-fallback"]
    if "--vp-fp16-prescaled-qk" not in command:
        command.append("--vp-fp16-prescaled-qk")
    return command


def load_csv(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def summarize_vp_method(rows: list[dict], method: str, label: str,
                        training_data_seed: int) -> dict:
    group = [row for row in rows if row.get("method") == method and row.get("status") == "complete"]
    if {int(row["evaluation_seed"]) for row in group} != set(SEEDS):
        raise RuntimeError(f"incomplete VP three-seed result: {method}")
    result = {
        "method": method, "label": label,
        "checkpoint_training_data_seed": training_data_seed,
        "seed_count": 3, "evaluation_seeds": "1,2,3",
        "evaluation_rng_mode": "continuous",
        "attention_score_mode": "fp16_prescaled_qk_with_fp32_retry",
        "active_rank_total": int(group[0]["active_rank_total"]),
    }
    for metric in ("mae", "rmse", "latency_mean_ms"):
        values = [float(row[metric]) for row in group]
        result[f"{metric}_mean"] = statistics.mean(values)
        result[f"{metric}_std"] = statistics.stdev(values)
    for metric in ("fp16_prescaled_qk_fallback_calls", "fp32_attention_fallback_calls"):
        result[f"{metric}_total"] = sum(int(row.get(metric, 0)) for row in group)
    return result


def evaluate_vp_method(args, method: str, checkpoint: Path,
                       training_data_seed: int, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    description = validate_exact_vp_budget(method, checkpoint)
    rows_path = output_dir / "per_seed_results.csv"
    rows = load_csv(rows_path) if args.resume else []
    rank_config = vp_lora.write_fixed_rank_config(description, checkpoint, output_dir)
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
            args, description, checkpoint.resolve(), seed, result_dir, rank_config,
        )
        command = force_vp_prescaled(command)
        print(f"[{method} seed={seed}] {shlex.join(command)}", flush=True)
        if args.dry_run:
            continue
        row = {
            "method": method, "checkpoint": str(checkpoint.resolve()),
            "checkpoint_training_data_seed": training_data_seed,
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
    label = {"adalora": "Stock AdaLoRA", "shapley": "ShapLoRA",
             "uniform": "Uniform LoRA", "eva": "EVA"}[method]
    summary = summarize_vp_method(rows, method, label, training_data_seed)
    write_rows(output_dir / f"{method}_three_seed_summary.csv", [summary])
    return summary


def validate_abr_checkpoint(path: Path, method: str) -> dict:
    required = ["adapter_config.json", "modules_except_plm.bin", "checkpoint_metadata.json"]
    if method == "eva":
        required.append("eva_state.pt")
    missing = [name for name in required if not (path / name).is_file()]
    if not any((path / name).is_file() for name in ("adapter_model.bin", "adapter_model.safetensors")):
        missing.append("adapter weights")
    if missing:
        raise FileNotFoundError(f"incomplete ABR {method}: {missing}")
    metadata = json.loads((path / "checkpoint_metadata.json").read_text(encoding="utf-8"))
    if int(metadata.get("effective_rank_budget", -1)) != 1536:
        raise ValueError(f"ABR {method} is not budget 1536: {metadata}")
    return metadata


def evaluate_abr_data3(args, checkpoints: dict[str, Path]) -> None:
    output = args.output_dir / "abr_data3" / "per_seed_results.csv"
    rows = abr_lora.load_rows(output) if args.resume else []
    command_args = argparse.Namespace(
        base_model_dir=args.base_model_dir, exp_pool_path=args.exp_pool_path,
        rank_budget=1536, trace="fcc-test", trace_num=100,
        video="video1", device=args.device,
    )
    completed = {
        (row.get("method"), int(row.get("evaluation_seed", -1)))
        for row in rows if row.get("status", "complete") == "complete"
    }
    for method in ("eva", "shapley"):
        checkpoint = checkpoints[method]
        validate_abr_checkpoint(checkpoint, method)
        for seed in SEEDS:
            if (method, seed) in completed:
                continue
            command = abr_lora.build_command(command_args, method, checkpoint, seed)
            print(f"[ABR {method} seed={seed}] {shlex.join(command)}", flush=True)
            if args.dry_run:
                continue
            started_at = time.time() - 1.0
            subprocess.run(command, cwd=ABR_ROOT, check=True)
            metrics_path = abr_lora.newest_metrics(started_at)
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            rows.append({
                "method": method, "label": abr_lora.METHODS[method]["label"],
                "checkpoint_training_data_seed": 3,
                "evaluation_seed": seed, "rank_budget": 1536,
                "physical_rank": 32, "checkpoint_dir": str(checkpoint.resolve()),
                "attention_score_mode": "fp16_prescaled_qk_with_fp32_retry",
                "metrics_path": str(metrics_path.resolve()), "status": "complete",
                **abr_lora.scalar_metrics(metrics),
            })
            abr_lora.write_rows(output, rows)
    if args.dry_run:
        return
    summaries = []
    for method in ("eva", "shapley"):
        group = [row for row in rows if row["method"] == method]
        item = {
            "method": method, "label": abr_lora.METHODS[method]["label"],
            "checkpoint_training_data_seed": 3, "num_seeds": 3,
            "evaluation_seeds": "1,2,3", "evaluation_rng_mode": "per-episode",
            "attention_score_mode": "fp16_prescaled_qk_with_fp32_retry",
        }
        for metric in abr_lora.SUMMARY_METRICS:
            values = [float(row[metric]) for row in group if row.get(metric) not in (None, "")]
            if len(values) == 3:
                item[f"{metric}_mean"] = statistics.mean(values)
                item[f"{metric}_std"] = statistics.stdev(values)
        summaries.append(item)
    abr_lora.write_rows(args.output_dir / "abr_data3/three_seed_summary.csv", summaries)


def vp_nbs_args(args):
    return argparse.Namespace(
        device=args.device, physical_rank=32, rank_budget=512,
        latency_warmup_steps=5, projector_checkpoint=args.tuned_projector,
        cache_dir=args.patch_cache, motion_threshold_deg=6.0,
        projector_cache_max_entries=512,
    )


def vp_module_command(args, compact: Path, kind: str, seed: int,
                      result_dir: Path, source: Path | None = None) -> list[str]:
    common_args = vp_nbs_args(args)
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
    return force_vp_prescaled(command)


MODULE_CASES = (
    ("pure_nbs_compact", "Pure NBS compact", "baseline"),
    ("patch_gated_k1_t6_skip0_cache", "NBS + Patch gated-k1/T6/skip0/cache", "patch"),
    ("token_k8", "NBS + Token K=8", "token"),
    ("spec_g6_t0p4", "NBS + Spec G=6/T=0.4", "speculative"),
    ("full_stack", "NBS + Patch + Token K=8 + Spec G=6/T=0.4", "full_stack"),
)


def compact_vp_nbs(args, source: Path) -> Path:
    compact = args.output_dir / "vp_data2_nbs_compact/compact_checkpoint"
    required = (compact / "compact_adapter.pt", compact / "modules_except_plm.bin")
    if all(path.is_file() for path in required):
        return compact
    result_dir = args.output_dir / "vp_data2_modules/seed_1/pure_nbs_compact"
    command = vp_module_command(args, compact, "baseline", 1, result_dir, source)
    print(f"[VP NBS compact] {shlex.join(command)}", flush=True)
    if not args.dry_run:
        run_case(command, result_dir, args.resume)
        if not all(path.is_file() for path in required):
            raise RuntimeError("compact VP NBS checkpoint was not produced")
    return compact


def summarize_module_rows(rows: list[dict]) -> list[dict]:
    summaries = []
    for case, label, _ in MODULE_CASES:
        group = [row for row in rows if row.get("case") == case and row.get("status") == "complete"]
        if {int(row["evaluation_seed"]) for row in group} != set(SEEDS):
            raise RuntimeError(f"incomplete module result: {case}")
        item = {
            "case": case, "label": label, "seed_count": 3,
            "evaluation_seeds": "1,2,3", "evaluation_rng_mode": "continuous",
            "attention_score_mode": "fp16_prescaled_qk_with_fp32_retry",
        }
        for metric in ("mae", "rmse", "latency_mean_ms", "mean_initial_token_count",
                       "mean_selected_token_count", "mean_target_forward_count"):
            values = [float(row[metric]) for row in group if row.get(metric) not in (None, "")]
            if len(values) == 3:
                item[f"{metric}_mean"] = statistics.mean(values)
                item[f"{metric}_std"] = statistics.stdev(values)
        summaries.append(item)
    return summaries


def evaluate_vp_modules(args, compact: Path) -> list[dict]:
    output = args.output_dir / "vp_data2_modules/per_seed_results.csv"
    rows = load_csv(output) if args.resume else []
    # Compaction already evaluated seed-1 baseline; import its metrics once.
    baseline_dir = args.output_dir / "vp_data2_modules/seed_1/pure_nbs_compact"
    if not any(row.get("case") == "pure_nbs_compact" and row.get("evaluation_seed") == "1" for row in rows):
        metrics_path = baseline_dir / "metrics.json"
        if metrics_path.is_file():
            row = {"case": "pure_nbs_compact", "label": "Pure NBS compact",
                   "evaluation_seed": 1, "status": "complete"}
            row.update(json.loads(metrics_path.read_text(encoding="utf-8")))
            row.update(read_safety(baseline_dir))
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
            result_dir = args.output_dir / f"vp_data2_modules/seed_{seed}/{case}"
            command = vp_module_command(args, compact, kind, seed, result_dir)
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
    summaries = summarize_module_rows(rows)
    write_rows(args.output_dir / "vp_data2_modules/three_seed_summary.csv", summaries)
    non_nbs_path = args.output_dir / "vp_data2_lora/non_nbs_three_seed_summary.csv"
    if non_nbs_path.is_file():
        baseline = dict(summaries[0])
        baseline["method"] = "nbs"
        baseline["label"] = "NBS-LoRA compact"
        baseline["checkpoint_training_data_seed"] = 2
        combined = [baseline, *load_csv(non_nbs_path)]
        write_rows(
            args.output_dir / "vp_data2_lora/five_method_three_seed_summary.csv",
            combined,
        )
    return summaries


def run_stage(args, state, state_path, name, action):
    if state["stages"].get(name, {}).get("status") == "complete":
        print(f"[{name}] already complete; skipping", flush=True)
        return
    print(f"\n===== {name} =====", flush=True)
    started = time.time()
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


def checkpoint_from_state(state, key):
    value = state["checkpoints"].get(key)
    if not value:
        raise RuntimeError(f"checkpoint unavailable: {key}")
    return Path(value)


def main(argv=None):
    args = parse_args(argv)
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    state_path, state = load_state(args)

    run_stage(args, state, state_path, "train_vp_adalora_data1",
              lambda: train_vp(args, state, state_path, "adalora"))
    run_stage(args, state, state_path, "evaluate_vp_adalora_data1",
              lambda: evaluate_vp_method(args, "adalora", checkpoint_from_state(state, "vp_adalora_data1"), 1,
                                         args.output_dir / "vp_new_data1"))
    run_stage(args, state, state_path, "train_vp_shapley_data1",
              lambda: train_vp(args, state, state_path, "shapley"))
    run_stage(args, state, state_path, "evaluate_vp_shapley_data1",
              lambda: evaluate_vp_method(args, "shapley", checkpoint_from_state(state, "vp_shapley_data1"), 1,
                                         args.output_dir / "vp_new_data1"))

    abr_checkpoints = {}
    def validate_abr():
        for method, fragment in ABR_DATA3_FRAGMENTS.items():
            path = find_checkpoint(ABR_MODEL_ROOT, fragment, ("early_stop_-1_best_model",))
            validate_abr_checkpoint(path, method)
            abr_checkpoints[method] = path
            state["checkpoints"][f"abr_{method}_data3"] = str(path.resolve())
        atomic_json(state_path, state)
    run_stage(args, state, state_path, "validate_abr_data3", validate_abr)
    run_stage(args, state, state_path, "evaluate_abr_data3_eva_shapley",
              lambda: evaluate_abr_data3(args, {
                  method: checkpoint_from_state(state, f"abr_{method}_data3")
                  for method in ("eva", "shapley")
              }))

    vp_data2 = {}
    def validate_vp_data2():
        for method, fragment in VP_DATA2_FRAGMENTS.items():
            path = find_checkpoint(VP_MODEL_ROOT, fragment, VP_TERMINALS[method])
            if method != "nbs":
                validate_exact_vp_budget(method, path)
            vp_data2[method] = path
            state["checkpoints"][f"vp_{method}_data2"] = str(path.resolve())
        atomic_json(state_path, state)
    run_stage(args, state, state_path, "validate_vp_data2", validate_vp_data2)

    compact_holder = {}
    def compact_stage():
        compact = compact_vp_nbs(args, checkpoint_from_state(state, "vp_nbs_data2"))
        compact_holder["path"] = compact
        if not args.dry_run:
            state["checkpoints"]["vp_nbs_data2_compact"] = str(compact.resolve())
            atomic_json(state_path, state)
    run_stage(args, state, state_path, "compact_vp_nbs_data2", compact_stage)

    def evaluate_data2_lora():
        summaries = []
        for method in ("uniform", "adalora", "shapley", "eva"):
            summaries.append(evaluate_vp_method(
                args, method, checkpoint_from_state(state, f"vp_{method}_data2"),
                2, args.output_dir / "vp_data2_lora",
            ))
        if not args.dry_run:
            write_rows(args.output_dir / "vp_data2_lora/non_nbs_three_seed_summary.csv", summaries)
    run_stage(args, state, state_path, "evaluate_vp_data2_lora_methods", evaluate_data2_lora)
    run_stage(args, state, state_path, "evaluate_vp_data2_nbs_modules",
              lambda: evaluate_vp_modules(args, checkpoint_from_state(state, "vp_nbs_data2_compact")))

    if not args.dry_run:
        failed = [name for name, item in state["stages"].items() if item.get("status") == "failed"]
        print(f"\nPipeline state: {state_path}", flush=True)
        print(f"Failed stages: {failed or 'none'}", flush=True)
        print(f"OUTPUT={args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
