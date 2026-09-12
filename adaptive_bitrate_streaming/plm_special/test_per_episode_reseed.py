"""Opt-in ABR evaluation with an independent RNG stream per episode.

The historical evaluator in :mod:`plm_special.test` is intentionally left
unchanged.  This wrapper resets all RNGs after each completed episode so that
different inference modules cannot shift the random stream used by later
traces.
"""

from contextlib import contextmanager
import json
import os


@contextmanager
def episode_reseed_context(model, base_seed, seed_fn):
    """Reseed after ``clear_dq`` while restoring the model method afterward."""
    original_clear_dq = model.clear_dq
    had_instance_override = "clear_dq" in getattr(model, "__dict__", {})
    original_instance_value = getattr(model, "__dict__", {}).get("clear_dq")
    state = {"completed_episodes": 0}

    def clear_dq_and_reseed(*args, **kwargs):
        result = original_clear_dq(*args, **kwargs)
        state["completed_episodes"] += 1
        seed_fn(int(base_seed) + state["completed_episodes"])
        return result

    model.clear_dq = clear_dq_and_reseed
    try:
        yield state
    finally:
        if had_instance_override:
            model.clear_dq = original_instance_value
        else:
            delattr(model, "clear_dq")


def test_on_env(
    args, model, results_dir, env_settings, target_return, max_ep_num=100,
    process_reward_fn=None, seed=0,
):
    """Run the unchanged evaluator with per-episode RNG isolation enabled."""
    from plm_special import test as continuous_test
    from plm_special.utils.utils import set_random_seed

    base_seed = int(getattr(args, "seed", seed))
    with episode_reseed_context(model, base_seed, set_random_seed) as state:
        results = continuous_test.test_on_env(
            args, model, results_dir, env_settings, target_return,
            max_ep_num=max_ep_num, process_reward_fn=process_reward_fn,
            seed=seed,
        )

    results["evaluation_rng_mode"] = "per-episode"
    results["episode_reseed_count"] = state["completed_episodes"]
    metrics_path = os.path.join(results_dir, "selector_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as stream:
        json.dump(results, stream, indent=2, sort_keys=True)
    return results
