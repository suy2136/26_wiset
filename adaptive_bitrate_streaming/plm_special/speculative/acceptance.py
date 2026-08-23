"""Pure acceptance and state-validation helpers for ABR speculation."""

from dataclasses import dataclass

import numpy as np

from baseline_special.utils.constants import BUFFER_NORM_FACTOR


@dataclass(frozen=True)
class AcceptancePlan:
    actions: tuple
    accepted_count: int
    mismatch_index: object

    @property
    def fully_accepted(self):
        return self.mismatch_index is None


def build_acceptance_plan(draft_actions, target_actions):
    """Accept the matching prefix and append one target correction on mismatch."""
    draft_actions = tuple(int(action) for action in draft_actions)
    target_actions = tuple(int(action) for action in target_actions)
    if not draft_actions or len(draft_actions) != len(target_actions):
        raise ValueError('draft_actions and target_actions need equal non-zero length')
    for index, (draft, target) in enumerate(zip(draft_actions, target_actions)):
        if draft != target:
            return AcceptancePlan(
                actions=draft_actions[:index] + (target,),
                accepted_count=index,
                mismatch_index=index,
            )
    return AcceptancePlan(
        actions=draft_actions,
        accepted_count=len(draft_actions),
        mismatch_index=None,
    )


def buffer_deviation_seconds(observed_state, predicted_state):
    """Return absolute buffer deviation using NetLLM's normalized state row."""
    if hasattr(observed_state, 'detach'):
        observed_state = observed_state.detach().cpu().numpy()
    if hasattr(predicted_state, 'detach'):
        predicted_state = predicted_state.detach().cpu().numpy()
    observed = np.asarray(observed_state)
    predicted = np.asarray(predicted_state)
    while observed.ndim > 2 and observed.shape[0] == 1:
        observed = observed[0]
    while predicted.ndim > 2 and predicted.shape[0] == 1:
        predicted = predicted[0]
    if observed.shape != (6, 6) or predicted.shape != (6, 6):
        raise ValueError('states must resolve to shape [6,6]')
    return abs(float(observed[1, -1] - predicted[1, -1])) * BUFFER_NORM_FACTOR
