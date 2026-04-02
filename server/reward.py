from dataclasses import dataclass


@dataclass
class RewardInputs:
    checklist_progress_delta: float
    task_score: float
    invalid_action: float
    noop: float
    destructive_action: float
    efficiency: float
    late_order_delta: float
    refund_delta: float


@dataclass
class RewardBreakdown:
    total: float
    progress_term: float
    score_term: float
    invalid_term: float
    noop_term: float
    destructive_term: float
    efficiency_term: float
    late_term: float
    refund_term: float


WEIGHTS = {
    "checklist_progress": 0.45,
    "task_score": 0.35,
    "invalid": 0.2,
    "noop": 0.08,
    "destructive": 0.25,
    "efficiency": 0.2,
    "late": 0.16,
    "refund": 0.06,
}


def compute_reward(inputs: RewardInputs) -> RewardBreakdown:
    progress_term = WEIGHTS["checklist_progress"] * inputs.checklist_progress_delta
    score_term = WEIGHTS["task_score"] * inputs.task_score
    invalid_term = WEIGHTS["invalid"] * inputs.invalid_action
    noop_term = WEIGHTS["noop"] * inputs.noop
    destructive_term = WEIGHTS["destructive"] * inputs.destructive_action
    efficiency_term = WEIGHTS["efficiency"] * inputs.efficiency
    late_term = WEIGHTS["late"] * inputs.late_order_delta
    refund_term = WEIGHTS["refund"] * inputs.refund_delta

    total = progress_term + score_term - invalid_term - noop_term - destructive_term + efficiency_term - late_term - refund_term
    return RewardBreakdown(
        total=total,
        progress_term=progress_term,
        score_term=score_term,
        invalid_term=invalid_term,
        noop_term=noop_term,
        destructive_term=destructive_term,
        efficiency_term=efficiency_term,
        late_term=late_term,
        refund_term=refund_term,
    )
