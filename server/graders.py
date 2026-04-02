from dataclasses import dataclass
from typing import Any, Mapping


@dataclass
class GradeResult:
    passed: bool
    score: float
    reason: str


def _checklist(state: Mapping[str, Any]) -> Mapping[str, bool]:
    return state.get("completed_checklist", {})


def _score_from_checklist(state: Mapping[str, Any], required_items: list[str]) -> float:
    completed = _checklist(state)
    completed_count = sum(1 for item in required_items if bool(completed.get(item, False)))
    base = completed_count / max(1, len(required_items))

    invalid_actions = int(state.get("invalid_actions", 0))
    noops = int(state.get("noops", 0))
    destructive = int(state.get("destructive_actions", 0))

    penalties = min(0.5, (0.03 * invalid_actions) + (0.02 * noops) + (0.08 * destructive))
    return max(0.0, min(1.0, base - penalties))


def _ops_adjustments(state: Mapping[str, Any], target_late: int, max_refund_ratio: float) -> tuple[float, float, float]:
    late_orders = int(state.get("late_orders", 0))
    refunds_total = float(state.get("refunds_total", 0.0))
    revenue_total = float(state.get("revenue_total", 0.0))
    delivered = max(1, int(state.get("delivered_orders", 0)))

    late_score = 1.0 if late_orders <= target_late else max(0.0, 1.0 - ((late_orders - target_late) / float(delivered)))

    refund_ratio = refunds_total / max(1.0, revenue_total + refunds_total)
    refund_score = 1.0 if refund_ratio <= max_refund_ratio else max(0.0, 1.0 - ((refund_ratio - max_refund_ratio) / 0.25))

    profit = revenue_total - refunds_total
    profit_score = max(0.0, min(1.0, profit / max(1.0, revenue_total)))
    return late_score, refund_score, profit_score


def grade_task_1(state: Mapping[str, Any]) -> GradeResult:
    required_items = [
        "all_orders_accepted",
        "all_orders_prepped",
        "all_orders_baked",
        "all_orders_dispatched",
        "all_orders_closed",
    ]
    checklist_score = _score_from_checklist(state, required_items)
    late_score, refund_score, profit_score = _ops_adjustments(state, target_late=1, max_refund_ratio=0.1)
    score = max(0.0, min(1.0, 0.65 * checklist_score + 0.2 * late_score + 0.05 * refund_score + 0.1 * profit_score))

    passed = score >= 0.95
    return GradeResult(passed=passed, score=score, reason="task_1_complete" if passed else "task_1_partial")


def grade_task_2(state: Mapping[str, Any]) -> GradeResult:
    required_items = [
        "all_orders_accepted",
        "all_orders_prepped",
        "all_orders_baked",
        "all_orders_dispatched",
        "all_orders_closed",
        "driver_capacity_respected",
    ]
    checklist_score = _score_from_checklist(state, required_items)
    late_score, refund_score, profit_score = _ops_adjustments(state, target_late=2, max_refund_ratio=0.12)
    score = max(0.0, min(1.0, 0.62 * checklist_score + 0.2 * late_score + 0.08 * refund_score + 0.1 * profit_score))

    passed = score >= 0.95
    return GradeResult(passed=passed, score=score, reason="task_2_complete" if passed else "task_2_partial")


def grade_task_3(state: Mapping[str, Any]) -> GradeResult:
    required_items = [
        "all_orders_accepted",
        "all_orders_prepped",
        "all_orders_baked",
        "all_orders_dispatched",
        "all_orders_closed",
        "driver_capacity_respected",
        "refunds_used_judiciously",
    ]
    checklist_score = _score_from_checklist(state, required_items)
    late_score, refund_score, profit_score = _ops_adjustments(state, target_late=2, max_refund_ratio=0.15)
    complaints = int(state.get("complaints", 0))
    complaint_penalty = min(0.3, complaints * 0.05)

    score = max(
        0.0,
        min(
            1.0,
            0.55 * checklist_score + 0.2 * late_score + 0.1 * refund_score + 0.15 * profit_score - complaint_penalty,
        ),
    )

    passed = score >= 0.95
    return GradeResult(passed=passed, score=score, reason="task_3_complete" if passed else "task_3_partial")


def grade_current_task(state: Mapping[str, Any]) -> GradeResult:
    task_level = int(state.get("task_level", 1))
    if task_level == 1:
        return grade_task_1(state)
    if task_level == 2:
        return grade_task_2(state)
    return grade_task_3(state)
