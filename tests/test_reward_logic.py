from server.reward import RewardInputs, compute_reward


def test_reward_increases_for_progress() -> None:
    base = compute_reward(
        RewardInputs(
            checklist_progress_delta=0.0,
            task_score=0.0,
            invalid_action=0.0,
            noop=0.0,
            destructive_action=0.0,
            efficiency=0.0,
            late_order_delta=0.0,
            refund_delta=0.0,
        )
    )
    improved = compute_reward(
        RewardInputs(
            checklist_progress_delta=0.3,
            task_score=0.6,
            invalid_action=0.0,
            noop=0.0,
            destructive_action=0.0,
            efficiency=0.0,
            late_order_delta=0.0,
            refund_delta=0.0,
        )
    )
    assert improved.total > base.total


def test_invalid_action_penalty_reduces_reward() -> None:
    clean = compute_reward(
        RewardInputs(
            checklist_progress_delta=0.2,
            task_score=0.5,
            invalid_action=0.0,
            noop=0.0,
            destructive_action=0.0,
            efficiency=0.0,
            late_order_delta=0.0,
            refund_delta=0.0,
        )
    )
    penalized = compute_reward(
        RewardInputs(
            checklist_progress_delta=0.2,
            task_score=0.5,
            invalid_action=1.0,
            noop=0.0,
            destructive_action=0.0,
            efficiency=0.0,
            late_order_delta=0.0,
            refund_delta=0.0,
        )
    )
    assert penalized.total < clean.total


def test_late_and_refund_penalties_reduce_reward() -> None:
    clean = compute_reward(
        RewardInputs(
            checklist_progress_delta=0.2,
            task_score=0.6,
            invalid_action=0.0,
            noop=0.0,
            destructive_action=0.0,
            efficiency=0.0,
            late_order_delta=0.0,
            refund_delta=0.0,
        )
    )
    penalized = compute_reward(
        RewardInputs(
            checklist_progress_delta=0.2,
            task_score=0.6,
            invalid_action=0.0,
            noop=0.0,
            destructive_action=0.0,
            efficiency=0.0,
            late_order_delta=2.0,
            refund_delta=15.0,
        )
    )
    assert penalized.total < clean.total
