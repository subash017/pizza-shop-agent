from server.graders import grade_current_task


def test_task_1_passes_with_full_workflow() -> None:
    state = {
        "task_level": 1,
        "completed_checklist": {
            "all_orders_accepted": True,
            "all_orders_prepped": True,
            "all_orders_baked": True,
            "all_orders_dispatched": True,
            "all_orders_closed": True,
            "driver_capacity_respected": True,
            "refunds_used_judiciously": True,
        },
        "late_orders": 0,
        "refunds_total": 0.0,
        "revenue_total": 50.0,
        "delivered_orders": 3,
        "invalid_actions": 0,
        "noops": 0,
        "destructive_actions": 0,
    }
    result = grade_current_task(state)
    assert result.passed is True
    assert 0.95 <= result.score < 1.0


def test_task_2_partial_score_with_late_orders() -> None:
    state = {
        "task_level": 2,
        "completed_checklist": {
            "all_orders_accepted": True,
            "all_orders_prepped": True,
            "all_orders_baked": True,
            "all_orders_dispatched": False,
            "all_orders_closed": False,
            "driver_capacity_respected": True,
            "refunds_used_judiciously": True,
        },
        "late_orders": 3,
        "refunds_total": 10.0,
        "revenue_total": 70.0,
        "delivered_orders": 4,
        "invalid_actions": 0,
        "noops": 0,
        "destructive_actions": 0,
    }
    result = grade_current_task(state)
    assert result.passed is False
    assert 0.0 < result.score < 1.0


def test_task_3_penalizes_destructive_actions() -> None:
    state = {
        "task_level": 3,
        "completed_checklist": {
            "all_orders_accepted": True,
            "all_orders_prepped": True,
            "all_orders_baked": True,
            "all_orders_dispatched": True,
            "all_orders_closed": True,
            "driver_capacity_respected": True,
            "refunds_used_judiciously": True,
        },
        "late_orders": 0,
        "refunds_total": 0.0,
        "revenue_total": 95.0,
        "delivered_orders": 5,
        "complaints": 0,
        "invalid_actions": 0,
        "noops": 0,
        "destructive_actions": 2,
    }
    result = grade_current_task(state)
    assert result.passed is False
    assert result.score < 0.95
