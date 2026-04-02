from openenv.core.client_types import StepResult
from openenv.core.env_client import EnvClient

from models import PizzaShopAction, PizzaShopObservation, PizzaShopState


class PizzaShopEnv(EnvClient[PizzaShopAction, PizzaShopObservation, PizzaShopState]):
    def _step_payload(self, action: PizzaShopAction) -> dict:
        return {
            "action_type": action.action_type,
            "order_id": action.order_id,
            "oven_slot": action.oven_slot,
            "driver_id": action.driver_id,
            "refund_reason": action.refund_reason,
        }

    def _parse_result(self, payload: dict) -> StepResult:
        obs_data = payload.get("observation", {})
        observation = PizzaShopObservation(
            done=payload.get("done", False),
            reward=payload.get("reward"),
            task_level=obs_data.get("task_level", 1),
            task_id=obs_data.get("task_id", ""),
            objective=obs_data.get("objective", ""),
            current_tick=obs_data.get("current_tick", 0),
            pending_orders=obs_data.get("pending_orders", []),
            oven_status=obs_data.get("oven_status", []),
            driver_status=obs_data.get("driver_status", []),
            completed_checklist=obs_data.get("completed_checklist", []),
            required_actions_remaining=obs_data.get("required_actions_remaining", 0),
            available_actions=obs_data.get("available_actions", []),
            progress=float(obs_data.get("progress", 0.0)),
            message=obs_data.get("message", ""),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: dict) -> PizzaShopState:
        return PizzaShopState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            task_level=payload.get("task_level", 1),
            task_id=payload.get("task_id", ""),
            objective=payload.get("objective", ""),
            max_steps=payload.get("max_steps", 16),
            ticks_total=payload.get("ticks_total", 8),
            current_tick=payload.get("current_tick", 0),
            orders=payload.get("orders", []),
            oven_slots=payload.get("oven_slots", []),
            drivers=payload.get("drivers", []),
            completed_checklist=payload.get("completed_checklist", {}),
            invalid_actions=payload.get("invalid_actions", 0),
            repeated_actions=payload.get("repeated_actions", 0),
            noops=payload.get("noops", 0),
            destructive_actions=payload.get("destructive_actions", 0),
            revenue_total=payload.get("revenue_total", 0.0),
            refunds_total=payload.get("refunds_total", 0.0),
            late_orders=payload.get("late_orders", 0),
            complaints=payload.get("complaints", 0),
            closed_orders=payload.get("closed_orders", 0),
            delivered_orders=payload.get("delivered_orders", 0),
            last_action=payload.get("last_action", ""),
            done_reason=payload.get("done_reason"),
        )
