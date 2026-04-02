import copy
import uuid
from typing import Dict, List

from openenv.core.env_server import Environment

from models import PizzaShopAction, PizzaShopObservation, PizzaShopState
from server.graders import grade_current_task
from server.reward import RewardInputs, compute_reward


TASK_SPECS: Dict[int, Dict[str, object]] = {
    1: {
        "task_id": "easy_lunch_shift",
        "objective": "Run a calm lunch shift and fulfill all pizza orders on time.",
        "max_steps": 18,
        "ticks_total": 8,
        "ovens": 2,
        "drivers": 1,
        "orders": [
            {"id": "P1", "price": 16.0, "due_tick": 4},
            {"id": "P2", "price": 18.0, "due_tick": 5},
            {"id": "P3", "price": 14.0, "due_tick": 6},
        ],
    },
    2: {
        "task_id": "medium_dinner_rush",
        "objective": "Handle dinner rush with tight oven and driver scheduling.",
        "max_steps": 24,
        "ticks_total": 10,
        "ovens": 2,
        "drivers": 2,
        "orders": [
            {"id": "P2A", "price": 19.0, "due_tick": 4},
            {"id": "P2B", "price": 22.0, "due_tick": 5},
            {"id": "P2C", "price": 17.0, "due_tick": 5},
            {"id": "P2D", "price": 21.0, "due_tick": 7},
        ],
    },
    3: {
        "task_id": "hard_storm_surge",
        "objective": "Survive surge demand and weather-delivery delays while protecting ratings and margin.",
        "max_steps": 28,
        "ticks_total": 12,
        "ovens": 2,
        "drivers": 2,
        "orders": [
            {"id": "P3A", "price": 22.0, "due_tick": 4},
            {"id": "P3B", "price": 20.0, "due_tick": 5},
            {"id": "P3C", "price": 24.0, "due_tick": 6},
            {"id": "P3D", "price": 18.0, "due_tick": 7},
            {"id": "P3E", "price": 23.0, "due_tick": 8},
        ],
    },
}


class PizzaShopEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self):
        self._state = PizzaShopState()
        self._last_action = ""

    def reset(self, seed=None, episode_id=None, **kwargs) -> PizzaShopObservation:
        task_level = int(kwargs.get("task_level", 1))
        task_level = max(1, min(3, task_level))
        spec = TASK_SPECS[task_level]

        orders = []
        for order in spec["orders"]:
            item = copy.deepcopy(order)
            item.update(
                {
                    "status": "new",
                    "accepted": False,
                    "prepped": False,
                    "baked": False,
                    "dispatched": False,
                    "closed": False,
                    "late": False,
                    "refunded": False,
                    "complaint": False,
                    "dispatched_tick": None,
                }
            )
            orders.append(item)

        ovens = [{"slot": i, "order_id": None} for i in range(int(spec["ovens"]))]
        drivers = [{"id": f"D{i + 1}", "order_id": None} for i in range(int(spec["drivers"]))]

        self._state = PizzaShopState(
            episode_id=episode_id or str(uuid.uuid4()),
            step_count=0,
            task_level=task_level,
            task_id=str(spec["task_id"]),
            objective=str(spec["objective"]),
            max_steps=int(spec["max_steps"]),
            ticks_total=int(spec["ticks_total"]),
            current_tick=0,
            orders=orders,
            oven_slots=ovens,
            drivers=drivers,
            completed_checklist={
                "all_orders_accepted": False,
                "all_orders_prepped": False,
                "all_orders_baked": False,
                "all_orders_dispatched": False,
                "all_orders_closed": False,
                "driver_capacity_respected": True,
                "refunds_used_judiciously": True,
            },
            invalid_actions=0,
            repeated_actions=0,
            noops=0,
            destructive_actions=0,
            revenue_total=0.0,
            refunds_total=0.0,
            late_orders=0,
            complaints=0,
            closed_orders=0,
            delivered_orders=0,
            last_action="",
            done_reason=None,
        )
        self._last_action = ""
        return self._build_observation(0.0, "Episode reset.", 0.0, False)

    def step(self, action: PizzaShopAction, timeout_s=None, **kwargs) -> PizzaShopObservation:
        self._state.step_count += 1
        invalid_action = 0.0
        noop = 0.0
        destructive_action = 0.0
        message = "Action accepted."

        action_fingerprint = f"{action.action_type}|{action.order_id}|{action.oven_slot}|{action.driver_id}|{action.refund_reason}"
        if action_fingerprint == self._last_action:
            self._state.repeated_actions += 1
        self._last_action = action_fingerprint
        self._state.last_action = action_fingerprint

        checklist_before = self._completed_ratio()
        late_before = self._state.late_orders
        refund_before = self._state.refunds_total

        if action.action_type == "noop":
            self._state.noops += 1
            noop = 1.0
            message = "No operation was taken."
        else:
            order = self._find_order(action.order_id)
            if action.action_type in {"accept_order", "start_prep", "load_oven", "dispatch_driver", "issue_refund", "close_order"} and order is None:
                self._state.invalid_actions += 1
                invalid_action = 1.0
                message = "Invalid action: unknown or missing order_id."
            else:
                if action.action_type == "accept_order":
                    if order["accepted"]:
                        self._state.invalid_actions += 1
                        invalid_action = 1.0
                        message = "Order already accepted."
                    else:
                        order["accepted"] = True
                        order["status"] = "accepted"
                        message = f"Order {order['id']} accepted."

                elif action.action_type == "start_prep":
                    if not order["accepted"]:
                        self._state.invalid_actions += 1
                        invalid_action = 1.0
                        message = "Cannot prep before accept."
                    elif order["prepped"]:
                        self._state.invalid_actions += 1
                        invalid_action = 1.0
                        message = "Order already prepped."
                    else:
                        order["prepped"] = True
                        order["status"] = "prepped"
                        message = f"Prep started for {order['id']}."

                elif action.action_type == "load_oven":
                    if action.oven_slot is None:
                        self._state.invalid_actions += 1
                        invalid_action = 1.0
                        message = "load_oven requires oven_slot."
                    elif not order["prepped"]:
                        self._state.invalid_actions += 1
                        invalid_action = 1.0
                        message = "Cannot bake before prep."
                    elif order["baked"]:
                        self._state.invalid_actions += 1
                        invalid_action = 1.0
                        message = "Order already baked."
                    else:
                        oven = self._find_oven(action.oven_slot)
                        if oven is None:
                            self._state.invalid_actions += 1
                            invalid_action = 1.0
                            message = "Invalid oven slot."
                        elif oven["order_id"] is not None:
                            self._state.destructive_actions += 1
                            destructive_action = 1.0
                            message = "Oven slot already in use."
                        else:
                            oven["order_id"] = order["id"]
                            # Simple one-step bake model: order leaves oven baked in same tick.
                            order["baked"] = True
                            order["status"] = "baked"
                            oven["order_id"] = None
                            message = f"Order {order['id']} baked in oven slot {action.oven_slot}."

                elif action.action_type == "dispatch_driver":
                    if action.driver_id is None:
                        self._state.invalid_actions += 1
                        invalid_action = 1.0
                        message = "dispatch_driver requires driver_id."
                    elif not order["baked"]:
                        self._state.invalid_actions += 1
                        invalid_action = 1.0
                        message = "Cannot dispatch before bake."
                    elif order["dispatched"]:
                        self._state.invalid_actions += 1
                        invalid_action = 1.0
                        message = "Order already dispatched."
                    else:
                        driver = self._find_driver(action.driver_id)
                        if driver is None:
                            self._state.invalid_actions += 1
                            invalid_action = 1.0
                            message = "Invalid driver id."
                        elif driver["order_id"] is not None:
                            self._state.destructive_actions += 1
                            destructive_action = 1.0
                            self._state.completed_checklist["driver_capacity_respected"] = False
                            message = f"Driver {action.driver_id} already busy."
                        else:
                            driver["order_id"] = order["id"]
                            order["dispatched"] = True
                            order["dispatched_tick"] = self._state.current_tick
                            order["status"] = "dispatched"
                            self._state.delivered_orders += 1
                            driver["order_id"] = None
                            message = f"Order {order['id']} dispatched with driver {action.driver_id}."

                elif action.action_type == "issue_refund":
                    if order["closed"]:
                        self._state.invalid_actions += 1
                        invalid_action = 1.0
                        message = "Cannot refund closed order."
                    elif order["refunded"]:
                        self._state.invalid_actions += 1
                        invalid_action = 1.0
                        message = "Order already refunded."
                    elif action.refund_reason is None:
                        self._state.invalid_actions += 1
                        invalid_action = 1.0
                        message = "issue_refund requires refund_reason."
                    else:
                        order["refunded"] = True
                        order["status"] = "refunded"
                        refund_amount = float(order["price"]) * 0.5
                        self._state.refunds_total += refund_amount
                        if action.refund_reason == "wrong_order":
                            order["complaint"] = True
                            self._state.complaints += 1
                        message = f"Refund issued for {order['id']}."

                elif action.action_type == "close_order":
                    if order["closed"]:
                        self._state.invalid_actions += 1
                        invalid_action = 1.0
                        message = "Order already closed."
                    elif not order["dispatched"] and not order["refunded"]:
                        self._state.destructive_actions += 1
                        destructive_action = 1.0
                        message = "Cannot close order before dispatch or refund."
                    else:
                        order["closed"] = True
                        order["status"] = "closed"
                        self._state.closed_orders += 1
                        if not order["refunded"]:
                            self._state.revenue_total += float(order["price"])
                        message = f"Order {order['id']} closed."

                else:
                    self._state.invalid_actions += 1
                    invalid_action = 1.0
                    message = "Unknown action type."

        self._advance_tick()
        self._update_order_sla()
        self._update_checklist()

        grade = grade_current_task(self._state.model_dump())
        checklist_after = self._completed_ratio()
        checklist_progress_delta = max(-1.0, min(1.0, checklist_after - checklist_before))

        done = grade.passed or self._state.step_count >= self._state.max_steps or self._state.current_tick >= self._state.ticks_total
        if done and grade.passed:
            self._state.done_reason = "task_completed"
        elif done:
            self._state.done_reason = "horizon_or_max_steps_reached"

        efficiency = 0.0
        if grade.passed:
            efficiency = max(0.0, 1.0 - (self._state.step_count / float(self._state.max_steps)))

        reward = compute_reward(
            RewardInputs(
                checklist_progress_delta=checklist_progress_delta,
                task_score=grade.score,
                invalid_action=invalid_action,
                noop=noop,
                destructive_action=destructive_action,
                efficiency=efficiency,
                late_order_delta=max(0.0, float(self._state.late_orders - late_before)),
                refund_delta=max(0.0, float(self._state.refunds_total - refund_before)),
            )
        )

        if grade.passed:
            message = f"Success: {grade.reason}."
        elif done:
            message = "Episode ended: shift horizon reached."

        return self._build_observation(grade.score, message, reward.total, done)

    @property
    def state(self) -> PizzaShopState:
        return self._state

    def task_descriptions(self) -> List[Dict[str, object]]:
        tasks = []
        for level, spec in TASK_SPECS.items():
            tasks.append(
                {
                    "task_level": level,
                    "task_id": spec["task_id"],
                    "objective": spec["objective"],
                    "max_steps": spec["max_steps"],
                    "actions": {
                        "action_type": [
                            "accept_order",
                            "start_prep",
                            "load_oven",
                            "dispatch_driver",
                            "issue_refund",
                            "close_order",
                            "noop",
                        ],
                        "order_id": "string (required except noop)",
                        "oven_slot": "int (required for load_oven)",
                        "driver_id": "string (required for dispatch_driver)",
                        "refund_reason": ["late", "quality", "wrong_order"],
                    },
                }
            )
        return tasks

    def current_grade(self) -> Dict[str, object]:
        result = grade_current_task(self._state.model_dump())
        return {
            "task_level": self._state.task_level,
            "task_id": self._state.task_id,
            "score": result.score,
            "passed": result.passed,
            "reason": result.reason,
            "done_reason": self._state.done_reason,
        }

    def _find_order(self, order_id: str | None) -> Dict[str, object] | None:
        if not order_id:
            return None
        for order in self._state.orders:
            if order["id"] == order_id:
                return order
        return None

    def _find_driver(self, driver_id: str | None) -> Dict[str, object] | None:
        if not driver_id:
            return None
        for driver in self._state.drivers:
            if driver["id"] == driver_id:
                return driver
        return None

    def _find_oven(self, slot_id: int | None) -> Dict[str, object] | None:
        if slot_id is None:
            return None
        for oven in self._state.oven_slots:
            if int(oven["slot"]) == int(slot_id):
                return oven
        return None

    def _advance_tick(self) -> None:
        self._state.current_tick = min(self._state.ticks_total, self._state.current_tick + 1)

    def _update_order_sla(self) -> None:
        self._state.late_orders = 0
        self._state.complaints = sum(1 for order in self._state.orders if order["complaint"])
        for order in self._state.orders:
            due_tick = int(order["due_tick"])
            dispatched_tick = order["dispatched_tick"]
            if order["dispatched"] and dispatched_tick is not None and int(dispatched_tick) > due_tick:
                order["late"] = True
            elif not order["dispatched"] and self._state.current_tick > due_tick:
                order["late"] = True
            if order["late"]:
                self._state.late_orders += 1
                if not order["complaint"] and self._state.current_tick > due_tick + 1:
                    order["complaint"] = True
                    self._state.complaints += 1

    def _update_checklist(self) -> None:
        orders = self._state.orders
        self._state.completed_checklist["all_orders_accepted"] = all(order["accepted"] for order in orders)
        self._state.completed_checklist["all_orders_prepped"] = all(order["prepped"] for order in orders)
        self._state.completed_checklist["all_orders_baked"] = all(order["baked"] for order in orders)
        self._state.completed_checklist["all_orders_dispatched"] = all(order["dispatched"] or order["refunded"] for order in orders)
        self._state.completed_checklist["all_orders_closed"] = all(order["closed"] for order in orders)

        total_value = sum(float(order["price"]) for order in orders)
        refund_ratio = self._state.refunds_total / max(1.0, total_value)
        self._state.completed_checklist["refunds_used_judiciously"] = refund_ratio <= {1: 0.2, 2: 0.25, 3: 0.3}[self._state.task_level]

    def _completed_ratio(self) -> float:
        total = max(1, len(self._state.completed_checklist))
        done = sum(1 for value in self._state.completed_checklist.values() if value)
        return done / total

    def _pending_summary(self) -> List[str]:
        lines = []
        for order in self._state.orders:
            if not order["closed"]:
                lines.append(
                    f"{order['id']} status={order['status']} due={order['due_tick']} late={order['late']} refunded={order['refunded']}"
                )
        return lines

    def _oven_summary(self) -> List[str]:
        lines = []
        for oven in self._state.oven_slots:
            lines.append(f"slot={oven['slot']} order={oven['order_id']}")
        return lines

    def _driver_summary(self) -> List[str]:
        lines = []
        for driver in self._state.drivers:
            lines.append(f"driver={driver['id']} order={driver['order_id']}")
        return lines

    def _build_observation(self, task_score: float, message: str, reward: float, done: bool) -> PizzaShopObservation:
        remaining = sum(1 for value in self._state.completed_checklist.values() if not value)
        completed = sorted([key for key, value in self._state.completed_checklist.items() if value])
        return PizzaShopObservation(
            done=done,
            reward=reward,
            task_level=self._state.task_level,
            task_id=self._state.task_id,
            objective=self._state.objective,
            current_tick=self._state.current_tick,
            pending_orders=self._pending_summary(),
            oven_status=self._oven_summary(),
            driver_status=self._driver_summary(),
            completed_checklist=completed,
            required_actions_remaining=remaining,
            available_actions=[
                "accept_order",
                "start_prep",
                "load_oven",
                "dispatch_driver",
                "issue_refund",
                "close_order",
                "noop",
            ],
            progress=task_score,
            message=message,
        )
