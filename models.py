from typing import Dict, List, Literal, Optional

from openenv.core.env_server import Action, Observation, State
from pydantic import Field


class PizzaShopAction(Action):
    action_type: Literal[
        "accept_order",
        "start_prep",
        "load_oven",
        "dispatch_driver",
        "issue_refund",
        "close_order",
        "noop",
    ]
    order_id: Optional[str] = None
    oven_slot: Optional[int] = None
    driver_id: Optional[str] = None
    refund_reason: Optional[Literal["late", "quality", "wrong_order"]] = None


class PizzaShopObservation(Observation):
    task_level: int
    task_id: str
    objective: str
    current_tick: int
    pending_orders: List[str]
    oven_status: List[str]
    driver_status: List[str]
    completed_checklist: List[str]
    required_actions_remaining: int
    available_actions: List[str]
    progress: float
    message: str


class PizzaShopState(State):
    task_level: int = 1
    task_id: str = "easy_lunch_shift"
    objective: str = ""
    max_steps: int = 16
    ticks_total: int = 8
    current_tick: int = 0
    orders: List[Dict[str, object]] = Field(default_factory=list)
    oven_slots: List[Dict[str, object]] = Field(default_factory=list)
    drivers: List[Dict[str, object]] = Field(default_factory=list)
    completed_checklist: Dict[str, bool] = Field(default_factory=dict)
    invalid_actions: int = 0
    repeated_actions: int = 0
    noops: int = 0
    destructive_actions: int = 0
    revenue_total: float = 0.0
    refunds_total: float = 0.0
    late_orders: int = 0
    complaints: int = 0
    closed_orders: int = 0
    delivered_orders: int = 0
    last_action: str = ""
    done_reason: Optional[str] = None
