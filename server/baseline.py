import json
import os
from typing import Any, Dict, List

from openai import OpenAI

from models import PizzaShopAction
from server.environment import PizzaShopEnvironment
from server.graders import grade_current_task

DEFAULT_MODEL = "gpt-4o-mini"
SYSTEM_PROMPT = (
    "You are a careful pizza shop operations agent. "
    "Return exactly one JSON object with fields: action_type, order_id, oven_slot, driver_id, refund_reason. "
    "Use valid enum actions and avoid extra keys."
)

VALID_ACTION_TYPES = {
    "accept_order",
    "start_prep",
    "load_oven",
    "dispatch_driver",
    "issue_refund",
    "close_order",
    "noop",
}
VALID_REFUND_REASONS = {"late", "quality", "wrong_order"}


def _norm(value: Any) -> str:
    return str(value or "").strip().lower()


def _normalize_action_type(value: Any) -> str:
    raw = _norm(value).replace("-", "_").replace(" ", "_")
    aliases = {
        "accept": "accept_order",
        "prep": "start_prep",
        "bake": "load_oven",
        "dispatch": "dispatch_driver",
        "refund": "issue_refund",
        "close": "close_order",
    }
    normalized = aliases.get(raw, raw)
    return normalized if normalized in VALID_ACTION_TYPES else "noop"


def _normalize_refund_reason(value: Any) -> str | None:
    raw = _norm(value).replace("-", "_").replace(" ", "_")
    return raw if raw in VALID_REFUND_REASONS else None


def _client_config() -> tuple[str, str | None, dict[str, str]]:
    openai_key = os.getenv("OPENAI_API_KEY", "").strip()
    openrouter_key = os.getenv("OPENROUTER_API_KEY", "").strip()

    api_key = openai_key or openrouter_key
    if not api_key:
        raise RuntimeError("Set OPENAI_API_KEY or OPENROUTER_API_KEY.")

    base_url = os.getenv("OPENAI_BASE_URL", "").strip() or None
    headers: dict[str, str] = {}

    if api_key.startswith("sk-or-") and base_url is None:
        base_url = "https://openrouter.ai/api/v1"

    if base_url and "openrouter.ai" in base_url:
        referer = os.getenv("OPENROUTER_HTTP_REFERER", "http://localhost").strip()
        title = os.getenv("OPENROUTER_X_TITLE", "pizza-shop-openenv").strip()
        headers = {
            "HTTP-Referer": referer,
            "X-OpenRouter-Title": title,
            "X-Title": title,
        }

    return api_key, base_url, headers


def _action_from_model(client: OpenAI, model: str, observation: Dict[str, Any], seed: int) -> PizzaShopAction:
    user_prompt = {
        "objective": observation.get("objective"),
        "task_level": observation.get("task_level"),
        "current_tick": observation.get("current_tick"),
        "pending_orders": observation.get("pending_orders"),
        "oven_status": observation.get("oven_status"),
        "driver_status": observation.get("driver_status"),
        "completed_checklist": observation.get("completed_checklist"),
        "required_actions_remaining": observation.get("required_actions_remaining"),
        "available_actions": observation.get("available_actions"),
        "notes": "Pick one valid next action. Prefer accept -> prep -> bake -> dispatch -> close flow.",
    }

    response = client.chat.completions.create(
        model=model,
        temperature=0,
        seed=seed,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(user_prompt)},
        ],
    )
    content = response.choices[0].message.content or "{}"

    try:
        raw = json.loads(content)
    except json.JSONDecodeError:
        raw = {"action_type": "noop"}

    action_type = _normalize_action_type(raw.get("action_type", "noop"))
    order_id_raw = raw.get("order_id")
    order_id = str(order_id_raw).strip() if order_id_raw is not None else None

    oven_slot_raw = raw.get("oven_slot")
    try:
        oven_slot = int(oven_slot_raw) if oven_slot_raw is not None else None
    except (TypeError, ValueError):
        oven_slot = None

    driver_id_raw = raw.get("driver_id")
    driver_id = str(driver_id_raw).strip() if driver_id_raw is not None else None

    refund_reason = _normalize_refund_reason(raw.get("refund_reason"))

    if action_type in {"accept_order", "start_prep", "load_oven", "dispatch_driver", "issue_refund", "close_order"} and not order_id:
        action_type = "noop"

    if action_type == "load_oven" and oven_slot is None:
        action_type = "noop"
        order_id = None

    if action_type == "dispatch_driver" and not driver_id:
        action_type = "noop"
        order_id = None

    if action_type == "issue_refund" and refund_reason is None:
        action_type = "noop"
        order_id = None

    return PizzaShopAction(
        action_type=action_type,
        order_id=order_id,
        oven_slot=oven_slot,
        driver_id=driver_id,
        refund_reason=refund_reason,
    )


def run_baseline(model: str = DEFAULT_MODEL, max_steps_override: int | None = None) -> Dict[str, Any]:
    api_key, base_url, headers = _client_config()
    client = OpenAI(api_key=api_key, base_url=base_url, default_headers=headers)
    per_task: List[Dict[str, Any]] = []

    for task_level in [1, 2, 3]:
        env = PizzaShopEnvironment()
        obs = env.reset(task_level=task_level)
        done = False
        steps = 0
        budget = max_steps_override if max_steps_override is not None else int(env.state.max_steps)
        reward_sum = 0.0

        while not done and steps < budget:
            action = _action_from_model(client, model, obs.model_dump(), seed=19 + task_level + steps)
            obs = env.step(action)
            done = obs.done
            steps += 1
            reward_sum += float(obs.reward or 0.0)

        grade = grade_current_task(env.state.model_dump())
        per_task.append(
            {
                "task_level": task_level,
                "task_id": env.state.task_id,
                "score": grade.score,
                "passed": grade.passed,
                "reason": grade.reason,
                "steps": steps,
                "reward_sum": reward_sum,
                "revenue_total": env.state.revenue_total,
                "refunds_total": env.state.refunds_total,
                "late_orders": env.state.late_orders,
                "complaints": env.state.complaints,
            }
        )

    aggregate = sum(task["score"] for task in per_task) / len(per_task)
    return {
        "model": model,
        "aggregate_score": aggregate,
        "tasks": per_task,
    }
