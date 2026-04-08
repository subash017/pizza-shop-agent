import json
import os
from typing import Any

from openai import OpenAI

from client import PizzaShopEnv
from models import PizzaShopAction

LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct").strip()
TASK_NAME = (os.getenv("PIZZA_SHOP_TASK") or "").strip()
BENCHMARK = (os.getenv("PIZZA_SHOP_BENCHMARK") or "pizza_shop_env").strip()
MAX_STEPS = int(os.getenv("MAX_STEPS") or "24")
SUCCESS_THRESHOLD = float(os.getenv("SUCCESS_THRESHOLD") or "0.95")

SYSTEM_PROMPT = (
    "You are a careful pizza shop operations agent. "
    "Return exactly one JSON object with fields: action_type, order_id, oven_slot, driver_id, refund_reason. "
    "Use only valid actions and avoid extra keys."
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
TASK_TO_LEVEL = {
    "easy_lunch_shift": 1,
    "medium_dinner_rush": 2,
    "hard_storm_surge": 3,
}


def _normalize_text(value: Any) -> str:
    return str(value or "").strip().lower()


def _normalize_action_type(value: Any) -> str:
    raw = _normalize_text(value).replace("-", "_").replace(" ", "_")
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
    raw = _normalize_text(value).replace("-", "_").replace(" ", "_")
    return raw if raw in VALID_REFUND_REASONS else None


def _make_action(client: OpenAI, observation: dict[str, Any], seed: int) -> PizzaShopAction:
    response = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=0,
        seed=seed,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "objective": observation.get("objective"),
                        "task_level": observation.get("task_level"),
                        "current_tick": observation.get("current_tick"),
                        "pending_orders": observation.get("pending_orders"),
                        "oven_status": observation.get("oven_status"),
                        "driver_status": observation.get("driver_status"),
                        "completed_checklist": observation.get("completed_checklist"),
                        "required_actions_remaining": observation.get("required_actions_remaining"),
                        "available_actions": observation.get("available_actions"),
                        "notes": "Prefer accept -> prep -> bake -> dispatch -> close progression.",
                    }
                ),
            },
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

    requires_order = {
        "accept_order",
        "start_prep",
        "load_oven",
        "dispatch_driver",
        "issue_refund",
        "close_order",
    }
    if action_type in requires_order and not order_id:
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


def _action_str(action: PizzaShopAction) -> str:
    return (
        f"{action.action_type}(order_id={action.order_id},"
        f"oven_slot={action.oven_slot},driver_id={action.driver_id},"
        f"refund_reason={action.refund_reason})"
    )


def _to_bool(value: bool) -> str:
    return "true" if value else "false"


def _fmt_reward(value: Any) -> str:
    return f"{float(value or 0.0):.2f}"


def _build_env() -> PizzaShopEnv:
    if LOCAL_IMAGE_NAME:
        try:
            return PizzaShopEnv.from_docker_image(LOCAL_IMAGE_NAME)
        except TypeError:
            return PizzaShopEnv.from_docker_image(image_name=LOCAL_IMAGE_NAME)
    base_url = os.getenv("BASE_URL") or os.getenv("OPENENV_BASE_URL") or os.getenv("SPACE_URL")
    if not base_url:
        base_url = "http://127.0.0.1:7860"
    return PizzaShopEnv(base_url=base_url)


def main() -> int:
    rewards: list[float] = []
    step_count = 0
    success = False
    score = 0.0
    env = None

    try:
        api_base_url = os.environ["API_BASE_URL"].strip()
        api_key = os.environ["API_KEY"].strip()

        client = OpenAI(base_url=api_base_url, api_key=api_key)
        # Force at least one proxy-visible LLM request before env interaction.
        client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": "ping"}],
            temperature=0,
            max_tokens=1,
        )

        env = _build_env()
        task_level = TASK_TO_LEVEL.get(TASK_NAME, 1)

        with env.sync() as running_env:
            observation = running_env.reset(task_level=task_level)
            done = bool(observation.done)
            score = max(0.0, min(1.0, float(observation.progress)))

            resolved_task = TASK_NAME or str(observation.task_id)
            print(f"[START] task={resolved_task} env={BENCHMARK} model={MODEL_NAME}")

            while not done and step_count < MAX_STEPS:
                action = _make_action(client, observation.model_dump(), seed=100 + step_count)
                result = running_env.step(action)
                step_count += 1

                reward = float(result.reward or 0.0)
                rewards.append(reward)
                done = bool(result.done)
                observation = result.observation
                score = max(0.0, min(1.0, float(observation.progress)))

                raw_error = getattr(result, "last_action_error", None)
                if raw_error is None:
                    raw_error = getattr(observation, "last_action_error", None)
                error_text = "null" if raw_error in (None, "") else str(raw_error)

                print(
                    f"[STEP] step={step_count} action={_action_str(action)} "
                    f"reward={_fmt_reward(reward)} done={_to_bool(done)} error={error_text}"
                )

            success = done and score >= SUCCESS_THRESHOLD

    except Exception:
        # Emit a valid START line even if reset/setup fails before episode loop.
        print(f"[START] task={TASK_NAME or 'unknown'} env={BENCHMARK} model={MODEL_NAME}")
        success = False
    finally:
        reward_csv = ",".join(f"{value:.2f}" for value in rewards)
        print(
            f"[END] success={_to_bool(success)} steps={step_count} "
            f"score={score:.2f} rewards={reward_csv}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
