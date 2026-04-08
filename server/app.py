from typing import Any, Dict, Optional
import os

import gradio as gr
from fastapi import HTTPException
from openenv.core.env_server import create_fastapi_app
from pydantic import BaseModel
from uvicorn import run as uvicorn_run

from models import PizzaShopAction, PizzaShopObservation
from server.baseline import run_baseline
from server.environment import PizzaShopEnvironment
from server.graders import grade_current_task, grade_task_by_level

app = create_fastapi_app(
    PizzaShopEnvironment,
    PizzaShopAction,
    PizzaShopObservation,
)

utility_env = PizzaShopEnvironment()
utility_env.reset(task_level=1)

ui_env = PizzaShopEnvironment()
ui_env.reset(task_level=1)


class GraderRequest(BaseModel):
    state: Optional[Dict[str, Any]] = None
    task_level: Optional[int] = None
    task_id: Optional[str] = None


def _render_ui_state(obs: PizzaShopObservation) -> Dict[str, Any]:
    grade = grade_current_task(ui_env.state.model_dump())
    return {
        "observation": obs.model_dump(),
        "state": ui_env.state.model_dump(),
        "grade": {
            "score": grade.score,
            "passed": grade.passed,
            "reason": grade.reason,
        },
    }


def _build_demo() -> gr.Blocks:
    with gr.Blocks(title="Pizza Shop OpenEnv UI") as demo:
        gr.Markdown("# Pizza Shop OpenEnv\nInteractive manual policy runner for tasks 1-3.")

        with gr.Row():
            task_level = gr.Dropdown(choices=[1, 2, 3], value=1, label="Task Level")
            reset_btn = gr.Button("Reset Episode")

        with gr.Row():
            action_type = gr.Dropdown(
                choices=[
                    "accept_order",
                    "start_prep",
                    "load_oven",
                    "dispatch_driver",
                    "issue_refund",
                    "close_order",
                    "noop",
                ],
                value="noop",
                label="Action Type",
            )
            order_id = gr.Textbox(label="Order ID", placeholder="P1 / P2A / P3C ...")

        with gr.Row():
            oven_slot = gr.Number(label="Oven Slot", value=0, precision=0)
            driver_id = gr.Textbox(label="Driver ID", placeholder="D1 / D2")
            refund_reason = gr.Dropdown(
                choices=["late", "quality", "wrong_order"],
                value="late",
                label="Refund Reason",
            )

        step_btn = gr.Button("Step")
        output = gr.JSON(label="Episode Snapshot")

        def on_reset(level: int) -> Dict[str, Any]:
            obs = ui_env.reset(task_level=int(level))
            return _render_ui_state(obs)

        def on_step(a_type: str, o_id: str, slot: float, d_id: str, reason: str) -> Dict[str, Any]:
            action = PizzaShopAction(
                action_type=a_type,
                order_id=o_id or None,
                oven_slot=int(slot) if a_type == "load_oven" else None,
                driver_id=d_id or None,
                refund_reason=reason if a_type == "issue_refund" else None,
            )
            obs = ui_env.step(action)
            return _render_ui_state(obs)

        reset_btn.click(on_reset, inputs=[task_level], outputs=[output])
        step_btn.click(on_step, inputs=[action_type, order_id, oven_slot, driver_id, refund_reason], outputs=[output])

    return demo


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/tasks")
def list_tasks() -> Dict[str, Any]:
    tasks = utility_env.task_descriptions()
    task_lookup = {str(task.get("task_id")): int(task.get("task_level", 1)) for task in tasks}
    return {
        "tasks": tasks,
        "graded_tasks": task_lookup,
        "grader_score_range": "(0,1)",
        "action_schema": PizzaShopAction.model_json_schema(),
    }


@app.post("/grader")
def grader(payload: Optional[GraderRequest] = None) -> Dict[str, Any]:
    state = utility_env.state.model_dump()

    if payload and payload.state:
        state = payload.state
        result = grade_current_task(state)
    else:
        task_level = 1
        if payload and payload.task_level is not None:
            task_level = int(payload.task_level)
        elif payload and payload.task_id:
            task_id_map = {
                "easy_lunch_shift": 1,
                "medium_dinner_rush": 2,
                "hard_storm_surge": 3,
            }
            task_level = task_id_map.get(str(payload.task_id), 1)

        obs = utility_env.reset(task_level=task_level)
        state = utility_env.state.model_dump()
        state["task_level"] = int(obs.task_level)
        result = grade_task_by_level(int(obs.task_level), state)

    return {
        "task_level": int(state.get("task_level", 1)),
        "task_id": str(state.get("task_id", "")),
        "score": result.score,
        "passed": result.passed,
        "reason": result.reason,
    }


@app.get("/grader")
def grader_get(task_level: int = 1) -> Dict[str, Any]:
    return grader(GraderRequest(task_level=task_level))


@app.get("/graders")
def graders_summary() -> Dict[str, Any]:
    results = []
    for task_level in [1, 2, 3]:
        row = grader(GraderRequest(task_level=task_level))
        results.append(row)
    return {
        "task_count": len(results),
        "score_range": "(0,1)",
        "tasks": results,
    }


@app.get("/baseline")
def baseline(model: str = "gpt-4o-mini") -> Dict[str, Any]:
    try:
        return run_baseline(model=model)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Baseline failed: {exc}") from exc


gradio_demo = _build_demo()
app = gr.mount_gradio_app(app, gradio_demo, path="/ui")


def main() -> None:
    port = int(os.getenv("PORT", "7860"))
    uvicorn_run("server.app:app", host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
