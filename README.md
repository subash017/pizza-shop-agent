---
title: Pizza Shop OpenEnv
emoji: "🍕"
colorFrom: red
colorTo: yellow
sdk: docker
pinned: false
---

# Pizza Shop OpenEnv Environment

A complete OpenEnv environment that simulates pizza store operations: accepting orders, preparing and baking pizzas, dispatching drivers, handling refunds, and closing checks while balancing speed, quality, and margin.

## Why This Environment

Pizza operations are realistic and sequential:
- Workflows must be completed in order (accept -> prep -> bake -> dispatch -> close).
- Resource constraints matter (oven slots and driver capacity).
- Business outcomes trade off (late deliveries, refunds, complaints, profit).

## OpenEnv Spec Compliance

Implemented:
- Typed models (`PizzaShopAction`, `PizzaShopObservation`, `PizzaShopState`).
- Standard API (`reset`, `step`, `state`) through `openenv.core.env_server`.
- `openenv.yaml` metadata.
- Concurrent sessions enabled.

## Action Space

`PizzaShopAction` fields:
- `action_type`: one of `accept_order`, `start_prep`, `load_oven`, `dispatch_driver`, `issue_refund`, `close_order`, `noop`
- `order_id`: required for order-specific actions
- `oven_slot`: required for `load_oven`
- `driver_id`: required for `dispatch_driver`
- `refund_reason`: one of `late`, `quality`, `wrong_order` for `issue_refund`

## Observation Space

`PizzaShopObservation` includes:
- `task_level`, `task_id`, `objective`
- `current_tick`
- `pending_orders`
- `oven_status`
- `driver_status`
- `completed_checklist`
- `required_actions_remaining`
- `available_actions`
- `progress` (grader score in [0, 1])
- `message`
- standard `done` and `reward`

## Task Ladder (Easy -> Medium -> Hard)

1. Easy (`easy_lunch_shift`)
- Low volume lunch orders.
- Learn full workflow reliably.

2. Medium (`medium_dinner_rush`)
- More concurrent orders.
- Tight oven and driver scheduling.

3. Hard (`hard_storm_surge`)
- Surge demand under delivery pressure.
- Manage lateness, complaints, and refund discipline.

## Deterministic Graders

Graders in `server/graders.py`:
- Output score in `[0.0, 1.0]`.
- Deterministic checklist + operational KPI scoring.
- Penalize invalid/noop/destructive behavior.
- Separate task graders (`grade_task_1`, `grade_task_2`, `grade_task_3`) plus dispatcher.

Success threshold: `score >= 0.95`.

## Reward Function

Reward is trajectory-aware:
- Positive for checklist progress and task score.
- Penalties for invalid actions, no-ops, destructive actions, newly late orders, and refund spend.
- Efficiency bonus for solving in fewer steps.

Implemented in `server/reward.py`.

## Required Endpoints

In addition to OpenEnv standard endpoints:
- `GET /tasks`: returns all tasks and action schema.
- `POST /grader`: deterministic grader output.
- `GET /baseline`: reproducible baseline run.

Also included:
- `GET /health`
- `GET /ui` (Gradio interface)

## Setup

```bash
cd pizza_shop_env
pip install -r requirements.txt
uvicorn server.app:app --host 0.0.0.0 --port 8000 --reload
```

Quick checks:

```bash
curl http://localhost:8000/health
curl http://localhost:8000/tasks
curl http://localhost:8000/ui
```

## Baseline Inference (OpenAI API)

```bash
set OPENAI_API_KEY=your_key_here
python scripts/baseline_inference.py --model gpt-4o-mini
```

OpenRouter keys are also supported:

```bash
set OPENROUTER_API_KEY=your_openrouter_key
set OPENAI_BASE_URL=https://openrouter.ai/api/v1
set OPENROUTER_HTTP_REFERER=http://localhost
set OPENROUTER_X_TITLE=pizza-shop-openenv
python scripts/baseline_inference.py --model openai/gpt-4o-mini
```

## Docker / Hugging Face Space

The repository includes a root `Dockerfile` configured for HF Spaces:

```bash
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

## Tests

```bash
python -m pytest -q
```
