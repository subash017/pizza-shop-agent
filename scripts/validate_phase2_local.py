import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from server.app import graders_summary  # noqa: E402


def main() -> int:
    summary = graders_summary()

    print(json.dumps(summary, indent=2))

    task_count = int(summary.get("task_count", 0))
    if task_count < 3:
        print("FAIL: task_count < 3")
        return 2

    tasks = summary.get("tasks", [])
    if len(tasks) < 3:
        print("FAIL: fewer than 3 task grader entries")
        return 2

    for item in tasks:
        score = float(item.get("score", -1.0))
        task_id = str(item.get("task_id", "unknown"))
        if not (0.0 < score < 1.0):
            print(f"FAIL: score out of strict range for task={task_id} score={score}")
            return 2

    print("PASS: local phase-2 grader checks satisfied")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
