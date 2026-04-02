import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from server.baseline import run_baseline


def main() -> int:
    parser = argparse.ArgumentParser(description="Run baseline inference for all pizza shop tasks.")
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI model name")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Optional custom max step budget per task for debugging.",
    )
    args = parser.parse_args()

    result = run_baseline(model=args.model, max_steps_override=args.max_steps)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
