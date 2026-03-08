from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ENDPOINTS_PATH = REPO_ROOT / "configs" / "endpoints.toml"


@dataclass(frozen=True)
class EvalMode:
    name: str
    output_mode: str


EVAL_MODES = {
    "json": EvalMode(name="json", output_mode="json"),
    "markdown": EvalMode(name="markdown", output_mode="markdown"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run chart extraction baseline evals.")
    parser.add_argument("--env-id", default="chart-extraction")
    parser.add_argument("--model", default="qwen/qwen3-vl-8b-instruct")
    parser.add_argument("--num-examples", type=int, default=8)
    parser.add_argument("--rollouts-per-example", type=int, default=1)
    parser.add_argument("--default-examples", type=int, default=64)
    parser.add_argument("--dataset-repo-id", default="")
    parser.add_argument("--dataset-local-path", default="")
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=sorted(EVAL_MODES),
        default=["json", "markdown"],
    )
    parser.add_argument("--endpoints-path", type=Path, default=DEFAULT_ENDPOINTS_PATH)
    parser.add_argument("--save-results", action="store_true")
    parser.add_argument("--skip-upload", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for mode_name in args.modes:
        mode = EVAL_MODES[mode_name]
        env_args = {"default_examples": args.default_examples, "output_mode": mode.output_mode}
        if args.dataset_repo_id:
            env_args["dataset_repo_id"] = args.dataset_repo_id
        if args.dataset_local_path:
            env_args["dataset_local_path"] = args.dataset_local_path

        command = [
            "prime",
            "eval",
            "run",
            args.env_id,
            "--endpoints-path",
            str(args.endpoints_path),
            "--model",
            args.model,
            "--num-examples",
            str(args.num_examples),
            "--rollouts-per-example",
            str(args.rollouts_per_example),
            "--env-args",
            json.dumps(env_args, sort_keys=True),
        ]
        if args.save_results:
            command.append("--save-results")
        if args.skip_upload:
            command.append("--skip-upload")

        print(f"\n=== Running {mode.name} baseline ===")
        print(shlex.join(command))
        subprocess.run(command, cwd=REPO_ROOT, check=True)


if __name__ == "__main__":
    main()
