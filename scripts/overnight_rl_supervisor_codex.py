#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
from dataclasses import dataclass
from datetime import datetime, time as dt_time, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import tomllib


LOCAL_TZ = ZoneInfo("America/New_York")
REPO_ROOT = Path(__file__).resolve().parents[1]
LOGS_DIR = REPO_ROOT / "logs"
OVERNIGHT_CONFIG_DIR = REPO_ROOT / "rl-configs" / "overnight"
PRIME_BIN = shutil.which("prime")
CODEX_BIN = shutil.which("codex")

SUCCESS_STATUSES = {"RUNNING", "COMPLETED"}
TERMINAL_FAILURE_STATUSES = {"FAILED", "STOPPED", "DELETED"}


@dataclass
class WatchSpec:
    run_id: str
    config_path: Path


def now_local() -> datetime:
    return datetime.now(tz=LOCAL_TZ)


def local_timestamp(value: datetime | None = None) -> str:
    dt = value or now_local()
    return dt.strftime("%Y-%m-%d %H:%M:%S %Z")


def repo_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path.resolve())


def ensure_dirs() -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    OVERNIGHT_CONFIG_DIR.mkdir(parents=True, exist_ok=True)


def deadline_from_now(target_hour: int = 10) -> datetime:
    current = now_local()
    deadline = datetime.combine(current.date(), dt_time(hour=target_hour), tzinfo=LOCAL_TZ)
    if deadline <= current:
        deadline += timedelta(days=1)
    return deadline


def append_markdown(log_path: Path, text: str) -> None:
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(text.rstrip() + "\n\n")


def save_state(state_path: Path, state: dict[str, Any]) -> None:
    state_path.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")


def load_state(state_path: Path, deadline: datetime, interval_minutes: int) -> dict[str, Any]:
    if state_path.exists():
        return json.loads(state_path.read_text(encoding="utf-8"))

    return {
        "started_at": local_timestamp(),
        "deadline": deadline.isoformat(),
        "interval_minutes": interval_minutes,
        "runs": {},
        "config_fingerprints": {},
        "config_history": [],
    }


def command_error(argv: list[str], proc: subprocess.CompletedProcess[str]) -> RuntimeError:
    pieces = [
        f"Command failed with exit code {proc.returncode}: {' '.join(argv)}",
    ]
    if proc.stdout.strip():
        pieces.append(f"stdout:\n{proc.stdout.strip()}")
    if proc.stderr.strip():
        pieces.append(f"stderr:\n{proc.stderr.strip()}")
    return RuntimeError("\n\n".join(pieces))


def run_command(
    argv: list[str],
    *,
    env: dict[str, str] | None = None,
    input_text: str | None = None,
    timeout: int = 600,
    check: bool = True,
) -> str:
    proc = subprocess.run(
        argv,
        cwd=REPO_ROOT,
        env=env,
        input=input_text,
        text=True,
        capture_output=True,
        timeout=timeout,
    )
    if check and proc.returncode != 0:
        raise command_error(argv, proc)
    return proc.stdout.strip()


def run_prime(args: list[str], *, env: dict[str, str] | None = None, timeout: int = 600) -> str:
    if not PRIME_BIN:
        raise FileNotFoundError("prime CLI is not available on PATH")
    return run_command([PRIME_BIN, *args], env=env, timeout=timeout)


def parse_watch(value: str) -> WatchSpec:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Expected RUN_ID=CONFIG_PATH")
    run_id, raw_path = value.split("=", 1)
    config_path = Path(raw_path)
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path
    return WatchSpec(run_id=run_id.strip(), config_path=config_path.resolve())


def parse_get_output(text: str) -> dict[str, Any]:
    data: dict[str, Any] = {"raw": text}
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("Status:"):
            data["status"] = stripped.split(":", 1)[1].strip()
        elif stripped.startswith("Model:"):
            data["model"] = stripped.split(":", 1)[1].strip()
        elif stripped.startswith("Max Steps:"):
            data["max_steps"] = stripped.split(":", 1)[1].strip()
        elif stripped.startswith("Batch Size:"):
            data["batch_size"] = stripped.split(":", 1)[1].strip()
        elif stripped.startswith("Rollouts per Example:"):
            data["rollouts_per_example"] = stripped.split(":", 1)[1].strip()
        elif stripped.startswith("Max Tokens:"):
            data["max_tokens"] = stripped.split(":", 1)[1].strip()
        elif stripped.startswith("Created:"):
            data["created"] = stripped.split(":", 1)[1].strip()
        elif stripped.startswith("Started:"):
            data["started"] = stripped.split(":", 1)[1].strip()
        elif stripped.startswith("Completed:"):
            data["completed"] = stripped.split(":", 1)[1].strip()
        elif stripped.startswith("Error:"):
            data["error"] = stripped.split(":", 1)[1].strip()
    return data


def get_run_details(run_id: str) -> dict[str, Any]:
    return parse_get_output(run_prime(["rl", "get", run_id]))


def get_run_progress(run_id: str) -> dict[str, Any]:
    output = run_prime(["rl", "progress", run_id])
    return json.loads(output)


def get_run_logs(run_id: str, tail: int = 250) -> str:
    return run_prime(["rl", "logs", run_id, "--tail", str(tail), "--raw"], timeout=900)


def load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("rb") as handle:
        return tomllib.load(handle)


def config_fingerprint(config: dict[str, Any]) -> str:
    normalized = {
        "model": config.get("model"),
        "batch_size": config.get("batch_size"),
        "rollouts_per_example": config.get("rollouts_per_example"),
        "learning_rate": config.get("learning_rate"),
        "sampling_max_tokens": config.get("sampling", {}).get("max_tokens"),
        "train_split": config.get("env", [{}])[0].get("args", {}).get("split"),
        "eval_num_examples": config.get("eval", {}).get("num_examples"),
        "eval_interval": config.get("eval", {}).get("interval"),
        "eval_split": config.get("eval", {}).get("env", [{}])[0].get("args", {}).get("split"),
        "val_num_examples": config.get("val", {}).get("num_examples"),
        "val_interval": config.get("val", {}).get("interval"),
    }
    return json.dumps(normalized, sort_keys=True)


def format_toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return format(value, "g")
    if isinstance(value, int):
        return str(value)
    return json.dumps(value)


def render_config(config: dict[str, Any]) -> str:
    lines = [
        f'model = {format_toml_value(config["model"])}',
        f'max_steps = {format_toml_value(config["max_steps"])}',
        f'batch_size = {format_toml_value(config["batch_size"])}',
        f'rollouts_per_example = {format_toml_value(config["rollouts_per_example"])}',
    ]
    if "learning_rate" in config:
        lines.append(f'learning_rate = {format_toml_value(config["learning_rate"])}')

    lines.extend(
        [
            "",
            "[sampling]",
            f'max_tokens = {format_toml_value(config["sampling"]["max_tokens"])}',
            "",
            "[[env]]",
            f'id = {format_toml_value(config["env"][0]["id"])}',
            f'args = {{ split = {format_toml_value(config["env"][0]["args"]["split"])} }}',
            "",
            "[wandb]",
            f'project = {format_toml_value(config["wandb"]["project"])}',
            f'name = {format_toml_value(config["wandb"]["name"])}',
            f'entity = {format_toml_value(config["wandb"]["entity"])}',
            "",
            "[eval]",
            f'interval = {format_toml_value(config["eval"]["interval"])}',
            f'num_examples = {format_toml_value(config["eval"]["num_examples"])}',
            f'rollouts_per_example = {format_toml_value(config["eval"]["rollouts_per_example"])}',
            f'eval_base_model = {format_toml_value(config["eval"]["eval_base_model"])}',
            "",
            "[[eval.env]]",
            f'id = {format_toml_value(config["eval"]["env"][0]["id"])}',
            f'args = {{ split = {format_toml_value(config["eval"]["env"][0]["args"]["split"])} }}',
            "",
            "[val]",
            f'num_examples = {format_toml_value(config["val"]["num_examples"])}',
            f'rollouts_per_example = {format_toml_value(config["val"]["rollouts_per_example"])}',
            f'interval = {format_toml_value(config["val"]["interval"])}',
            "",
            "# Optional: buffer configuration for difficulty filtering",
            "# [buffer]",
            "# easy_threshold = 0.8",
            "# hard_threshold = 0.2",
            "# easy_fraction = 0.0",
            "# hard_fraction = 0.0",
            "# online_difficulty_filtering = false",
            "# seed = 42",
        ]
    )
    return "\n".join(lines) + "\n"


def merge_proposal(base_config: dict[str, Any], proposal: dict[str, Any], attempt_index: int) -> dict[str, Any]:
    merged = json.loads(json.dumps(base_config))
    changes = proposal.get("proposed_changes", {})

    if changes.get("batch_size") is not None:
        merged["batch_size"] = int(changes["batch_size"])
    if changes.get("rollouts_per_example") is not None:
        merged["rollouts_per_example"] = int(changes["rollouts_per_example"])
    if changes.get("learning_rate") is not None:
        merged["learning_rate"] = float(changes["learning_rate"])
    elif "learning_rate" in merged and proposal.get("clear_learning_rate"):
        merged.pop("learning_rate", None)
    if changes.get("sampling_max_tokens") is not None:
        merged.setdefault("sampling", {})["max_tokens"] = int(changes["sampling_max_tokens"])
    if changes.get("train_split"):
        merged.setdefault("env", [{}])[0].setdefault("args", {})["split"] = changes["train_split"]
    if changes.get("eval_num_examples") is not None:
        merged.setdefault("eval", {})["num_examples"] = int(changes["eval_num_examples"])
    if changes.get("eval_interval") is not None:
        merged.setdefault("eval", {})["interval"] = int(changes["eval_interval"])
    if changes.get("eval_split"):
        merged.setdefault("eval", {}).setdefault("env", [{}])[0].setdefault("args", {})["split"] = changes["eval_split"]
    if changes.get("val_num_examples") is not None:
        merged.setdefault("val", {})["num_examples"] = int(changes["val_num_examples"])
    if changes.get("val_interval") is not None:
        merged.setdefault("val", {})["interval"] = int(changes["val_interval"])

    train_split = merged["env"][0]["args"]["split"]
    split_slug = re.sub(r"[^a-zA-Z0-9]+", "-", train_split).strip("-").lower()[:24] or "train"
    merged["wandb"]["name"] = (
        f"qwen3-vl-8b-bs{merged['batch_size']}-rpe{merged['rollouts_per_example']}"
        f"-{split_slug}-a{attempt_index}"
    )[:128]
    return merged


def write_config_file(config: dict[str, Any], attempt_index: int) -> Path:
    timestamp = now_local().strftime("%Y%m%d-%H%M%S")
    train_split = config["env"][0]["args"]["split"]
    split_slug = re.sub(r"[^a-zA-Z0-9]+", "-", train_split).strip("-").lower()[:24] or "train"
    config_path = OVERNIGHT_CONFIG_DIR / (
        f"{timestamp}-a{attempt_index}-bs{config['batch_size']}-"
        f"rpe{config['rollouts_per_example']}-{split_slug}.toml"
    )
    config_path.write_text(render_config(config), encoding="utf-8")
    return config_path


def submit_run(config_path: Path, env: dict[str, str]) -> tuple[str, str]:
    output = run_prime(["rl", "run", "-e", "WANDB_API_KEY", repo_relative(config_path)], env=env, timeout=900)
    match = re.search(r"/training/([a-z0-9]+)", output)
    if not match:
        match = re.search(r"prime rl logs ([a-z0-9]+) -f", output)
    if not match:
        raise RuntimeError(f"Unable to parse run id from output:\n{output}")
    return match.group(1), output


def recent_attempts_summary(state: dict[str, Any], limit: int = 8) -> list[dict[str, Any]]:
    attempts = state.get("config_history", [])[-limit:]
    return attempts


def codex_schema() -> dict[str, Any]:
    nullable_string = {"type": ["string", "null"]}
    nullable_integer = {"type": ["integer", "null"]}
    nullable_number = {"type": ["number", "null"]}
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "should_retry",
            "summary",
            "error_signature",
            "likely_cause",
            "proposed_changes",
        ],
        "properties": {
            "should_retry": {"type": "boolean"},
            "summary": {"type": "string"},
            "error_signature": {"type": "string"},
            "likely_cause": {"type": "string"},
            "clear_learning_rate": {"type": "boolean"},
            "proposed_changes": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "batch_size",
                    "rollouts_per_example",
                    "learning_rate",
                    "sampling_max_tokens",
                    "train_split",
                    "eval_num_examples",
                    "eval_interval",
                    "eval_split",
                    "val_num_examples",
                    "val_interval",
                    "reason",
                ],
                "properties": {
                    "batch_size": nullable_integer,
                    "rollouts_per_example": nullable_integer,
                    "learning_rate": nullable_number,
                    "sampling_max_tokens": nullable_integer,
                    "train_split": nullable_string,
                    "eval_num_examples": nullable_integer,
                    "eval_interval": nullable_integer,
                    "eval_split": nullable_string,
                    "val_num_examples": nullable_integer,
                    "val_interval": nullable_integer,
                    "reason": {"type": "string"},
                },
            },
        },
    }


def call_codex_analysis(
    *,
    run_id: str,
    run_details: dict[str, Any],
    progress: dict[str, Any],
    config_path: Path,
    config_text: str,
    logs_tail: str,
    state: dict[str, Any],
) -> dict[str, Any]:
    if not CODEX_BIN:
        raise FileNotFoundError("codex CLI is not available on PATH")

    prompt = textwrap.dedent(
        f"""
        You are helping tune hosted Prime RL runs overnight.

        Output JSON only, matching the provided schema exactly.

        Goal:
        - Get any run past 100 steps before 10:00 AM America/New_York.
        - Do not propose code edits or environment implementation changes.
        - Only propose the next config tweak to maximize the chance of a run getting past startup and then reaching 100 steps.
        - Avoid repeating configs that are already in the recent attempt history.
        - If the logs show startup or health-check timeout before any training steps, prioritize smaller train/eval splits and lighter eval/val, then smaller batch/max tokens.

        Failed run id: {run_id}
        Current run details:
        {json.dumps(run_details, indent=2)}

        Current progress:
        {json.dumps(progress, indent=2)}

        Current config path: {repo_relative(config_path)}
        Current config TOML:
        ```toml
        {config_text}
        ```

        Recent attempts:
        {json.dumps(recent_attempts_summary(state), indent=2)}

        Recent raw log tail:
        ```text
        {logs_tail}
        ```
        """
    ).strip()

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as schema_file:
        json.dump(codex_schema(), schema_file)
        schema_path = Path(schema_file.name)

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as output_file:
        output_path = Path(output_file.name)

    try:
        run_command(
            [
                CODEX_BIN,
                "exec",
                "-C",
                str(REPO_ROOT),
                "-s",
                "read-only",
                "--color",
                "never",
                "--ephemeral",
                "--output-schema",
                str(schema_path),
                "-o",
                str(output_path),
                "-",
            ],
            input_text=prompt,
            timeout=900,
        )
        return json.loads(output_path.read_text(encoding="utf-8"))
    finally:
        schema_path.unlink(missing_ok=True)
        output_path.unlink(missing_ok=True)


def sliced_split(base: str, size: int) -> str:
    if ":" in base:
        prefix = base.split("[", 1)[0]
    else:
        prefix = base
    return f"{prefix}[:{size}]"


def fallback_analysis(base_config: dict[str, Any], failure_text: str) -> dict[str, Any]:
    failure_lower = failure_text.lower()
    current_split = base_config["env"][0]["args"]["split"]
    current_eval_split = base_config["eval"]["env"][0]["args"]["split"]
    batch_size = int(base_config["batch_size"])
    rollouts_per_example = int(base_config["rollouts_per_example"])
    max_tokens = int(base_config["sampling"]["max_tokens"])

    if "did not become healthy within 10m" in failure_lower or "backofflimitexceeded" in failure_lower:
        if "[:2000]" not in current_split and "[:1000]" not in current_split and "[:500]" not in current_split:
            train_split = sliced_split(current_split, 2000)
            eval_split = sliced_split(current_eval_split, 200)
            next_batch = min(batch_size, 64)
            next_rollouts = min(rollouts_per_example, 4)
            next_tokens = min(max_tokens, 4000)
            eval_examples = min(int(base_config["eval"]["num_examples"]), 100)
            val_examples = min(int(base_config["val"]["num_examples"]), 25)
        elif "[:500]" not in current_split:
            train_split = sliced_split(current_split, 500)
            eval_split = sliced_split(current_eval_split, 100)
            next_batch = min(batch_size, 32)
            next_rollouts = min(rollouts_per_example, 4)
            next_tokens = min(max_tokens, 2500)
            eval_examples = min(int(base_config["eval"]["num_examples"]), 50)
            val_examples = min(int(base_config["val"]["num_examples"]), 10)
        else:
            train_split = sliced_split(current_split, 100)
            eval_split = sliced_split(current_eval_split, 50)
            next_batch = min(batch_size, 16)
            next_rollouts = min(rollouts_per_example, 2)
            next_tokens = min(max_tokens, 2000)
            eval_examples = min(int(base_config["eval"]["num_examples"]), 25)
            val_examples = min(int(base_config["val"]["num_examples"]), 5)

        return {
            "should_retry": True,
            "summary": "Fallback picked a much lighter config because the run never became healthy before the hosted startup timeout.",
            "error_signature": "env-startup-timeout",
            "likely_cause": "Environment startup is too heavy for the hosted timeout window, so the next attempt needs a much smaller train/eval footprint.",
            "clear_learning_rate": False,
            "proposed_changes": {
                "batch_size": next_batch,
                "rollouts_per_example": next_rollouts,
                "learning_rate": None,
                "sampling_max_tokens": next_tokens,
                "train_split": train_split,
                "eval_num_examples": eval_examples,
                "eval_interval": max(int(base_config["eval"]["interval"]), 50),
                "eval_split": eval_split,
                "val_num_examples": val_examples,
                "val_interval": max(int(base_config["val"]["interval"]), 50),
                "reason": "Reduce startup work and rollout pressure so the job can get past hosted env initialization and reach 100 steps overnight.",
            },
        }

    return {
        "should_retry": True,
        "summary": "Fallback applied a conservative downshift after a non-specific failure.",
        "error_signature": "generic-failure",
        "likely_cause": "Unknown failure, so the next attempt lowers throughput and token pressure while keeping the task setup intact.",
        "clear_learning_rate": False,
        "proposed_changes": {
            "batch_size": max(16, batch_size // 2),
            "rollouts_per_example": max(2, min(rollouts_per_example, 4)),
            "learning_rate": None,
            "sampling_max_tokens": max(2000, min(max_tokens, 4000)),
            "train_split": None,
            "eval_num_examples": min(int(base_config["eval"]["num_examples"]), 100),
            "eval_interval": max(int(base_config["eval"]["interval"]), 50),
            "eval_split": None,
            "val_num_examples": min(int(base_config["val"]["num_examples"]), 25),
            "val_interval": max(int(base_config["val"]["interval"]), 50),
            "reason": "Back off the most expensive knobs first while keeping the same environment contract.",
        },
    }


def ensure_unique_proposal(base_config: dict[str, Any], proposal: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    tried = set(state.get("config_fingerprints", {}).keys())
    candidate = merge_proposal(base_config, proposal, attempt_index=len(state.get("config_history", [])) + 1)
    fingerprint = config_fingerprint(candidate)
    if fingerprint not in tried:
        return candidate

    fallback = fallback_analysis(base_config, proposal.get("error_signature", ""))
    candidate = merge_proposal(base_config, fallback, attempt_index=len(state.get("config_history", [])) + 1)
    fingerprint = config_fingerprint(candidate)
    if fingerprint in tried:
        candidate["batch_size"] = max(8, int(candidate["batch_size"]) // 2)
        candidate["rollouts_per_example"] = max(1, int(candidate["rollouts_per_example"]) // 2)
        candidate["sampling"]["max_tokens"] = max(1000, int(candidate["sampling"]["max_tokens"]) // 2)
    return candidate


def register_config(state: dict[str, Any], config_path: Path, config: dict[str, Any]) -> None:
    fingerprint = config_fingerprint(config)
    state.setdefault("config_fingerprints", {})[fingerprint] = repo_relative(config_path)
    history_entry = {
        "timestamp": local_timestamp(),
        "config_path": repo_relative(config_path),
        "batch_size": config["batch_size"],
        "rollouts_per_example": config["rollouts_per_example"],
        "sampling_max_tokens": config["sampling"]["max_tokens"],
        "train_split": config["env"][0]["args"]["split"],
        "eval_num_examples": config["eval"]["num_examples"],
        "val_num_examples": config["val"]["num_examples"],
    }
    if history_entry not in state.setdefault("config_history", []):
        state["config_history"].append(history_entry)


def register_run(state: dict[str, Any], run_id: str, config_path: Path, status: str = "PENDING") -> None:
    state.setdefault("runs", {})[run_id] = {
        "config_path": repo_relative(config_path),
        "status": status,
        "latest_step": None,
        "last_checked_at": local_timestamp(),
        "handled_terminal_signature": None,
    }


def prime_env() -> dict[str, str]:
    env = os.environ.copy()
    if not env.get("WANDB_API_KEY"):
        raise EnvironmentError("WANDB_API_KEY must be set in the environment before running the supervisor.")
    return env


def failure_signature(details: dict[str, Any], logs_tail: str) -> str:
    if details.get("error"):
        return str(details["error"])
    if logs_tail.strip():
        return logs_tail.splitlines()[-1]
    return str(details.get("status", "unknown"))


def handle_failed_run(
    *,
    run_id: str,
    state: dict[str, Any],
    log_path: Path,
    env: dict[str, str],
) -> str | None:
    run_state = state["runs"][run_id]
    config_path = REPO_ROOT / run_state["config_path"]
    base_config = load_config(config_path)
    config_text = config_path.read_text(encoding="utf-8")
    details = get_run_details(run_id)
    progress = get_run_progress(run_id)
    logs_tail = get_run_logs(run_id)
    combined_failure = "\n".join(
        [
            details.get("error", ""),
            logs_tail,
        ]
    )

    try:
        analysis = call_codex_analysis(
            run_id=run_id,
            run_details=details,
            progress=progress,
            config_path=config_path,
            config_text=config_text,
            logs_tail=logs_tail,
            state=state,
        )
        analysis_source = "codex"
    except Exception as exc:
        analysis = fallback_analysis(base_config, combined_failure)
        analysis_source = f"fallback ({exc})"

    if not analysis.get("should_retry", True):
        append_markdown(
            log_path,
            textwrap.dedent(
                f"""
                ### {local_timestamp()} failure analysis for `{run_id}`
                - source: `{analysis_source}`
                - config: `{repo_relative(config_path)}`
                - summary: {analysis.get("summary", "no summary")}
                - likely cause: {analysis.get("likely_cause", "unknown")}
                - action: no retry requested
                """
            ),
        )
        return None

    next_config = ensure_unique_proposal(base_config, analysis, state)
    attempt_index = len(state.get("config_history", [])) + 1
    next_config_path = write_config_file(next_config, attempt_index)
    register_config(state, next_config_path, next_config)

    try:
        new_run_id, submission_output = submit_run(next_config_path, env)
        register_run(state, new_run_id, next_config_path)
    except Exception as exc:
        append_markdown(
            log_path,
            textwrap.dedent(
                f"""
                ### {local_timestamp()} retry launch failed for `{run_id}`
                - next config: `{repo_relative(next_config_path)}`
                - error: `{exc}`
                """
            ),
        )
        return None

    append_markdown(
        log_path,
        textwrap.dedent(
            f"""
            ### {local_timestamp()} failure analysis for `{run_id}`
            - source: `{analysis_source}`
            - failed config: `{repo_relative(config_path)}`
            - error signature: {analysis.get("error_signature", "unknown")}
            - likely cause: {analysis.get("likely_cause", "unknown")}
            - summary: {analysis.get("summary", "no summary")}
            - next config: `{repo_relative(next_config_path)}`
            - new run id: `{new_run_id}`
            - proposal reason: {analysis.get("proposed_changes", {}).get("reason", "n/a")}

            ```text
            {submission_output}
            ```
            """
        ),
    )
    return new_run_id


def poll_runs(
    *,
    state: dict[str, Any],
    log_path: Path,
    env: dict[str, str],
    success_steps: int,
) -> bool:
    active_run_ids = list(state.get("runs", {}).keys())
    if not active_run_ids:
        append_markdown(log_path, f"## {local_timestamp()}\n- no runs are registered")
        return False

    lines = [f"## {local_timestamp()}", "- poll summary:"]
    success_reached = False

    for run_id in active_run_ids:
        run_state = state["runs"][run_id]
        try:
            details = get_run_details(run_id)
            progress = get_run_progress(run_id)
            latest_step = progress.get("latest_step")
            status = details.get("status", "UNKNOWN")
            error = details.get("error")
            run_state["status"] = status
            run_state["latest_step"] = latest_step
            run_state["last_checked_at"] = local_timestamp()
            if error:
                run_state["error"] = error

            lines.append(
                f"  - `{run_id}` from `{run_state['config_path']}` -> status `{status}`, latest_step `{latest_step}`"
                + (f", error `{error}`" if error else "")
            )

            if status in SUCCESS_STATUSES and latest_step is not None and int(latest_step) >= success_steps:
                success_reached = True

            if status in TERMINAL_FAILURE_STATUSES:
                signature = f"{status}:{failure_signature(details, error or '')}"
                if run_state.get("handled_terminal_signature") != signature:
                    new_run_id = handle_failed_run(
                        run_id=run_id,
                        state=state,
                        log_path=log_path,
                        env=env,
                    )
                    run_state["handled_terminal_signature"] = signature
                    if new_run_id:
                        lines.append(f"  - launched retry `{new_run_id}`")
        except Exception as exc:
            run_state["last_checked_at"] = local_timestamp()
            lines.append(f"  - `{run_id}` poll error: `{exc}`")

    append_markdown(log_path, "\n".join(lines))
    return success_reached


def initialize_log(log_path: Path, deadline: datetime, watches: list[WatchSpec], interval_minutes: int) -> None:
    if log_path.exists():
        return
    watch_lines = "\n".join(
        f"- `{watch.run_id}` from `{repo_relative(watch.config_path)}`" for watch in watches
    ) or "- none"
    append_markdown(
        log_path,
        textwrap.dedent(
            f"""
            # Overnight RL Supervisor

            - started: {local_timestamp()}
            - deadline: {local_timestamp(deadline)}
            - repo: `{REPO_ROOT}`
            - watch interval: `{interval_minutes} minutes`
            - initial runs:
            {watch_lines}
            """
        ),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Monitor hosted Prime RL runs overnight and retry with Codex-guided configs.")
    parser.add_argument("--watch", action="append", type=parse_watch, default=[], help="Watch an existing run: RUN_ID=CONFIG_PATH")
    parser.add_argument("--interval-minutes", type=int, default=15, help="Polling interval in minutes")
    parser.add_argument("--success-steps", type=int, default=100, help="Stop once any run reaches at least this many steps")
    parser.add_argument("--deadline-hour", type=int, default=10, help="Local hour to stop at on the next occurrence")
    parser.add_argument("--log-path", type=Path, default=LOGS_DIR / "overnight_rl_supervisor.md")
    parser.add_argument("--state-path", type=Path, default=LOGS_DIR / "overnight_rl_supervisor_state.json")
    parser.add_argument("--once", action="store_true", help="Run a single polling cycle and exit")
    args = parser.parse_args()

    ensure_dirs()
    deadline = deadline_from_now(args.deadline_hour)
    env = prime_env()
    state = load_state(args.state_path, deadline, args.interval_minutes)
    initialize_log(args.log_path, deadline, args.watch, args.interval_minutes)

    for watch in args.watch:
        config = load_config(watch.config_path)
        register_config(state, watch.config_path, config)
        state.setdefault("runs", {}).setdefault(
            watch.run_id,
            {
                "config_path": repo_relative(watch.config_path),
                "status": "PENDING",
                "latest_step": None,
                "last_checked_at": None,
                "handled_terminal_signature": None,
            },
        )

    save_state(args.state_path, state)

    try:
        while True:
            current = now_local()
            if current >= deadline:
                append_markdown(args.log_path, f"## {local_timestamp()}\n- deadline reached, stopping supervisor")
                save_state(args.state_path, state)
                return 0

            success = poll_runs(
                state=state,
                log_path=args.log_path,
                env=env,
                success_steps=args.success_steps,
            )
            save_state(args.state_path, state)

            if success:
                append_markdown(
                    args.log_path,
                    f"## {local_timestamp()}\n- success condition met: at least one run reached `{args.success_steps}` steps",
                )
                save_state(args.state_path, state)
                return 0

            if args.once:
                return 0

            sleep_seconds = max(60, args.interval_minutes * 60)
            append_markdown(
                args.log_path,
                f"## {local_timestamp()}\n- sleeping for `{args.interval_minutes}` minutes before the next check",
            )
            save_state(args.state_path, state)
            time.sleep(sleep_seconds)
    except KeyboardInterrupt:
        append_markdown(args.log_path, f"## {local_timestamp()}\n- interrupted by user")
        save_state(args.state_path, state)
        return 130
    except Exception as exc:
        append_markdown(
            args.log_path,
            textwrap.dedent(
                f"""
                ## {local_timestamp()}
                - supervisor crashed: `{exc}`
                """
            ),
        )
        save_state(args.state_path, state)
        raise


if __name__ == "__main__":
    sys.exit(main())
