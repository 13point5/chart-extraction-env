from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EVAL_ROOT = REPO_ROOT / "environments" / "chart_extraction" / "outputs" / "evals"


@dataclass(frozen=True)
class RunSummary:
    run_dir: Path
    model: str
    output_mode: str
    avg_reward: float
    avg_metrics: dict[str, float]
    avg_output_tokens: float
    notes: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize chart extraction eval runs.")
    parser.add_argument("run_dirs", nargs="*", type=Path)
    parser.add_argument("--eval-root", type=Path, default=DEFAULT_EVAL_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dirs = args.run_dirs or discover_run_dirs(args.eval_root)
    summaries = [load_summary(run_dir) for run_dir in run_dirs]
    if not summaries:
        raise SystemExit("No eval runs found.")

    print("run_id\tmode\treward\tpoint_value\tx_type\toutput_tokens\tnotes")
    for summary in summaries:
        point_value = summary.avg_metrics.get("point_value_score", 0.0)
        x_type = summary.avg_metrics.get("x_type_score_metric", 0.0)
        print(
            "\t".join(
                [
                    summary.run_dir.name,
                    summary.output_mode,
                    f"{summary.avg_reward:.4f}",
                    f"{point_value:.4f}",
                    f"{x_type:.4f}",
                    f"{summary.avg_output_tokens:.1f}",
                    "; ".join(summary.notes) or "-",
                ]
            )
        )


def discover_run_dirs(eval_root: Path) -> list[Path]:
    metadata_files = sorted(eval_root.glob("**/metadata.json"))
    return [path.parent for path in metadata_files]


def load_summary(run_dir: Path) -> RunSummary:
    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    avg_metrics = {key: float(value) for key, value in metadata.get("avg_metrics", {}).items()}
    output_mode = str(metadata.get("env_args", {}).get("output_mode", "unknown"))
    notes = detect_notes(avg_metrics)
    return RunSummary(
        run_dir=run_dir,
        model=str(metadata["model"]),
        output_mode=output_mode,
        avg_reward=float(metadata["avg_reward"]),
        avg_metrics=avg_metrics,
        avg_output_tokens=float(metadata.get("usage", {}).get("output_tokens", 0.0)),
        notes=notes,
    )


def detect_notes(avg_metrics: dict[str, float]) -> list[str]:
    notes: list[str] = []
    if avg_metrics.get("series_name_score", 1.0) < 0.5:
        notes.append("series-name mismatch")
    if avg_metrics.get("chart_type_score", 1.0) < 0.9:
        notes.append("chart-type confusion")
    if avg_metrics.get("point_x_score_metric", 1.0) > 0.95 and avg_metrics.get("point_y_score_metric", 1.0) < 0.8:
        notes.append("x good, y weak")
    if avg_metrics.get("exact_answer_score_metric", 1.0) == 0.0 and avg_metrics.get("point_value_score", 0.0) > 0.95:
        notes.append("approximate-only")
    return notes


if __name__ == "__main__":
    main()
