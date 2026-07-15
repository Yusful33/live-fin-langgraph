#!/usr/bin/env python3
"""
Generate experiment runs for Arize dataset `2026-ipo-stock-questions`.

Exports examples via `ax datasets export`, runs the fin agent on each
`question`, and writes `example_id` + `output` rows for
`ax experiments create`.

Usage:
  export ARIZE_API_KEY=... ARIZE_SPACE=... OPENAI_API_KEY=... FMP_API_KEY=...
  ax profiles create default --api-key "$ARIZE_API_KEY"   # once
  python scripts/run_ipo_dataset_experiment.py --output runs.json
  ax experiments create -n "my-run" --dataset 2026-ipo-stock-questions \\
    --space "$ARIZE_SPACE" -f runs.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = "2026-ipo-stock-questions"
DEFAULT_SPACE = "U3BhY2U6MjEwMzU6MEtGdA=="  # Live Demo


def _space() -> str:
    return (
        os.environ.get("ARIZE_SPACE")
        or os.environ.get("ARIZE_SPACE_ID")
        or os.environ.get("ARIZE_SPACE_KEY")
        or DEFAULT_SPACE
    )


def export_dataset(dataset: str, space: str) -> list[dict]:
    proc = subprocess.run(
        [
            "ax",
            "datasets",
            "export",
            dataset,
            "--space",
            space,
            "--stdout",
        ],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        print(proc.stderr or proc.stdout, file=sys.stderr)
        sys.exit(proc.returncode or 1)

    text = (proc.stdout or "").strip()
    if not text.startswith("["):
        idx = text.find("[")
        if idx >= 0:
            text = text[idx:]
    if not text:
        print("Dataset export returned empty stdout.", file=sys.stderr)
        sys.exit(1)
    return json.loads(text)


def example_question(row: dict) -> str | None:
    props = row.get("additional_properties") or {}
    q = props.get("question") or row.get("question")
    if isinstance(q, str) and q.strip():
        return q.strip()
    return None


async def run_agent_on_question(app, question: str, thread_id: str) -> str:
    from langchain_core.messages import HumanMessage

    inputs = {"messages": [HumanMessage(content=question)]}
    config = {"configurable": {"thread_id": thread_id}}
    final = ""
    async for chunk in app.astream(inputs, config, stream_mode="values"):
        messages = chunk.get("messages") or []
        if not messages:
            continue
        content = getattr(messages[-1], "content", None)
        if content is not None:
            final = content if isinstance(content, str) else str(content)
    return final


async def collect_runs(
    rows: list[dict],
    *,
    pr_number: str | None,
    git_sha: str | None,
) -> list[dict]:
    # Import graph only — avoid main.py module-level Arize register side effects.
    sys.path.insert(0, str(ROOT))
    from langgraph_fin_agent.graph import build_app

    app = build_app()
    runs: list[dict] = []

    for i, row in enumerate(rows):
        eid = row.get("id")
        question = example_question(row)
        if not eid or not question:
            print(f"  skip row missing id/question: {row!r}", file=sys.stderr)
            continue

        print(f"  [{i + 1}/{len(rows)}] {eid}: {question[:80]}...", file=sys.stderr)
        start = time.time()
        try:
            output = await run_agent_on_question(app, question, thread_id=str(eid))
        except Exception as exc:  # noqa: BLE001 — record failure as output for the run
            print(f"  error on {eid}: {exc}", file=sys.stderr)
            output = f"ERROR: {exc}"
        latency_ms = round((time.time() - start) * 1000)

        metadata: dict = {
            "model": "gpt-4o-mini",
            "latency_ms": latency_ms,
            "question": question,
        }
        if pr_number:
            metadata["pr_number"] = pr_number
        if git_sha:
            metadata["git_sha"] = git_sha

        props = row.get("additional_properties") or {}
        if props.get("company"):
            metadata["company"] = props["company"]
        if props.get("expected_ticker"):
            metadata["expected_ticker"] = props["expected_ticker"]

        runs.append(
            {
                "example_id": eid,
                "output": output,
                "metadata": metadata,
            }
        )
        print(f"  done {eid}: {latency_ms}ms", file=sys.stderr)

    return runs


def main() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is required.", file=sys.stderr)
        sys.exit(1)

    parser = argparse.ArgumentParser(
        description="Run fin agent on 2026-ipo-stock-questions for Arize experiments"
    )
    parser.add_argument(
        "--dataset",
        default=os.environ.get("ARIZE_DATASET", DEFAULT_DATASET),
        help="Dataset name or ID",
    )
    parser.add_argument(
        "--space",
        default=_space(),
        help="Arize space name or ID",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("runs.json"),
        help="Output runs JSON path",
    )
    parser.add_argument(
        "--pr-number",
        default=os.environ.get("PR_NUMBER") or os.environ.get("GITHUB_PR_NUMBER"),
    )
    parser.add_argument(
        "--git-sha",
        default=os.environ.get("GIT_SHA") or os.environ.get("GITHUB_SHA"),
    )
    args = parser.parse_args()

    print(
        f"Exporting dataset {args.dataset!r} from space {args.space!r}...",
        file=sys.stderr,
    )
    rows = export_dataset(args.dataset, args.space)
    print(f"Loaded {len(rows)} examples", file=sys.stderr)

    runs = asyncio.run(
        collect_runs(rows, pr_number=args.pr_number, git_sha=args.git_sha)
    )
    if not runs:
        print("No runs produced.", file=sys.stderr)
        sys.exit(1)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(runs, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(runs)} runs to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
