#!/usr/bin/env python3
"""Recompute scene-transfer results and the overview figure."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def run(*args: str) -> None:
    print("Running " + " ".join(args), flush=True)
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    subprocess.run([sys.executable, *args], cwd=ROOT, env=env, check=True)


def compare(actual: dict, expected: dict) -> int:
    """Compare every reported primary metric and source-choice summary."""
    checked = 0
    def visit(a, e, path):
        nonlocal checked
        if isinstance(e, dict):
            for key, value in e.items():
                visit(a[key], value, path + "/" + key)
        elif isinstance(e, (float, int)) and not isinstance(e, bool):
            if not math.isclose(a, e, rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError(f"Result differs at {path}: {a} versus {e}")
            checked += 1
        elif a != e:
            raise ValueError(f"Result differs at {path}: {a} versus {e}")
    for outcome, item in expected["outcomes"].items():
        produced = actual["outcomes"][outcome]
        visit(produced["primary_summary"], item["primary_summary"], outcome)
        for rule, data in item["fractions"]["0.20"]["decision_consequence"]["rules"].items():
            visit(produced["fractions"]["0.20"]["decision_consequence"]["rules"][rule]["summary"],
                  data["summary"], outcome + "/" + rule)
    return checked


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--references", action="store_true",
                        help="Also recompute item-level failure references and risk ranking")
    args = parser.parse_args()
    (ROOT / "artifacts").mkdir(exist_ok=True)
    total = 0
    for name, data in (("development", "data"), ("followup", "data/confirmation")):
        output = f"artifacts/{name}.json"
        run("analysis/transfer.py", "--primary", f"{data}/primary_checkpoint.json",
            "--mechanism", f"{data}/mechanism.json", "--output", output,
            "--report", f"artifacts/{name}.md")
        run("analysis/validate.py", "--primary", f"{data}/primary_checkpoint.json",
            "--mechanism", f"{data}/mechanism.json", "--corrected", output,
            "--output", f"artifacts/{name}-validation.json")
        total += compare(json.loads((ROOT / output).read_text()),
                         json.loads((ROOT / f"results/{name}.json").read_text()))
        if args.references:
            run("analysis/reference_models.py", "--mechanism", f"{data}/mechanism.json",
                "--target-table", f"{data}/target_table.parquet",
                "--target-sensitivity-table", f"{data}/target_sensitivity.parquet",
                "--corrected", output, "--output", f"artifacts/{name}-references.json",
                "--summary", f"artifacts/{name}-references.md")
    run("analysis/figures.py", "--results", "artifacts", "--output", "artifacts/figures")
    print(json.dumps({"numeric_results_reproduced": total,
                      "development_cells": 216, "followup_cells": 108,
                      "figures": "artifacts/figures"}, indent=2))


if __name__ == "__main__":
    main()
