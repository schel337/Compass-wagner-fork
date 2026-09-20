#!/usr/bin/env python3
"""
Measure wall-clock runtimes of the integration-test pipelines and write them
to tests/reference/timings.json as baselines.

Run locally (on a reasonably loaded machine) after algorithm changes to
update the baselines:

    python scripts/regen_baselines.py

The baselines are used by test_integration_submodel.py to detect
regressions with a generous 3x factor (CI hardware varies).
"""
import json
import os
import subprocess
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASELINE_FILE = os.path.join(REPO_ROOT, "tests", "reference", "timings.json")
DATA_FILE = os.path.join(REPO_ROOT, "tests", "data", "synthetic_expression.tsv")


def time_run(name, *extra_args):
    out_dir = f"/tmp/baseline_{name}_{int(time.time())}"
    cmd = [
        sys.executable, "-m", "compass.main",
        "--data", DATA_FILE,
        "--species", "homo_sapiens",
        "--model", "SubModel_cc",
        "--optimizer", "highs",
        "--output-dir", out_dir,
        "--temp-dir", os.path.join(out_dir, "_tmp"),
        "--num-processes", "2",
    ] + list(extra_args)
    start = time.time()
    result = subprocess.run(
        cmd, cwd=REPO_ROOT, input="y\n" * 100, text=True,
        stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    elapsed = time.time() - start
    assert result.returncode == 0, f"{name} failed"
    print(f"{name}: {elapsed:.1f}s")
    return elapsed


def main():
    timings = {
        "single_sample": time_run("single", "--single-sample", "1"),
        "turbo": time_run("turbo", "--turbo", "0.9",
                          "--turbo-increments", "0.1", "--turbo-max-iters", "2"),
    }
    os.makedirs(os.path.dirname(BASELINE_FILE), exist_ok=True)
    with open(BASELINE_FILE, "w") as f:
        json.dump(timings, f, indent=2)
    print(f"Wrote {BASELINE_FILE}")


if __name__ == "__main__":
    main()
