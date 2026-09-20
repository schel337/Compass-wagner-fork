#!/usr/bin/env python3
"""
Paired in-silico metabolic knockout (KO) experiment.

Simulates a KO (e.g. a mouse with a metabolic gene deleted) and isolates its
effect with two Compass runs on the same cells:

    Run A (with KO):   Compass run with the KO reactions blocked (zero flux
                       in every LP). v_r^opt is therefore computed with the
                       KO active and stored in Run A's preprocess cache.
    Run B (no KO):     Compass run on the unblocked model in which the
                       per-reaction max problems are NOT solved; the v_r^opt
                       values from Run A (exported to a TSV) are used as the
                       high-activity targets. The activity targets are
                       therefore identical across the two runs, so the only
                       difference between them is the blocked reactions
                       themselves (apples to apples).

Steps:
    1. Run A:  compass --block-reactions KO --precache (isolated cache dir)
    2. Export v_r^opt from Run A's cache -> <out>/activity_v_r_opt.tsv
    3. Run B:  compass --activity-file <out>/activity_v_r_opt.tsv
    4. Compare: <out>/paired_comparison.tsv (side-by-side scores + delta)

Example:
    python scripts/ko_paired_run.py \
        --data tests/data/synthetic_expression.tsv \
        --model SubModel_cc --species homo_sapiens \
        --block-reactions my_ko.txt \
        --output-dir ko_run --optimizer gurobi
"""
import argparse
import os
import shutil
import subprocess
import sys

import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _ensure_gurobi_license_for_main(optimizer):
    """
    compass/main.py requires a license file inside the compass package dir
    when running with --optimizer gurobi. If the user has dropped a
    (gitignored) gurobi.lic in the repo root, copy it there.
    """
    if optimizer != "gurobi":
        return
    compass_lic = os.path.join(REPO_ROOT, "compass", "gurobi.lic")
    if not os.path.exists(compass_lic):
        root_lic = os.path.join(REPO_ROOT, "gurobi.lic")
        if os.path.exists(root_lic):
            shutil.copy(root_lic, compass_lic)


def run_compass(cmd, timeout):
    """Runs a compass.main subprocess (see test_integration_submodel)."""
    print("[ko_paired_run] Running:", " ".join(cmd), flush=True)
    result = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        input="y\n" * 100,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout,
    )
    tail = result.stdout[-5000:]
    if result.returncode != 0 or "COMPASS Completed Successfully" not in result.stdout:
        print(tail)
        raise RuntimeError("Compass run failed (see output above)")
    print(tail[-2000:])
    return result


def find_reactions_file(out_dir):
    """Locates the reaction score matrix of a completed run."""
    for candidate in (os.path.join(out_dir, "reactions.tsv"),
                      os.path.join(out_dir, "_tmp", "reactions.txt")):
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(f"No reaction scores found under {out_dir} "
                            "(expected reactions.tsv or _tmp/reactions.txt)")


def main():
    parser = argparse.ArgumentParser(
        description="Paired in-silico metabolic knockout experiment "
                    "(KO run + comparison run with matched v_r^opt activities)")
    parser.add_argument("--data", required=True, help="Gene expression matrix (tsv/mtx)")
    parser.add_argument("--model", required=True, help="Metabolic model (e.g. SubModel_cc)")
    parser.add_argument("--species", required=True, choices=["homo_sapiens", "mus_musculus"])
    parser.add_argument("--media", default="default-media")
    parser.add_argument("--block-reactions", required=True,
                        help="KO file: one reaction ID per line; undirected IDs block both directions")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--optimizer", default="gurobi", choices=["gurobi", "highs", "cuopt"])
    parser.add_argument("--num-processes", type=int, default=4)
    parser.add_argument("--lambda", type=float, default=0, dest="lambda_",
                        help="Smoothing factor (default 0)")
    parser.add_argument("--single-sample", type=int, default=None,
                        help="Restrict both runs to a single sample index")
    parser.add_argument("--calc-metabolites", action="store_true",
                        help="Also compute and compare uptake/secretion scores")
    parser.add_argument("--timeout", type=int, default=7200,
                        help="Per-run subprocess timeout (seconds)")
    args = parser.parse_args()

    out = os.path.abspath(args.output_dir)
    os.makedirs(out, exist_ok=True)
    _ensure_gurobi_license_for_main(args.optimizer)

    ko_file = os.path.abspath(args.block_reactions)
    activity_file = os.path.join(out, "activity_v_r_opt.tsv")
    a_cache_dir = os.path.join(out, "runA_preprocess_cache")
    b_cache_dir = os.path.join(out, "runB_preprocess_cache")
    a_out = os.path.join(out, "with_ko")
    b_out = os.path.join(out, "no_ko")

    base_cmd = [
        sys.executable, "-m", "compass.main",
        "--data", os.path.abspath(args.data),
        "--model", args.model,
        "--species", args.species,
        "--media", args.media,
        "--optimizer", args.optimizer,
        "--num-processes", str(args.num_processes),
        "--lambda", str(args.lambda_),
    ]
    if args.single_sample is not None:
        base_cmd += ["--single-sample", str(args.single_sample)]
    if args.calc_metabolites:
        base_cmd += ["--calc-metabolites"]

    # ---- Step 1: Run A (with KO) ------------------------------------------
    print(f"\n[1/4] Run A: Compass WITH the KO ({ko_file})", flush=True)
    run_compass(
        base_cmd + [
            "--block-reactions", ko_file,
            "--precache",
            "--preprocess-cache-dir", a_cache_dir,
            "--output-dir", a_out,
            "--temp-dir", os.path.join(a_out, "_tmp"),
        ],
        timeout=args.timeout,
    )

    # ---- Step 2: export v_r^opt from Run A's cache ------------------------
    print(f"\n[2/4] Exporting v_r^opt from Run A's cache -> {activity_file}", flush=True)
    sys.path.insert(0, REPO_ROOT)
    from compass.compass.activity import export_activity_file
    export_activity_file(model=args.model, media=args.media, out_path=activity_file,
                         preprocess_cache_dir=a_cache_dir)

    # Seed Run B's isolated cache dir with Run A's cache (auditability: the
    # comparison run carries exactly the activity values it was given).
    src_cache = os.path.join(a_cache_dir, args.model, args.media, "preprocess.json")
    dst_cache_dir = os.path.join(b_cache_dir, args.model, args.media)
    os.makedirs(dst_cache_dir, exist_ok=True)
    shutil.copy(src_cache, os.path.join(dst_cache_dir, "preprocess.json"))

    # ---- Step 3: Run B (no KO, Run A's activities) -------------------------
    print("\n[3/4] Run B: Compass WITHOUT the KO, using Run A's v_r^opt", flush=True)
    run_compass(
        base_cmd + [
            "--activity-file", activity_file,
            "--preprocess-cache-dir", b_cache_dir,
            "--output-dir", b_out,
            "--temp-dir", os.path.join(b_out, "_tmp"),
        ],
        timeout=args.timeout,
    )

    # ---- Step 4: compare ----------------------------------------------------
    print("\n[4/4] Comparing the paired runs", flush=True)
    a_file = find_reactions_file(a_out)
    b_file = find_reactions_file(b_out)
    scores_a = pd.read_csv(a_file, sep="\t", index_col=0).iloc[:, 0]
    scores_b = pd.read_csv(b_file, sep="\t", index_col=0).iloc[:, 0]
    scores_a.name, scores_b.name = "with_ko", "no_ko"
    scores_a.index.name = "reaction"

    comparison = pd.concat([scores_a, scores_b], axis=1)
    comparison["delta"] = comparison["no_ko"] - comparison["with_ko"]
    comparison.to_csv(os.path.join(out, "paired_comparison.tsv"), sep="\t")

    n_diff = int((comparison["delta"].abs() > 1e-9).sum())
    max_diff = float(comparison["delta"].abs().max())
    print(f"  Reactions compared:        {len(comparison)}")
    print(f"  Reactions with delta:      {n_diff}")
    print(f"  Max |delta|:               {max_diff:.6g}")
    if n_diff > 0:
        print("  Largest score changes (no_ko - with_ko):")
        print(comparison.reindex(comparison["delta"].abs().sort_values(ascending=False)
                                 .index).head(20).to_string())
    # Report the KO reactions themselves
    ko_ids = [line.strip() for line in open(ko_file)
              if line.strip() and not line.strip().startswith("#")]
    print("\n  KO reactions (scores should be 0 in both runs when blocked):")
    for rxn in ko_ids:
        for d in (rxn, rxn + "_pos", rxn + "_neg"):
            if d in comparison.index:
                print(f"    {d}: with_ko={comparison.loc[d, 'with_ko']:.6g} "
                      f"no_ko={comparison.loc[d, 'no_ko']:.6g}")

    print(f"\nDone. Outputs in {out}:")
    print("  with_ko/          Run A scores")
    print("  no_ko/            Run B scores")
    print("  activity_v_r_opt.tsv   v_r^opt used by Run B (from Run A's cache)")
    print("  paired_comparison.tsv  side-by-side scores + delta")


if __name__ == "__main__":
    main()
