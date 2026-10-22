#!/usr/bin/env python3
"""
Paired in-silico metabolic knockout (KO) experiment.

Simulates a KO (e.g. a mouse with a metabolic gene deleted) and isolates its
effect with two Compass runs on the same cells:

    Run A (with KO):   Compass run with the KO reactions blocked (zero flux
                       in every LP). v_r^opt is therefore computed with the
                       KO active and stored in Run A's preprocess cache.
    Run B (no KO):     Compass run on the unblocked model in which the
                       per-reaction max problems are NOT solved; Run A's
                       v_r^opt values are used as the high-activity targets.
                       Standard Compass passes them through an activity TSV.
                       Module-Compass copies each module's cache so targets
                       remain scoped to that module. Blocked directions in
                       Run B receive the configured patch value.

Steps:
    1. Run A:  compass --block-reactions KO --precache (isolated cache dir)
    2. Export KO-derived v_r^opt and prepare patched control targets.
    3. Run B:  use the patched activity TSV (standard) or per-module caches
    4. Compare: <out>/paired_comparison.tsv (side-by-side scores + delta)

Example:
    python scripts/ko_paired_run.py \
        --data tests/data/synthetic_expression.tsv \
        --model SubModel_cc --species homo_sapiens \
        --block-reactions my_ko.txt \
        --output-dir ko_run --optimizer gurobi

    Module-Compass example:
    python scripts/ko_paired_run.py \
        --data cells.tsv --model Mouse1 --species mus_musculus \
        --select-meta-subsystems lipid_modules.txt \
        --block-reactions scap_ko.txt --output-dir ko_modules
"""
import argparse
import json
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


def read_module_names(path):
    """Return meta-subsystem IDs in the format accepted by Module-Compass."""
    names = []
    with open(path) as fin:
        for lineno, line in enumerate(fin, 1):
            line = line.strip()
            if not line:
                continue
            if ":" not in line:
                raise ValueError(
                    f"Malformed meta-subsystem selection on line {lineno} of {path}: "
                    "expected 'MODULE_ID: subsystem; subsystem'")
            name = line.split(":", 1)[0].strip()
            if not name:
                raise ValueError(f"Missing module ID on line {lineno} of {path}")
            names.append(name)
    if not names:
        raise ValueError(f"Meta-subsystem selection file {path} contains no modules")
    if len(names) != len(set(names)):
        raise ValueError(f"Meta-subsystem selection file {path} contains duplicate module IDs")
    return names


def write_activity_audit(path, rows, module_scoped=False):
    """Write raw or patched activity values as an auditable TSV."""
    with open(path, "w") as fout:
        if module_scoped:
            fout.write("# module\treaction_id\tv_r_opt\n")
            for module, reaction_id, value in rows:
                fout.write(f"{module}\t{reaction_id}\t{value}\n")
        else:
            fout.write("# reaction_id\tv_r_opt\n")
            for reaction_id, value in rows:
                fout.write(f"{reaction_id}\t{value}\n")


def seed_module_caches(source_root, destination_root, module_names, media,
                       raw_audit_path, control_audit_path, blocked_reactions,
                       patch_value):
    """Copy KO caches into Run B, patching blocked targets in the copies.

    Module cache keys are scoped by model name, so copying each cache keeps
    targets distinct when reaction IDs occur in multiple modules. Missing
    directions are ignored because a reaction may be one-way or absent from a
    selected module; the KO file is not validated against each partition.
    """
    from compass.compass.activity import patch_activity_values

    raw_rows = []
    control_rows = []
    patched_count = 0
    for module in module_names:
        relative_path = os.path.join(module, media, "preprocess.json")
        source = os.path.join(source_root, relative_path)
        if not os.path.isfile(source):
            raise FileNotFoundError(
                f"Run A did not produce the expected activity cache for module "
                f"{module!r}: {source}")
        with open(source) as fin:
            values = json.load(fin)
        if not values:
            raise ValueError(f"Run A's activity cache is empty for module {module!r}: {source}")

        control_values, patched_reactions = patch_activity_values(
            values, blocked_reactions, patch_value)
        patched_count += len(patched_reactions)

        destination = os.path.join(destination_root, relative_path)
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        with open(destination, "w") as fout:
            json.dump(control_values, fout, indent=1, sort_keys=True)
        raw_rows.extend((module, reaction_id, value)
                        for reaction_id, value in sorted(values.items()))
        control_rows.extend((module, reaction_id, value)
                            for reaction_id, value in sorted(control_values.items()))

    write_activity_audit(raw_audit_path, raw_rows, module_scoped=True)
    write_activity_audit(control_audit_path, control_rows, module_scoped=True)
    return len(raw_rows), patched_count


def find_module_reactions_file(out_dir, module):
    """Find a module's reaction matrix in batch or single-sample output."""
    candidates = (
        os.path.join(out_dir, module, "reactions.tsv"),
        os.path.join(out_dir, "_tmp", module, "reactions.txt"),
    )
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    raise FileNotFoundError(
        f"No reaction scores found for module {module!r} under {out_dir}; "
        f"looked for {candidates[0]} and {candidates[1]}")


def compare_module_scores(a_out, b_out, module_names, output_path):
    """Write all module/cell comparisons in long form and return summary stats."""
    n_pairs = 0
    n_diff = 0
    max_diff = 0.0
    write_header = True
    for module in module_names:
        scores_a = pd.read_csv(find_module_reactions_file(a_out, module), sep="\t", index_col=0)
        scores_b = pd.read_csv(find_module_reactions_file(b_out, module), sep="\t", index_col=0)
        if not scores_a.index.equals(scores_b.index) or not scores_a.columns.equals(scores_b.columns):
            raise ValueError(
                f"Run A and Run B score axes differ for module {module!r}; "
                "the paired comparison requires matching reaction IDs and samples")

        # Retain every cell. Write bounded reaction chunks so large modules
        # don't require constructing a multi-million-row table in memory.
        sample_names = scores_a.columns.tolist()
        for start in range(0, len(scores_a), 128):
            stop = min(start + 128, len(scores_a))
            chunk_a = scores_a.iloc[start:stop]
            chunk_b = scores_b.iloc[start:stop]
            comparison = pd.DataFrame({
                "module": module,
                "reaction": chunk_a.index.repeat(len(sample_names)),
                "sample": sample_names * len(chunk_a),
                "with_ko": chunk_a.to_numpy().ravel(),
                "no_ko": chunk_b.to_numpy().ravel(),
            })
            comparison["delta"] = comparison["no_ko"] - comparison["with_ko"]
            deltas = comparison["delta"].abs()
            n_pairs += len(comparison)
            n_diff += int((deltas > 1e-9).sum())
            max_diff = max(max_diff, float(deltas.max()))
            comparison.to_csv(output_path, sep="\t", index=False,
                              mode="w" if write_header else "a", header=write_header)
            write_header = False

    return n_pairs, n_diff, max_diff


def main():
    parser = argparse.ArgumentParser(
        description="Paired in-silico metabolic knockout experiment "
                    "(KO run + comparison run with matched v_r^opt activities)")
    parser.add_argument("--data", required=True, help="Gene expression matrix (tsv/mtx)")
    parser.add_argument("--model", required=True, help="Metabolic model (e.g. SubModel_cc)")
    parser.add_argument("--species", required=True, choices=["homo_sapiens", "mus_musculus"])
    parser.add_argument("--media", default="default-media")
    parser.add_argument("--select-meta-subsystems", default=None,
                        help="Module-Compass selection file (one module definition per line)")
    parser.add_argument("--block-reactions", required=True,
                        help="KO file: one reaction ID per line; undirected IDs block both directions")
    parser.add_argument("--patch-value", type=float, default=1.0,
                        help="Control activity assigned to blocked directions (default: 1.0)")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--optimizer", default="gurobi", choices=["gurobi", "highs", "cuopt"])
    parser.add_argument("--num-processes", type=int, default=4)
    parser.add_argument("--num-neighbors", type=int, default=None,
                        help="Neighbor count for penalty diffusion (Compass default: 30)")
    parser.add_argument("--penalty-diffusion", choices=["gaussian", "knn"], default=None,
                        help="Penalty diffusion mode (Compass default: knn)")
    parser.add_argument("--input-knn", default=None,
                        help="Precomputed sample neighbor graph passed to both Compass runs")
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
    sys.path.insert(0, REPO_ROOT)
    from compass.compass.activity import patch_activity_values, read_block_reactions_file
    blocked_reactions = read_block_reactions_file(ko_file)
    module_selection = (os.path.abspath(args.select_meta_subsystems)
                        if args.select_meta_subsystems else None)
    module_names = read_module_names(module_selection) if module_selection else None
    activity_file = os.path.join(out, "activity_v_r_opt.tsv")
    control_activity_file = os.path.join(out, "control_activity_v_r_opt.tsv")
    a_cache_dir = os.path.join(out, "runA_preprocess_cache")
    b_cache_dir = os.path.join(out, "runB_preprocess_cache")
    a_out = os.path.join(out, "with_ko")
    b_out = os.path.join(out, "no_ko")
    if os.path.exists(a_out) or os.path.exists(b_out):
        raise FileExistsError(
            f"Paired-run output already contains a with_ko/ or no_ko/ directory: {out}. "
            "Choose a new output directory so previous scores or caches cannot be reused.")

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
    if module_selection:
        base_cmd += ["--select-meta-subsystems", module_selection]
    if args.num_neighbors is not None:
        base_cmd += ["--num-neighbors", str(args.num_neighbors)]
    if args.penalty_diffusion is not None:
        base_cmd += ["--penalty-diffusion", args.penalty_diffusion]
    if args.input_knn:
        base_cmd += ["--input-knn", os.path.abspath(args.input_knn)]
    if args.single_sample is not None:
        base_cmd += ["--single-sample", str(args.single_sample)]
    if args.calc_metabolites:
        base_cmd += ["--calc-metabolites"]

    # ---- Step 1: Run A (with KO) ------------------------------------------
    print(f"\n[1/4] Run A: Compass WITH the KO ({ko_file})", flush=True)
    run_a_cmd = base_cmd + [
        "--block-reactions", ko_file,
        "--precache",
        "--output-dir", a_out,
        "--temp-dir", os.path.join(a_out, "_tmp"),
    ]
    if not module_selection:
        run_a_cmd += ["--preprocess-cache-dir", a_cache_dir]
    run_compass(run_a_cmd, timeout=args.timeout)

    # ---- Step 2: export raw KO activities and prepare control targets -----
    print("\n[2/4] Exporting KO activities and preparing control targets", flush=True)
    if module_selection:
        # Module-Compass keeps a cache per partitioned model. Preserve that
        # namespace when passing Run A's targets to Run B; a flat activity
        # mapping cannot distinguish duplicate IDs across modules. Keep raw
        # KO values in the audit file and patch only the copied control cache.
        source_cache_root = os.path.join(a_out, "meta_subsystem_cache")
        destination_cache_root = os.path.join(b_out, "meta_subsystem_cache")
        n_activities, n_patched = seed_module_caches(
            source_cache_root, destination_cache_root, module_names, args.media,
            activity_file, control_activity_file, blocked_reactions, args.patch_value)
        print(f"  Exported {n_activities} module-scoped activities", flush=True)
        print(f"  Patched {n_patched} cache entries to {args.patch_value} for Run B", flush=True)
    else:
        from compass.compass.activity import export_activity_file
        export_activity_file(model=args.model, media=args.media, out_path=activity_file,
                             preprocess_cache_dir=a_cache_dir)

        # Seed Run B's isolated cache dir from Run A, patching only the
        # blocked directions. Keep the raw export above unchanged for audit.
        src_cache = os.path.join(a_cache_dir, args.model, args.media, "preprocess.json")
        dst_cache_dir = os.path.join(b_cache_dir, args.model, args.media)
        os.makedirs(dst_cache_dir, exist_ok=True)
        with open(src_cache) as fin:
            raw_values = json.load(fin)
        control_values, patched_ids = patch_activity_values(
            raw_values, blocked_reactions, args.patch_value)
        with open(os.path.join(dst_cache_dir, "preprocess.json"), "w") as fout:
            json.dump(control_values, fout, indent=1, sort_keys=True)
        write_activity_audit(
            control_activity_file, sorted(control_values.items()), module_scoped=False)
        print(f"  Patched {len(patched_ids)} cache entries to {args.patch_value} for Run B",
              flush=True)

    # ---- Step 3: Run B (no KO, Run A's activities) -------------------------
    print("\n[3/4] Run B: Compass WITHOUT the KO, using Run A's v_r^opt", flush=True)
    run_b_cmd = base_cmd + [
        "--output-dir", b_out,
        "--temp-dir", os.path.join(b_out, "_tmp"),
    ]
    if not module_selection:
        run_b_cmd += ["--activity-file", control_activity_file,
                      "--preprocess-cache-dir", b_cache_dir]
    run_compass(run_b_cmd, timeout=args.timeout)

    # ---- Step 4: compare ----------------------------------------------------
    print("\n[4/4] Comparing the paired runs", flush=True)
    if module_selection:
        n_pairs, n_diff, max_diff = compare_module_scores(
            a_out, b_out, module_names, os.path.join(out, "paired_comparison.tsv"))
        print(f"  Reaction-cell pairs compared: {n_pairs}")
        print(f"  Pairs with delta:             {n_diff}")
        print(f"  Max |delta|:                   {max_diff:.6g}")
    else:
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
        print("\n  KO reactions (Run A is blocked; Run B uses patched activity targets):")
        for reaction_id in sorted(blocked_reactions):
            if reaction_id in comparison.index:
                print(f"    {reaction_id}: with_ko={comparison.loc[reaction_id, 'with_ko']:.6g} "
                      f"no_ko={comparison.loc[reaction_id, 'no_ko']:.6g}")

    print(f"\nDone. Outputs in {out}:")
    print("  with_ko/          Run A scores")
    print("  no_ko/            Run B scores")
    if module_selection:
        print("  activity_v_r_opt.tsv           raw KO-derived per-module activities")
        print("  control_activity_v_r_opt.tsv   patched control targets (per module)")
    else:
        print("  activity_v_r_opt.tsv           raw KO-derived v_r^opt values")
        print("  control_activity_v_r_opt.tsv   patched values used by Run B")
    print("  paired_comparison.tsv  side-by-side scores + delta")


if __name__ == "__main__":
    main()
