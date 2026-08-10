"""
Orchestration script for in silico knockout comparisons.

Runs COMPASS twice — first on the KO model to obtain ``v_r^opt``, then on the
baseline with the KO's ``v_r^opt`` injected so that both scores share the same
normalization reference.  Then outputs a diff table.

Usage::

    compass-ko --ko-reactions KO.txt --data expr.tsv \\
               --species homo_sapiens [any compass args]

The ``--ko-reactions`` file contains one **directional** reaction ID per line
(e.g. ``MAR09048_pos``).  An optional second column sets a partial upper-bound
(default is ``0.0`` for a full KO).

Output layout::

    output_dir/
    ├── baseline/          ← baseline COMPASS output (scored with KO v_r^opt)
    ├── ko/                ← KO COMPASS output
    └── ko_diff.tsv        ← baseline − KO per reaction per cell
"""
from __future__ import print_function, division, absolute_import

import json
import logging
import os
import sys
from unittest.mock import patch

import pandas as pd
from tqdm import tqdm

import compass.main
from compass import globals
from compass.models import init_model
from compass.models.knockout import (
    KoSpec,
    load_ko_reactions,
    generate_ko_media,
    find_model_media_dir,
    validate_ko_spec,
)
from compass.compass import cache
from compass.compass.cache import PREPROCESS_CACHE_DIR
from compass.globals import MODEL_DIR, EXCHANGE_LIMIT
import compass.utils as utils


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def _parse_full_args():
    """Parse all Compass args plus KO args.

    Re-uses the shared ``build_parser`` from ``compass.main`` so that every
    Compass argument is automatically forwarded — no manual argv surgery needed.
    """
    parser = compass.main.build_parser(prog="compass-ko")
    parser.add_argument(
        "--ko-reactions",
        required=True,
        help="Text file with one directional reaction ID per line "
             "(e.g. MAR09048_pos).  Optional second column sets the "
             "reduced upper-bound (default 0.0 = full KO).",
        metavar="FILE",
    )
    args, unknown = parser.parse_known_args()
    if unknown:
        logger = logging.getLogger("compass")
        logger.warning("Unrecognized arguments ignored: %s", unknown)
    args = vars(args)  # Convert to dict, matching compass.main.parseArgs()
    return args


# ---------------------------------------------------------------------------
# Core orchestration
# ---------------------------------------------------------------------------


def load_scores(output_dir, filename="reactions.tsv"):
    """Load reaction scores from a COMPASS output directory.

    Handles both plain output directories and meta-subsystem layouts
    where scores live under <output_dir>/<meta_subsystem>/reactions.tsv.
    If multiple meta-subsystem subdirectories are found the scores are
    concatenated (reactions from different subsystems are non-overlapping).
    """
    path = os.path.join(output_dir, filename)
    if os.path.exists(path):
        return pd.read_csv(path, sep="\t", index_col=0)
    # Try reactions.txt (single-sample format)
    path = os.path.join(output_dir, "reactions.txt")
    if os.path.exists(path):
        return pd.read_csv(path, sep="\t", index_col=0)

    # Meta-subsystem layout: look for subdirectories containing the file
    scores = []
    for entry in sorted(os.listdir(output_dir)):
        sub_dir = os.path.join(output_dir, entry)
        if os.path.isdir(sub_dir):
            sub_path = os.path.join(sub_dir, filename)
            if os.path.exists(sub_path):
                scores.append(pd.read_csv(sub_path, sep="\t", index_col=0))
            else:
                sub_path_txt = os.path.join(sub_dir, "reactions.txt")
                if os.path.exists(sub_path_txt):
                    scores.append(pd.read_csv(sub_path_txt, sep="\t", index_col=0))

    if scores:
        return pd.concat(scores)
    return pd.DataFrame()


def create_ko_media(ko_spec, args):
    """Generate KO media JSON from the baseline media + KO spec."""
    media_dir = find_model_media_dir(args["model"], MODEL_DIR)
    base_media_path = os.path.join(media_dir, f"{args['media']}.json")

    if not os.path.exists(base_media_path):
        raise FileNotFoundError(
            f"Baseline media file not found: {base_media_path}\n"
            f"Make sure --media '{args['media']}' exists for model '{args['model']}'."
        )

    ko_media_name = f"{args['media']}_ko"
    ko_media_path = os.path.join(media_dir, f"{ko_media_name}.json")

    # Only create if not already present
    if not os.path.exists(ko_media_path):
        generate_ko_media(ko_spec, base_media_path, ko_media_path)

    return ko_media_name, ko_media_path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def entry():
    """Main entry point for compass-ko."""
    args = _parse_full_args()

    if args["ko_reactions"] is None:
        print("Error: --ko-reactions is required", file=sys.stderr)
        sys.exit(1)

    # Resolve output directories
    output_dir = args["output_dir"]
    baseline_output_dir = os.path.join(output_dir, "baseline")
    baseline_temp_dir = os.path.join(baseline_output_dir, "_tmp")
    ko_output_dir = os.path.join(output_dir, "ko")
    ko_temp_dir = os.path.join(ko_output_dir, "_tmp")

    os.makedirs(baseline_output_dir, exist_ok=True)
    os.makedirs(baseline_temp_dir, exist_ok=True)
    os.makedirs(ko_output_dir, exist_ok=True)
    os.makedirs(ko_temp_dir, exist_ok=True)

    # Initialize logger
    globals.init_logger(output_dir)
    logger = __import__('logging').getLogger('compass')

    # ── Validate KO spec ──
    logger.info("Loading KO reaction list from: %s", args["ko_reactions"])
    ko_spec = load_ko_reactions(args["ko_reactions"])
    logger.info("KO spec: %d reaction(s) blocked", len(ko_spec.ko_reactions))

    # Validate against model
    model = init_model(
        model=args["model"],
        species=args["species"],
        exchange_limit=EXCHANGE_LIMIT,
        media=args["media"],
        isoform_summing=args.get("isoform_summing", "remove-summing"),
    )
    reaction_ids = set(model.reactions.keys())
    unknown = validate_ko_spec(ko_spec, reaction_ids)
    if unknown:
        logger.warning(
            "The following KO reaction IDs were not found in the model and will be skipped: %s",
            unknown
        )

    # ── Generate KO media ──
    logger.info("Generating KO media...")
    ko_media_name, ko_media_path = create_ko_media(ko_spec, args)
    logger.info("KO media written to: %s", ko_media_path)

    # ── Phase 1: KO run (computes KO v_r^opt) ──
    logger.info("=" * 60)
    logger.info("PHASE 1: Running KO COMPASS (KO reactions blocked)")
    logger.info("=" * 60)
    logger.info("KO media: %s", ko_media_name)
    logger.info("Output: %s", ko_output_dir)

    ko_argv = [
        "compass",
        "--output-dir", ko_output_dir,
        "--temp-dir", ko_temp_dir,
        "--data", args["data"][0] if isinstance(args["data"], list) else args["data"],
        "--model", args["model"],
        "--species", args["species"],
        "--media", ko_media_name,
        "--precache",
    ]

    # Forward relevant args
    if args.get("num_processes") is not None:
        ko_argv.extend(["--num-processes", str(args["num_processes"])])
    if args.get("lambda") is not None and args["lambda"] != 0:
        ko_argv.extend(["--lambda", str(args["lambda"])])
    if args.get("num_neighbors") is not None:
        ko_argv.extend(["--num-neighbors", str(args["num_neighbors"])])
    if args.get("calc_metabolites"):
        ko_argv.append("--calc-metabolites")
    if args.get("penalty_diffusion") is not None:
        ko_argv.extend(["--penalty-diffusion", args["penalty_diffusion"]])
    if args.get("isoform_summing") is not None:
        ko_argv.extend(["--isoform-summing", args["isoform_summing"]])
    if args.get("select_meta_subsystems"):
        ko_argv.extend(["--select-meta-subsystems", args["select_meta_subsystems"]])
    if args.get("select_reactions"):
        ko_argv.extend(["--select-reactions", args["select_reactions"]])
    if args.get("select_subsystems"):
        ko_argv.extend(["--select-subsystems", args["select_subsystems"]])

    with patch.object(sys, 'argv', ko_argv):
        compass.main.entry()

    # ── Load KO cache (v_r^opt) ──
    logger.info("Loading KO v_opt cache...")

    # Module-Compass stores the cache in output_dir/meta_subsystem_cache/META_ID/media_name/
    # Determine the correct cache directory and model names
    ko_cache_dir = None
    ko_model_names = None
    if args.get("select_meta_subsystems"):
        ko_cache_dir = os.path.join(ko_output_dir, "meta_subsystem_cache")
        # Parse meta-subsystem names from the file
        with open(args["select_meta_subsystems"]) as f:
            text = [line.strip() for line in f.readlines()]
        ko_model_names = []
        for line in text:
            if line:
                ko_model_names.append(line.split(':')[0].strip())

    # Load cache for each meta-subsystem model
    ko_cache = {}
    if ko_model_names is not None:
        for meta_model_name in ko_model_names:
            ko_cache.update(cache.load(
                meta_model_name,
                media=ko_media_name,
                preprocess_cache_dir=ko_cache_dir,
            ))
    else:
        ko_cache = cache.load(
            args["model"],
            media=ko_media_name,
            preprocess_cache_dir=ko_cache_dir if ko_cache_dir else PREPROCESS_CACHE_DIR,
        )
    logger.info("KO cache has %d entries", len(ko_cache))

    if len(ko_cache) == 0:
        logger.error(
            "KO cache is empty. The KO COMPASS run may have failed. "
            "Check %s/compass.log for details.", ko_output_dir
        )
        sys.exit(1)

    # ── Phase 2: Baseline run (uses KO v_r^opt for comparison) ──
    logger.info("=" * 60)
    logger.info("PHASE 2: Running baseline COMPASS (no KO, with KO v_r^opt)")
    logger.info("=" * 60)
    logger.info("Output: %s", baseline_output_dir)

    baseline_argv = [
        "compass",
        "--output-dir", baseline_output_dir,
        "--temp-dir", baseline_temp_dir,
        "--data", args["data"][0] if isinstance(args["data"], list) else args["data"],
        "--model", args["model"],
        "--species", args["species"],
        "--media", args["media"],
        "--ko-baseline-cache-media", ko_media_name,
    ]

    # Pass the KO cache directory so the baseline can load the v_opt from it
    if ko_cache_dir is not None:
        baseline_argv.extend(["--ko-baseline-cache-dir", ko_cache_dir])

    # Forward relevant args
    if args.get("num_processes") is not None:
        baseline_argv.extend(["--num-processes", str(args["num_processes"])])
    if args.get("lambda") is not None and args["lambda"] != 0:
        baseline_argv.extend(["--lambda", str(args["lambda"])])
    if args.get("num_neighbors") is not None:
        baseline_argv.extend(["--num-neighbors", str(args["num_neighbors"])])
    if args.get("calc_metabolites"):
        baseline_argv.append("--calc-metabolites")
    if args.get("penalty_diffusion") is not None:
        baseline_argv.extend(["--penalty-diffusion", args["penalty_diffusion"]])
    if args.get("isoform_summing") is not None:
        baseline_argv.extend(["--isoform-summing", args["isoform_summing"]])
    if args.get("select_meta_subsystems"):
        baseline_argv.extend(["--select-meta-subsystems", args["select_meta_subsystems"]])
    if args.get("select_reactions"):
        baseline_argv.extend(["--select-reactions", args["select_reactions"]])
    if args.get("select_subsystems"):
        baseline_argv.extend(["--select-subsystems", args["select_subsystems"]])

    with patch.object(sys, 'argv', baseline_argv):
        compass.main.entry()

    # ── Compare ──
    logger.info("=" * 60)
    logger.info("Comparing baseline vs KO scores")
    logger.info("=" * 60)

    baseline_scores = load_scores(baseline_output_dir)
    ko_scores = load_scores(ko_output_dir)

    if baseline_scores.empty or ko_scores.empty:
        logger.error("Could not load scores from baseline or KO output.")
        sys.exit(1)

    # Align on common reactions
    common_idx = baseline_scores.index.intersection(ko_scores.index)
    baseline_aligned = baseline_scores.loc[common_idx]
    ko_aligned = ko_scores.loc[common_idx]

    # Compute diff
    diff = ko_aligned.subtract(baseline_aligned, fill_value=0)

    diff_path = os.path.join(output_dir, "ko_diff.tsv")
    diff.to_csv(diff_path, sep="\t")
    logger.info("KO diff written to: %s", diff_path)

    # Summary stats
    n_reactions = len(diff.index)
    n_cells = len(diff.columns)
    mean_abs_diff = diff.abs().mean().mean()
    logger.info(
        "Summary: %d reactions x %d cells, mean |KO - baseline| = %.4f",
        n_reactions, n_cells, mean_abs_diff
    )

    # Highlight KO'd reactions
    logger.info("\nKO reaction scores (baseline → KO):")
    for rxn_id, ub in ko_spec.ko_reactions.items():
        if rxn_id in diff.index:
            bl_vals = baseline_scores.loc[rxn_id].dropna()
            ko_vals = ko_scores.loc[rxn_id].dropna()
            logger.info(
                "  %s (bound=%.1f): baseline=%.3f → KO=%.3f (Δ=%.3f)",
                rxn_id, ub,
                bl_vals.mean() if len(bl_vals) > 0 else float('nan'),
                ko_vals.mean() if len(ko_vals) > 0 else float('nan'),
                (ko_vals - bl_vals).mean() if len(ko_vals) > 0 and len(bl_vals) > 0 else float('nan'),
            )

    logger.info("\nIn silico KO comparison complete.")


if __name__ == "__main__":
    entry()