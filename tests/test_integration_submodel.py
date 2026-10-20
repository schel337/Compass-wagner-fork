"""
Integration tests on the bundled central-carbon submodel (SubModel_cc)
with the committed synthetic expression matrix (tests/data/synthetic_expression.tsv).

These exercise the full pipeline end-to-end, solver included, in a few
minutes rather than the ~30 minutes a full RECON2.2 sample takes. They are
marked ``slow`` and deselected by default (see pyproject.toml):

    pytest -m slow tests/test_integration_submodel.py

They also serve as the fastest feedback loop for tuning the algorithm on
a *real* (if small) metabolic model, which the unit/micro tiers cannot
provide.
"""
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FILE = os.path.join(REPO_ROOT, "tests", "data", "synthetic_expression.tsv")
MODEL_NAME = "SubModel_cc"
TIMINGS_FILE = os.path.join(REPO_ROOT, "tests", "reference", "timings.json")
# Order-of-magnitude guard: CI hardware and shared-node load vary, and this is
# a smoke test for accidentally doing substantially more work, not a benchmark.
# Regenerate baselines with scripts/regen_baselines.py.
TIMING_TOLERANCE = 10.0


def _ensure_gurobi_license_for_main():
    """
    compass/main.py requires a license file inside the compass package dir
    when running with --optimizer gurobi. If the user has dropped a
    (gitignored) gurobi.lic in the repo root, copy it there.
    """
    compass_lic = os.path.join(REPO_ROOT, "compass", "gurobi.lic")
    if not os.path.exists(compass_lic):
        root_lic = os.path.join(REPO_ROOT, "gurobi.lic")
        if os.path.exists(root_lic):
            shutil.copy(root_lic, compass_lic)


def run_compass(tmp_path, *extra_args, timeout=1200, optimizer="highs"):
    """
    Runs compass.main in a subprocess with a fresh temp dir.

    Feeds 'y' on stdin so the "arguments differ from cached run" prompt
    (triggered by the nested single-sample Compass runs inside the turbo
    oracle) never blocks.

    Returns (subprocess result, output dir).
    """
    if optimizer == "gurobi":
        _ensure_gurobi_license_for_main()
    out_dir = os.path.join(str(tmp_path), "out")
    cmd = [
        sys.executable, "-m", "compass.main",
        "--data", DATA_FILE,
        "--species", "homo_sapiens",
        "--model", MODEL_NAME,
        "--optimizer", optimizer,
        "--output-dir", out_dir,
        "--temp-dir", os.path.join(out_dir, "_tmp"),
        "--num-processes", "2",
    ] + list(extra_args)
    result = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        input="y\n" * 100,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout,
    )
    return result, out_dir


def _load_penalties(out_dir):
    return pd.read_csv(os.path.join(out_dir, "_tmp", "penalties.txt.gz"),
                       sep="\t", compression="gzip", index_col=0)


@pytest.mark.slow
class TestSingleSamplePipeline:
    def test_runs_and_writes_outputs(self, tmp_path):
        result, out_dir = run_compass(tmp_path, "--single-sample", "1")
        assert result.returncode == 0, result.stdout[-5000:]
        assert "COMPASS Completed Successfully" in result.stdout

        # Success token is written
        assert os.path.exists(os.path.join(out_dir, "_tmp", "success_token"))

        # Penalty matrix: one row per reaction, one column per cell
        penalties = _load_penalties(out_dir)
        assert penalties.shape[1] == 24  # number of synthetic cells
        assert penalties.shape[0] > 100  # submodel has ~190 unidirectional reactions
        # Penalties are bounded in [0, 1] and not all identical
        assert np.all(penalties.values >= 0)
        assert np.all(penalties.values <= 1)
        assert penalties.stack().nunique() > 10

        # Reaction scores exist and have some non-zero (expressed) reactions
        scores = pd.read_csv(os.path.join(out_dir, "_tmp", "reactions.txt"),
                             sep="\t", index_col=0).iloc[:, 0]
        assert len(scores) == penalties.shape[0]
        assert (scores > 0).any()
        assert (scores == 0).any()  # unexpressed reactions score 0

    def test_deterministic(self, tmp_path):
        """Two independent runs give identical reaction scores."""
        result1, out1 = run_compass(os.path.join(str(tmp_path), "a"),
                                    "--single-sample", "3")
        result2, out2 = run_compass(os.path.join(str(tmp_path), "b"),
                                    "--single-sample", "3")
        assert result1.returncode == 0, result1.stdout[-5000:]
        assert result2.returncode == 0, result2.stdout[-5000:]
        scores1 = pd.read_csv(os.path.join(out1, "_tmp", "reactions.txt"),
                              sep="\t", index_col=0).iloc[:, 0]
        scores2 = pd.read_csv(os.path.join(out2, "_tmp", "reactions.txt"),
                              sep="\t", index_col=0).iloc[:, 0]
        pd.testing.assert_series_equal(scores1, scores2)


@pytest.mark.slow
class TestTurboPipeline:
    def test_turbo_completes_and_imputes(self, tmp_path):
        # NB: --single-sample is incompatible with --turbo (the oracle matrix
        # would be 1 x C and no column could be sampled at these densities).
        # NB: --turbo-increments must be > 1/n_cells or no entries would be
        # sampled per column.
        result, out_dir = run_compass(
            tmp_path,
            "--turbo", "0.9",
            "--turbo-increments", "0.1",
            "--turbo-max-iters", "2",
            timeout=2400,
        )
        assert result.returncode == 0, result.stdout[-5000:]
        assert "Turbo Compass completed successfully!" in result.stdout

        imputed = pd.read_csv(os.path.join(out_dir, "reactions.tsv"),
                              sep="\t", index_col=0)
        assert imputed.shape == (190, 24) or imputed.shape[1] == 24
        assert imputed.shape[0] > 100
        # Imputation filled every entry
        assert not imputed.isnull().any().any()
        # Not a constant matrix
        assert imputed.stack().nunique() > 10


@pytest.mark.slow
class TestKoFeatureCombo:
    """
    In-silico KO feature (--block-reactions / --activity-file) combined
    with Turbo-Compass. Turbo's oracle re-runs the full compass.main
    entry point in-process for every observed matrix entry, so the KO
    flags must survive the round trip through the oracle's per-iteration
    argument re-parsing and its in-process cache state.

    The KO target is R_GLCGLUT2_pos (glucose uptake, a live reaction
    with baseline v_r^opt = 1000 in the committed SubModel_cc cache).
    """

    KO_RXN = "R_GLCGLUT2_pos"
    KO_SIBLING = "R_L_LACt4r_pos"  # live reaction, baseline v_r^opt = 1000
    BASELINE_V_R_OPT = 1000.0
    COMMITTED_CACHE = os.path.join(REPO_ROOT, "compass", "Resources", "COMPASS",
                                   MODEL_NAME, "default-media", "preprocess.json")

    def _turbo_args(self):
        return ("--turbo", "0.9", "--turbo-increments", "0.1",
                "--turbo-max-iters", "2")

    def test_turbo_with_activity_file(self, tmp_path):
        """Paired-run Run-B leg via Turbo: max problems are not solved."""
        # Activity file simulating Run A's KO'd cache: the baseline values
        # with the KO reaction zeroed.
        with open(self.COMMITTED_CACHE) as f:
            baseline = json.load(f)
        assert abs(baseline[self.KO_RXN] - self.BASELINE_V_R_OPT) < 1e-9
        act = tmp_path / "activity.tsv"
        with open(act, "w") as f:
            for rxn, val in sorted(baseline.items()):
                f.write(f"{rxn}\t{val if rxn != self.KO_RXN else 0.0}\n")

        result, out_dir = run_compass(tmp_path, *self._turbo_args(),
                                      "--activity-file", str(act),
                                      timeout=2400)
        assert result.returncode == 0, result.stdout[-5000:]
        assert "Turbo Compass completed successfully!" in result.stdout
        # The activity file was honored: cache check skipped in the oracle
        # and the per-reaction max problems were not solved in the
        # oracle's in-process compass runs
        assert "skipping cache check" in result.stdout
        assert "will not be solved" in result.stdout

        imputed = pd.read_csv(os.path.join(out_dir, "reactions.tsv"),
                              sep="\t", index_col=0)
        assert not imputed.isnull().any().any()
        # The KO reaction's activity target is 0, so it is skipped in
        # every oracle run: score 0 in every observed cell, and constant
        # columns are imputed exactly as their constant value
        assert (imputed.loc[self.KO_RXN].abs() < 1e-6).all()
        # ... while a control reaction keeps its normal activity target
        # and scores as usual
        assert (imputed.loc[self.KO_SIBLING] > 0).any()

    def test_turbo_with_block_reactions(self, tmp_path):
        """Paired-run Run-A leg via Turbo: KO blocked in every oracle LP."""
        ko = tmp_path / "ko.txt"
        ko.write_text("R_GLCGLUT2\n")
        cache_dir = str(tmp_path / "runA_cache")

        # Step 1: build Run A's (KO'd) cache in an isolated dir
        result, _ = run_compass(
            tmp_path / "precache",
            "--single-sample", "3", "--precache",
            "--block-reactions", str(ko),
            "--preprocess-cache-dir", cache_dir,
            timeout=1200)
        assert result.returncode == 0, result.stdout[-5000:]
        a_cache = json.load(open(os.path.join(cache_dir, MODEL_NAME,
                                              "default-media", "preprocess.json")))
        # The KO zeroed its own max LP; siblings are unaffected
        assert abs(a_cache[self.KO_RXN]) < 1e-9
        assert abs(a_cache[self.KO_SIBLING] - self.BASELINE_V_R_OPT) < 1e-9

        # Step 2: Turbo with the KO applied to every oracle LP, reading
        # Run A's cache
        result, out_dir = run_compass(
            tmp_path / "turbo",
            *self._turbo_args(),
            "--block-reactions", str(ko),
            "--preprocess-cache-dir", cache_dir,
            timeout=2400)
        assert result.returncode == 0, result.stdout[-5000:]
        assert "Turbo Compass completed successfully!" in result.stdout
        # The oracle's compass runs applied the KO to their LPs ...
        assert "in-silico KO" in result.stdout
        # ... and their max problems were solved (from Run A's cache),
        # not overridden
        assert "will not be solved" not in result.stdout

        imputed = pd.read_csv(os.path.join(out_dir, "reactions.tsv"),
                              sep="\t", index_col=0)
        assert not imputed.isnull().any().any()
        # KO reaction: blocked in every LP and its (KO'd) activity target
        # is 0 -> skipped -> 0 in every observed cell -> imputed row is 0
        assert (imputed.loc[self.KO_RXN].abs() < 1e-6).all()
        assert (imputed.loc[self.KO_SIBLING] > 0).any()


@pytest.mark.slow
class TestCrossSolverParity:
    """
    LP objective values are solver-independent: running the same pipeline
    with Gurobi and with HiGHS must yield (up to solver tolerances) the
    same reaction scores. This is the strongest end-to-end check that the
    HiGHS backend is a faithful substitute for Gurobi.
    """

    def test_gurobi_and_highs_agree(self, tmp_path, gurobi_licensed):
        if not gurobi_licensed:
            pytest.skip("Gurobi not available or unlicensed")
        res_h, out_h = run_compass(os.path.join(str(tmp_path), "h"),
                                   "--single-sample", "1", optimizer="highs")
        res_g, out_g = run_compass(os.path.join(str(tmp_path), "g"),
                                   "--single-sample", "1", optimizer="gurobi")
        assert res_h.returncode == 0, res_h.stdout[-5000:]
        assert res_g.returncode == 0, res_g.stdout[-5000:]
        h = pd.read_csv(os.path.join(out_h, "_tmp", "reactions.txt"),
                        sep="\t", index_col=0).iloc[:, 0]
        g = pd.read_csv(os.path.join(out_g, "_tmp", "reactions.txt"),
                        sep="\t", index_col=0).iloc[:, 0]
        assert h.index.equals(g.index)
        assert (h > 0).sum() == (g > 0).sum()  # same reactions are active
        # Tolerance ~1000x the solver tolerances (1e-9)
        assert (h - g).abs().max() < 1e-6, (
            f"max abs diff {(h - g).abs().max():.3e}")


@pytest.mark.slow
class TestTimingBaselines:
    """
    Wall-clock regression checks against tests/reference/timings.json.
    These catch algorithmic slowdowns (e.g. a penalty computation that
    accidentally runs twice), not small constant-factor changes.
    """

    def _baseline(self, key):
        if not os.path.exists(TIMINGS_FILE):
            pytest.skip("No timing baselines (run scripts/regen_baselines.py)")
        with open(TIMINGS_FILE) as f:
            return json.load(f)[key]

    def test_single_sample_timing(self, tmp_path):
        baseline = self._baseline("single_sample")
        start = time.time()
        result, _ = run_compass(tmp_path, "--single-sample", "1")
        elapsed = time.time() - start
        assert result.returncode == 0, result.stdout[-5000:]
        assert elapsed < baseline * TIMING_TOLERANCE, (
            f"single-sample run took {elapsed:.1f}s "
            f"(baseline {baseline:.1f}s)")

    def test_turbo_timing(self, tmp_path):
        baseline = self._baseline("turbo")
        start = time.time()
        result, _ = run_compass(
            tmp_path, "--turbo", "0.9",
            "--turbo-increments", "0.1", "--turbo-max-iters", "2",
            timeout=2400)
        elapsed = time.time() - start
        assert result.returncode == 0, result.stdout[-5000:]
        assert elapsed < baseline * TIMING_TOLERANCE, (
            f"turbo run took {elapsed:.1f}s (baseline {baseline:.1f}s)")
