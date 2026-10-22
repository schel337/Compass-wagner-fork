"""
Tests for in-silico metabolic knockout (KO) simulation
(--block-reactions / --activity-file, see scripts/ko_paired_run.py).

The feature
==========
A paired KO experiment compares two runs on the same cells:

* **Run A (with KO):** the KO reactions are blocked in *every* LP of the
  run, so the per-reaction v_r^opt maximizations reflect the KO'd network.
* **Run B (no KO):** the per-reaction maximizations are *not* solved;
  v_r^opt is taken from the activity file (Run A's values), so the
  "high metabolic activity" targets are identical across the pair and the
  only difference is the blocked reactions themselves.

Micro-tier tests run the two-stage pipeline (penalties -> scoring) in-
process on deterministic hand-built models (see conftest.build_chain_model
and build_branch_model below), where every consequence of a KO is
analytically knowable:

* The chain E1 -> R1 -> R2 <-> R3 -> E4 is killed by any KO, so blocking
  e.g. R3_pos makes every v_r^opt (and therefore every score) exactly 0.
* Forcing v_r >= f on any chain reaction forces the whole chain (mass
  balance), so an activity value of f gives a score of
  f * (3 * p_int + 2 * p_E) for the five exchange-limited reactions.
* The branch model E1 -> R1 -> M2 -> {R2 -> R3 -> E4 | R5 -> E5} proves
  the core paired-run semantics: KO'ing R2_pos lowers v_r^opt of the
  R2/R3/E4 arm to 0 in run A, and in run B (unblocked, run A's activity
  values) the min-penalty LPs of E1 and R1 reroute their M2 absorption
  through the cheaper R2 arm - so delta = BETA * (p_R5 + p_E5 -
  p_R2 - p_R3 - p_E4) > 0 for exactly those two reactions.

The slow-tier test drives the full CLI wrapper (scripts/ko_paired_run.py)
on the central-carbon submodel; it validates the plumbing end to end
(subprocesses, cache dirs, file formats, logging) and asserts the
strongest sanity checks on the KO outputs available there: the KO
reaction (a live reaction with baseline v_r^opt = 1000) is zeroed in
the exported activity file while control reactions keep 1000, and the
KO changes the scores of at least one coupled reaction between the two
runs (a zero delta everywhere would mean the KO had no effect).
"""
import json
import os
import subprocess
import sys
from unittest import mock

import pandas as pd
import pytest

import compass.global_state as global_state
from compass import utils
from compass.compass import algorithm, cache
from compass.compass.activity import (
    export_activity_file,
    patch_activity_values,
    read_activity_file,
    read_block_reactions_file,
)
from compass.compass.algorithm import singleSampleCompass
from compass.compass.penalties import eval_reaction_penalties
from compass.globals import BETA, EXCHANGE_LIMIT
from compass.main import parseArgs, save_single_sample_penalties
from compass.models.MetabolicModel import (
    MetabolicModel,
    Reaction,
    Species,
    Gene,
    Association,
)

RXNS = ['E1_pos', 'R1_pos', 'R2_pos', 'R2_neg', 'R3_pos', 'E4_pos']
# Exchange-added reactions the scoring stage maximizes for the chain model
# (M1 has an uptake but no secretion rxn; M4 the reverse)
EXTRA_UPTAKE_SECRETION = ['M1_SECRETION', 'M4_UPTAKE']
# Media name the chain model carries when no media is loaded
# (MetabolicModel's default); it forms the preprocess-cache key
CACHE_MEDIA = 'NoMedia'
P_INT = 1.0 / 3.0  # internal-reaction penalty in CELL_EXPRESSED
P_UNEX = 1.0       # inaccessible / unassociated-reaction penalty


def _fresh_cache_state(model, media):
    """
    Resets the cache module's process-global state for (model, media) to
    what a *fresh process* would have: an empty in-memory cache plus a
    pending save so that on-the-fly values reach disk. Needed because
    run_scoring simulates several independent runs (with different cache
    dirs) inside one process, and the in-memory cache is keyed by
    (model, media) only.
    """
    key = (model, media)
    cache._cache.pop(key, None)
    cache._new_cache.pop(key, None)


# singleSampleCompass skips a sample whose directory already contains a
# success_token, so each simulated run needs its own sample directories
_run_counter = 0


def run_scoring(expression_file, tmp_path, args, solver, cache_dir):
    """
    Runs penalties + per-sample scoring on the chain model (mirrors the
    regular non-turbo compass_work flow), into a fresh cache dir.

    The cache module keys its in-memory cache by (model, media) only, so
    the entry is cleared before every sample: otherwise a previous
    "run" in the same process would silently supply its v_r^opt values.
    """
    args = dict(args)
    args['optimizer'] = solver

    global_state.init_selected_reactions_for_each_cell(None)

    global _run_counter
    _run_counter += 1
    # Per-run sample root: success_tokens from an earlier simulated run
    # must not make singleSampleCompass skip this run's samples
    sample_root = os.path.join(args['temp_dir'], f'run{_run_counter}')
    temp_dir = args['temp_dir']
    os.makedirs(temp_dir, exist_ok=True)

    penalties = eval_reaction_penalties(
        [expression_file], args['model'], None, args['species'], args)
    penalties.to_csv(os.path.join(temp_dir, 'penalties.txt.gz'),
                     sep='\t', compression='gzip')
    os.makedirs(args['penalties_dir'], exist_ok=True)
    for i in range(penalties.shape[1]):
        save_single_sample_penalties(i, temp_dir, penalties.iloc[:, i])

    data = utils.read_data([expression_file])
    sample_names, sample_dirs = [], {}
    for i, sample_name in enumerate(list(data.columns)):
        directory = os.path.join(sample_root, 'sample' + str(i))
        os.makedirs(directory, exist_ok=True)
        _fresh_cache_state(args['model'], CACHE_MEDIA)
        singleSampleCompass(
            data=[expression_file], model=args['model'], media=None,
            directory=directory, sample_name=sample_name, sample_index=i,
            args=args,
            preprocess_cache_dir=str(cache_dir),
        )
        sample_names.append(sample_name)
        sample_dirs[sample_name] = directory

    return sample_names, sample_dirs


def load_cache(cache_dir, model_name='TinyChain'):
    with open(os.path.join(str(cache_dir), model_name, CACHE_MEDIA,
                           'preprocess.json')) as f:
        return json.load(f)


def build_branch_model():
    r"""
    Parallel-route variant of the chain model, built the same way as
    conftest.build_chain_model:

        E1 (uptake of M1) -> R1 -> M2 -> R2 -> R3 -> M4 -> E4 (secretion)
                               |        |
                               +-- R5 -> M5 -> E5 (secretion)

    R1, R3, R5, E1, E4, E5 are forward-only. After
    limitExchangeReactions(1.0) + make_unidirectional(), every
    exchange-limited reaction has maximal flux 1.0. With the expression
    penalties of make_chain_expression_df (GENE_A/B/C expressed in
    CELL_EXPRESSED -> penalty 1/3; R5 and the exchange reactions
    unexpressed -> penalty 1.0):

      * v_r^opt with R2_pos KO'd: E1/R1/R5/E5 = 1.0, R2/R3/E4 = 0.
      * Run B (unblocked) reroutes M2 absorption for r in {E1, R1}
        through the R2 arm when p_R2 + p_R3 + p_E4 < p_R5 + p_E5,
        i.e. in CELL_EXPRESSED (5/3 < 2) but not CELL_ZERO (3 > 2),
        giving delta = BETA * (2 - 5/3) = BETA / 3 for exactly those
        two reactions in CELL_EXPRESSED.
    """
    model = MetabolicModel('TinyBranch')

    def make_reaction(rid, lb, ub, reactants, products, gene=None):
        r = Reaction()
        r.id = rid
        r.lower_bound = lb
        r.upper_bound = ub
        r.reactants = dict(reactants)
        r.products = dict(products)
        if gene is not None:
            g = Gene()
            g.id = gene
            g.name = gene
            a = Association()
            a.type = 'gene'
            a.gene = g
            r.gene_associations = a
        return r

    model.reactions = {
        x.id: x for x in [
            make_reaction('E1', 0.0, 10.0, {}, {'M1': 1.0}),
            make_reaction('R1', 0.0, 10.0, {'M1': 1.0}, {'M2': 1.0},
                          gene='GENE_A'),
            make_reaction('R2', 0.0, 10.0, {'M2': 1.0}, {'M3': 1.0},
                          gene='GENE_B'),
            make_reaction('R3', 0.0, 10.0, {'M3': 1.0}, {'M4': 1.0},
                          gene='GENE_C'),
            make_reaction('E4', 0.0, 10.0, {'M4': 1.0}, {}),
            make_reaction('R5', 0.0, 10.0, {'M2': 1.0}, {'M5': 1.0}),
            make_reaction('E5', 0.0, 10.0, {'M5': 1.0}, {}),
        ]
    }
    model.species = {}
    for sp_id in ['M1', 'M2', 'M3', 'M4', 'M5']:
        sp = Species()
        sp.id = sp_id
        model.species[sp_id] = sp

    model.limitExchangeReactions(limit=EXCHANGE_LIMIT)
    model.make_unidirectional()
    return model


@pytest.fixture()
def branch_model_loader(monkeypatch):
    """Like conftest.chain_model_loader, but returns the branch model."""
    import compass.models
    from compass.globals import MODEL_DIR

    def load(name, species="homo_sapiens", metabolic_model_dir=MODEL_DIR):
        return build_branch_model()

    monkeypatch.setattr(compass.models, "load_metabolic_model", load)
    return load


class TestBlockReactionsFile:
    def test_expansion(self, tmp_path):
        path = tmp_path / "ko.txt"
        path.write_text("# comment line\nR1\n\nR2_pos\nR3_neg\nR1\n")
        assert read_block_reactions_file(str(path)) == {
            'R1_pos', 'R1_neg', 'R2_pos', 'R3_neg'}

    def test_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            read_block_reactions_file(str(tmp_path / "nope.txt"))


class TestApplyBlockedReactions:
    def test_applies_zero_bounds_only_to_present_directions(self):
        model = build_chain_model()
        model.reactions['R2_pos'].lower_bound = 0.25

        applied = algorithm.apply_blocked_reactions(
            {'block_reactions': ['R2_pos', 'R2_neg', 'MISSING_pos']}, model)

        assert applied == {'R2_pos', 'R2_neg'}
        for reaction_id in ('R2_pos', 'R2_neg'):
            assert model.reactions[reaction_id].lower_bound == 0.0
            assert model.reactions[reaction_id].upper_bound == 0.0

    def test_ignores_absent_direction_of_one_way_reaction(self):
        model = build_branch_model()
        assert 'R2_pos' in model.reactions
        assert 'R2_neg' not in model.reactions

        applied = algorithm.apply_blocked_reactions(
            {'block_reactions': ['R2_pos', 'R2_neg']}, model)

        assert applied == {'R2_pos'}
        assert model.reactions['R2_pos'].lower_bound == 0.0
        assert model.reactions['R2_pos'].upper_bound == 0.0


class TestPatchActivityValues:
    def test_patches_present_directions_without_mutating_raw_values(self):
        raw = {'R1_pos': 0.0, 'R1_neg': 0.0, 'OTHER_pos': 2.5}

        control, patched = patch_activity_values(
            raw, {'R1_pos', 'R1_neg', 'MISSING_pos'}, patch_value=0.6)

        assert raw == {'R1_pos': 0.0, 'R1_neg': 0.0, 'OTHER_pos': 2.5}
        assert control == {'R1_pos': 0.6, 'R1_neg': 0.6, 'OTHER_pos': 2.5}
        assert patched == {'R1_pos', 'R1_neg'}

    def test_default_patch_value_is_one(self):
        control, patched = patch_activity_values({'R1_pos': 0.0}, {'R1_pos'})

        assert control == {'R1_pos': 1.0}
        assert patched == {'R1_pos'}


class TestActivityFile:
    def test_roundtrip(self, tmp_path):
        path = tmp_path / "activity.tsv"
        path.write_text("# reaction_id\tv_r_opt\nA\t1.5\n\nB\t-0.25\n")
        assert read_activity_file(str(path)) == {'A': 1.5, 'B': -0.25}

    @pytest.mark.parametrize("contents, message", [
        ("A\n", "Malformed line"),
        ("A\tbanana\n", "Non-numeric"),
        ("", "no reactions"),
        ("# only comments\n", "no reactions"),
    ])
    def test_malformed(self, tmp_path, contents, message):
        path = tmp_path / "activity.tsv"
        path.write_text(contents)
        with pytest.raises(ValueError, match=message):
            read_activity_file(str(path))

    def test_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            read_activity_file(str(tmp_path / "nope.tsv"))


class TestExportActivityFile:
    def test_export(self, tmp_path):
        cache_dir = tmp_path / "cache"
        model_dir = cache_dir / "ModelA" / "mediaX"
        model_dir.mkdir(parents=True)
        (model_dir / "preprocess.json").write_text(
            json.dumps({'r1': 1.0, 'r2': 0.0, 'r3': 2.5}))

        _fresh_cache_state('ModelA', 'mediaX')

        out = tmp_path / "activity.tsv"
        n = export_activity_file('ModelA', 'mediaX', str(out),
                                 preprocess_cache_dir=str(cache_dir))

        assert n == 3
        assert read_activity_file(str(out)) == {'r1': 1.0, 'r2': 0.0, 'r3': 2.5}

    def test_empty_cache(self, tmp_path):
        cache_dir = tmp_path / "cache"
        (cache_dir / "ModelA" / "mediaX").mkdir(parents=True)
        (cache_dir / "ModelA" / "mediaX" / "preprocess.json").write_text("{}")
        _fresh_cache_state('ModelA', 'mediaX')
        with pytest.raises(ValueError, match="empty"):
            export_activity_file('ModelA', 'mediaX',
                                 str(tmp_path / "a.tsv"),
                                 preprocess_cache_dir=str(cache_dir))


class TestParseArgs:
    def test_ko_flags(self, monkeypatch):
        monkeypatch.setattr(
            "sys.argv",
            ["compass", "--data", "x.tsv", "--species", "homo_sapiens",
             "--block-reactions", "ko.txt", "--activity-file", "act.tsv",
             "--preprocess-cache-dir", "cachedir"])
        args = parseArgs()
        assert args["block_reactions"] == "ko.txt"
        assert args["activity_file"] == "act.tsv"
        assert args["preprocess_cache_dir"] == "cachedir"

    @pytest.mark.parametrize("extra", ["--precache", "--generate-cache"])
    def test_activity_file_conflicts(self, monkeypatch, extra):
        monkeypatch.setattr(
            "sys.argv",
            ["compass", "--data", "x.tsv", "--species", "homo_sapiens",
             "--activity-file", "act.tsv", extra])
        with pytest.raises(SystemExit):
            parseArgs()


class TestKoScoring:
    """Run A of the pair: KO reactions blocked in every LP."""

    def test_ko_zeroes_v_r_opt_and_scores(self, solver, chain_model_loader,
                                          expression_file, compass_args,
                                          tmp_path):
        """
        Blocking R3 (undirected -> R3_pos; R3_neg does not exist and is
        ignored) kills the whole chain: every v_r^opt is 0, so every score
        is 0 and the (KO-applied) cache records all zeros.
        """
        args = dict(compass_args)
        args['block_reactions'] = ['R3_pos', 'R3_neg']
        cache_dir = tmp_path / "ko_cache"

        sample_names, sample_dirs = run_scoring(
            expression_file, tmp_path, args, solver, cache_dir)

        for sample_name in sample_names:
            scores = pd.read_csv(
                os.path.join(sample_dirs[sample_name], 'reactions.txt'),
                sep='\t', index_col=0).iloc[:, 0]
            assert set(scores.index) == set(RXNS)
            assert (scores == 0.0).all(), f"{sample_name}: {scores.to_dict()}"

        model_cache = load_cache(cache_dir)
        assert set(model_cache.keys()) >= set(RXNS)
        assert all(v == 0.0 for v in model_cache.values()), model_cache


class TestActivityScoring:
    """Run B of the pair: v_r^opt taken from an activity file."""

    def test_activity_reproduces_baseline_without_max_solves(
            self, solver, chain_model_loader, expression_file,
            compass_args, tmp_path):
        """
        The core apples-to-apples guarantee: a run whose activity file is
        exactly the (unblocked) baseline's v_r^opt values must reproduce
        the baseline scores while solving zero maximization problems.
        """
        # Baseline run (no overrides)
        _, baseline_dirs = run_scoring(
            expression_file, tmp_path, dict(compass_args), solver,
            tmp_path / "base_cache")
        baseline = {
            name: pd.read_csv(
                os.path.join(baseline_dirs[name], 'reactions.txt'),
                sep='\t', index_col=0).iloc[:, 0]
            for name in baseline_dirs}
        activities = {k: float(v)
                      for k, v in load_cache(tmp_path / "base_cache").items()}
        assert set(RXNS) <= set(activities.keys())

        # Run with the same activities; count maximization solves
        real_wrapper = algorithm.solve_model_wrapper
        n_max = {'n': 0}

        def counting(opt, delta):
            if delta.sense == 'max':
                n_max['n'] += 1
            return real_wrapper(opt, delta)

        args = dict(compass_args)
        args['activity_override'] = activities
        with mock.patch.object(algorithm, 'solve_model_wrapper', counting):
            _, activity_dirs = run_scoring(
                expression_file, tmp_path, args, solver,
                tmp_path / "act_cache")

        assert n_max['n'] == 0, "max problems must not be solved with an activity file"

        for name, expected in baseline.items():
            scores = pd.read_csv(
                os.path.join(activity_dirs[name], 'reactions.txt'),
                sep='\t', index_col=0).iloc[:, 0]
            pd.testing.assert_series_equal(
                scores.sort_index(), expected.sort_index(),
                check_names=False, rtol=1e-6, atol=1e-9)

    def test_activity_half_flux_scales_scores(self, solver,
                                              chain_model_loader,
                                              expression_file, compass_args,
                                              tmp_path):
        """
        Forcing v_r >= BETA * 0.5 on a chain reaction forces the whole
        chain to BETA * 0.5 (mass balance), so the score of the five
        exchange-limited reactions is (BETA * 0.5) * (3 * p_int + 2 * p_E)
        per cell. R2_neg (activity 0) short-circuits to 0.
        """
        activities = {
            'E1_pos': 0.5, 'R1_pos': 0.5, 'R2_pos': 0.5,
            'R2_neg': 0.0, 'R3_pos': 0.5, 'E4_pos': 0.5,
            'M1_SECRETION': 0.0, 'M4_UPTAKE': 0.0,
        }
        factor = BETA * 0.5
        args = dict(compass_args)
        args['activity_override'] = activities
        sample_names, sample_dirs = run_scoring(
            expression_file, tmp_path, args, solver, tmp_path / "act_cache")

        for sample_name in sample_names:
            p_int = P_INT if sample_name == 'CELL_EXPRESSED' else P_UNEX
            scores = pd.read_csv(
                os.path.join(sample_dirs[sample_name], 'reactions.txt'),
                sep='\t', index_col=0).iloc[:, 0]
            for rxn in RXNS:
                want = 0.0 if rxn == 'R2_neg' else factor * (3 * p_int + 2 * P_UNEX)
                assert scores[rxn] == pytest.approx(want, abs=1e-6), (
                    f"{sample_name}/{rxn}: got {scores[rxn]}, want {want}")

    def test_ko_paired_roundtrip(self, solver, chain_model_loader,
                                 expression_file, compass_args, tmp_path):
        """
        The full paired experiment on the chain model:
        Run A (KO) -> export activity from Run A's cache -> Run B (no KO,
        Run A's activities). The KO kills the whole chain, so Run A's
        activities are all zero and Run B scores everything at 0 while
        solving no max problems.
        """
        # Run A: KO
        args_a = dict(compass_args)
        args_a['block_reactions'] = ['R3_pos', 'R3_neg']
        run_scoring(expression_file, tmp_path, args_a, solver,
                    tmp_path / "runA_cache")

        # Export v_r^opt from Run A's (KO-applied) cache
        activity_path = tmp_path / "activity.tsv"
        export_activity_file('TinyChain', CACHE_MEDIA, str(activity_path),
                             preprocess_cache_dir=str(tmp_path / "runA_cache"))
        activities = read_activity_file(str(activity_path))
        assert set(RXNS) <= set(activities.keys())
        assert all(v == 0.0 for v in activities.values()), activities

        # Run B: no KO, Run A's activities
        args_b = dict(compass_args)
        args_b['activity_override'] = activities
        real_wrapper = algorithm.solve_model_wrapper
        n_max = {'n': 0}

        def counting(opt, delta):
            if delta.sense == 'max':
                n_max['n'] += 1
            return real_wrapper(opt, delta)

        with mock.patch.object(algorithm, 'solve_model_wrapper', counting):
            sample_names, sample_dirs = run_scoring(
                expression_file, tmp_path, args_b, solver,
                tmp_path / "runB_cache")

        assert n_max['n'] == 0
        for sample_name in sample_names:
            scores = pd.read_csv(
                os.path.join(sample_dirs[sample_name], 'reactions.txt'),
                sep='\t', index_col=0).iloc[:, 0]
            assert (scores == 0.0).all()


class TestKoDelta:
    """
    The paired experiment's payoff, on the branch model: the KO reaction
    (R2_pos) is the cheap way to absorb M2 in run B (unblocked, penalties
    of the expressed GENE_B/GENE_C arm), while run A (blocked) must use
    the expensive R5/E5 arm. For the reactions whose activity targets
    force M2 absorption (E1, R1) the paired scores therefore differ by
    exactly BETA * (p_R5 + p_E5 - p_R2 - p_R3 - p_E4) = BETA/3 in
    CELL_EXPRESSED and by 0 in CELL_ZERO (all penalties 1.0, so the R2
    arm costs 3 > 2 there).
    """

    KO = 'R2_pos'

    def test_ko_changes_paired_scores(self, solver, branch_model_loader,
                                      expression_file, compass_args,
                                      tmp_path):
        # Run A: KO blocked in every LP
        args_a = dict(compass_args)
        args_a['model'] = 'TinyBranch'
        args_a['optimizer'] = solver
        args_a['block_reactions'] = [self.KO]
        sample_names, dirs_a = run_scoring(
            expression_file, tmp_path, args_a, solver, tmp_path / "runA_cache")

        a_cache = load_cache(tmp_path / "runA_cache", 'TinyBranch')
        for rxn, want in {
            'E1_pos': 1.0, 'R1_pos': 1.0, 'R5_pos': 1.0, 'E5_pos': 1.0,
            'R2_pos': 0.0, 'R3_pos': 0.0, 'E4_pos': 0.0,
        }.items():
            assert a_cache[rxn] == pytest.approx(want, abs=1e-9), (
                f"run A v_r^opt {rxn}: got {a_cache[rxn]}, want {want}")

        scores_a = {
            name: pd.read_csv(os.path.join(d, 'reactions.txt'),
                              sep='\t', index_col=0).iloc[:, 0]
            for name, d in zip(sample_names, dirs_a.values())}

        # Run A's min-LPs used the expensive R5/E5 arm for M2 absorption
        assert scores_a['CELL_EXPRESSED']['E1_pos'] == pytest.approx(
            BETA * (1 + P_INT + 2.0), abs=1e-6)

        # Run B: no KO, run A's activity values, zero max solves
        args_b = dict(compass_args)
        args_b['model'] = 'TinyBranch'
        args_b['optimizer'] = solver
        args_b['activity_override'] = {k: float(v)
                                       for k, v in a_cache.items()}
        real_wrapper = algorithm.solve_model_wrapper
        n_max = {'n': 0}

        def counting(opt, delta):
            if delta.sense == 'max':
                n_max['n'] += 1
            return real_wrapper(opt, delta)

        with mock.patch.object(algorithm, 'solve_model_wrapper', counting):
            _, dirs_b = run_scoring(
                expression_file, tmp_path, args_b, solver,
                tmp_path / "runB_cache")
        assert n_max['n'] == 0

        scores_b = {
            name: pd.read_csv(os.path.join(d, 'reactions.txt'),
                              sep='\t', index_col=0).iloc[:, 0]
            for name, d in zip(sample_names, dirs_b.values())}

        for name in sample_names:
            # Wrapper convention: delta = with_ko (run A) - no_ko (run B);
            # positive where the unblocked network scores cheaper
            delta = scores_a[name] - scores_b[name]
            if name == 'CELL_EXPRESSED':
                # Rerouted reactions differ by exactly BETA/3 ...
                assert delta['E1_pos'] == pytest.approx(BETA / 3.0, abs=1e-6)
                assert delta['R1_pos'] == pytest.approx(BETA / 3.0, abs=1e-6)
                # ... and run B scored them via the cheap R2 arm
                assert scores_b[name]['E1_pos'] == pytest.approx(
                    BETA * 3.0, abs=1e-6)
                # ... while every other reaction has an identical score
                assert (delta.drop(['E1_pos', 'R1_pos']).abs() < 1e-6).all(), \
                    delta[delta.abs() > 1e-6]
            else:
                # CELL_ZERO: all penalties 1.0, the R2 arm (cost 3) is
                # more expensive than R5/E5 (cost 2) -> identical scores
                assert (delta.abs() < 1e-6).all(), \
                    delta[delta.abs() > 1e-6]


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FILE = os.path.join(REPO_ROOT, "tests", "data", "synthetic_expression.tsv")


@pytest.mark.slow
class TestKoPairedRunWrapper:
    """
    End-to-end paired KO experiment on the central-carbon submodel via
    scripts/ko_paired_run.py (subprocesses, full CLI plumbing:
    --block-reactions, --activity-file, --preprocess-cache-dir).

    This test validates the plumbing and performs the strongest sanity
    checks available on this model. The KO target is a *live* reaction
    (R_GLCGLUT2, glucose uptake, baseline v_r^opt = 1000), so the exported
    activity file proves through the full CLI path that the KO actually
    changed what Run A computed: the KO reaction's v_r^opt is 0 because
    its maximization LP was blocked, while control reactions (which the
    KO does not affect) keep their baseline 1000 - i.e. Run A's activity
    targets are identical to a no-KO baseline except for the KO itself.

    The score-level effect of the KO is asserted to be non-zero for at
    least one non-KO reaction: with the KO unblocked, Run B's per-cell
    penalty LPs can route flux through the KO reaction, changing the
    optimal penalties (scores) of coupled reactions (observed for
    R_GLCt4_pos and R_CITt4_4_neg on this model). A zero delta would
    indicate the KO had no effect on either leg.
    """

    KO = "R_GLCGLUT2"  # live reaction: baseline v_r^opt = 1000, unidirectional
    CONTROLS = ("R_L_LACt4r_pos", "R_PYRt2r_pos")  # live reactions, baseline 1000
    BASELINE_V_R_OPT = 1000.0

    def test_wrapper_end_to_end(self, tmp_path, gurobi_licensed):
        if not gurobi_licensed:
            pytest.skip("Gurobi not available or unlicensed")

        ko_file = tmp_path / "ko.txt"
        ko_file.write_text(self.KO + "\n")
        out_dir = str(tmp_path / "ko_run")

        result = subprocess.run(
            [
                sys.executable, "scripts/ko_paired_run.py",
                "--data", DATA_FILE,
                "--model", "SubModel_cc",
                "--species", "homo_sapiens",
                "--block-reactions", str(ko_file),
                "--output-dir", out_dir,
                "--optimizer", "gurobi",
                "--num-processes", "2",
                "--single-sample", "3",
                "--timeout", "3600",
            ],
            cwd=REPO_ROOT,
            input="y\n" * 100,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=7200,
        )
        assert result.returncode == 0, result.stdout[-10000:]

        # Run A log: the KO was applied to every LP ... the blocking log
        # line lists exactly the directed IDs present in the model ...
        with open(os.path.join(out_dir, "with_ko", "compass.log")) as f:
            log_a = f.read()
        blocking_lines = [ln for ln in log_a.splitlines() if "in-silico KO" in ln]
        assert any(f"{self.KO}_pos" in ln for ln in blocking_lines)
        assert all(f"{self.KO}_neg" not in ln for ln in blocking_lines)
        # ... and the nonexistent R_GLCGLUT2_neg direction was filtered
        # (dropped with a log message) against the model
        assert any("Ignoring" in ln and f"{self.KO}_neg" in ln
                   for ln in log_a.splitlines())

        # Run B log: activities were taken from the file, no max solves
        with open(os.path.join(out_dir, "no_ko", "compass.log")) as f:
            log_b = f.read()
        assert "activity_v_r_opt.tsv" in log_b and \
            "will not be solved" in log_b

        comparison = pd.read_csv(os.path.join(out_dir, "paired_comparison.tsv"),
                                 sep="\t", index_col=0)
        assert {'with_ko', 'no_ko', 'delta'} <= set(comparison.columns)

        # The KO reaction itself: its max LP was blocked in Run A, so
        # v_r^opt = 0 (baseline is 1000 - the activity file proves the
        # KO was active in Run A's maximization LPs, not a model trait)
        # and its score is 0 in both runs ...
        activities = pd.read_csv(os.path.join(out_dir, "activity_v_r_opt.tsv"),
                                 sep="\t", index_col=0, comment="#")
        assert abs(activities.loc[f"{self.KO}_pos"].iloc[0]) < 1e-9

        # ... while the control reactions kept their baseline activity:
        # the KO changed exactly one v_r^opt value, so Run B's activity
        # targets are identical to a no-KO baseline (paired comparison)
        for control in self.CONTROLS:
            assert abs(activities.loc[control].iloc[0] - self.BASELINE_V_R_OPT) < 1e-9

        assert comparison.loc[f"{self.KO}_pos", "with_ko"] == 0.0
        assert comparison.loc[f"{self.KO}_pos", "no_ko"] == 0.0
        for control in self.CONTROLS:
            assert abs(comparison.loc[control, "with_ko"] - comparison.loc[control, "no_ko"]) < 1e-9

        # ... and the KO changed the scores of at least one non-KO
        # reaction (Run B's penalty LPs can use the unblocked KO
        # reaction, shifting the optimal penalties of coupled reactions;
        # a fully zero delta would mean the KO had no effect on either
        # leg)
        n_diff = int((comparison["delta"].abs() > 1e-9).sum())
        assert n_diff >= 1, (
            "KO had no score-level effect on either run:\n"
            f"{comparison}")
