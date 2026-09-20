"""
Micro-tier tests for the two-stage single-sample Compass pipeline:

    eval_reaction_penalties  ->  singleSampleCompass

run in-process on the deterministic chain model (see
``build_chain_model`` in conftest). This exercises the reaction-penalty
math and the LP-based scoring of ``algorithm.py`` — the code the
subprocess integration tests only cover at ~190-reaction scale.

Analytic ground truth
=====================
After unidirectionalization the chain model has
E1_pos (0..1), R1_pos, R2_pos, R2_neg, R3_pos, E4_pos (0..10), and every
reaction has max flux ``1.0`` in the directions the exchanges allow
(except R2_neg, whose partner-blocked max is 0, below).

With ``lambda = 0`` the reaction penalty of a reaction is
``1 / (1 + log2(1 + expr))`` where ``expr`` is the aggregated gene
expression; an unexpressed or gene-unassociated reaction therefore gets
the "inaccessible" penalty ``1.0``. In CELL_EXPRESSED (GENE_A/B/C = 3.0)
the four internal reactions get ``1/3`` and the exchanges get ``1.0``;
in CELL_ZERO everything is ``1.0``.

``singleSampleCompass`` scores each non-exchange reaction R by
minimizing the total penalty subject to ``v_R >= BETA * r_max``
(``BETA = 0.95``, ``r_max = 1.0`` for every exchange-limited reaction here)
with R's reverse blocked, and each exchange reaction by the analogous
max-uptake/max-secretion LPs of the exchange stage.

Mass balance (M1: E1 = R1; M2: R1 + R2_neg = R2_pos; M3: R2_pos = R2_neg + R3;
M4: R3 = E4) implies that forcing *any* one of E1, R1, R2_pos, R3 or E4 to
0.95 forces the whole chain: R1 = R2_pos = R3 = E1 = E4 = 0.95 and
R2_neg = 0 (any R2_neg flux would force R2_pos above R3). The minimal
total penalty is therefore, for every one of those five reactions,

    BETA * (3 * p_int + 2 * p_E)

= 0.95 * 3 = 2.85 (CELL_EXPRESSED) or 0.95 * 5 = 4.75 (CELL_ZERO).

R2_pos's reverse (R2_neg) has ``r_max = 0`` once its partner R2_pos is
blocked, so it is short-circuited to score 0 before any LP is solved.
"""
import os

import pandas as pd
import pytest

import compass.global_state as global_state
from compass import utils
from compass.compass.algorithm import singleSampleCompass
from compass.compass.penalties import eval_reaction_penalties
from compass.globals import BETA
from compass.main import save_single_sample_penalties

RXNS = ['E1_pos', 'R1_pos', 'R2_pos', 'R2_neg', 'R3_pos', 'E4_pos']
P_INT = 1.0 / 3.0  # internal-reaction penalty in CELL_EXPRESSED
P_UNEX = 1.0       # inaccessible / unassociated-reaction penalty


def expected_scores(expressed: bool) -> dict:
    """
    All five exchange-limited reactions force the whole chain (see module
docstring), so they share one score: BETA * (3*p_int + 2*p_E).
    R2_neg short-circuits to 0 (r_max = 0 with its partner blocked).
    """
    p_int = P_INT if expressed else P_UNEX
    value = BETA * (3.0 * p_int + 2.0 * P_UNEX)
    return {
        'R1_pos': value,
        'R2_pos': value,
        'R3_pos': value,
        'E1_pos': value,
        'E4_pos': value,
        'R2_neg': 0.0,
    }


def run_pipeline(expression_file, tmp_path, args, solver):
    """
    Runs both pipeline stages in-process on the chain model, mirroring the
    regular (non-turbo) compass_work flow of compass.main: penalties for
    all samples first, then per-sample scoring with one output directory
    per sample (the runCompassParallel layout).
    """
    args = dict(args)
    args['optimizer'] = solver

    # main.py calls this once at startup; required by solve_model_wrapper.
    global_state.init_selected_reactions_for_each_cell(None)

    temp_dir = args['temp_dir']
    os.makedirs(temp_dir, exist_ok=True)

    # Stage 1: reaction penalties for all samples
    penalties = eval_reaction_penalties(
        [expression_file], args['model'], None, args['species'], args)
    penalties.to_csv(os.path.join(temp_dir, 'penalties.txt.gz'),
                     sep='\t', compression='gzip')

    # Per-sample penalty files (main.py plumbing read by singleSampleCompass)
    os.makedirs(args['penalties_dir'], exist_ok=True)
    for i in range(penalties.shape[1]):
        save_single_sample_penalties(i, temp_dir, penalties.iloc[:, i])

    # Stage 2: per-sample scoring
    data = utils.read_data([expression_file])
    sample_names = list(data.columns)
    sample_dirs = {}
    for i, sample_name in enumerate(sample_names):
        directory = os.path.join(temp_dir, 'sample' + str(i))
        singleSampleCompass(
            data=[expression_file], model=args['model'], media=None,
            directory=directory, sample_name=sample_name, sample_index=i,
            args=args,
            preprocess_cache_dir=str(tmp_path / 'preprocess_cache'),
        )
        sample_dirs[sample_name] = directory

    return args, penalties, sample_names, sample_dirs


class TestReactionPenalties:
    """The penalty stage is solver-free and hand-computable."""

    def test_penalties_match_hand_computation(self, chain_model_loader,
                                              expression_file, compass_args):
        penalties = eval_reaction_penalties(
            [expression_file], 'TinyChain', None, 'homo_sapiens',
            compass_args)

        expected = pd.DataFrame({
            'CELL_EXPRESSED': {'R1_pos': P_INT, 'R2_pos': P_INT,
                               'R2_neg': P_INT, 'R3_pos': P_INT,
                               'E1_pos': P_UNEX, 'E4_pos': P_UNEX},
            'CELL_ZERO': {rxn: P_UNEX for rxn in RXNS},
        })

        assert set(penalties.index) == set(RXNS)
        assert list(penalties.columns) == list(expected.columns)
        pd.testing.assert_frame_equal(
            penalties.reindex(index=RXNS, columns=expected.columns),
            expected.reindex(index=RXNS, columns=expected.columns),
            check_exact=False, atol=1e-12, check_names=False)


class TestSingleSampleScoring:
    @pytest.fixture()
    def _pipeline(self, solver, chain_model_loader, expression_file,
                  compass_args, tmp_path):
        return run_pipeline(expression_file, tmp_path, compass_args, solver)

    def test_scores_match_hand_computation(self, _pipeline):
        _, _, sample_names, sample_dirs = _pipeline
        assert list(sample_names) == ['CELL_EXPRESSED', 'CELL_ZERO']

        for sample_name in sample_names:
            scores = pd.read_csv(
                os.path.join(sample_dirs[sample_name], 'reactions.txt'),
                sep='\t', index_col=0).iloc[:, 0]
            expected = expected_scores(sample_name == 'CELL_EXPRESSED')

            assert set(scores.index) == set(RXNS)
            for rxn, value in expected.items():
                assert scores[rxn] == pytest.approx(value, abs=1e-6), (
                    f"{sample_name}/{rxn}: got {scores[rxn]}, want {value}")

            assert os.path.exists(
                os.path.join(sample_dirs[sample_name], 'success_token'))

    def test_scores_ordering(self, _pipeline):
        """
        Expressed reactions carry lower total penalty than inaccessible
        ones, in every position of the chain.
        """
        _, _, sample_names, sample_dirs = _pipeline

        def scores_of(name):
            return pd.read_csv(
                os.path.join(sample_dirs[name], 'reactions.txt'),
                sep='\t', index_col=0).iloc[:, 0]

        expr, zero = scores_of('CELL_EXPRESSED'), scores_of('CELL_ZERO')
        for rxn in RXNS:
            assert expr[rxn] <= zero[rxn], (
                f"{rxn}: expressed {expr[rxn]} > unexpressed {zero[rxn]}")

    def test_second_run_is_noop(self, solver, chain_model_loader,
                                 expression_file, compass_args, tmp_path,
                                 _pipeline):
        """With a success_token present, singleSampleCompass must skip."""
        args, _, sample_names, sample_dirs = _pipeline
        directory = sample_dirs[sample_names[0]]
        reactions_file = os.path.join(directory, 'reactions.txt')
        mtime_before = os.path.getmtime(reactions_file)

        singleSampleCompass(
            data=[expression_file], model=args['model'], media=None,
            directory=directory, sample_name=sample_names[0], sample_index=0,
            args=args,
            preprocess_cache_dir=str(tmp_path / 'preprocess_cache'),
        )

        # The success token made the run a no-op: no file was rewritten
        assert os.path.getmtime(reactions_file) == mtime_before

    def test_penalty_stage_is_solver_free(self, chain_model_loader,
                                          expression_file, compass_args):
        """
        The penalty stage touches no solver: it must work with the
        optimizer set to one that is not installed (guaranteed-absent name).
        """
        args = dict(compass_args)
        args['optimizer'] = 'definitely-not-installed'
        penalties = eval_reaction_penalties(
            [expression_file], 'TinyChain', None, 'homo_sapiens', args)
        assert penalties.shape == (6, 2)
