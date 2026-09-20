"""
Tests for the Gurobi/HiGHS optimizer backends against the tiny chain model,
whose reaction maxima are analytically known (see build_chain_model in conftest).
"""
import numpy as np
import pytest

from compass.opt.base import LinearProgramDelta


def test_chain_model_shape(tiny_model):
    """The chain model unidirectionalizes to 6 reactions, all with lower bound 0."""
    assert set(tiny_model.reactions) == {
        'E1_pos', 'R1_pos', 'R2_pos', 'R2_neg', 'R3_pos', 'E4_pos'
    }
    for r in tiny_model.reactions.values():
        assert r.lower_bound == 0.0
    # Exchanges were limited to the exchange limit; internal reactions kept their bounds
    assert tiny_model.reactions['E1_pos'].upper_bound == 1.0
    assert tiny_model.reactions['R1_pos'].upper_bound == 10.0


@pytest.mark.parametrize("rxn_id, expected_max", [
    ('E1_pos', 1.0),
    ('R1_pos', 1.0),
    ('R2_pos', 10.0),  # R2 / R2_neg form a flux cycle, so its own bound applies
    ('R2_neg', 10.0),
    ('R3_pos', 1.0),
    ('E4_pos', 1.0),
])
def test_reaction_maxima(optimizer, rxn_id, expected_max):
    """Reaction maxima match the analytic ground truth of the chain model."""
    delta = LinearProgramDelta(objective={rxn_id: 1.0}, sense="max")
    sol = optimizer.solve(delta)
    assert sol.success, f"solver failed with status {sol.status}"
    assert sol.obj_value == pytest.approx(expected_max, abs=1e-6)


@pytest.mark.parametrize("met_id", ['M1', 'M2', 'M3', 'M4'])
def test_metabolite_uptake_maxima(optimizer, met_id):
    """The maximal uptake of every metabolite is 1.0 (exchange limit)."""
    delta = LinearProgramDelta(
        objective={f"{met_id}_UPTAKE": 1.0},
        sense="max",
        added_uptake={met_id: f"{met_id}_UPTAKE"},
    )
    sol = optimizer.solve(delta)
    assert sol.success, f"solver failed with status {sol.status}"
    assert sol.obj_value == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize("met_id", ['M1', 'M2', 'M3', 'M4'])
def test_metabolite_secretion_maxima(optimizer, met_id):
    """The maximal secretion of every metabolite is 1.0 (exchange limit)."""
    delta = LinearProgramDelta(
        objective={f"{met_id}_SECRETION": 1.0},
        sense="max",
        added_secretion={met_id: f"{met_id}_SECRETION"},
    )
    sol = optimizer.solve(delta)
    assert sol.success, f"solver failed with status {sol.status}"
    assert sol.obj_value == pytest.approx(1.0, abs=1e-6)


def test_blocked_reactions(optimizer):
    """Blocking E1 (the only source of M1) shuts R1_pos down to 0."""
    delta = LinearProgramDelta(
        objective={'R1_pos': 1.0},
        sense="max",
        blocked_reactions={'E1_pos'},
    )
    sol = optimizer.solve(delta)
    assert sol.success, f"solver failed with status {sol.status}"
    # M1 balance is E1_pos - R1_pos = 0; with E1_pos blocked, R1_pos must be 0.
    assert sol.obj_value == pytest.approx(0.0, abs=1e-9)


def test_delta_is_tabula_rasa(optimizer):
    r"""
    The core invariant of the Optimizer interface: after solving a delta, the
    model must be back to its pristine base state, so repeated solves of the
    same base objective give the same answer and the model size is unchanged.
    """
    def base_obj():
        delta = LinearProgramDelta(
            objective={'R1_pos': 1.0, 'R2_pos': 1.0, 'R3_pos': 1.0},
            sense="max",
        )
        sol = optimizer.solve(delta)
        assert sol.success
        return sol.obj_value

    n_vars_before = _n_vars(optimizer)
    obj_before = base_obj()

    # A delta that touches everything: added uptake, added secretion, blocked
    # reactions and a high-flux constraint.
    delta = LinearProgramDelta(
        objective={'R1_pos': 1.0},
        sense="max",
        added_uptake={'M3': 'M3_UPTAKE'},
        added_secretion={'M2': 'M2_SECRETION'},
        blocked_reactions={'E4_pos'},
        high_flux={'R2_pos': 0.5},
    )
    sol = optimizer.solve(delta)
    assert sol.success

    # Model must be back to base state.
    assert _n_vars(optimizer) == n_vars_before
    assert base_obj() == pytest.approx(obj_before, abs=1e-9)


def _n_vars(optimizer):
    """Number of variables currently in the backend model (test helper)."""
    if hasattr(optimizer, "highs_model"):
        return optimizer.highs_model.getNumCol()
    if hasattr(optimizer, "gp_model"):
        return optimizer.gp_model.NumVars
    if hasattr(optimizer, "problem"):
        return optimizer.problem.numVars
    raise TypeError("Unknown optimizer backend")


def test_infeasible_delta_reports_failure(optimizer):
    """A delta that makes the LP infeasible must report success=False."""
    # R1_pos >= 5 is infeasible: the whole chain (and thus every reaction
    # out of M1/M2) is limited to 1.0 by the exchange reactions.
    delta = LinearProgramDelta(
        objective={'R1_pos': 1.0},
        sense="max",
        high_flux={'R1_pos': 5.0},
    )
    sol = optimizer.solve(delta)
    assert not sol.success
    # And the model is still usable afterwards.
    sol2 = optimizer.solve(LinearProgramDelta(objective={'E1_pos': 1.0}, sense="max"))
    assert sol2.success
    assert sol2.obj_value == pytest.approx(1.0, abs=1e-6)
