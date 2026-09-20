"""
Shared fixtures for the Compass test suite.
"""
import os

import numpy as np
import pandas as pd
import pytest

from compass import globals as compass_globals
from compass.models.MetabolicModel import (
    MetabolicModel,
    Reaction,
    Species,
    Gene,
    Association,
)


def build_chain_model():
    r"""
    Builds a small, fully deterministic metabolic network:

        E1 (uptake of M1) -> R1 -> M2 -> R2 <-> M3 -> R3 -> M4 -> E4 (secretion)

    R1 and R3 are forward-only, R2 is bidirectional (R2 and its reverse form
    a flux cycle, so its maximal flux is its own bound of 10.0); exchange
    reactions E1/E4 are forward-only. After applying
    ``limitExchangeReactions(1.0)`` and ``make_unidirectional()``, every
    reaction that is limited by the exchanges has maximal flux exactly
    ``1.0``, and every metabolite uptake/secretion (bounded by the exchange
    limit) is exactly ``1.0``. This gives us analytic ground truth for
    optimizer / preprocessing tests without any LP guessing.
    """
    model = MetabolicModel('TinyChain')

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
            make_reaction('R1', 0.0, 10.0, {'M1': 1.0}, {'M2': 1.0}, gene='GENE_A'),
            make_reaction('R2', -10.0, 10.0, {'M2': 1.0}, {'M3': 1.0}, gene='GENE_B'),
            make_reaction('R3', 0.0, 10.0, {'M3': 1.0}, {'M4': 1.0}, gene='GENE_C'),
            make_reaction('E4', 0.0, 10.0, {'M4': 1.0}, {}),
        ]
    }
    model.species = {}
    for sp_id in ['M1', 'M2', 'M3', 'M4']:
        sp = Species()
        sp.id = sp_id
        model.species[sp_id] = sp

    # Bring the model into the same state Compass runs it in
    model.limitExchangeReactions(limit=1.0)
    model.make_unidirectional()
    return model


@pytest.fixture(scope="session")
def tiny_model():
    r"""
    Deterministic chain model, shared across the session.
    Optimizer solves must leave the model in its pristine state
    (tabula rasa invariant), so sharing is safe; the optimizer tests
    pin exactly that behavior.
    """
    return build_chain_model()


def make_chain_expression_df():
    r"""
    Generates the synthetic expression DataFrame for the chain model.

    Two cells are defined, with hand-computable reaction penalties
    (lambda=0, so penalty = 1 / (1 + log2(1 + expression))):

      CELL_EXPRESSED: GENE_A/B/C all expressed at 3.0
          -> log2(1+3) = 2 -> penalty 1/3 for R1_pos, R2_pos, R2_neg, R3_pos
      CELL_ZERO: everything unexpressed
          -> log2(1+0) = 0 -> penalty 1.0 (the "inaccessible" value)

    Reactions without gene associations (E1_pos, E4_pos) always get
    penalty 1.0. A filler gene verifies that genes not in the model
    are ignored.
    """
    return pd.DataFrame({
        "CELL_EXPRESSED": {"GENE_A": 3.0, "GENE_B": 3.0, "GENE_C": 3.0, "FILLER": 1.0},
        "CELL_ZERO": {"GENE_A": 0.0, "GENE_B": 0.0, "GENE_C": 0.0, "FILLER": 0.0},
    })


@pytest.fixture()
def expression_file(tmp_path):
    """The chain-model expression matrix as a TSV readable by utils.read_data."""
    path = tmp_path / "expression.tsv"
    make_chain_expression_df().to_csv(path, sep="\t")
    return str(path)


@pytest.fixture()
def chain_model_loader(monkeypatch):
    r"""
    Patches compass.models.load_metabolic_model so that init_model() returns
    a freshly built chain model regardless of the requested name. Both
    eval_reaction_penalties and singleSampleCompass load the model through
    init_model, so patching the loader exercises the full pipeline in-memory
    without a model on disk. Each call rebuilds the model, so the two
    pipeline stages (and different tests) cannot leak state into each other.
    """
    import compass.models
    from compass.globals import MODEL_DIR

    def load(name, species="homo_sapiens", metabolic_model_dir=MODEL_DIR):
        return build_chain_model()

    monkeypatch.setattr(compass.models, "load_metabolic_model", load)
    return load


@pytest.fixture()
def compass_args(tmp_path):
    r"""
    An args dict mirroring parseArgs() defaults (plus test-specific values)
    with every key read by the penalty stage (eval_reaction_penalties) and
    the scoring stage (singleSampleCompass) of compass.main.compass_work.

    lambda=0 means no TSNE/kNN sample blending: penalties are purely per-cell,
    which is what makes them hand-computable.
    """
    temp_dir = str(tmp_path / "temp")
    return {
        # penalty stage
        'lambda': 0,
        'num_neighbors': 30,
        'symmetric_kernel': False,
        'input_weights': None,
        'penalty_diffusion': 'knn',
        'and_function': 'mean',
        'input_knn': None,
        'output_knn': None,
        'latent_space': None,
        'num_processes': 1,
        # model / data
        'model': 'TinyChain',
        'species': 'homo_sapiens',
        'media': None,
        'isoform_summing': 'remove-summing',
        # scoring stage
        'no_reactions': False,
        'calc_metabolites': False,
        'test_mode': False,
        'detailed_perf': False,
        'generate_cache': False,
        'save_argmaxes': False,
        'save_argmaxes_dir': None,
        'select_reactions': None,
        'select_subsystems': None,
        # plumbing
        'temp_dir': temp_dir,
        'penalties_dir': os.path.join(temp_dir, 'penalties_tmp', 'penalties'),
        # optimizer (tests set this via parameterization)
        'optimizer': 'highs',
        'num_threads': None,
        'lpmethod': None,
    }


# Prefer the repo's WLS license if present. Without GRB_LICENSE_FILE the pip
# wheel's bundled size-limited license (in gurobipy/.libs) silently wins over
# the license in $HOME, capping models at 2000 vars / 200 quadratic vars.
_GUROBI_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_GUROBI_LIC = os.path.join(_GUROBI_ROOT, "gurobi.lic")
if os.path.isfile(_GUROBI_LIC):
    os.environ.setdefault("GRB_LICENSE_FILE", _GUROBI_LIC)


def solver_available(name: str) -> bool:
    r"""
    Checks that a solver can actually solve a trivial LP (i.e. it is
    installed *and* licensed). Used to skip LP-dependent tests gracefully.
    """
    try:
        if name == "highs":
            import highspy as h
            m = h.Highs()
            m.setOptionValue("output_flag", False)
            v = m.addVariable(0.0, 1.0, name="x")
            m.addConstr(v == 0.0)
            m.setObjective(v, h.ObjSense.kMaximize)
            m.solve()
            return m.getModelStatus() == h.HighsModelStatus.kOptimal
        elif name == "gurobi":
            import gurobipy as gp
            env = gp.Env(empty=True)
            env.start()
            m = gp.Model(env=env)
            v = m.addVar(lb=0.0, ub=1.0)
            m.setObjective(v, gp.GRB.MAXIMIZE)
            m.optimize()
            return m.status == gp.GRB.OPTIMAL
        else:
            return False
    except Exception:
        return False


@pytest.fixture(scope="session")
def gurobi_licensed():
    r"""
    Whether Gurobi can actually solve a trivial LP (installed *and*
    licensed). Slow-tier cross-solver tests skip when this is False.
    """
    return solver_available("gurobi")


@pytest.fixture(scope="session", params=["highs", "gurobi"])
def optimizer(request, tiny_model):
    r"""
    Yields a working optimizer (HiGHS always if installed; Gurobi only if a
    license is available) over the tiny chain model.
    """
    name = request.param
    if not solver_available(name):
        pytest.skip(f"{name} solver not available or unlicensed")
    if name == "highs":
        from compass.opt.highs import HighsOptimizer, get_highs_config
        return HighsOptimizer(tiny_model, get_highs_config())
    elif name == "gurobi":
        from compass.opt.gurobi import GurobiOptimizer, get_gurobi_config
        return GurobiOptimizer(tiny_model, get_gurobi_config())


@pytest.fixture(params=["highs", "gurobi"])
def solver(request):
    r"""
    Like ``optimizer`` but yields only the solver *name* (for tests that
    build their own optimizer, e.g. through initialize_optimization).
    Skips when the solver is not installed or unlicensed.
    """
    if not solver_available(request.param):
        pytest.skip(f"{request.param} solver not available or unlicensed")
    return request.param
