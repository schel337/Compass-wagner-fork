"""
Shared fixtures for the Compass test suite.
"""
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


def make_expression_df(n_cells=16, n_genes=200, seed=0):
    r"""
    Generates a synthetic expression DataFrame in the same units Compass
    expects (CPM-like scaled values): one row per gene, one column per cell.

    Gene expression levels are drawn log-normally so that a few genes are
    highly expressed, mirroring the heavy-tailed distribution of real
    scRNA-seq CPM data.
    """
    rng = np.random.default_rng(seed)
    genes = [f"GENE_{i:04d}" for i in range(n_genes)]
    cells = [f"CELL_{i:04d}" for i in range(n_cells)]
    log_means = rng.normal(-4.0, 2.0, size=n_genes)
    data = np.clip(np.exp(log_means[:, None] + rng.normal(0.0, 0.5, size=(n_genes, n_cells))), 0.0, None)
    # Scale columns to CPM-like totals
    data = data / data.sum(axis=0) * 1e5
    return pd.DataFrame(data, index=genes, columns=cells)


@pytest.fixture()
def expression_df():
    return make_expression_df()


@pytest.fixture()
def expression_file(tmp_path, expression_df):
    path = tmp_path / "expression.tsv"
    expression_df.to_csv(path, sep="\t")
    return str(path)


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
