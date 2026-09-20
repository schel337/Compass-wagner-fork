import logging
from typing import Any

from compass.globals import EXCHANGE_LIMIT
from compass.models.MetabolicModel import MetabolicModel
from .base import Optimizer, LinearProgramDelta, Solution

logger = logging.getLogger("compass")


def get_highs_config(threads: int | None = None, method: int | None = None) -> dict[str, Any]:
    """
    Returns the HiGHS option values for Compass.

    Note that HiGHS (unlike Gurobi) does not expose a choice of
    algorithm for LPs; the 'threads' and 'method' arguments are
    accepted for interface compatibility but have no effect.
    """
    return {
        "output_flag": False,  # Disable all output
        "time_limit": 120,  # Each problem should take well under 120 seconds
    }


class HighsOptimizer(Optimizer):
    """
    HiGHS-based implementation of the Optimizer.
    """

    def __init__(self, model: MetabolicModel, config: dict[str, Any] | None = None, credentials: dict[str, str] | None = None,
                 blocked_reactions: set[str] | None = None):
        super().__init__(model, blocked_reactions=blocked_reactions)

        if config is None:
            config = get_highs_config()
        self.config = config

        self.highs_model, self._metab_constr = self._build_base_model()
        # Cache variable handles at build time. Name-based lookups must NOT be
        # used during solve(): after addVariable()/deleteVariable() rounds,
        # highspy's getColByName intermittently returns HighsStatus.kError
        # together with an *uninitialized* index (observed values in the
        # 32764-32767 range, i.e. freed memory), and the wrapper swallows the
        # error. changeColBounds/changeColsCost on such an index silently
        # no-ops, so the LP would be solved with the wrong objective or
        # bounds while still reporting "Optimal". Base-variable indices are
        # stable across solves because added variables are always appended
        # (highest indices) and deleted in descending order.
        import highspy as h

        self._rxn_var = {}
        for rid in self.model.reactions.keys():
            _, idx = self.highs_model.getColByName(rid)
            self._rxn_var[rid] = h.highs_var(idx, self.highs_model)

    def _build_base_model(self):
        """
        Builds the initial HiGHS model from the provided metabolic model.
        Returns the model and a mapping of metabolite id to constraint index.
        """
        import highspy as h

        highs_model = h.Highs()
        for k, v in self.config.items():
            highs_model.setOptionValue(k, v)

        # Define minimum and maximum flux for each reaction
        for x in self.model.reactions.values():
            highs_model.addVariable(lb=x.lower_bound, ub=x.upper_bound, name=x.id)

        # Add stoichiometry constraints
        metab_constr = {}
        row_index = 0
        for metab_id, stoichiometry in self.model.SMAT.items():
            # If there is no reaction associated with the given metabolite, then skip
            if len(stoichiometry) == 0:
                continue

            expr = 0.0
            for [rxn_id, coeff] in stoichiometry:
                _, var_index = highs_model.getColByName(rxn_id)
                expr = expr + coeff * h.highs_var(var_index, highs_model)

            # Each metabolite must obey mass conservation
            highs_model.addConstr(expr == 0.0, name=metab_id)
            metab_constr[metab_id] = row_index
            row_index += 1

        return highs_model, metab_constr

    def _get_bounds(self, var):
        """Returns (lower, upper) bounds of a variable."""
        col = self.highs_model.getCol(var.index)
        # getCol returns (status, objective, lower, upper, integrality)
        return col[2], col[3]

    def solve(self, delta: LinearProgramDelta) -> Solution:
        """
        Applies the delta, solves the model, and returns the solution.
        Reverts changes to the model after solving, to serve as tabula rasa for the next delta.
        """
        import highspy as h

        # Store original state to revert later
        original_ub = {}
        original_lb = {}
        # Name -> highs_var handle map for this solve: cached base handles
        # plus the handles returned by addVariable() below. Never use
        # getColByName here (see the note in __init__ about the highspy bug).
        vars_ = dict(self._rxn_var)
        added_vars = []

        for met_id, rxn_id in delta.added_secretion.items():
            rxn_var = self.highs_model.addVariable(lb=0.0, ub=self.model.maximum_flux, name=rxn_id)
            added_vars.append(rxn_var)
            vars_[rxn_id] = rxn_var
            # Add secretion to metabolite's constraint as a reduction in metabolite
            self.highs_model.changeCoeff(self._metab_constr[met_id], rxn_var.index, -1.0)

        for met_id, rxn_id in delta.added_uptake.items():
            rxn_var = self.highs_model.addVariable(lb=0.0, ub=EXCHANGE_LIMIT, name=rxn_id)
            added_vars.append(rxn_var)
            vars_[rxn_id] = rxn_var
            # Add uptake to metabolite's constraint as an increase in metabolite
            self.highs_model.changeCoeff(self._metab_constr[met_id], rxn_var.index, 1.0)

        # Close all blocked reactions (per-problem deltas plus run-level,
        # e.g. metabolic knockouts) by setting upper bound to lower bound
        # and store previous bounds so they can be restored later
        for rxn_id in set(delta.blocked_reactions) | self.blocked_reactions:
            var = vars_[rxn_id]
            lb, old_ub = self._get_bounds(var)
            original_ub[rxn_id] = old_ub
            self.highs_model.changeColBounds(var, lb, lb)

        for rxn_id, limit in delta.high_flux.items():
            var = vars_[rxn_id]
            old_lb, ub = self._get_bounds(var)
            original_lb[rxn_id] = old_lb
            self.highs_model.changeColBounds(var, limit, ub)

        # Construct objective linear expression by summing all coefficient * reaction pairs.
        obj_expr = 0.0
        for rxn_id, coeff in delta.objective.items():
            rxn_var = vars_[rxn_id]
            obj_expr = obj_expr + coeff * rxn_var

        if delta.sense == "max":
            sense = h.ObjSense.kMaximize
        else:
            sense = h.ObjSense.kMinimize
        self.highs_model.setObjective(obj_expr, sense)

        self.highs_model.solve()

        status = self.highs_model.getModelStatus()
        obj_value = self.highs_model.getObjectiveValue()
        if status == h.HighsModelStatus.kOptimal:
            success = True
        else:
            success = False

        # Revert the delta

        # Restore bounds
        for rxn_id in original_ub.keys():
            var = vars_[rxn_id]
            lb, ub = self._get_bounds(var)
            self.highs_model.changeColBounds(var, lb, original_ub[rxn_id])

        for rxn_id in original_lb.keys():
            var = vars_[rxn_id]
            _, ub = self._get_bounds(var)
            self.highs_model.changeColBounds(var, original_lb[rxn_id], ub)

        # Remove added variables (highest index first, since deletion shifts
        # the indices of all subsequent variables)
        for var in sorted(added_vars, key=lambda v: v.index, reverse=True):
            self.highs_model.deleteVariable(var)

        return Solution(
            success=success,
            status=self.highs_model.modelStatusToString(status),
            obj_value=obj_value,
        )
