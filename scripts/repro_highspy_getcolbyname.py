"""
Minimal repro of an upstream HiGHS name-lookup bug, as exposed by highspy.

Symptom
-------
After a model goes through an addVariable()/deleteVariable() round and a
variable is re-added under a previously-used name, getColByName() for
*other*, never-touched base variables returns HighsStatus.kError together
with an UNINITIALIZED index (observed values in the 32764-32767 range,
i.e. whatever happened to be on the stack). The bug is 100% deterministic.

    $ python scripts/repro_highspy_getcolbyname.py
    getColByName('base_0') -> (HighsStatus.kError, 32765)   # expected (kOk, 0)
    REPRODUCED

Reproducible with highspy 1.15.1 (the latest release on PyPI as of
2026-08-19) and confirmed present in the C++ core on HiGHS master
(ERGO-Code/HiGHS @ 73cac48, 2026-08-19); no newer release exists to
upgrade to.

Mechanism (HiGHS C++ core, highspy just exposes it)
---------------------------------------------------
1. Highs::deleteColsInterface() clears lp.col_hash_.name2index after
   deleting columns (highs/lp_data/HighsInterface.cpp).
2. Highs::passColName() does NOT re-form the hash; it only incrementally
   calls col_hash_.update(col, old_name, new_name)
   (highs/lp_data/Highs.cpp). So after a delete, only names passed
   *after* the delete end up in the hash - the hash is non-empty but
   incomplete.
3. Highs::getColByName() only rebuilds the hash when it is EMPTY:
       if (!lp.col_hash_.name2index.size()) lp.col_hash_.form(lp.col_names_);
   (highs/lp_data/Highs.cpp). With the non-empty-but-incomplete hash from
   step 2 it never rebuilds, so lookups of the pre-existing names fail
   with kError.
4. The pybind binding (highspy/highs_bindings.cpp) and the C API
   (Highs_getColByName in highs/interfaces/highs_c_api.cpp) declare the
   out-parameter on the stack and return it *uninitialized* on error, so
   callers that ignore the status silently operate on a garbage column:
   changeColBounds / changeColsCost become no-ops and LPs solve with the
   wrong objective or bounds while still reporting "Optimal".

Two variants that do NOT trigger it (pinning down the trigger):
  - delete WITHOUT re-adding a named variable: the hash stays empty and
    is lazily re-formed on the next getColByName -> works.
  - a getColByName call *between* the delete and the re-add: forces the
    lazy re-form, after which the incremental update is consistent ->
    works.
So the trigger is exactly: deleteVariable() followed by a named
addVariable() (passColName) before any name lookup.

Compass impact
--------------
HighsOptimizer solves many small LP deltas by adding uptake/secretion
variables per solve and deleting them again (tabula rasa). The first delta
worked; from the second solve on, every name-based objective/bounds change
was silently dropped. Fixed in Compass by caching highs_var handles at
model build time and never calling getColByName during solve()
(see compass/opt/highs.py). A regression test pinning this is
tests/test_optimizer.py::test_sequential_solves_keep_objective_correct.

A minimal C++-level repro against the core (no Python) would be:

    Highs highs;
    highs.addVar(0, 1);   highs.passColName(0, "base_0");
    highs.addVar(0, 1);   highs.passColName(1, "base_1");
    highs.addVar(0, 1);   highs.passColName(2, "tmp");
    highs.deleteCol(2);
    highs.addVar(0, 1);   highs.passColName(2, "tmp");   // re-add under old name
    HighsInt col;
    highs.getColByName("base_0", col);   // -> kError, col uninitialized

A possible upstream fix: treat the name hash as stale after any
deleteCols/deleteRows (re-form on next use) instead of relying on
"empty == needs forming", and initialize the out-parameter to -1 on
error in Highs_getColByName / the pybind binding.
"""
import sys

import highspy as h


def main() -> int:
    import importlib.metadata as md

    m = h.Highs()
    m.setOptionValue("output_flag", False)

    m.addVariable(lb=0.0, ub=1.0, name="base_0")
    m.addVariable(lb=0.0, ub=1.0, name="base_1")

    d = m.addVariable(lb=0.0, ub=1.0, name="tmp")
    m.deleteVariable(d)
    m.addVariable(lb=0.0, ub=1.0, name="tmp")  # re-add under the deleted name

    ok, idx = m.getColByName("base_0")
    print(f"highspy {md.version('highspy')}: getColByName('base_0') -> ({ok}, {idx})")
    if ok == h.HighsStatus.kOk and idx == 0:
        print("not reproduced (upstream fixed?)")
        return 0
    print("REPRODUCED")
    return 1


if __name__ == "__main__":
    sys.exit(main())
