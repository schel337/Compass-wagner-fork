"""
Upstream-bug tracker tests (not Compass bugs).

These tests document known defects in solver-backend *dependencies* and
are marked xfail so the suite stays green. They will start passing (xpass)
once the dependency is fixed upstream, at which point the xfail marker
should be removed. See scripts/repro_highspy_getcolbyname.py for the full
standalone repro and the C++-level mechanism.
"""
import pytest

import highspy as h


@pytest.mark.xfail(
    reason=(
        "highspy/HiGHS bug: after addVariable()/deleteVariable() rounds with "
        "name reuse, getColByName() returns kError plus an *uninitialized* "
        "index for pre-existing variables (highspy 1.15.1, latest; also on "
        "HiGHS master 73cac48 2026-08-19). The name hash is cleared on "
        "delete but only re-formed on lookup when empty, while passColName "
        "fills it incrementally, so it is left non-empty but incomplete. "
        "Compass works around it by caching highs_var handles at build time "
        "and never calling getColByName during solve() (compass/opt/highs.py)."
    ),
    strict=False,
)
def test_highspy_getcolbyname_after_delete_and_readd():
    """
    getColByName() must find pre-existing named columns after a named
    variable is deleted and re-added. Fails with (kError, garbage) on
    highspy 1.15.1.
    """
    m = h.Highs()
    m.setOptionValue("output_flag", False)

    m.addVariable(lb=0.0, ub=1.0, name="base_0")
    m.addVariable(lb=0.0, ub=1.0, name="base_1")

    d = m.addVariable(lb=0.0, ub=1.0, name="tmp")
    m.deleteVariable(d)
    m.addVariable(lb=0.0, ub=1.0, name="tmp")  # re-add under the deleted name

    for name, expected in (("base_0", 0), ("base_1", 1)):
        ok, idx = m.getColByName(name)
        assert ok == h.HighsStatus.kOk, (
            f"getColByName({name!r}) -> {ok} (upstream bug; see "
            f"scripts/repro_highspy_getcolbyname.py)"
        )
        assert idx == expected
