"""
Unit tests for MetabolicModel invariants, using the deterministic chain model
from conftest.
"""
import json

import pytest


def test_smat_structure(tiny_model):
    """The stoichiometric matrix matches the chain topology."""
    smat = tiny_model.SMAT
    # M1: produced by E1_pos, consumed by R1_pos
    assert {(r, c) for r, c in smat["M1"]} == {("E1_pos", 1.0), ("R1_pos", -1.0)}
    # M2: produced by R1_pos and R2_neg, consumed by R2_pos
    assert {(r, c) for r, c in smat["M2"]} == {
        ("R1_pos", 1.0), ("R2_neg", 1.0), ("R2_pos", -1.0)
    }
    # M4: produced by R3_pos, consumed by E4_pos
    assert {(r, c) for r, c in smat["M4"]} == {("R3_pos", 1.0), ("E4_pos", -1.0)}


def test_exchange_detection():
    from compass.models.MetabolicModel import Reaction

    e = Reaction()
    e.id = "E"
    e.products = {"M1": 1.0}
    assert e.is_exchange

    r = Reaction()
    r.id = "R"
    r.reactants = {"M1": 1.0}
    r.products = {"M2": 1.0}
    assert not r.is_exchange


def test_to_json_round_trip(tiny_model):
    payload = json.loads(tiny_model.to_JSON())
    assert payload["name"] == "TinyChain"
    assert set(payload["reactions"]) == set(tiny_model.reactions)
    for rid, rxn in payload["reactions"].items():
        assert rxn["lower_bound"] == tiny_model.reactions[rid].lower_bound
        assert rxn["upper_bound"] == tiny_model.reactions[rid].upper_bound
    assert set(payload["species"]) == {"M1", "M2", "M3", "M4"}


def test_to_json_includes_gene_associations(tiny_model):
    payload = json.loads(tiny_model.to_JSON())
    # R1 was associated with GENE_A
    assert "GENE_A" in payload["reactions"]["R1_pos"]["genes"]
    assert "GENE_B" in payload["reactions"]["R2_pos"]["genes"]
    assert "GENE_C" in payload["reactions"]["R3_pos"]["genes"]


def test_maximum_flux(tiny_model):
    # The maximum flux is the maximum of the reaction upper bounds
    expected = max(r.upper_bound for r in tiny_model.reactions.values())
    assert tiny_model.maximum_flux == expected


def test_make_unidirectional_is_idempotent(tiny_model):
    n_before = len(tiny_model.reactions)
    tiny_model.make_unidirectional()
    assert len(tiny_model.reactions) == n_before  # nothing new to split
    for r in tiny_model.reactions.values():
        assert r.lower_bound == 0.0


def test_is_pos_unidirectional(tiny_model):
    for r in tiny_model.reactions.values():
        assert r.is_pos_unidirectional
