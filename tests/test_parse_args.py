"""
Unit tests for compass.main.parseArgs().
"""
import pytest

from compass.main import parseArgs


def test_defaults(monkeypatch):
    monkeypatch.setattr("sys.argv", ["compass", "--data", "x.tsv", "--species", "homo_sapiens"])
    args = parseArgs()
    # --data is stored as a list of file path(s) (one for tsv, three for mtx)
    assert args["data"][-1].endswith("x.tsv")
    assert args["species"] == "homo_sapiens"
    assert args["optimizer"] == "gurobi"
    # Gurobi's default LP method
    assert args["lpmethod"] == 4
    assert args["num_processes"] is None
    assert args["lambda"] == 0


@pytest.mark.parametrize("optimizer, expected_method", [
    ("gurobi", 4),
    ("cuopt", 1),
    ("highs", None),
])
def test_optimizer_selection(monkeypatch, optimizer, expected_method):
    monkeypatch.setattr("sys.argv", ["compass", "--data", "x.tsv", "--species", "homo_sapiens", "--optimizer", optimizer])
    args = parseArgs()
    assert args["optimizer"] == optimizer
    assert args["lpmethod"] == expected_method


def test_turbo_arguments(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        ["compass", "--data", "x.tsv", "--species", "homo_sapiens", "--turbo", "0.95", "--turbo-max-iters", "2"],
    )
    args = parseArgs()
    assert args["turbo"] == 0.95
    assert args["turbo_max_iters"] == 2


def test_single_sample(monkeypatch):
    monkeypatch.setattr("sys.argv", ["compass", "--data", "x.tsv", "--species", "homo_sapiens", "--single-sample", "3"])
    args = parseArgs()
    assert args["single_sample"] == 3
