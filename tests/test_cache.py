"""
Unit tests for the preprocess cache (compass.compass.cache).

The cache maps (model, media) -> dict of reaction/uptake/secretion ids to
their maximal flux, computed once and reused across runs.
"""
import json
import os

import pytest

from compass.compass import cache


@pytest.fixture()
def cache_dir(tmp_path):
    return str(tmp_path / "preprocess_cache")


def test_load_new_model_reports_new_cache(cache_dir):
    # Use a made-up model name so we never touch the real caches
    loaded = cache.load("FakeModel_1", "default-media", preprocess_cache_dir=cache_dir)
    assert loaded == {}
    assert cache.is_new_cache("FakeModel_1", "default-media") is True


def test_save_and_reload_round_trip(cache_dir):
    cache_dir_ = cache_dir
    loaded = cache.load("FakeModel_2", "default-media", preprocess_cache_dir=cache_dir_)
    loaded["R1_pos"] = 1.0
    loaded["R2_pos"] = 2.5
    cache.save("FakeModel_2", "default-media", preprocess_cache_dir=cache_dir_)

    cache_file = os.path.join(cache_dir_, "FakeModel_2", "default-media", "preprocess.json")
    assert os.path.exists(cache_file)
    with open(cache_file) as f:
        on_disk = json.load(f)
    assert on_disk == {"R1_pos": 1.0, "R2_pos": 2.5}

    # A fresh load (simulating a new process) sees the saved data
    cache._cache.pop(("FakeModel_2", "default-media"))
    reloaded = cache.load("FakeModel_2", "default-media", preprocess_cache_dir=cache_dir_)
    assert reloaded == {"R1_pos": 1.0, "R2_pos": 2.5}
    # NB: is_new_cache tracks whether the cache had to be *built* in this
    # process, and stays True for the rest of the process even after the
    # cache file now exists on disk.


def test_clear_empties_cache(cache_dir):
    loaded = cache.load("FakeModel_3", "default-media", preprocess_cache_dir=cache_dir)
    loaded["R1_pos"] = 1.0
    cache.clear("FakeModel_3", "default-media", preprocess_cache_dir=cache_dir)
    assert cache.load("FakeModel_3", "default-media", preprocess_cache_dir=cache_dir) == {}


def test_model_object_as_key(cache_dir, tiny_model):
    """The model can be passed as a MetabolicModel object (not just a name)."""
    loaded = cache.load(tiny_model, "default-media", preprocess_cache_dir=cache_dir)
    assert loaded == {}
