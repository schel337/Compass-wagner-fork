"""Tests for the in silico KO module (compass.models.knockout)."""
from __future__ import print_function, division, absolute_import

import json
import os
import tempfile
import pytest

from compass.models.MetabolicModel import MetabolicModel, Reaction, Species
from compass.models.knockout import (
    KoSpec,
    load_ko_reactions,
    generate_ko_media,
    find_model_media_dir,
    validate_ko_spec,
)


# ── Fixtures ──


@pytest.fixture
def ko_text_file(tmp_path):
    """Minimal KO reaction list (full KOs)."""
    path = tmp_path / "ko.txt"
    path.write_text("RXN1_pos\nRXN2_neg\n# comment line\n\nRXN3_pos 0.1\n")
    return path


@pytest.fixture
def base_media_file(tmp_path):
    """Minimal baseline media JSON."""
    path = tmp_path / "default-media.json"
    path.write_text(json.dumps({"RXN1_pos": 1000, "RXN2_neg": 1000, "RXN3_pos": 1000}))
    return path


@pytest.fixture
def simple_model():
    """Minimal metabolic model for validation tests."""
    model = MetabolicModel("Test")
    r1 = Reaction()
    r1.id = "R1_pos"
    r1.upper_bound = 1000
    r1.lower_bound = 0
    model.reactions = {"R1_pos": r1}
    return model


# ── KoSpec ──


class TestKoSpec:
    def test_empty(self):
        s = KoSpec()
        assert s.is_empty()

    def test_non_empty(self):
        s = KoSpec(ko_reactions={"A": 0.0})
        assert not s.is_empty()


# ── load_ko_reactions ──


class TestLoadKoReactions:
    def test_full_kos(self, ko_text_file):
        spec = load_ko_reactions(str(ko_text_file))
        assert spec.ko_reactions["RXN1_pos"] == 0.0
        assert spec.ko_reactions["RXN2_neg"] == 0.0

    def test_partial_ko(self, ko_text_file):
        spec = load_ko_reactions(str(ko_text_file))
        assert spec.ko_reactions["RXN3_pos"] == 0.1

    def test_ignores_comments_and_blanks(self, ko_text_file):
        spec = load_ko_reactions(str(ko_text_file))
        assert "#" not in spec.ko_reactions
        assert "" not in spec.ko_reactions
        assert len(spec.ko_reactions) == 3


# ── generate_ko_media ──


class TestGenerateKoMedia:
    def test_overwrites_bounds(self, ko_text_file, base_media_file, tmp_path):
        spec = load_ko_reactions(str(ko_text_file))
        out = tmp_path / "ko_media.json"
        generate_ko_media(spec, str(base_media_file), str(out))
        media = json.loads(out.read_text())
        assert media["RXN1_pos"] == 0.0
        assert media["RXN2_neg"] == 0.0
        assert media["RXN3_pos"] == 0.1

    def test_preserves_non_ko_reactions(self, base_media_file, tmp_path):
        spec = KoSpec(ko_reactions={"RXN1_pos": 0.0})
        out = tmp_path / "ko_media.json"
        generate_ko_media(spec, str(base_media_file), str(out))
        media = json.loads(out.read_text())
        assert media["RXN2_neg"] == 1000  # unchanged


# ── find_model_media_dir ──


class TestFindModelMediaDir:
    def test_existing_dir(self, tmp_path):
        media = tmp_path / "MyModel" / "media"
        media.mkdir(parents=True)
        assert find_model_media_dir("MyModel", str(tmp_path)) == str(media)

    def test_missing(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            find_model_media_dir("Missing", str(tmp_path))


# ── validate_ko_spec ──


class TestValidateKoSpec:
    def test_valid(self, simple_model):
        spec = KoSpec(ko_reactions={"R1_pos": 0.0})
        assert validate_ko_spec(spec, set(simple_model.reactions.keys())) == []

    def test_unknown_reactions(self, simple_model):
        spec = KoSpec(ko_reactions={"R1_pos": 0.0, "FAKE_neg": 0.0})
        unknown = validate_ko_spec(spec, set(simple_model.reactions.keys()))
        assert unknown == ["FAKE_neg"]