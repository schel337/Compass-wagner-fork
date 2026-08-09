"""
In silico knockout support.

Blocks specific reactions by generating a KO-specific media JSON that overrides
their upper bounds.  This media can then be loaded via the normal
``MetabolicModel.load_media()`` path — no core algorithm changes are required
beyond the baseline v_opt injection layer.
"""
from __future__ import print_function, division, absolute_import

from dataclasses import dataclass, field
import json
import os
from typing import Optional


@dataclass
class KoSpec:
    """Mapping of directional reaction IDs → upper-bound values for a KO.

    A value of ``0.0`` means a full knockout; any positive float gives a
    partial reduction.
    """
    ko_reactions: dict[str, float] = field(default_factory=dict)

    def is_empty(self) -> bool:
        return len(self.ko_reactions) == 0


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def load_ko_reactions(path: str) -> KoSpec:
    """Load a KO reaction list from a plain-text file.

    Supported formats (one reaction per line)::

        MAR09048_pos
        SOME_RXN_neg   0.1

    The second column is *optional* and specifies the reduced upper-bound.
    When omitted the default upper-bound is ``0.0`` (full KO).

    Reaction IDs must be **directional** (ending in ``_pos`` or ``_neg``).

    Blank lines and lines starting with ``#`` are ignored.
    """
    ko: dict[str, float] = {}
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            rxn_id = parts[0]
            ub = float(parts[1]) if len(parts) > 1 else 0.0
            ko[rxn_id] = ub
    return KoSpec(ko_reactions=ko)


def generate_ko_media(
    ko_spec: KoSpec,
    base_media_path: str,
    output_path: str,
) -> str:
    """Merge *ko_spec* bounds into a copy of the baseline media file.

    Parameters
    ----------
    ko_spec : KoSpec
        Reaction IDs → upper bounds to override.
    base_media_path : str
        Path to the baseline ``<media_name>.json``.
    output_path : str
        Destination path for the KO media JSON.

    Returns
    -------
    str
        *output_path* (for convenience).
    """
    with open(base_media_path) as fh:
        media: dict = json.load(fh)

    for rxn_id, ub in ko_spec.ko_reactions.items():
        media[rxn_id] = ub

    with open(output_path, "w") as fh:
        json.dump(media, fh, indent=1)

    return output_path


def find_model_media_dir(
    model_name: str,
    metabolic_model_dir: str,
) -> str:
    """Return the path to the ``media/`` directory for a given model.

    Handles both MATLAB-style (``model_name`` dir) and SBML-style
    (``model_name`` dir with ``.xml`` file) conventions.
    """
    # Try direct child first (MATLAB and most SBML)
    candidate = os.path.join(metabolic_model_dir, model_name, "media")
    if os.path.isdir(candidate):
        return candidate

    # Fallback: search siblings
    for entry in os.listdir(metabolic_model_dir):
        media = os.path.join(metabolic_model_dir, entry, "media")
        if os.path.isdir(media):
            # Only match if the model name appears in the directory name
            if model_name in entry:
                return media

    raise FileNotFoundError(
        f"Could not find media directory for model '{model_name}' "
        f"under '{metabolic_model_dir}'"
    )


def validate_ko_spec(
    ko_spec: KoSpec,
    reaction_ids: set[str],
) -> list[str]:
    """Return reaction IDs in *ko_spec* that are **not** present in the model.

    An empty return means the spec is valid.
    """
    return [rid for rid in ko_spec.ko_reactions if rid not in reaction_ids]