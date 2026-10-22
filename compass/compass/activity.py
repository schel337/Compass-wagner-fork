r"""
Support for in-silico metabolic knockout (KO) experiments.

A paired KO experiment compares two Compass runs on the same cells:

* **Run A (with KO):** Compass run with the KO reactions blocked (zero flux
  in every linear program). The per-reaction maximum flux values v_r^opt
  are therefore computed *with the KO active* and are written to the
  preprocess cache.
* **Run B (without KO):** Compass run on the unblocked model in which the
  per-reaction maximum problems are *not* solved; instead the v_r^opt
  values from Run A (exported as an "activity file") are used as the
  high-activity targets (v_r >= BETA * v_r^opt). A paired-run wrapper may
  patch blocked directions in Run B's targets (default 1.0) while preserving
  the raw KO-derived values in Run A's cache.

This module holds the file-format helpers shared by both runs:

* :func:`read_block_reactions_file` - KO reaction list (one ID per line;
  a directed ID such as ``R123_pos`` blocks that direction only, while an
  undirected ID such as ``R123`` blocks both ``R123_pos`` and ``R123_neg``).
* :func:`read_activity_file` - reaction ID -> v_r^opt mapping (TSV).
* :func:`export_activity_file` - dump a preprocess cache (Run A's v_r^opt
  values) to an activity TSV.
"""
import os
import logging

from . import cache

logger = logging.getLogger("compass")


def read_block_reactions_file(path):
    """
    Reads a KO block-reactions file and returns the set of *directed*
    reaction IDs to block.

    File format: one reaction ID per line. Lines starting with ``#`` and
    blank lines are ignored. An ID ending in ``_pos`` or ``_neg`` is used
    as-is; any other ID is treated as undirected and expands to both
    ``<ID>_pos`` and ``<ID>_neg`` (matching IDs against the model happens
    later, so unknown directions are simply dropped there).
    """
    if not os.path.exists(path):
        raise FileNotFoundError("Cannot find block reactions file: " + path)

    blocked = set()
    with open(path) as fin:
        for line in fin:
            rxn_id = line.strip()
            if not rxn_id or rxn_id.startswith("#"):
                continue
            if rxn_id.endswith("_pos") or rxn_id.endswith("_neg"):
                blocked.add(rxn_id)
            else:
                blocked.add(rxn_id + "_pos")
                blocked.add(rxn_id + "_neg")
    return blocked


def read_activity_file(path):
    """
    Reads a fixed-activity file mapping reaction IDs to v_r^opt values.

    File format: TSV with two columns, ``reaction_id<TAB>v_r_opt``. Lines
    starting with ``#`` and blank lines are ignored.

    Returns
    -------
    dict[str, float]
    """
    if not os.path.exists(path):
        raise FileNotFoundError("Cannot find activity file: " + path)

    activities = {}
    with open(path) as fin:
        for lineno, line in enumerate(fin, 1):
            rxn_id = line.rstrip("\n")
            if not rxn_id.strip() or rxn_id.lstrip().startswith("#"):
                continue
            parts = rxn_id.split("\t")
            if len(parts) != 2:
                raise ValueError(
                    f"Malformed line {lineno} in activity file {path}: "
                    f"{line.rstrip()!r} (expected 'reaction_id<TAB>v_r_opt')")
            rxn, value = parts[0].strip(), parts[1].strip()
            try:
                activities[rxn] = float(value)
            except ValueError:
                raise ValueError(
                    f"Non-numeric v_r^opt value {value!r} for reaction "
                    f"{rxn!r} on line {lineno} of activity file {path}")
    if not activities:
        raise ValueError("Activity file " + path + " contains no reactions")
    return activities


def export_activity_file(model, media, out_path,
                         preprocess_cache_dir=cache.PREPROCESS_CACHE_DIR):
    """
    Exports the v_r^opt values stored in a preprocess cache as an activity
    file. This is how Run B of a paired KO experiment obtains its activity
    targets: Run A's (KO-applied) cache is exported to a TSV and passed to
    Run B via ``--activity-file``.

    Returns the number of exported reactions.
    """
    model_cache = cache.load(model, media, preprocess_cache_dir=preprocess_cache_dir)
    if not model_cache:
        raise ValueError(
            f"Preprocess cache for model {model!r}, media {media!r} "
            f"(in {preprocess_cache_dir}) is empty; nothing to export")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    n = 0
    with open(out_path, "w") as fout:
        fout.write("# reaction_id\tv_r_opt\n")
        for rxn_id in sorted(model_cache.keys()):
            fout.write(f"{rxn_id}\t{model_cache[rxn_id]}\n")
            n += 1
    logger.info("Exported v_r^opt for %d reactions to %s", n, out_path)
    return n
