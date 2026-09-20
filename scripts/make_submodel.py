#!/usr/bin/env python3
"""
Build a reduced RECON2.2 submodel of central carbon metabolism for fast
Compass integration tests (see TESTING_PLAN.md, section 7).

Selection rule:
  1. Seed: the reactions listed in
     compass/Resources/Recon2_export/central_carbon_rxns.txt.
  2. Boundary: every boundary (extracellular) reaction transporting a
     seed metabolite, excluding ubiquitous carriers (H+, ATP/ADP, H2O, Pi)
     which would otherwise pull in hundreds of unrelated transports.

The result (~120 reactions) is written as a standalone SBML file under
Resources/Metabolic Models/<model-dir>/ plus the default media file, so
it can be loaded like any other Compass model:
    python compass/main.py --model <model-dir> ...

Preprocessing a ~120-reaction model takes ~1 minute versus ~30 minutes
for the full 7785-reaction RECON2.2.

Usage:
    python scripts/make_submodel.py                 # build + verify
    python scripts/make_submodel.py --print-only    # just show the selection
"""
import argparse
import os
import shutil

import compass
import libsbml

MODEL_DIR = os.path.join(os.path.dirname(compass.__file__), "Resources",
                         "Metabolic Models")
SEED_FILE = os.path.join(os.path.dirname(compass.__file__), "Resources",
                         "Recon2_export", "central_carbon_rxns.txt")
SOURCE_MODEL = "RECON2.2"


def get_seed_rxns():
    with open(SEED_FILE) as f:
        return {line.strip() for line in f if line.strip()}


def select_rxns(model, seed):
    """
    Returns the ordered dict of reaction id -> reaction to keep.
    """
    # RECON2.2 prefixes reaction ids with R_; the seed file uses bare names
    seed = {rid if rid in model.reactions else "R_" + rid for rid in seed}
    kept = {rid: model.reactions[rid] for rid in seed if rid in model.reactions}

    def touched(rxn):
        return set(rxn.reactants) | set(rxn.products)

    def is_boundary(rxn):
        # In RECON2.2's SBML, boundary reactions are extracellular transport
        # pairs (M_x_e <-> M_x_c) plus pseudo-reactions, not strict exchanges
        if rxn.is_exchange:
            return True
        t = touched(rxn)
        if any(mid.endswith("_e") for mid in t):
            return True
        return "biomass" in rxn.id.lower()

    # Ubiquitous carrier metabolites: excluding them keeps the transporter
    # selection small and focused on actual central carbon substrates
    ubiq = ({f"M_{s}_c" for s in ("h", "h2o", "atp", "adp", "pi", "ppi")}
            | {f"M_{s}_e" for s in ("h", "h2o", "atp", "adp", "pi", "ppi")})

    # Boundary reactions transporting seed metabolites
    seed_metab = set().union(*[touched(r) for r in kept.values()]) if kept else set()
    specific_metab = seed_metab - ubiq
    for rid, r in model.reactions.items():
        if rid in kept:
            continue
        if is_boundary(r) and (touched(r) & specific_metab):
            kept[rid] = r
    return kept


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default="SubModel_cc",
                        help="Model directory name under Resources/Metabolic Models")
    parser.add_argument("--print-only", action="store_true")
    args = parser.parse_args()

    from compass.models import load_metabolic_model
    model = load_metabolic_model(SOURCE_MODEL)

    seed = get_seed_rxns()
    kept = select_rxns(model, seed)
    print(f"Selected {len(kept)} reactions")

    if args.print_only:
        return

    # Write the reduced SBML
    model_xml = os.path.join(MODEL_DIR, SOURCE_MODEL)
    sbml_file = [x for x in os.listdir(model_xml)
                 if x.lower().endswith((".xml", ".xml.gz"))][0]
    doc = libsbml.readSBMLFromFile(os.path.join(model_xml, sbml_file))
    xml_model = doc.getModel()

    # Remove unselected reactions (descending index order to keep indices valid)
    remove_rxn_idx = [i for i in range(xml_model.getNumReactions())
                      if xml_model.getReaction(i).getId() not in kept]
    for i in sorted(remove_rxn_idx, reverse=True):
        xml_model.removeReaction(i)

    # Remove species no longer referenced
    kept_metab = set().union(*[set(r.reactants) | set(r.products)
                               for r in kept.values()])
    remove_sp_idx = [i for i in range(xml_model.getNumSpecies())
                     if xml_model.getSpecies(i).getId() not in kept_metab]
    for i in sorted(remove_sp_idx, reverse=True):
        xml_model.removeSpecies(i)

    out_dir = os.path.join(MODEL_DIR, args.model_dir)
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f"{args.model_dir}.xml")
    libsbml.writeSBMLToFile(doc, out_file)

    # Copy the default media
    media_src = os.path.join(MODEL_DIR, SOURCE_MODEL, "media", "default-media.json")
    media_dst_dir = os.path.join(out_dir, "media")
    os.makedirs(media_dst_dir, exist_ok=True)
    shutil.copy(media_src, os.path.join(media_dst_dir, "default-media.json"))

    print(f"Wrote {out_file} with {xml_model.getNumReactions()} reactions, "
          f"{xml_model.getNumSpecies()} species")

    # Verify it loads through Compass
    m2 = load_metabolic_model(args.model_dir)
    print(f"Verification: loaded {args.model_dir} with "
          f"{len(m2.reactions)} reactions")


if __name__ == "__main__":
    main()
