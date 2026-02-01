#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from common import (
    read_json, write_json, ensure_dir, canonicalize_species_to_h5adify,
    detect_technology_from_string, pick_best_gold_key, doi_slug, now_iso
)

from h5adify.core import ensure_user_metadata_vocab
from h5adify.core.metadata_harmonize import load_metadata_vocab


FIELD_SYNONYMS = {
    "batch":   ["batch", "batch_id", "library", "library_id", "run", "lane", "seq_batch", "chemistry_batch"],
    "sample":  ["sample", "sample_id", "sample_name", "biosample_id", "specimen", "specimen_id"],
    "donor":   ["donor", "donor_id", "patient", "patient_id", "individual", "individual_id", "subject", "subject_id"],
    "domain":  ["domain", "region", "anatomical_region", "area", "cluster", "subclass", "compartment"],
    "sex":     ["sex", "gender", "donor_sex", "biological_sex"],
    # species/technology are often dataset-level or free text; keep for completeness
    "species": ["species", "organism"],
    "technology": ["technology", "assay", "platform", "method"],
}


def candidates_from_obs(obs_columns: List[str], field: str) -> List[str]:
    # stable candidate list: exact synonyms + fuzzy contains
    syns = FIELD_SYNONYMS.get(field, [field])
    syns_norm = {s.lower().replace("-", "_") for s in syns}
    out = []
    for k in obs_columns:
        kn = k.lower().replace("-", "_")
        if kn in syns_norm:
            out.append(k)
    if not out:
        for k in obs_columns:
            kn = k.lower().replace("-", "_")
            for s in syns_norm:
                if s in kn or kn in s:
                    out.append(k); break
    # stable unique
    seen = set()
    out2 = []
    for x in out:
        if x not in seen:
            seen.add(x); out2.append(x)
    return out2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/doi20/manifest.json")
    ap.add_argument("--out", default="gold/doi20_gold.json")
    ap.add_argument("--use-small", action="store_true", help="Prefer .small.h5ad when available")
    args = ap.parse_args()

    ensure_dir(Path(args.out).parent)

    man = read_json(args.manifest)
    ensure_user_metadata_vocab(overwrite=False)
    vocab = load_metadata_vocab(None)
    tech_keywords = vocab.get("technology_keywords", {})

    gold: Dict[str, Any] = {
        "created_at": now_iso(),
        "manifest": args.manifest,
        "fields": ["batch", "sample", "donor", "domain", "sex", "species", "technology"],
        "items": [],
    }

    import anndata as ad

    for it in man["items"]:
        path = it.get("small_h5ad") if args.use_small and it.get("small_h5ad") else it.get("source_h5ad")
        path = Path(path)
        adata = ad.read_h5ad(path, backed="r")  # fast header-only
        obs_cols = list(adata.obs.columns)

        # gold keys for obs-mapped fields
        chosen = {}
        for f in gold["fields"]:
            cand = candidates_from_obs(obs_cols, f)
            chosen[f] = pick_best_gold_key(obs_cols, cand)

        # gold canonical species/technology from Census metadata
        species_canon = canonicalize_species_to_h5adify(it.get("organism", ""))
        tech_raw = it.get("assay", "") or ""
        tech_canon = detect_technology_from_string(tech_raw, tech_keywords)

        gold["items"].append({
            "doi": it["doi"],
            "dataset_id": it["dataset_id"],
            "h5ad_path": str(path),
            "obs_columns": obs_cols,
            "gold_key": chosen,                    # obs key expected (or None)
            "gold_species_canon": species_canon,   # "human"/"mouse"/"rat"/...
            "gold_technology_canon": tech_canon,   # e.g. "Visium"/"MERFISH"/...
            "meta": {
                "organism_raw": it.get("organism", ""),
                "assay_raw": tech_raw,
                "dataset_title": it.get("dataset_title", ""),
                "collection_name": it.get("collection_name", ""),
            }
        })

    write_json(args.out, gold)
    print(f"[ok] wrote {args.out} with {len(gold['items'])} items")


if __name__ == "__main__":
    main()
