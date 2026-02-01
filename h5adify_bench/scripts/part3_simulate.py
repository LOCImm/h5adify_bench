#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import anndata as ad

from common import ensure_dir, write_json, now_iso


SEX_GENES = {
    "human": {"female": ["XIST"], "male": ["RPS4Y1", "DDX3Y", "KDM5D", "UTY"]},
    "mouse": {"female": ["Xist"], "male": ["Rps4y1", "Ddx3y", "Kdm5d", "Uty"]},
    "rat":   {"female": ["Xist"], "male": ["Ddx3y", "Kdm5d", "Uty"]},
}

HOUSEKEEPING_HUMAN = ["ACTB", "GAPDH", "RPLP0", "EEF1A1", "B2M"]
HOUSEKEEPING_MOUSE = ["Actb", "Gapdh", "Rplp0", "Eef1a1", "B2m"]
HOUSEKEEPING_RAT   = ["Actb", "Gapdh", "Rplp0", "Eef1a1", "B2m"]

TECHS = ["10x", "Visium", "MERFISH", "Multiome"]
SPECIES = ["human", "mouse", "rat"]

# Different obs naming schemes to challenge mapping
OBS_SCHEMES = [
    {"batch": "batch", "sample": "sample", "donor": "donor_id", "domain": "region", "sex": "sex"},
    {"batch": "library", "sample": "sample_id", "donor": "patient", "domain": "anatomical_region", "sex": "gender"},
    {"batch": "run", "sample": "specimen_id", "donor": "subject_id", "domain": "area", "sex": "donor_sex"},
]


def gene_panel(species: str) -> List[str]:
    if species == "human":
        base = HOUSEKEEPING_HUMAN
    elif species == "mouse":
        base = HOUSEKEEPING_MOUSE
    else:
        base = HOUSEKEEPING_RAT

    # add some “marker-ish” genes
    extra = (["MKI67", "SOX2", "GFAP", "AQP4", "PDGFRA"] if species == "human"
             else ["Mki67", "Sox2", "Gfap", "Aqp4", "Pdgfra"])
    # add sex genes
    sg = SEX_GENES[species]["female"] + SEX_GENES[species]["male"]
    genes = list(dict.fromkeys(base + extra + sg))
    return genes


def simulate_counts(n_cells: int, n_genes: int, mean: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    # Negative-binomial-ish via Gamma-Poisson
    lam = rng.gamma(shape=2.0, scale=mean/2.0, size=(n_cells, n_genes))
    x = rng.poisson(lam)
    return x.astype(np.int32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="data/sim")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    outdir = ensure_dir(args.outdir)
    rng = np.random.default_rng(args.seed)

    gold = {"created_at": now_iso(), "items": []}

    for i in range(args.n):
        species = rng.choice(SPECIES).item()
        tech = rng.choice(TECHS).item()
        scheme = OBS_SCHEMES[int(rng.integers(0, len(OBS_SCHEMES)))]
        n_cells = int(rng.integers(800, 3000))
        genes = gene_panel(species)
        n_genes = len(genes)

        # donors / batches / samples
        n_donors = int(rng.integers(3, 10))
        n_batches = int(rng.integers(2, 5))
        n_samples = int(rng.integers(3, 12))
        donors = [f"D{j}" for j in range(n_donors)]
        batches = [f"B{j}" for j in range(n_batches)]
        samples = [f"S{j}" for j in range(n_samples)]
        domains = ["core", "edge", "white_matter", "immune_niche"]

        donor = rng.choice(donors, size=n_cells, replace=True)
        batch = rng.choice(batches, size=n_cells, replace=True)
        sample = rng.choice(samples, size=n_cells, replace=True)
        domain = rng.choice(domains, size=n_cells, replace=True)

        # sex at donor level
        donor_sex = {d: ("female" if rng.random() < 0.5 else "male") for d in donors}
        sex = np.array([donor_sex[d] for d in donor], dtype=object)

        X = simulate_counts(n_cells, n_genes, mean=2.0 if tech != "MERFISH" else 1.0, seed=args.seed + i)

        # boost sex genes
        sex_g = SEX_GENES[species]
        gene_to_idx = {g: k for k, g in enumerate(genes)}
        for c in range(n_cells):
            if sex[c] == "female":
                for g in sex_g["female"]:
                    if g in gene_to_idx:
                        X[c, gene_to_idx[g]] += 8
            else:
                for g in sex_g["male"]:
                    if g in gene_to_idx:
                        X[c, gene_to_idx[g]] += 8

        obs = pd.DataFrame(index=[f"cell{i}_{j}" for j in range(n_cells)])
        obs[scheme["batch"]] = batch
        obs[scheme["sample"]] = sample
        obs[scheme["donor"]] = donor
        obs[scheme["domain"]] = domain
        obs[scheme["sex"]] = sex

        # dataset-level fields in uns (toolkit may or may not use these)
        uns = {"species": species, "technology": tech}

        adata = ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=genes), uns=uns)

        fn = f"sim_{i:02d}_{species}_{tech}.h5ad".replace(" ", "_")
        fp = outdir / fn
        adata.write_h5ad(fp)

        gold["items"].append({
            "file": str(fp),
            "species": species,
            "technology": tech,
            "gold_key": {
                "batch": scheme["batch"],
                "sample": scheme["sample"],
                "donor": scheme["donor"],
                "domain": scheme["domain"],
                "sex": scheme["sex"],
                "species": None,       # dataset-level, not obs key here
                "technology": None,
            },
            "notes": {"scheme": scheme}
        })

    write_json(outdir / "sim_gold.json", gold)
    print(f"[ok] wrote {outdir}/sim_gold.json with {len(gold['items'])} sims")


if __name__ == "__main__":
    main()
