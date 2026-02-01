#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import anndata as ad

from common import read_yaml, read_json, ensure_dir
from h5adify.core import harmonize_metadata, DEFAULT_METADATA_FIELDS


FIELDS = list(DEFAULT_METADATA_FIELDS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="configs/models.yaml")
    ap.add_argument("--sim-gold", default="data/sim/sim_gold.json")
    ap.add_argument("--outdir", default="results/part3_sim_scores")
    ap.add_argument("--prompt-name", default="metadata_harmonize_v1_default")
    ap.add_argument("--use-llm", action="store_true")
    args = ap.parse_args()

    cfg = read_yaml(args.models)
    base_url = cfg.get("ollama_base_url", "http://localhost:11434")
    models = [m["name"] for m in cfg["models"]]

    sim = read_json(args.sim_gold)
    outdir = ensure_dir(args.outdir)

    rows = []

    for model in models:
        for it in sim["items"]:
            fp = Path(it["file"])
            gold_key = it["gold_key"]

            adata = ad.read_h5ad(fp)
            adata2, report = harmonize_metadata(
                adata,
                fields=FIELDS,
                use_llm=bool(args.use_llm),
                llm_prompt_name=args.prompt_name,
                sex_from_expression=True,
                ollama_base_url=base_url,
                ollama_model=model,
                inplace=False,
            )

            chosen = report.get("chosen_keys", {})
            # mapping accuracy for the 5 obs-key fields (batch/sample/donor/domain/sex)
            acc = {}
            for f in ["batch", "sample", "donor", "domain", "sex"]:
                acc[f] = int(chosen.get(f, None) == gold_key.get(f, None))

            # sex inference accuracy (cell-level) since sim has true sex column
            true_sex_col = gold_key["sex"]
            pred_col = "h5adify_sex"
            sex_acc = np.nan
            if true_sex_col in adata2.obs.columns and pred_col in adata2.obs.columns:
                t = adata2.obs[true_sex_col].astype(str).values
                p = adata2.obs[pred_col].astype(str).values
                sex_acc = float((t == p).mean())

            rows.append({
                "model": model,
                "file": str(fp),
                "species_truth": it["species"],
                "technology_truth": it["technology"],
                "use_llm": bool(args.use_llm),
                **{f"acc_{k}": acc[k] for k in acc},
                "sex_cell_accuracy": sex_acc,
            })

    df = pd.DataFrame(rows)
    df.to_csv(outdir / "sim_scores_long.csv", index=False)
    summ = df.groupby(["model", "use_llm"]).mean(numeric_only=True).reset_index()
    summ.to_csv(outdir / "sim_scores_summary.csv", index=False)

    print(f"[ok] wrote {outdir}/sim_scores_long.csv and summary")


if __name__ == "__main__":
    main()
