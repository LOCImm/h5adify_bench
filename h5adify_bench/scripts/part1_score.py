#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from common import read_json, read_yaml, ensure_dir, write_json


FIELDS = ["batch", "sample", "donor", "domain", "sex", "species", "technology"]


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    # Jensen-Shannon divergence (base e)
    p = p / (p.sum() + 1e-12)
    q = q / (q.sum() + 1e-12)
    m = 0.5 * (p + q)
    kl = lambda a, b: float(np.sum(a * np.log((a + 1e-12) / (b + 1e-12))))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def load_all_preds(results_dir: Path) -> List[Path]:
    return list(results_dir.glob("**/pred.json"))


def extract_pred_mapping(pred: Dict[str, Any]) -> Dict[str, Optional[str]]:
    rep = pred.get("report", {})
    chosen = rep.get("chosen_keys", {}) if isinstance(rep, dict) else {}
    out = {}
    for f in FIELDS:
        v = chosen.get(f, None)
        out[f] = v if (isinstance(v, str) or v is None) else None
    return out


def extract_pred_species_tech(pred: Dict[str, Any]) -> Dict[str, str]:
    # from canon_preview examples
    cp = pred.get("canon_preview", {}) if isinstance(pred.get("canon_preview", {}), dict) else {}
    out = {}
    for f in ["species", "technology"]:
        ex = ""
        try:
            ex_list = cp.get(f, {}).get("examples", [])
            if ex_list:
                ex = str(ex_list[0])
        except Exception:
            ex = ""
        out[f] = ex
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="configs/models.yaml")
    ap.add_argument("--gold", default="gold/doi20_gold.json")
    ap.add_argument("--results", default="results/part1")
    ap.add_argument("--outdir", default="results/part1_scores")
    args = ap.parse_args()

    ensure_dir(args.outdir)

    gold = read_json(args.gold)
    gold_index = {(it["doi"], it["dataset_id"]): it for it in gold["items"]}

    preds = load_all_preds(Path(args.results))
    rows = []

    for pj in preds:
        pred = read_json(pj)
        doi = pred.get("doi")
        dsid = pred.get("dataset_id")
        key = (doi, dsid)
        if key not in gold_index:
            continue
        g = gold_index[key]

        pred_map = extract_pred_mapping(pred)
        pred_st = extract_pred_species_tech(pred)

        # mapping accuracy per field
        acc_fields = {}
        halluc_fields = {}
        for f in FIELDS:
            gk = g["gold_key"].get(f, None)
            pk = pred_map.get(f, None)

            # hallucination: predicted key not in obs_columns
            halluc = False
            if pk is not None and pk not in g["obs_columns"]:
                halluc = True
            halluc_fields[f] = int(halluc)

            # scoring:
            if gk is None:
                acc_fields[f] = int(pk is None)
            else:
                acc_fields[f] = int(pk == gk)

        # completeness: fraction of fields where pk != None
        completeness = float(np.mean([pred_map[f] is not None for f in FIELDS]))

        # value accuracy for species/technology (canonical string compare, best-effort)
        species_gold = str(g.get("gold_species_canon", "")).strip().lower()
        species_pred = str(pred_st.get("species", "")).strip().lower()
        species_val_acc = int(species_gold != "" and species_pred != "" and species_gold == species_pred)

        tech_gold = str(g.get("gold_technology_canon", "")).strip().lower()
        tech_pred = str(pred_st.get("technology", "")).strip().lower()
        tech_val_acc = int(tech_gold != "" and tech_pred != "" and tech_gold == tech_pred)

        rows.append({
            "model": pred.get("model", ""),
            "prompt_name": pred.get("prompt_name", ""),
            "use_llm": bool(pred.get("use_llm", False)),
            "doi": doi,
            "dataset_id": dsid,
            "status": pred.get("status", ""),
            "elapsed_sec": float(pred.get("elapsed_sec", np.nan)),
            "completeness": completeness,
            "species_val_acc": species_val_acc,
            "tech_val_acc": tech_val_acc,
            **{f"acc_{f}": acc_fields[f] for f in FIELDS},
            **{f"halluc_{f}": halluc_fields[f] for f in FIELDS},
        })

    df = pd.DataFrame(rows)
    out_csv = Path(args.outdir) / "scores_long.csv"
    df.to_csv(out_csv, index=False)

    # summary by model/prompt
    grp = df[df["status"] == "ok"].groupby(["model", "prompt_name", "use_llm"], dropna=False)
    summ = grp.mean(numeric_only=True).reset_index()
    out_sum = Path(args.outdir) / "scores_summary.csv"
    summ.to_csv(out_sum, index=False)

    print(f"[ok] wrote {out_csv}")
    print(f"[ok] wrote {out_sum}")


if __name__ == "__main__":
    main()
