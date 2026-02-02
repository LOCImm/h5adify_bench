#!/usr/bin/env python3
# scripts/part2_eval_prompt_variants.py

from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import yaml

from eval_normalize import load_synonyms
from eval_adapters import gold_to_canonical, pred_to_canonical
from part1_eval_compare_methods import score_one, aggregate  # reuse (no changes)

def read_json(p: Path) -> Any:
    return json.loads(p.read_text(encoding="utf-8"))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True)
    ap.add_argument("--field-map", required=True)
    ap.add_argument("--normalize", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--root", required=True, help="results_part2/")
    ap.add_argument("--models", nargs="+", required=True, help="subdirs under root (e.g. qwen2.5_3b llama3_latest)")
    ap.add_argument("--variants", nargs="+", default=["baseline", "avatar", "textgrad"])
    ap.add_argument("--outdir", default="eval_part2")
    args = ap.parse_args()

    gold_json = read_json(Path(args.gold))
    gold_items = gold_json.get("items") or gold_json.get("datasets") or []
    gold_by_doi = {it["doi"]: gold_to_canonical(it) for it in gold_items if it.get("doi")}

    field_map = yaml.safe_load(Path(args.field_map).read_text(encoding="utf-8"))
    inv = load_synonyms(yaml.safe_load(Path(args.normalize).read_text(encoding="utf-8")))
    weights = yaml.safe_load(Path(args.weights).read_text(encoding="utf-8")).get("weights", {})

    root = Path(args.root)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    big = {"models": {}}

    for model in args.models:
        model_dir = root / model
        big["models"][model] = {"variants": {}}

        for variant in args.variants:
            vdir = model_dir / variant
            per_doi = {}

            for fp in sorted(vdir.glob("*.json")):
                obj = read_json(fp)
                doi = obj.get("doi") or (obj.get("schema", {}) or {}).get("doi") or obj.get("deterministic_facts", {}).get("doi")
                if not doi or doi not in gold_by_doi:
                    continue

                pred_can = pred_to_canonical(obj, field_map, "llm")  # Part2 outputs are typically schema-rooted
                gold_can = gold_by_doi[doi]
                per_doi[doi] = score_one(pred_can, gold_can, inv)

            summ = aggregate(per_doi, weights)
            big["models"][model]["variants"][variant] = {
                "n_compared": len(per_doi),
                "summary": summ
            }

        # compute deltas vs baseline
        base = big["models"][model]["variants"].get("baseline", {}).get("summary", {})
        for variant in args.variants:
            if variant == "baseline":
                continue
            cur = big["models"][model]["variants"][variant]["summary"]
            big["models"][model]["variants"][variant]["delta_vs_baseline"] = {
                "composite_weighted_score": cur["composite_weighted_score"] - base.get("composite_weighted_score", 0.0),
                "macro_hallucination_fp_rate": cur["macro_hallucination_fp_rate"] - base.get("macro_hallucination_fp_rate", 0.0)
            }

    (outdir / "part2_prompt_variant_summary.json").write_text(json.dumps(big, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"✅ Part2 evaluation summary written to {outdir / 'part2_prompt_variant_summary.json'}")

if __name__ == "__main__":
    main()
