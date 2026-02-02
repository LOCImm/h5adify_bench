#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
part2_build_training_jsonl_paperaware.py

Builds a JSONL file for prompt optimization:
Each line:
{
  "id": "...",
  "evidence": "...",           # includes paper_fulltext chunks + dataset facts summary
  "dataset_facts": {...},      # DatasetFacts.to_dict()
  "expected_extraction": {...} # your gold JSON for this dataset
}

You must provide a gold file mapping dataset_id -> expected_extraction.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List

import yaml

from h5adify.annotation.evidence_store import EvidenceStore, SourceType
from h5adify.annotation.deterministic import DeterministicExtractor
from h5adify.annotation.rag import RAGRetriever


def doi_slug(doi: str) -> str:
    doi = doi.strip().lower()
    doi = re.sub(r"^https?://doi\.org/", "", doi)
    doi = re.sub(r"^doi:", "", doi)
    return re.sub(r"[^a-z0-9._-]+", "_", doi)


def read_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_gold(path: str) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def add_long_text_in_chunks(store: EvidenceStore, text: str, stype: SourceType, sname: str, section: str, chunk_chars: int = 2500) -> None:
    text = (text or "").strip()
    if not text:
        return
    for i in range(0, len(text), chunk_chars):
        store.add_text(text[i : i + chunk_chars], source_type=stype, source_name=sname, section=section, chunk_index=i // chunk_chars)


def pick_items(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    if isinstance(cfg, dict) and "papers" in cfg:
        return cfg["papers"]
    if isinstance(cfg, list):
        return cfg
    raise ValueError("Config must contain 'papers' list or be a list.")


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--doi-config", required=True)
    ap.add_argument("--papers-dir", default="papers")
    ap.add_argument("--gold-json", required=True, help="dataset_id -> expected_extraction dict")
    ap.add_argument("--out-jsonl", default="train_paperaware.jsonl")
    ap.add_argument("--max-tokens", type=int, default=3000, help="RAG context size")
    args = ap.parse_args()

    gold = load_gold(args.gold_json)

    cfg = read_yaml(args.doi_config)
    items = pick_items(cfg)

    det = DeterministicExtractor()
    outp = Path(args.out_jsonl)
    outp.parent.mkdir(parents=True, exist_ok=True)

    n_written = 0
    with open(outp, "w", encoding="utf-8") as f:
        for p in items:
            doi = str(p.get("doi") or "").strip()
            if not doi:
                continue
            datasets = p.get("datasets") or []
            for ds in datasets:
                h5ad_path = ds.get("h5ad_path") or ds.get("local_h5ad") or ds.get("path")
                ds_id = ds.get("id") or ds.get("name") or Path(h5ad_path).stem
                if not h5ad_path or ds_id not in gold:
                    continue

                import anndata as ad
                adata = ad.read_h5ad(h5ad_path)

                ev = EvidenceStore()
                facts, ev = det.extract_from_h5ad(adata, ev)
                doi_facts, ev = det.extract_from_doi(doi, ev)
                facts.doi = doi_facts.doi
                facts.title = doi_facts.title
                facts.year = doi_facts.year
                facts.journal = doi_facts.journal

                slug = doi_slug(doi)
                txt_path = Path(args.papers_dir) / slug / "paper_fulltext.txt"
                if txt_path.exists():
                    paper_text = txt_path.read_text(encoding="utf-8", errors="ignore")
                    add_long_text_in_chunks(ev, paper_text, SourceType.PDF, "paper_fulltext_local", "paper_fulltext_local")

                rag = RAGRetriever(ev)
                evidence_text = rag.build_full_context(max_tokens=int(args.max_tokens))

                obj = {
                    "id": ds_id,
                    "evidence": evidence_text,
                    "dataset_facts": facts.to_dict(),
                    "expected_extraction": gold[ds_id],
                }
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")
                n_written += 1

    print(f"✅ Wrote {n_written} examples to {outp}")


if __name__ == "__main__":
    main()
