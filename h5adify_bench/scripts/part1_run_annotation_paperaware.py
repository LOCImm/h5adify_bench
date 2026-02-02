#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
part1_run_annotation_paperaware.py

Paper-aware run:
- loads each dataset h5ad
- runs DeterministicExtractor on h5ad
- runs DeterministicExtractor on DOI (Crossref/PMC/EuropePMC/Unpaywall HTML)
- additionally loads papers/<doi_slug>/paper_fulltext.txt (PDF+HTML extraction) and adds it as EvidenceStore chunks
- runs LLM extraction (Stage C) on enriched evidence

Outputs one JSON per dataset into outdir.
"""

from __future__ import annotations

import os
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from h5adify.annotation.evidence_store import EvidenceStore, SourceType
from h5adify.annotation.deterministic import DeterministicExtractor
from h5adify.annotation.llm_extractor import LLMExtractor, OllamaClient
from h5adify.annotation.prompt_store import PromptStore
from h5adify.annotation.verifier import AnnotationVerifier


def doi_slug(doi: str) -> str:
    doi = doi.strip().lower()
    doi = re.sub(r"^https?://doi\.org/", "", doi)
    doi = re.sub(r"^doi:", "", doi)
    return re.sub(r"[^a-z0-9._-]+", "_", doi)


def read_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def add_long_text_in_chunks(
    store: EvidenceStore,
    text: str,
    source_type: SourceType,
    source_name: str,
    section: str,
    chunk_chars: int = 2500,
) -> None:
    text = (text or "").strip()
    if not text:
        return
    for i in range(0, len(text), chunk_chars):
        sub = text[i : i + chunk_chars]
        store.add_text(sub, source_type=source_type, source_name=source_name, section=section, chunk_index=i // chunk_chars)


def pick_items(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    # expected:
    # papers: [{doi, datasets:[{h5ad_path or url or local_path, id/name}]}]
    # but we keep it flexible:
    if isinstance(cfg, dict) and "papers" in cfg:
        return cfg["papers"]
    if isinstance(cfg, list):
        return cfg
    raise ValueError("Config must contain 'papers' list or be a list.")


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--doi-config", required=True, help="configs/doi20.yaml (must include local h5ad paths)")
    ap.add_argument("--papers-dir", default="papers", help="papers/<doi_slug>/paper_fulltext.txt")
    ap.add_argument("--outdir", default="results_part1_paperaware", help="Output folder")
    ap.add_argument("--ollama-url", default="http://localhost:11434")
    ap.add_argument("--ollama-model", default=os.environ.get("H5ADIFY_MODEL", "qwen2.5:3b"))
    ap.add_argument("--prompt-name", default="extraction_v2_default")
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    cfg = read_yaml(args.doi_config)
    items = pick_items(cfg)

    det = DeterministicExtractor()
    client = OllamaClient(base_url=args.ollama_url, model=args.ollama_model)
    store = PromptStore()
    extractor = LLMExtractor(client=client, prompt_store=store, prompt_name=args.prompt_name)

    for p in items:
        doi = str(p.get("doi") or "").strip()
        if not doi:
            continue

        datasets = p.get("datasets") or []
        for ds in datasets:
            h5ad_path = ds.get("h5ad_path") or ds.get("local_h5ad") or ds.get("path")
            ds_id = ds.get("id") or ds.get("name") or Path(h5ad_path).stem

            if not h5ad_path:
                continue

            import anndata as ad
            adata = ad.read_h5ad(h5ad_path)

            ev = EvidenceStore()

            # Stage A: h5ad
            facts, ev = det.extract_from_h5ad(adata, ev)

            # Stage A: DOI (Crossref/PMC/EuropePMC/Unpaywall HTML when accessible)
            doi_facts, ev = det.extract_from_doi(doi, ev)
            facts.doi = doi_facts.doi
            facts.title = doi_facts.title
            facts.year = doi_facts.year
            facts.journal = doi_facts.journal

            # Add PDF/HTML fulltext extracted locally
            slug = doi_slug(doi)
            txt_path = Path(args.papers_dir) / slug / "paper_fulltext.txt"
            if txt_path.exists():
                paper_text = txt_path.read_text(encoding="utf-8", errors="ignore")
                add_long_text_in_chunks(ev, paper_text, SourceType.PDF, "paper_fulltext_local", "paper_fulltext_local")

            # Stage C: constrained extraction
            extraction = extractor.extract(ev, facts)
            schema = extractor.build_schema(extraction, facts)

            # Stage D: optional verification
            if args.verify and client.available:
                verifier = AnnotationVerifier(ev, client)
                schema = verifier.verify_schema(schema)

            result = {
                "dataset_id": ds_id,
                "h5ad_path": h5ad_path,
                "doi": doi,
                "model": client.model,
                "prompt_name": args.prompt_name,
                "evidence_sources": ev.get_source_summary(),
                "deterministic_facts": facts.to_dict(),
                "extraction_raw": extraction,
                "schema": schema.to_dict(),
            }

            out_path = outdir / f"{ds_id}.paperaware.json"
            out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"✅ Done. Results in: {outdir}")


if __name__ == "__main__":
    main()
