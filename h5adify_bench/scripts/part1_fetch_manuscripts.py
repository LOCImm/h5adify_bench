#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
part1_fetch_manuscripts.py

Given a DOI list/config, fetch manuscript assets:
- Crossref metadata (title, journal, year)
- Unpaywall best OA location (landing page + PDF URL if available)
- Save:
    papers/<doi_slug>/crossref.json
    papers/<doi_slug>/unpaywall.json
    papers/<doi_slug>/paper.html        (if available)
    papers/<doi_slug>/paper.pdf         (if available)
    papers/<doi_slug>/manifest.json
"""

from __future__ import annotations

import os
import re
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
import yaml
from tqdm import tqdm


def doi_slug(doi: str) -> str:
    doi = doi.strip().lower()
    doi = re.sub(r"^https?://doi\.org/", "", doi)
    doi = re.sub(r"^doi:", "", doi)
    return re.sub(r"[^a-z0-9._-]+", "_", doi)


def read_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def safe_write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def http_get(url: str, headers: Optional[Dict[str, str]] = None, timeout: int = 30) -> requests.Response:
    return requests.get(url, headers=headers or {}, timeout=timeout, allow_redirects=True)


def download_file(url: str, out_path: Path, headers: Optional[Dict[str, str]] = None, timeout: int = 60) -> bool:
    try:
        r = requests.get(url, headers=headers or {}, timeout=timeout, stream=True, allow_redirects=True)
        if r.status_code != 200:
            return False
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 256):
                if chunk:
                    f.write(chunk)
        return out_path.exists() and out_path.stat().st_size > 50_000
    except Exception:
        return False


def crossref_work(doi: str) -> Optional[Dict[str, Any]]:
    doi_clean = re.sub(r"^https?://doi\.org/", "", doi.strip())
    url = f"https://api.crossref.org/works/{doi_clean}"
    headers = {"User-Agent": "h5adify-bench/1.0 (mailto:you@example.org)"}
    try:
        r = http_get(url, headers=headers, timeout=30)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


def unpaywall(doi: str, email: str) -> Optional[Dict[str, Any]]:
    doi_clean = re.sub(r"^https?://doi\.org/", "", doi.strip())
    url = f"https://api.unpaywall.org/v2/{doi_clean}?email={email}"
    try:
        r = http_get(url, headers={"User-Agent": "h5adify-bench/1.0"}, timeout=30)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


def pick_dois(cfg: Dict[str, Any]) -> List[str]:
    # supports:
    # - papers: [{doi: ...}, ...]
    # - doi_list: [...]
    # - top-level list
    if isinstance(cfg, list):
        return [str(x.get("doi") if isinstance(x, dict) else x) for x in cfg]
    if "papers" in cfg and isinstance(cfg["papers"], list):
        return [str(p.get("doi")) for p in cfg["papers"] if p.get("doi")]
    if "doi_list" in cfg and isinstance(cfg["doi_list"], list):
        return [str(x) for x in cfg["doi_list"]]
    raise ValueError("Could not find DOIs in config. Expected keys: papers / doi_list / list.")


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--doi-config", required=True, help="YAML with list of DOIs (papers/doi_list)")
    ap.add_argument("--outdir", default="papers", help="Output folder for paper assets")
    ap.add_argument("--unpaywall-email", default=os.environ.get("UNPAYWALL_EMAIL", "h5adify@example.org"))
    ap.add_argument("--sleep", type=float, default=0.2, help="polite delay between requests")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    cfg = read_yaml(args.doi_config)
    dois = [d for d in pick_dois(cfg) if d and d != "None"]

    headers_html = {
        "User-Agent": "h5adify-bench/1.0 (Academic Research)",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }
    headers_pdf = {"User-Agent": "h5adify-bench/1.0 (Academic Research)", "Accept": "application/pdf,*/*"}

    for doi in tqdm(dois, desc="Fetch manuscript assets"):
        slug = doi_slug(doi)
        pdir = outdir / slug
        pdir.mkdir(parents=True, exist_ok=True)

        # Crossref
        cr = crossref_work(doi)
        if cr:
            safe_write_json(pdir / "crossref.json", cr)

        # Unpaywall
        up = unpaywall(doi, email=args.unpaywall_email)
        if up:
            safe_write_json(pdir / "unpaywall.json", up)

        # Try OA HTML landing + OA PDF
        html_url = None
        pdf_url = None
        if up and isinstance(up, dict):
            best = up.get("best_oa_location") or {}
            pdf_url = best.get("url_for_pdf") or None
            html_url = best.get("url_for_landing_page") or best.get("url") or None

        # Download HTML
        if html_url:
            try:
                r = http_get(html_url, headers=headers_html, timeout=30)
                ct = (r.headers.get("content-type") or "").lower()
                if r.status_code == 200 and ("html" in ct or "<html" in r.text.lower()):
                    (pdir / "paper.html").write_text(r.text, encoding="utf-8", errors="ignore")
            except Exception:
                pass

        # Download PDF
        if pdf_url:
            _ = download_file(pdf_url, pdir / "paper.pdf", headers=headers_pdf, timeout=90)

        # Minimal manifest
        manifest = {
            "doi": doi,
            "slug": slug,
            "html_url": html_url,
            "pdf_url": pdf_url,
            "has_html": (pdir / "paper.html").exists(),
            "has_pdf": (pdir / "paper.pdf").exists(),
        }
        safe_write_json(pdir / "manifest.json", manifest)

        time.sleep(float(args.sleep))


if __name__ == "__main__":
    main()
