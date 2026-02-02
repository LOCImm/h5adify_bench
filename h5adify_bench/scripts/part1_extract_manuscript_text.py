#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
part1_extract_manuscript_text.py

Reads papers/<doi_slug>/paper.pdf and/or paper.html and writes:
- papers/<doi_slug>/paper_pdf.txt   (from PDF)
- papers/<doi_slug>/paper_html.txt  (from HTML)
- papers/<doi_slug>/paper_fulltext.txt (best available concatenation)

Uses PyMuPDF for PDF text extraction.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from bs4 import BeautifulSoup

try:
    import pymupdf  # new name
except Exception:
    import fitz as pymupdf  # fallback


def clean_text(s: str) -> str:
    s = s.replace("\x00", " ")
    s = re.sub(r"\s+\n", "\n", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    s = re.sub(r"[ \t]{2,}", " ", s)
    return s.strip()


def extract_pdf_text(pdf_path: Path, max_pages: Optional[int] = None) -> str:
    doc = pymupdf.open(pdf_path)
    texts = []
    n = doc.page_count
    if max_pages is not None:
        n = min(n, int(max_pages))
    for i in range(n):
        page = doc.load_page(i)
        texts.append(page.get_text("text"))
    doc.close()
    return clean_text("\n".join(texts))


def extract_html_text(html_path: Path) -> str:
    html = html_path.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    return clean_text(text)


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--papers-dir", default="papers", help="Folder created by part1_fetch_manuscripts.py")
    ap.add_argument("--max-pdf-pages", type=int, default=None, help="Optional page cap for PDF parsing")
    args = ap.parse_args()

    papers_dir = Path(args.papers_dir)
    for pdir in sorted([p for p in papers_dir.iterdir() if p.is_dir()]):
        pdf = pdir / "paper.pdf"
        html = pdir / "paper.html"

        pdf_txt = ""
        html_txt = ""

        if pdf.exists():
            try:
                pdf_txt = extract_pdf_text(pdf, max_pages=args.max_pdf_pages)
                (pdir / "paper_pdf.txt").write_text(pdf_txt, encoding="utf-8")
            except Exception:
                pass

        if html.exists():
            try:
                html_txt = extract_html_text(html)
                (pdir / "paper_html.txt").write_text(html_txt, encoding="utf-8")
            except Exception:
                pass

        full = "\n\n".join([t for t in [pdf_txt, html_txt] if t])
        if full:
            (pdir / "paper_fulltext.txt").write_text(full, encoding="utf-8")


if __name__ == "__main__":
    main()
