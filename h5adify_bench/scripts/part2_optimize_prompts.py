#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from common import read_yaml, read_json, ensure_dir, write_json, now_iso

from h5adify.annotation.llm_extractor import LLMExtractor, OllamaClient
from h5adify.annotation.prompt_store import PromptStore
from h5adify.annotation.optimization import optimize_prompt_avatar_style, optimize_prompt_textgrad_ollama
from h5adify.core.metadata_harmonize import load_metadata_vocab


FIELDS = ["batch", "sample", "donor", "domain", "sex", "species", "technology"]


def build_candidates(obs_columns: List[str], vocab: Dict[str, Any]) -> Dict[str, List[str]]:
    # mimic h5adify.core.metadata_harmonize candidate logic (stable & local)
    def norm(s: str) -> str:
        return (s or "").strip().lower().replace("-", "_").replace(" ", "_")

    field_syn = vocab.get("field_synonyms", {})
    out = {}
    for f in FIELDS:
        syns = field_syn.get(f, []) or [f]
        syns_norm = {norm(s) for s in syns}
        hits = []
        for k in obs_columns:
            kn = norm(k)
            if kn in syns_norm:
                hits.append(k)
        if not hits:
            for k in obs_columns:
                kn = norm(k)
                for s in syns_norm:
                    if s and (s in kn or kn in s):
                        hits.append(k); break
        # unique preserve order
        seen = set()
        uniq = []
        for x in hits:
            if x not in seen:
                seen.add(x); uniq.append(x)
        out[f] = uniq
    return out


def make_train_jsonl(gold: Dict[str, Any], out_path: Path, seed: int = 42, train_frac: float = 0.6) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    items = gold["items"]
    idx = np.arange(len(items))
    rng.shuffle(idx)
    n_train = int(len(items) * train_frac)
    train_idx = set(idx[:n_train].tolist())

    vocab = load_metadata_vocab(None)
    lines = []
    split = {"train": [], "test": []}

    for i, it in enumerate(items):
        obs_cols = it["obs_columns"]
        cand = build_candidates(obs_cols, vocab)
        evidence = json.dumps({"obs_columns": obs_cols, "candidates": cand}, indent=2, ensure_ascii=False)

        expected = {}
        for f in FIELDS:
            # expected = gold_key mapping (None allowed)
            expected[f] = it["gold_key"].get(f, None)

        obj = {
            "id": f"{it['doi']}::{it['dataset_id']}",
            "evidence": evidence,
            "dataset_facts": {"obs_columns": obs_cols, "key_candidates": cand},
            "expected_extraction": expected,
        }
        lines.append(obj)
        (split["train"] if i in train_idx else split["test"]).append(obj["id"])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for obj in lines:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    return {"train_ids": split["train"], "test_ids": split["test"], "n": len(lines)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="configs/models.yaml")
    ap.add_argument("--gold", default="gold/doi20_gold.json")
    ap.add_argument("--method", choices=["avatar", "textgrad"], required=True)
    ap.add_argument("--base-prompt", default="metadata_harmonize_v1_default")
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--worst-k", type=int, default=3)
    ap.add_argument("--opt-model", default="qwen2.5:3b", help="Model used to optimize the prompt")
    ap.add_argument("--outdir", default="prompts_opt")
    args = ap.parse_args()

    cfg = read_yaml(args.models)
    base_url = cfg.get("ollama_base_url", "http://localhost:11434")

    gold = read_json(args.gold)

    outdir = ensure_dir(args.outdir)
    train_jsonl = outdir / "train_metadata_mapping.jsonl"
    split = make_train_jsonl(gold, train_jsonl)
    write_json(outdir / "split.json", split)

    store = PromptStore()
    base_text = store.load_prompt_text(args.base_prompt)

    client = OllamaClient(base_url=base_url, model=args.opt_model)
    extractor = LLMExtractor(client=client, prompt_store=store, prompt_template=base_text)

    if args.method == "avatar":
        best_prompt, hist = optimize_prompt_avatar_style(
            extractor=extractor,
            train_jsonl=str(train_jsonl),
            base_prompt_name=args.base_prompt,
            prompt_store=store,
            steps=args.steps,
            temperature=args.temperature,
        )
        new_name = "metadata_harmonize_v1_avatar_opt"
    else:
        best_prompt, hist = optimize_prompt_textgrad_ollama(
            extractor=extractor,
            train_jsonl=str(train_jsonl),
            base_prompt_name=args.base_prompt,
            prompt_store=store,
            steps=args.steps,
            temperature=args.temperature,
            worst_k=args.worst_k,
        )
        new_name = "metadata_harmonize_v1_textgrad_opt"

    # Save to user prompt store (h5adify home) and to local folder
    store.save_prompt_text(new_name, best_prompt, activate=False, overwrite=True)
    (outdir / f"{new_name}.txt").write_text(best_prompt, encoding="utf-8")

    # history
    hist_out = [h.__dict__ for h in hist]
    write_json(outdir / f"{new_name}.history.json", {"created_at": now_iso(), "method": args.method, "history": hist_out})

    print(f"[ok] wrote optimized prompt: {new_name}")
    print(f"[ok] local copy: {outdir}/{new_name}.txt")
    print("[next] re-run Part 1 with: --prompt-name", new_name)


if __name__ == "__main__":
    main()
