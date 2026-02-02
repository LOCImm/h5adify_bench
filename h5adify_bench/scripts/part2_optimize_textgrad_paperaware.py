#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import os

from h5adify.annotation.llm_extractor import OllamaClient, LLMExtractor
from h5adify.annotation.prompt_store import PromptStore
from h5adify.annotation.optimization import optimize_prompt_textgrad_ollama


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--train-jsonl", required=True)
    ap.add_argument("--ollama-url", default="http://localhost:11434")
    ap.add_argument("--ollama-model", default=os.environ.get("H5ADIFY_MODEL", "qwen2.5:3b"))
    ap.add_argument("--base-prompt-name", default="extraction_v2_default")
    ap.add_argument("--new-prompt-name", default="extraction_v2_paperaware_textgrad")
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--temperature", type=float, default=0.1)
    args = ap.parse_args()

    store = PromptStore()
    client = OllamaClient(base_url=args.ollama_url, model=args.ollama_model)
    extractor = LLMExtractor(client=client, prompt_store=store, prompt_name=args.base_prompt_name)

    best_text, history = optimize_prompt_textgrad_ollama(
        extractor=extractor,
        train_jsonl_path=args.train_jsonl,
        steps=int(args.steps),
        temperature=float(args.temperature),
    )

    prompt_path = store.prompts_dir / f"{args.new_prompt_name}.txt"
    prompt_path.write_text(best_text, encoding="utf-8")
    store.set_active_prompt_name(args.new_prompt_name)

    print(f"✅ Saved optimized prompt to {prompt_path}")
    print(f"✅ Activated prompt: {args.new_prompt_name}")
    print(f"History steps: {len(history)}")


if __name__ == "__main__":
    main()
