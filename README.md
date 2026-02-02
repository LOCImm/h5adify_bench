## Initial installation

```
bash scripts/install.sh /absolute/path/to/h5adify_v0.0.7_final.zip
```

For the analysis of the pdf and html documents, this should be installed.

```
python -m pip install -U pyyaml requests tqdm beautifulsoup4 lxml pymupdf
```


## Part 1-DOI20 benchmark: download → gold → run models → score

Download the datasets

```
source .venv/bin/activate
python scripts/part1_download_doi20.py --doi-config configs/doi20.yaml
```

### Run the part one

```
python scripts/part1_make_gold.py --manifest data/doi20/manifest.json --out gold/doi20_gold.json --use-small
```

### Run llm benchmark

```
python scripts/part1_run_benchmark.py --use-llm --prompt-name metadata_harmonize_v1_default --save-h5ad --prefer-small
```

### Run deterministic (no benchmark)

```
python scripts/part1_run_benchmark.py --prompt-name metadata_harmonize_v1_default --prefer-small
```

### Run scores of part 1

```
python scripts/part1_score.py --gold gold/doi20_gold.json --results results/part1 --outdir results/part1_scores
```

Analysis using the text / pdf or html of the associated manuscripts

```
python scripts/part1_fetch_manuscripts.py --doi-config configs/doi20.yaml --outdir papers
python scripts/part1_extract_manuscript_text.py --papers-dir papers
```

Run example with qwen2.5:3b

```
python scripts/part1_run_annotation_paperaware.py \
  --doi-config configs/doi20.yaml \
  --papers-dir papers \
  --outdir results_part1_paperaware \
  --ollama-model qwen2.5:3b \
  --verify
```



## Avatar vs TextGrad prompt optimization (metadata mapping prompt)

```
python scripts/part2_optimize_prompts.py --method avatar --opt-model qwen2.5:3b
python scripts/part2_optimize_prompts.py --method textgrad --opt-model qwen2.5:3b
```

Build JSONL training set (paper-aware)

```
python scripts/part2_build_training_jsonl_paperaware.py \
  --doi-config configs/doi20.yaml \
  --papers-dir papers \
  --gold-json gold/doi20_gold.json \
  --out-jsonl train_paperaware.jsonl
```

### The re-run Part 1 with each optimized prompt

```
python scripts/part1_run_benchmark.py --use-llm --prompt-name metadata_harmonize_v1_avatar_opt --save-h5ad --prefer-small
python scripts/part1_run_benchmark.py --use-llm --prompt-name metadata_harmonize_v1_textgrad_opt --save-h5ad --prefer-small
python scripts/part1_score.py --outdir results/part1_scores
```

Run Avatar-style optimization (paper-aware)

```
python scripts/part2_optimize_avatar_paperaware.py \
  --train-jsonl train_paperaware.jsonl \
  --ollama-model qwen2.5:3b
```

Run TextGrad-style optimization (paper-aware)

```
python scripts/part2_optimize_textgrad_paperaware.py \
  --train-jsonl train_paperaware.jsonl \
  --ollama-model qwen2.5:3b
```


## Part 3 — Simulations (5–10 synthetic h5ad) + evaluation

```
python scripts/part3_eval_simulations.py --use-llm --prompt-name metadata_harmonize_v1_default
```

## Part 4 — GBM analysis: GBM-Space + extra GBM datasets → merge → integration → sex markers

```
python scripts/part4_gbm_pipeline.py --config configs/gbm.yaml --models configs/models.yaml
```

Probably here there could be used other additional ways to deal with the batch effects as analysis using [Harmony](https://github.com/lilab-bcb/harmony-pytorch) or [scVI](https://docs.scvi-tools.org/en/stable/tutorials/notebooks/quick_start/api_overview.html)


