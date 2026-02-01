#!/usr/bin/env python3
from __future__ import annotations

import argparse
import tarfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import anndata as ad
from tqdm import tqdm

from common import (
    read_yaml, read_json, write_json, ensure_dir, now_iso,
    canonicalize_species_to_h5adify, knn_batch_entropy
)

from h5adify.core import merge_datasets, harmonize_metadata, DEFAULT_METADATA_FIELDS, infer_sex_from_expression

import cellxgene_census


def download_file(url: str, out_path: Path, chunk: int = 1024 * 1024) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and out_path.stat().st_size > 0:
        return
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(out_path, "wb") as f:
            for b in r.iter_content(chunk_size=chunk):
                if b:
                    f.write(b)


def extract_tar(tar_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    marker = out_dir / ".extracted.ok"
    if marker.exists():
        return
    with tarfile.open(tar_path, "r:gz") as tf:
        tf.extractall(path=out_dir)
    marker.write_text("ok", encoding="utf-8")


def find_h5ads(root: Path) -> List[Path]:
    return sorted(root.glob("**/*.h5ad"))


def census_find_gbm_datasets(df: pd.DataFrame, terms: List[str], species: str) -> pd.DataFrame:
    # filters by organism + query in title/collection/citation fields
    d = df.copy()
    if "organism" in d.columns:
        d = d[d["organism"].astype(str) == species]
    # query terms in title or citation
    mask = None
    for t in terms:
        t = t.lower()
        m = None
        for col in ["dataset_title", "collection_name", "citation"]:
            if col in d.columns:
                mm = d[col].astype(str).str.lower().str.contains(t, na=False)
                m = mm if m is None else (m | mm)
        mask = m if mask is None else (mask | m)
    if mask is not None:
        d = d[mask]
    # sort by size if possible
    for size_col in ["dataset_total_cell_count", "n_obs", "dataset_n_obs"]:
        if size_col in d.columns:
            d = d.sort_values(size_col, ascending=False)
            break
    return d


def download_census_h5ads(out_dir: Path, census_version: str, terms: List[str], species_list: List[str],
                         max_per_species: int, max_obs: int, seed: int = 0) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    with cellxgene_census.open_soma(census_version=census_version) as census:
        df = census["census_info"]["datasets"].read().concat().to_pandas()

    out_paths = []
    for sp in species_list:
        sub = census_find_gbm_datasets(df, terms, sp).head(max_per_species)
        for _, row in sub.iterrows():
            dsid = str(row["dataset_id"])
            p = out_dir / sp.replace(" ", "_") / f"{dsid}.source.h5ad"
            p.parent.mkdir(parents=True, exist_ok=True)
            if not p.exists():
                cellxgene_census.download_source_h5ad(dsid, to_path=str(p), census_version=census_version)

            # make a small copy for faster merging
            small = p.with_suffix("").with_suffix(".small.h5ad")
            if not small.exists():
                a = ad.read_h5ad(p)
                if a.n_obs > max_obs:
                    rng = np.random.default_rng(seed)
                    idx = rng.choice(a.n_obs, size=max_obs, replace=False)
                    a = a[idx].copy()
                a.write_h5ad(small)
            out_paths.append(small)
    return out_paths


def compute_embeddings_and_entropy(adata: ad.AnnData, batch_col: str, out_dir: Path) -> Dict[str, float]:
    out = {}
    # PCA baseline
    try:
        import scanpy as sc
        b = adata.obs[batch_col].astype(str).values
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        sc.pp.highly_variable_genes(adata, n_top_genes=3000)
        ad2 = adata[:, adata.var["highly_variable"]].copy()
        sc.pp.scale(ad2, max_value=10)
        sc.tl.pca(ad2, n_comps=50)
        emb = ad2.obsm["X_pca"]
        out["pca_knn_batch_entropy"] = knn_batch_entropy(emb, b, k=30)
    except Exception:
        pass
    return out


def pseudo_bulk_by_donor(adata: ad.AnnData, donor_col: str, sex_col: str) -> Tuple[pd.DataFrame, pd.Series]:
    # returns (expr_df donors x genes, sex_series)
    donors = adata.obs[donor_col].astype(str)
    sexes = adata.obs[sex_col].astype(str)
    # mean expression per donor (works on dense or sparse)
    X = adata.X
    genes = adata.var_names.astype(str)
    df = pd.DataFrame(index=sorted(donors.unique()), columns=genes, dtype=float)

    for d in df.index:
        m = (donors.values == d)
        if m.sum() < 10:
            continue
        sub = X[m]
        # mean row
        try:
            mu = np.asarray(sub.mean(axis=0)).ravel()
        except Exception:
            mu = np.array(sub.mean(axis=0)).ravel()
        df.loc[d, :] = mu

    sex_by_donor = sexes.groupby(donors).agg(lambda x: x.value_counts().index[0])
    return df.dropna(axis=0, how="all"), sex_by_donor


def de_sex_meta(df_expr: pd.DataFrame, sex: pd.Series) -> pd.DataFrame:
    # Simple Wilcoxon per gene across donors (female vs male), with BH FDR
    from scipy.stats import ranksums

    common = df_expr.index.intersection(sex.index)
    df_expr = df_expr.loc[common]
    sex = sex.loc[common].astype(str)

    f = df_expr[sex == "female"]
    m = df_expr[sex == "male"]
    if len(f) < 2 or len(m) < 2:
        return pd.DataFrame()

    rows = []
    for g in df_expr.columns:
        xf = f[g].astype(float).dropna()
        xm = m[g].astype(float).dropna()
        if len(xf) < 2 or len(xm) < 2:
            continue
        stat, p = ranksums(xf, xm)
        l2fc = float(np.log2((xm.mean() + 1e-6) / (xf.mean() + 1e-6)))
        rows.append({"gene": g, "p": float(p), "log2fc_male_vs_female": l2fc})

    out = pd.DataFrame(rows).sort_values("p")
    # BH FDR
    pvals = out["p"].values
    n = len(pvals)
    rank = np.arange(1, n + 1)
    q = pvals * n / rank
    q = np.minimum.accumulate(q[::-1])[::-1]
    out["q_bh"] = q
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="configs/models.yaml")
    ap.add_argument("--config", default="configs/gbm.yaml")
    args = ap.parse_args()

    cfg_models = read_yaml(args.models)
    base_url = cfg_models.get("ollama_base_url", "http://localhost:11434")
    # pick a default metadata model for this pipeline (you can change)
    meta_model = (cfg_models["models"][0]["name"] if cfg_models.get("models") else "qwen2.5:3b")

    cfg = read_yaml(args.config)
    out_root = ensure_dir(cfg["merge"]["out_dir"])
    run_meta = {"created_at": now_iso(), "steps": []}

    # 1) GBM-Space download/extract
    gbm_dir = ensure_dir(cfg["gbmspace"]["out_dir"])
    sn_tar = gbm_dir / "GBM_space_snRNA.tar.gz"
    vi_tar = gbm_dir / "spatial_data_visium.tar.gz"

    print("[i] Downloading GBM-Space tarballs ...")
    download_file(cfg["gbmspace"]["snrna_tar_url"], sn_tar)
    download_file(cfg["gbmspace"]["visium_tar_url"], vi_tar)

    sn_ex = gbm_dir / "snrna_extracted"
    vi_ex = gbm_dir / "visium_extracted"
    extract_tar(sn_tar, sn_ex)
    extract_tar(vi_tar, vi_ex)

    sn_h5ads = find_h5ads(sn_ex)
    vi_h5ads = find_h5ads(vi_ex)
    run_meta["steps"].append({"gbmspace_snrna_h5ads": [str(p) for p in sn_h5ads]})
    run_meta["steps"].append({"gbmspace_visium_h5ads": [str(p) for p in vi_h5ads]})

    # 2) Census GBM datasets (human/mouse/rat)
    census_cfg = cfg["census_gbm"]
    census_paths = download_census_h5ads(
        out_dir=Path(census_cfg["out_dir"]),
        census_version=census_cfg.get("census_version", "stable"),
        terms=census_cfg["query_terms"],
        species_list=census_cfg["species"],
        max_per_species=int(census_cfg["max_datasets_per_species"]),
        max_obs=int(census_cfg["max_obs_per_dataset"]),
        seed=42,
    )
    run_meta["steps"].append({"census_h5ads": [str(p) for p in census_paths]})

    # 3) Load a subset of GBM-Space + census into AnnData list
    adatas = []
    labels = []

    # Choose first snRNA h5ad if many
    for p in (sn_h5ads[:1] + vi_h5ads[:1] + census_paths):
        try:
            a = ad.read_h5ad(p)
            # metadata harmonize on each
            a, _ = harmonize_metadata(
                a,
                fields=list(DEFAULT_METADATA_FIELDS),
                use_llm=bool(cfg["merge"].get("metadata_use_llm", True)),
                llm_prompt_name=cfg["merge"].get("metadata_prompt_name", "metadata_harmonize_v1_default"),
                sex_from_expression=True,
                ollama_base_url=base_url,
                ollama_model=meta_model,
                inplace=False,
            )
            # ensure batch label exists
            if "h5adify_batch" not in a.obs.columns:
                a.obs["h5adify_batch"] = Path(p).stem
            adatas.append(a)
            labels.append(Path(p).stem)
        except Exception as e:
            run_meta["steps"].append({"load_error": str(p), "error": f"{type(e).__name__}: {e}"})

    # 4) Merge (uses h5adify.core.merge_datasets)
    if len(adatas) < 2:
        write_json(out_root / "run_meta.json", run_meta)
        print("[warn] not enough datasets loaded to merge")
        return

    print("[i] Merging datasets ...")
    merged = merge_datasets(
        datasets=adatas,
        batch_key="h5adify_batch",
        batch_labels=labels,
        join=cfg["merge"].get("join", "inner"),
        harmonize_first=False,
        target_species=cfg["merge"].get("target_species", "human"),
        harmonize_metadata_first=False,
        metadata_use_llm=False,
    )
    merged_path = out_root / "gbm_merged_raw.h5ad"
    merged.write_h5ad(merged_path)
    run_meta["steps"].append({"merged_h5ad": str(merged_path), "n_obs": int(merged.n_obs), "n_vars": int(merged.n_vars)})

    # 5) Batch mixing entropy (PCA)
    metrics = compute_embeddings_and_entropy(merged, batch_col="h5adify_batch", out_dir=out_root)
    write_json(out_root / "batch_metrics.json", metrics)

    # 6) Sex assignment (prefer h5adify_sex; else infer from expression)
    if "h5adify_sex" not in merged.obs.columns or merged.obs["h5adify_sex"].isna().all():
        # infer per donor if donor exists else per cell
        if "h5adify_donor" in merged.obs.columns and merged.obs["h5adify_donor"].notna().any():
            inferred = infer_sex_from_expression(merged, groupby="h5adify_donor", min_frac=0.02, min_group_obs=25)
            # broadcast to cells
            merged.obs["h5adify_sex"] = merged.obs["h5adify_donor"].astype(str).map(inferred).fillna("unknown")
        else:
            inferred = infer_sex_from_expression(merged, groupby="auto", min_frac=0.02, min_group_obs=25)
            merged.obs["h5adify_sex"] = merged.obs.index.astype(str).map(inferred).fillna("unknown")

    # 7) Sex-driven markers via pseudo-bulk by donor (if donor available)
    if "h5adify_donor" in merged.obs.columns and merged.obs["h5adify_donor"].notna().any():
        expr_df, sex_by_donor = pseudo_bulk_by_donor(merged, donor_col="h5adify_donor", sex_col="h5adify_sex")
        de = de_sex_meta(expr_df, sex_by_donor)
        de_path = out_root / "sex_markers_pseudobulk.csv"
        de.to_csv(de_path, index=False)
        run_meta["steps"].append({"sex_markers": str(de_path), "n_genes_tested": int(de.shape[0])})
    else:
        run_meta["steps"].append({"sex_markers": "skipped (no donor column)"})


    write_json(out_root / "run_meta.json", run_meta)
    print(f"[ok] outputs in {out_root}")


if __name__ == "__main__":
    main()
