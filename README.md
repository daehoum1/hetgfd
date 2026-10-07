# HetGFD (ICLR 2025)

Code for reproducing the experiments of

> Daeho Um, Yoonji Lee, Jiwoong Park, Seulki Park, Yuneil Yeo, Seong Jin Ahn.
> **Relation-Aware Diffusion for Heterogeneous Graphs with Partially Observed Features.** ICLR 2025.

| Paper | Experiment | Command |
|---|---|---|
| Table 1 | node classification with HGT | `scripts/run_hgt.py --tasks nc` |
| Table 2 | link prediction with HGT | `scripts/run_hgt.py --tasks lp` |
| Figure 4 / Table 18 | node classification with HGNN-AC | `scripts/run_hgnnac.py` |

## Setup

```bash
pip install -r requirements.txt     # PyTorch, PyTorch Geometric (DGL is not needed)
python scripts/download_data.py      # ACM / DBLP / IMDB (HGNN-AC preprocessed) -> data/preprocessed
bash scripts/setup_hgnnac.sh         # official HGNN-AC code -> third_party/HGNN-AC (Figure 4 / Table 18 only)
```

If Google Drive is not reachable, download the archive manually
(https://drive.google.com/file/d/1PqUjvSViICa8yOszqDrw-j96hXVJ0MHR/view) and run `python scripts/download_data.py --zip <file>`.

## Run

`scripts/launch.sh` starts one worker per GPU on a shared, resumable work queue (re-run the same command to resume).

```bash
scripts/launch.sh run_hgt.py    runs/table1 --tasks nc --hgt text --methods zero mean knn svd fp pcfi hetgfd
scripts/launch.sh run_hgt.py    runs/table2 --tasks lp --methods zero mean knn svd fp pcfi hetgfd
scripts/launch.sh run_hgnnac.py runs/figure4
```

Single GPU: `CUDA_VISIBLE_DEVICES=0 python scripts/run_hgt.py --workdir runs/table1 --worker gpu0 --launch L1 --tasks nc --hgt text`.
Quick check: add `--smoke` (tiny grids, few epochs). Other options: `--datasets`, `--modes node entry`, `--methods`,
`--sel-seeds`, `--final-seeds`; for HGNN-AC `--cells`, `--sel-masks`, `--final-masks`.

Each run is appended to `<workdir>/results/<setting>/*.jsonl`; `<workdir>/results/summary_<setting>.csv` holds mean ± std per method.

## Protocol

* Missing rate 0.995 for HGT (structural = whole nodes, uniform = single entries); 0.5 / 0.9 / 0.995 for HGNN-AC.
* Hyper-parameters of all imputation methods (HetGFD α, β; PCFI+VF α, γ; kNN k; SVD rank) are selected on the validation set
  using seeds 0–2 (HGNN-AC: masks 0–1); final results are averaged over 10 seeds / masks.
* HGT: grid over layers {1, 2, 3} × learning rate {0.1, 0.01, 0.001, 0.0001} selected on validation.
  Node classification uses `--hgt text` (2 heads, dropout 0.2, GELU, selection by validation macro-F1).
* Link prediction: ROC-AUC on held-out target edges (ACM paper–author, DBLP author–paper, IMDB movie–director; 85/5/10 split).
* HGNN-AC: official models and hyper-parameters, macro-F1 on the official test split.

## Layout

```
hetgfd_repro/   imputation (hetgfd.py, baselines.py), masks, data loading, HGT node classification, link prediction
scripts/        run_hgt.py, run_hgnnac.py, launch.sh, download_data.py, setup_hgnnac.sh
dgl_shim/       minimal DGL stand-in used by the official HGNN-AC code
results/        result summaries (CSV)
```

## Citation

```bibtex
@inproceedings{um2025hetgfd,
  title     = {Relation-Aware Diffusion for Heterogeneous Graphs with Partially Observed Features},
  author    = {Um, Daeho and Lee, Yoonji and Park, Jiwoong and Park, Seulki and Yeo, Yuneil and Ahn, Seong Jin},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2025}
}
```
