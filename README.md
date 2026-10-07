# HetGFD — reproduction code

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

## Results

Mean ± std over 10 seeds (S = structural, U = uniform missing). Full numbers and selected hyper-parameters: `results/`.

**Table 1 — node classification (HGT, macro-F1)**

| Method | ACM (S) | ACM (U) | DBLP (S) | DBLP (U) | IMDB (S) | IMDB (U) |
|---|---|---|---|---|---|---|
| Zero | 80.00 ± 1.44 | 80.18 ± 1.10 | 90.78 ± 1.13 | 89.83 ± 1.23 | 44.08 ± 1.37 | 43.96 ± 1.53 |
| Mean | 80.03 ± 1.18 | 80.34 ± 1.23 | 89.98 ± 2.07 | 90.06 ± 1.49 | 43.64 ± 1.22 | 43.58 ± 1.82 |
| kNN | 80.26 ± 1.54 | 79.80 ± 1.71 | 90.83 ± 0.99 | 88.61 ± 2.37 | 43.72 ± 1.24 | 43.42 ± 1.70 |
| SVD | 80.28 ± 1.21 | 80.46 ± 1.15 | 90.99 ± 0.70 | 90.21 ± 1.01 | 43.73 ± 1.43 | 44.07 ± 1.97 |
| FP | 82.35 ± 1.48 | 83.34 ± 1.08 | 89.99 ± 3.34 | 89.86 ± 1.60 | 43.97 ± 1.36 | 43.97 ± 1.10 |
| PCFI+VF | 83.33 ± 1.09 | 83.60 ± 1.43 | 90.80 ± 1.61 | 91.51 ± 0.69 | 45.74 ± 1.38 | 46.58 ± 1.80 |
| HetGFD | 84.02 ± 1.24 | 86.52 ± 1.06 | 91.98 ± 0.69 | 92.35 ± 0.78 | 44.71 ± 2.06 | 46.62 ± 1.24 |

**Table 2 — link prediction (HGT, ROC-AUC)**

| Method | ACM (S) | ACM (U) | DBLP (S) | DBLP (U) | IMDB (S) | IMDB (U) |
|---|---|---|---|---|---|---|
| Zero | 71.23 ± 1.79 | 71.62 ± 2.17 | 73.56 ± 1.37 | 73.29 ± 1.74 | 93.67 ± 1.10 | 93.47 ± 1.35 |
| Mean | 71.84 ± 2.50 | 71.18 ± 1.87 | 73.00 ± 1.77 | 72.09 ± 2.21 | 90.56 ± 6.94 | 93.06 ± 2.23 |
| kNN | 70.82 ± 1.62 | 71.64 ± 1.22 | 72.73 ± 1.42 | 73.40 ± 1.39 | 92.82 ± 2.44 | 93.86 ± 0.98 |
| SVD | 71.76 ± 1.90 | 72.07 ± 1.49 | 73.01 ± 1.49 | 73.38 ± 1.25 | 92.66 ± 3.52 | 92.77 ± 3.12 |
| FP | 73.13 ± 1.23 | 74.26 ± 1.31 | 72.84 ± 1.69 | 73.13 ± 1.58 | 91.97 ± 5.02 | 92.52 ± 3.01 |
| PCFI+VF | 75.50 ± 1.24 | 72.86 ± 1.39 | 72.61 ± 1.65 | 73.42 ± 2.24 | 92.79 ± 3.30 | 90.76 ± 6.44 |
| HetGFD | 82.30 ± 0.67 | 82.80 ± 1.08 | 93.70 ± 0.50 | 92.23 ± 0.60 | 91.38 ± 2.46 | 90.70 ± 7.67 |

**Figure 4 / Table 18 — node classification (HGNN-AC, macro-F1, average over ACM / DBLP / IMDB; per dataset in `results/hgnnac.csv`)**

| Missing (rate) | Zero | Mean | FP | PCFI+VF | HetGFD |
|---|---|---|---|---|---|
| S (0.5) | 72.79 | 71.94 | 76.15 | 76.12 | 75.87 |
| S (0.9) | 61.44 | 59.43 | 72.65 | 72.99 | 73.27 |
| S (0.995) | 24.24 | 17.28 | 34.79 | 65.85 | 66.30 |
| U (0.5) | 74.41 | 74.94 | 77.76 | 77.48 | 76.90 |
| U (0.9) | 63.82 | 62.83 | 74.01 | 74.48 | 75.48 |
| U (0.995) | 31.57 | 17.28 | 43.56 | 71.76 | 71.54 |

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
