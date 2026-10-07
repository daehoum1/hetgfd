"""HetGFD Table 1 (node classification) / Table 2 (link prediction) with HGT, one unified protocol
alpha·beta chosen on the validation set, 10 seeds.

Settings: task {nc, lp} x dataset {ACM, DBLP, IMDB} x missing type {node (structural), entry (uniform)}, r_m 0.995.

  stage 1 (selection): seeds SEL_SEEDS, one HGT config (2 layers, lr 0.01)
      HetGFD : paper grid alpha {0.9,...,0.1} x beta {0.99,...,0.05}       (40)
      PCFI+VF: alpha {0.9,0.7,0.5,0.3,0.1} x gamma {0.02, 0.1}             (10)
      -> pick the combination with the best mean validation score
         (nc: validation accuracy = the original code's selection score; lp: corrected-evaluation val AUC)
      kNN    : k {1,3,5,10}; SVD: rank {F/5, F-1}                          (only if in --methods)
  stage 2 (final): seeds FINAL_SEEDS, full HGT grid layers {1,2,3} x lr {0.1,0.01,0.001,0.0001}
      every method in --methods (default: HetGFD, PCFI+VF (selected), FP (original, symmetric), Zero;
      also available: Mean, kNN, SVD)

--hgt orig (default): HGT as in the original code (8 heads, no dropout, ReLU, epoch/config chosen by val accuracy).
--hgt text (node classification only): HGT as written in the paper (2 heads, dropout 0.2, GELU, chosen by val
  macro-F1; non-attributed DBLP nodes get per-type one-hot). Imputers, masks and graph are the same for both.
--refine-top K (default 0 = off): "stage 3" between selection and final — the K best combinations of stage 1
  (by mean validation score) of every tuned method are re-evaluated on the selection seeds with the FULL HGT grid,
  and the final choice is the best of those (same conditions as the final evaluation). Used where the single
  stage-1 HGT config is not representative (IMDB link prediction).
CSV: results/summary_<setting>.csv (final-stage mean/std per method) is rewritten after every run.

Parallel GPUs: one worker process per GPU (launched with CUDA_VISIBLE_DEVICES=<gpu>) pulls work units from a
shared file-based queue (gpu_queue.py). A unit is (setting, stage, seed): stage-1 units (50 runs each) for every
setting first, then stage-2 units (4 runs each) seeds 0-4, then 5-9; a stage-2 unit becomes ready once all
stage-1 units of its setting are finished (possibly on other GPUs).
Every finished run is appended to results/<task>_<ds>_<mode>/stage<k>_seed<s>.jsonl (one file per unit, so
workers never write the same file); finished runs are skipped on restart, a half-written last line is ignored.
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import os
import sys
import threading
import time
import traceback
from pathlib import Path

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--workdir", default=None, help="output folder (default: <repo>/runs/hgt)")
ap.add_argument("--data", default=None, help="folder with ACM_processed / DBLP_processed / IMDB_processed (default: <repo>/data/preprocessed)")
ap.add_argument("--worker", required=True, help="name, e.g. gpu0")
ap.add_argument("--launch", required=True, help="launch id shared by the workers of one launch")
ap.add_argument("--tasks", nargs="+", default=["nc", "lp"], choices=["nc", "lp"])
ap.add_argument("--datasets", nargs="+", default=["ACM", "DBLP", "IMDB"])
ap.add_argument("--modes", nargs="+", default=["node", "entry"])
ap.add_argument("--sel-seeds", type=int, nargs="+", default=[0, 1, 2])
ap.add_argument("--final-seeds", type=int, nargs="+", default=list(range(10)))
ap.add_argument("--methods", nargs="+", default=["hetgfd", "pcfi", "fp", "zero"],
                choices=["hetgfd", "pcfi", "fp", "zero", "mean", "knn", "svd"])
ap.add_argument("--hgt", default="orig", choices=["orig", "text"])
ap.add_argument("--refine-top", type=int, default=0, help="re-evaluate the top-K stage-1 combos with the full HGT grid")
ap.add_argument("--smoke", action="store_true", help="tiny run to check the pipeline (separate results)")
args = ap.parse_args()

REPO = Path(__file__).resolve().parents[1]
W = Path(args.workdir) if args.workdir else REPO / "runs" / "hgt"
DATA = Path(args.data) if args.data else REPO / "data" / "preprocessed"
# early "alive" signal, written before the slow imports (torch_geometric etc.), so the launcher can tell
# a worker that is still starting from one that never started; the queue's heartbeat replaces it
_st = W / ("smoke" if args.smoke else "") / "state"
_st.mkdir(parents=True, exist_ok=True)
(_st / f"{args.worker}.json").write_text(json.dumps({"worker": args.worker, "launch": args.launch, "pid": os.getpid(), "host": os.uname().nodename,
                                                     "time": time.time(), "current": "starting (loading libraries)"}))
sys.path[:0] = [str(REPO)]
from hetgfd_repro.downstream import evaluate_features  # noqa: E402
from hetgfd_repro.hetgfd import edge_type_homophily, fp_orig, hetgfd, pcfi_orig, preliminary_diffusion, rank_edge_types  # noqa: E402
from hetgfd_repro.hetgraph import load_hgnnac  # noqa: E402
from hetgfd_repro.linkpred_orig import evaluate_features_orig, make_orig_split  # noqa: E402
from hetgfd_repro.masking import make_observed_mask_orig  # noqa: E402
from hetgfd_repro.baselines import iterative_svd_impute, knn_impute  # noqa: E402
from hetgfd_repro.gpu_queue import Queue, append_jsonl, atomic_write, read_jsonl, run_queue  # noqa: E402

ALPHAS = (0.9, 0.7, 0.5, 0.3, 0.1)
BETAS = (0.99, 0.9, 0.8, 0.5, 0.4, 0.2, 0.1, 0.05)
PCFI_GRID = [(a, gm) for a in (0.9, 0.7, 0.5, 0.3, 0.1) for gm in (0.02, 0.1)]
LAYERS, LRS = (1, 2, 3), (0.1, 0.01, 0.001, 0.0001)
SEL_LAYERS, SEL_LRS = (2,), (0.01,)
FINAL_METHODS = list(args.methods)
KNN_GRID = [(k, 0.0) for k in (1, 3, 5, 10)]
SVD_GRID = [(0.2, 0.0), (1.0, 0.0)]  # p1 = rank / F: 0.2 -> F // 5, 1.0 -> F - 1 (HetGFD paper's SVD grid)
K = 100
RATE = 0.995
HGT_KW = dict(heads=8, dropout=0.0, weight_decay=0.0, act="relu", select="acc", nonattr_feat="node_onehot")  # original code
if args.hgt == "text":  # HGT as described in the paper text
    assert args.tasks == ["nc"], "--hgt text is for node classification only"
    HGT_KW = dict(heads=2, dropout=0.2, weight_decay=0.0, act="gelu", select="macro")
NONATTR_TEXT = {"ACM": "node_onehot", "DBLP": "type_onehot", "IMDB": "node_onehot"}
EPOCH_KW = {}
if args.smoke:
    ALPHAS, BETAS, PCFI_GRID = (0.9,), (0.9, 0.1), [(0.5, 0.02)]
    KNN_GRID, SVD_GRID = [(1, 0.0), (5, 0.0)], [(0.2, 0.0)]
    LAYERS, LRS = (1, 2), (0.01,)
    EPOCH_KW = dict(epochs=20, patience=10)

# paper (HetGFD ICLR 2025): Table 5 alpha, beta and Table 1 / 2 numbers (nc: macro, micro F1; lp: AUC, AP)
PAPER_AB = {"nc": {("ACM", "node"): (0.7, 0.2), ("DBLP", "node"): (0.1, 0.4), ("IMDB", "node"): (0.7, 0.1),
                   ("ACM", "entry"): (0.1, 0.4), ("DBLP", "entry"): (0.1, 0.4), ("IMDB", "entry"): (0.3, 0.5)},
            "lp": {("ACM", "node"): (0.1, 0.4), ("DBLP", "node"): (0.5, 0.8), ("IMDB", "node"): (0.9, 0.8),
                   ("ACM", "entry"): (0.1, 0.5), ("DBLP", "entry"): (0.9, 0.9), ("IMDB", "entry"): (0.9, 0.99)}}
PAPER = {
    "nc": {("ACM", "node"): {"mean": (83.25, 83.90), "knn": (82.67, 83.56), "svd": (82.71, 83.20), "zero": (82.75, 83.44), "fp": (83.15, 83.60), "pcfi": (84.47, 85.06), "hetgfd": (85.87, 86.10)},
           ("DBLP", "node"): {"mean": (89.88, 90.74), "knn": (90.37, 91.04), "svd": (90.11, 90.90), "zero": (90.11, 90.90), "fp": (90.30, 91.07), "pcfi": (90.41, 91.14), "hetgfd": (90.88, 91.53)},
           ("IMDB", "node"): {"mean": (45.38, 48.21), "knn": (44.88, 48.22), "svd": (44.28, 47.91), "zero": (44.29, 47.91), "fp": (45.37, 48.30), "pcfi": (46.64, 49.69), "hetgfd": (46.88, 49.81)},
           ("ACM", "entry"): {"mean": (82.76, 83.48), "knn": (81.91, 82.43), "svd": (80.45, 81.52), "zero": (81.85, 82.67), "fp": (83.89, 84.49), "pcfi": (86.26, 86.61), "hetgfd": (88.14, 88.14)},
           ("DBLP", "entry"): {"mean": (90.54, 91.32), "knn": (90.50, 91.20), "svd": (90.33, 91.07), "zero": (90.19, 90.99), "fp": (90.15, 90.91), "pcfi": (90.49, 91.14), "hetgfd": (90.65, 91.40)},
           ("IMDB", "entry"): {"mean": (46.33, 48.32), "knn": (45.81, 48.70), "svd": (44.53, 48.38), "zero": (46.54, 48.49), "fp": (45.58, 48.61), "pcfi": (47.25, 49.94), "hetgfd": (48.57, 50.57)}},
    "lp": {("ACM", "node"): {"mean": (71.64, 71.66), "knn": (72.04, 72.55), "svd": (71.49, 72.29), "zero": (71.65, 71.74), "fp": (73.40, 74.03), "pcfi": (73.41, 73.22), "hetgfd": (78.25, 78.62)},
           ("DBLP", "node"): {"mean": (72.49, 74.20), "knn": (71.96, 69.86), "svd": (72.49, 74.21), "zero": (72.49, 74.21), "fp": (71.58, 70.01), "pcfi": (71.37, 66.78), "hetgfd": (91.94, 91.88)},
           ("IMDB", "node"): {"mean": (91.78, 85.80), "knn": (91.10, 84.44), "svd": (92.48, 86.95), "zero": (92.48, 86.95), "fp": (92.50, 86.99), "pcfi": (91.71, 85.37), "hetgfd": (92.50, 86.99)},
           ("ACM", "entry"): {"mean": (71.98, 72.02), "knn": (71.02, 72.49), "svd": (70.49, 70.70), "zero": (70.69, 70.17), "fp": (73.18, 73.77), "pcfi": (74.94, 73.80), "hetgfd": (76.96, 77.19)},
           ("DBLP", "entry"): {"mean": (72.48, 74.20), "knn": (72.72, 70.29), "svd": (72.48, 74.20), "zero": (72.48, 74.20), "fp": (71.86, 70.03), "pcfi": (70.76, 68.97), "hetgfd": (92.17, 92.12)},
           ("IMDB", "entry"): {"mean": (91.40, 85.33), "knn": (91.15, 84.50), "svd": (92.50, 86.99), "zero": (92.50, 86.99), "fp": (91.52, 85.67), "pcfi": (91.54, 85.70), "hetgfd": (91.95, 86.72)}},
}

RES = W / ("smoke" if args.smoke else "") / "results"
LOGS = W / ("smoke" if args.smoke else "") / "logs"
STATE = W / ("smoke" if args.smoke else "") / "state"
for d in (RES, LOGS, STATE):
    d.mkdir(parents=True, exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                    handlers=[logging.FileHandler(LOGS / f"{args.worker}.log"), logging.StreamHandler(sys.stdout)])
log = logging.getLogger()
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------- records (one file per unit)
def unit_path(task, ds, mode, stage, seed):
    return RES / f"{task}_{ds}_{mode}" / f"stage{stage}_seed{seed}.jsonl"


def read_rows(task, ds, mode):
    return read_jsonl(sorted((RES / f"{task}_{ds}_{mode}").glob("*.jsonl")))


def key(r):
    return (r["stage"], r["method"], r["seed"], r["p1"], r["p2"])


def score(task, r):
    """validation score used for every selection."""
    return r["val"] if task == "nc" else r["fixed_val_auc"]


def select(task, rows, method, seeds):
    """(p1, p2) with the best mean stage-1 validation score; None until every combo is done on every seed."""
    rs = [r for r in rows if r["stage"] == 1 and r["method"] == method]
    combos = sorted({(r["p1"], r["p2"]) for r in rs})
    grid = GRIDS[method]
    table = {}
    for c in combos:
        v = {r["seed"]: score(task, r) for r in rs if (r["p1"], r["p2"]) == c}
        if all(s in v for s in seeds):
            table[c] = float(np.mean([v[s] for s in seeds]))
    if len(table) < len(grid):
        return None, table
    return max(table, key=table.get), table


GRIDS = {"hetgfd": list(itertools.product(ALPHAS, BETAS)), "pcfi": PCFI_GRID, "knn": KNN_GRID, "svd": SVD_GRID}
SEL_METHODS = [m for m in ("hetgfd", "pcfi", "knn", "svd") if m in FINAL_METHODS]  # methods with a stage-1 grid


def refine_candidates(task, rows, method):
    """top-K stage-1 combinations (by mean validation score); None until stage 1 is complete."""
    sel, table = select(task, rows, method, args.sel_seeds)
    if sel is None:
        return None
    return sorted(table, key=lambda c: -table[c])[: args.refine_top]


def final_choice(task, rows, method):
    """the (p1, p2) used in stage 2: stage-1 winner, or (with --refine-top) the best stage-3 re-evaluation."""
    if not args.refine_top:
        return select(task, rows, method, args.sel_seeds)[0]
    cands = refine_candidates(task, rows, method)
    if cands is None:
        return None
    t3 = {}
    for r in rows:
        if r["stage"] == 3 and r["method"] == method and (r["p1"], r["p2"]) in cands:
            t3.setdefault((r["p1"], r["p2"]), {})[r["seed"]] = score(task, r)
    if any(len(t3.get(c, {})) < len(args.sel_seeds) for c in cands):
        return None
    return max(cands, key=lambda c: np.mean(list(t3[c].values())))


# ---------------------------------------------------------------- one evaluation
G_CACHE, SETUP_CACHE = {}, {}


def graph(ds):
    if ds not in G_CACHE:
        G_CACHE[ds] = load_hgnnac(ds, root=str(DATA), same_type_directed=True)
    return G_CACHE[ds]


def setup(task, ds, mode, seed):
    k = (task, ds, mode, seed)
    if k not in SETUP_CACHE:
        SETUP_CACHE.clear()  # one (setting, seed) at a time keeps memory flat
        g = graph(ds)
        obs = make_observed_mask_orig(*g.x.shape, RATE, mode, seed=seed, device=dev)
        split = make_orig_split(g, ds, seed) if task == "lp" else None
        SETUP_CACHE[k] = {"obs": obs, "split": split, "rank": None}
    return SETUP_CACHE[k]


def impute(task, ds, mode, seed, method, p1, p2):
    g, st = graph(ds), setup(task, ds, mode, seed)
    obs, x = st["obs"], graph(ds).x.to(dev)
    if method == "zero":
        return torch.where(obs, x, torch.zeros_like(x)), ""
    if method == "fp":
        return fp_orig(g, obs, K, dev), ""
    if method == "pcfi":
        return pcfi_orig(g, obs, p1, p2, K, dev), ""
    if method == "mean":  # missing entries <- mean of the observed values of that channel
        fill = (torch.where(obs, x, torch.zeros_like(x)).sum(0) / obs.float().sum(0).clamp_min(1.0)).expand_as(x)
        return torch.where(obs, x, fill), ""
    if method == "knn":
        return knn_impute(x, obs, int(p1)), ""
    if method == "svd":
        f = x.shape[1]
        return iterative_svd_impute(x, obs, f // 5 if p1 < 1 else f - 1), ""
    if st["rank"] is None:  # edge-type ranking does not depend on (alpha, beta)
        st["rank"] = rank_edge_types(edge_type_homophily(preliminary_diffusion(g, obs, K, dev), g, seed=seed))
    xh, _ = hetgfd(g, obs, p1, p2, K, dev, seed=seed, rank=st["rank"])
    return xh, ">".join(sorted(st["rank"], key=st["rank"].get))


def evaluate(task, ds, mode, seed, xh, layers, lrs):
    if task == "nc":
        na = NONATTR_TEXT[ds] if args.hgt == "text" else HGT_KW["nonattr_feat"]
        kw = {k: v for k, v in HGT_KW.items() if k != "nonattr_feat"}
        r = evaluate_features(graph(ds), xh.cpu(), layers, lrs, seed=seed, device=dev, nonattr_feat=na, **kw, **EPOCH_KW)
        return {"val": r["val_macro"], "test_macro": r["test_macro"], "test_micro": r["test_micro"],
                "layers": r["layers"], "lr": r["lr"]}
    r = evaluate_features_orig(setup(task, ds, mode, seed)["split"], xh.cpu(), layers, lrs, seed=seed, device=dev, **EPOCH_KW)
    return {f"{m}_{k}": r[m][k] for m in ("fixed", "orig") for k in ("val_auc", "test_auc", "test_ap", "layers", "lr")}


def run_one(task, ds, mode, stage, method, seed, p1, p2):
    t0 = time.time()
    Q.status["current"] = f"{task} {ds} {mode} stage{stage} {method} seed={seed} p=({p1},{p2})"
    for attempt in range(3):
        try:
            xh, rank = impute(task, ds, mode, seed, method, p1, p2)
            res = (evaluate(task, ds, mode, seed, xh, SEL_LAYERS, SEL_LRS) if stage == 1
                   else evaluate(task, ds, mode, seed, xh, LAYERS, LRS))
            break
        except torch.cuda.OutOfMemoryError:  # noqa: PERF203
            log.warning(f"GPU out of memory ({Q.status['current']}), attempt {attempt + 1}/3 — waiting 120 s")
            xh = None
            torch.cuda.empty_cache()
            time.sleep(120)
    else:
        log.error(f"skipped after 3 OOM: {Q.status['current']} (will be retried on the next launch)")
        return None
    del xh
    torch.cuda.empty_cache()
    row = dict(task=task, dataset=ds, mode=mode, stage=stage, method=method, seed=seed, p1=p1, p2=p2, rank=rank,
               **res, sec=round(time.time() - t0, 1), host=os.uname().nodename, finished=time.strftime("%Y-%m-%d %H:%M:%S"))
    append_jsonl(unit_path(task, ds, mode, stage, seed), row)
    return row


def fmt(task, r):
    if task == "nc":
        return f"val={r['val']*100:.2f} test macro={r['test_macro']*100:.2f} micro={r['test_micro']*100:.2f}"
    return (f"fixed val={r['fixed_val_auc']*100:.2f} test auc={r['fixed_test_auc']*100:.2f} ap={r['fixed_test_ap']*100:.2f}"
            f" | orig auc={r['orig_test_auc']*100:.2f}")


# ---------------------------------------------------------------- summaries
def summarize(task, ds, mode):
    rows = read_rows(task, ds, mode)
    out = {"task": task, "dataset": ds, "missing": {"node": "structural", "entry": "uniform"}[mode], "n_rows": len(rows),
           "paper_ab": PAPER_AB[task][(ds, mode)], "paper": PAPER[task][(ds, mode)], "stage1": {}, "stage2": {}}
    m1, m2 = ("val", "test_macro") if task == "nc" else ("fixed_val_auc", "fixed_test_auc")
    for method in SEL_METHODS:
        sel, table = select(task, rows, method, args.sel_seeds)
        rs = [r for r in rows if r["stage"] == 1 and r["method"] == method]
        if not rs:
            continue
        tests = {}
        for r in rs:
            tests.setdefault((r["p1"], r["p2"]), []).append(r[m2])
        info = {"n_done": len(rs), "selected": sel,
                "top5_by_val": [[c[0], c[1], round(v * 100, 2), round(float(np.mean(tests[c])) * 100, 2)]
                                for c, v in sorted(table.items(), key=lambda t: -t[1])[:5]],
                "best_test": max(([c[0], c[1], round(float(np.mean(v)) * 100, 2)] for c, v in tests.items()), key=lambda t: t[2])}
        if method == "hetgfd":
            pa = tuple(PAPER_AB[task][(ds, mode)])
            if pa in tests:
                info["paper_ab_test"] = round(float(np.mean(tests[pa])) * 100, 2)
        if args.refine_top:
            cands = refine_candidates(task, rows, method) or []
            r3 = {}
            for r in rows:
                if r["stage"] == 3 and r["method"] == method:
                    r3.setdefault((r["p1"], r["p2"]), []).append(r)
            info["refined"] = {"candidates": cands, "chosen": final_choice(task, rows, method),
                               "full_grid_on_sel_seeds": [[c[0], c[1], len(r3.get(c, [])),
                                                           round(float(np.mean([score(task, x) for x in r3[c]])) * 100, 2) if c in r3 else None,
                                                           round(float(np.mean([x[m2] for x in r3[c]])) * 100, 2) if c in r3 else None]
                                                          for c in cands]}
        out["stage1"][method] = info
    s2 = [r for r in rows if r["stage"] == 2]
    metrics = ("test_macro", "test_micro") if task == "nc" else ("fixed_test_auc", "fixed_test_ap", "orig_test_auc", "orig_test_ap")
    for method in FINAL_METHODS:
        rs = sorted([r for r in s2 if r["method"] == method], key=lambda r: r["seed"])
        fc = final_choice(task, rows, method) if method in SEL_METHODS else None
        if fc is not None:  # only runs with the final choice (older runs with another choice may sit in the same files)
            rs = [r for r in rs if (r["p1"], r["p2"]) == tuple(fc)]
        if rs:
            out["stage2"][method] = {"n": len(rs), "p": [rs[0]["p1"], rs[0]["p2"]],
                                     **{m: [round(float(np.mean([r[m] for r in rs])) * 100, 2),
                                            round(float(np.std([r[m] for r in rs])) * 100, 2)] for m in metrics},
                                     "per_seed": {r["seed"]: round(r[metrics[0]] * 100, 2) for r in rs}}
    h, p = out["stage2"].get("hetgfd"), out["stage2"].get("pcfi")
    if h and p:
        common = sorted(set(h["per_seed"]) & set(p["per_seed"]))
        out["hetgfd_minus_pcfi"] = {"n": len(common),
                                    "mean_diff": round(float(np.mean([h["per_seed"][s] - p["per_seed"][s] for s in common])), 2) if common else None,
                                    "wins": sum(h["per_seed"][s] > p["per_seed"][s] for s in common)}
    atomic_write(RES / f"summary_{task}_{ds}_{mode}.json", json.dumps(out, indent=1, ensure_ascii=False))
    # CSV: one line per method (final stage), easy to paste back
    hdr = ["task", "hgt", "dataset", "missing", "method", "n_seeds", "p1", "p2"]
    hdr += [f"{m}_{s}" for m in metrics for s in ("mean", "std")] + ["paper_" + metrics[0], "paper_" + metrics[1]]
    lines = [",".join(hdr)]
    for method, s in out["stage2"].items():
        pap = PAPER[task][(ds, mode)].get(method, ("", ""))
        lines.append(",".join(map(str, [task, args.hgt, ds, out["missing"], method, s["n"], s["p"][0], s["p"][1],
                                        *[x for m in metrics for x in s[m]], pap[0], pap[1]])))
    atomic_write(RES / f"summary_{task}_{ds}_{mode}.csv", "\n".join(lines) + "\n")
    return out


# ---------------------------------------------------------------- work units + main loop
SETTINGS = [(t_, d, m) for t_ in args.tasks for d in args.datasets for m in args.modes]


def unit_jobs(u):
    task, ds, mode, stage, seed = u
    if stage == 1:
        return [(m, a, b) for m in SEL_METHODS for a, b in GRIDS[m]]
    rows = read_rows(task, ds, mode)
    if stage == 3:
        return [(m, a, b) for m in SEL_METHODS for a, b in (refine_candidates(task, rows, m) or [])]
    sel = {m: final_choice(task, rows, m) for m in SEL_METHODS}
    return [(m, *(sel[m] if m in sel else (0.0, 0.0))) for m in FINAL_METHODS]


def uid(u):
    return "{}_{}_{}_stage{}_seed{}".format(*u)


def build_units():
    fs = args.final_seeds
    units = [(t_, d, m, 1, s) for (t_, d, m) in SETTINGS for s in args.sel_seeds]  # setting-major: GPUs share a setting
    if args.refine_top:
        units += [(t_, d, m, 3, s) for (t_, d, m) in SETTINGS for s in args.sel_seeds]
    for chunk in (fs[:5], fs[5:]):
        units += [(t_, d, m, 2, s) for s in chunk for (t_, d, m) in SETTINGS]
    return units


def main():
    global Q
    Q = Queue(STATE, args.launch, args.worker)
    log.info(f"==== worker {args.worker} (launch {args.launch}) start | CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')} | "
             f"torch {torch.__version__} | {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'} | "
             f"sel seeds {args.sel_seeds} | final seeds {args.final_seeds} | methods {FINAL_METHODS} | hgt {args.hgt} | refine top {args.refine_top} | smoke={args.smoke}")
    units = build_units()
    by_id = {uid(u): u for u in units}

    def done_keys(u):
        task, ds, mode, stage, seed = u
        return {(r["method"], r["p1"], r["p2"]) for r in read_jsonl([unit_path(task, ds, mode, stage, seed)])}

    def is_ready(i):
        u = by_id[i]
        if u[3] == 1:
            return True
        rows = read_rows(*u[:3])
        if u[3] == 3:
            return all(refine_candidates(u[0], rows, m) is not None for m in SEL_METHODS)
        return all(final_choice(u[0], rows, m) is not None for m in SEL_METHODS)

    def is_done(i):
        u = by_id[i]
        if not is_ready(i):
            return False
        return all(tuple(j) in done_keys(u) for j in unit_jobs(u))

    def deps(i):
        u = by_id[i]
        if u[3] == 1:
            return []
        d = [uid((*u[:3], 1, s)) for s in args.sel_seeds]
        return d + ([uid((*u[:3], 3, s)) for s in args.sel_seeds] if u[3] == 2 and args.refine_top else [])

    def run_unit(i):
        u = by_id[i]
        task, ds, mode, stage, seed = u
        done = done_keys(u)
        todo = [j for j in unit_jobs(u) if tuple(j) not in done]
        log.info(f"---- unit {i}: {len(todo)} runs to do")
        for n, (m, p1, p2) in enumerate(todo, 1):
            try:
                r = run_one(task, ds, mode, stage, m, seed, p1, p2)
            except Exception:
                log.error(f"error in {i} {m} ({p1},{p2}):\n{traceback.format_exc()}")
                continue
            if r is not None:
                log.info(f"[{i} {n}/{len(todo)}] {m} ({p1},{p2}) {fmt(task, r)} ({r['sec']:.0f}s)")
                summarize(task, ds, mode)

    run_queue(Q, [uid(u) for u in units], is_done, is_ready, run_unit, deps, log)


if __name__ == "__main__":
    main()
