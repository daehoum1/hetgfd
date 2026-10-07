"""HGNN-AC downstream (HetGFD Figure 4 / Table 18), one protocol for every cell — final version.

Cells: {ACM, DBLP, IMDB} x {node (structural), entry (uniform)} x missing rate {0.5, 0.9, 0.995} = 18.
  stage 1 (selection, masks --sel-masks, default 0 1):
      HetGFD alpha x beta paper grid (40), PCFI+VF alpha {0.9,0.7,0.5,0.3,0.1} x gamma {0.02,0.1} (10)
      (+ kNN k {1,3,5,10}, SVD rank {F/5, F-1} if in --methods)
      -> best mean validation macro-F1 over the selection masks (tie: lower validation loss)
  stage 2 (final, masks --final-masks, default 0-9): every method in --methods
      (default Zero, Mean, FP (original), PCFI+VF (selected), HetGFD (selected))
One official HGNN-AC training per run (model-prediction macro / micro F1 on the official test split, early stopping
on validation loss as in the official code), training seed 123 + mask.

One worker per GPU pulls units (cell, stage, mask) from a file-based queue (gpu_queue.py); a stage-2 unit starts
once stage 1 of its cell is complete. Each run is appended to results/<cell>/stage<k>_mask<m>.jsonl (finished runs
are skipped on restart); results/summary_<cell>.json and .csv are rewritten after every run.
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import os
import random
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score

ap = argparse.ArgumentParser()
ap.add_argument("--workdir", default=None, help="output folder (default: <repo>/runs/hgnnac)")
ap.add_argument("--data", default=None, help="folder with ACM_processed / DBLP_processed / IMDB_processed (default: <repo>/data/preprocessed)")
ap.add_argument("--hgnnac-dir", default=None, help="clone of the official HGNN-AC repo (default: <repo>/third_party/HGNN-AC)")
ap.add_argument("--worker", required=True, help="name, e.g. gpu0")
ap.add_argument("--launch", required=True, help="launch id shared by the workers of one launch")
ap.add_argument("--cells", nargs="+", default=None, help="cell names, e.g. ACM_entry_0.995 (default: all 18)")
ap.add_argument("--methods", nargs="+", default=["hetgfd", "pcfi", "fp", "mean", "zero"],
                choices=["hetgfd", "pcfi", "fp", "mean", "zero", "knn", "svd"])
ap.add_argument("--sel-masks", type=int, nargs="+", default=[0, 1])
ap.add_argument("--final-masks", type=int, nargs="+", default=list(range(10)))
ap.add_argument("--epochs", type=int, default=100, help="HGNN-AC max epochs (official: 100)")
ap.add_argument("--smoke", action="store_true", help="tiny grid, few epochs, separate output folder")
args = ap.parse_args()

REPO = Path(__file__).resolve().parents[1]
W = Path(args.workdir) if args.workdir else REPO / "runs" / "hgnnac"
DATA = Path(args.data) if args.data else REPO / "data" / "preprocessed"
O = W / "smoke" if args.smoke else W
for d_ in ("logs", "results", "state"):
    (O / d_).mkdir(parents=True, exist_ok=True)
(O / "state" / f"{args.worker}.json").write_text(json.dumps({"worker": args.worker, "launch": args.launch, "pid": os.getpid(), "host": os.uname().nodename,
                                                             "time": time.time(), "current": "starting (loading libraries)"}))
HGNNAC = Path(args.hgnnac_dir) if args.hgnnac_dir else REPO / "third_party" / "HGNN-AC"
sys.path[:0] = [str(REPO / "dgl_shim"), str(HGNNAC), str(REPO)]
from model import MAGNN_nc_AC, MAGNN_nc_mb_AC  # noqa: E402  official HGNN-AC
from utils.data import load_ACM_data, load_DBLP_data, load_IMDB_data  # noqa: E402
from utils.tools import index_generator, parse_mask, parse_minibatch  # noqa: E402

from hetgfd_repro.baselines import iterative_svd_impute, knn_impute  # noqa: E402
from hetgfd_repro.hetgfd import edge_type_homophily, fp_orig, hetgfd, pcfi_orig, preliminary_diffusion, rank_edge_types  # noqa: E402
from hetgfd_repro.hetgraph import load_hgnnac  # noqa: E402
from hetgfd_repro.masking import make_observed_mask_orig  # noqa: E402
from hetgfd_repro.gpu_queue import Queue, append_jsonl, atomic_write, read_jsonl, run_queue  # noqa: E402

ALL_CELLS = [f"{d}_{m}_{r}" for d in ("DBLP", "ACM", "IMDB") for m in ("node", "entry") for r in (0.995, 0.9, 0.5)]
CELLS = args.cells or ALL_CELLS
ALPHAS = (0.9, 0.7, 0.5, 0.3, 0.1)
BETAS = (0.99, 0.9, 0.8, 0.5, 0.4, 0.2, 0.1, 0.05)
PCFI_GRID = [(a, gm) for a in (0.9, 0.7, 0.5, 0.3, 0.1) for gm in (0.02, 0.1)]
KNN_GRID = [(k, 0.0) for k in (1, 3, 5, 10)]
SVD_GRID = [(0.2, 0.0), (1.0, 0.0)]  # p1 = rank / F: 0.2 -> F // 5, 1.0 -> F - 1
if args.smoke:
    ALPHAS, BETAS, PCFI_GRID, KNN_GRID, SVD_GRID = (0.9,), (0.9, 0.1), [(0.5, 0.02)], [(5, 0.0)], [(0.2, 0.0)]
GRIDS = {"hetgfd": list(itertools.product(ALPHAS, BETAS)), "pcfi": PCFI_GRID, "knn": KNN_GRID, "svd": SVD_GRID}
METHODS = list(args.methods)
SEL_METHODS = [m for m in ("hetgfd", "pcfi", "knn", "svd") if m in METHODS]
# paper: Table 5/6 alpha, beta (HGNN-AC rows; uniform 0.9/0.5 not given) and Table 18 (r_m 0.995, macro-F1)
PAPER_AB = {("ACM", "node", 0.995): (0.7, 0.8), ("DBLP", "node", 0.995): (0.1, 0.8), ("IMDB", "node", 0.995): (0.7, 0.5),
            ("ACM", "node", 0.9): (0.7, 0.8), ("DBLP", "node", 0.9): (0.1, 0.99), ("IMDB", "node", 0.9): (0.9, 0.2),
            ("ACM", "node", 0.5): (0.9, 0.99), ("DBLP", "node", 0.5): (0.1, 0.99), ("IMDB", "node", 0.5): (0.9, 0.8),
            ("ACM", "entry", 0.995): (0.5, 0.9), ("DBLP", "entry", 0.995): (0.1, 0.9), ("IMDB", "entry", 0.995): (0.5, 0.99)}
PAPER_T18 = {("ACM", "node"): {"pcfi": 69.25, "hetgfd": 76.23}, ("DBLP", "node"): {"pcfi": 93.02, "hetgfd": 93.26},
             ("IMDB", "node"): {"pcfi": 34.29, "hetgfd": 35.05}, ("ACM", "entry"): {"pcfi": 84.04, "hetgfd": 85.27},
             ("DBLP", "entry"): {"pcfi": 93.77, "hetgfd": 94.03}, ("IMDB", "entry"): {"pcfi": 41.06, "hetgfd": 43.52}}

# official per-dataset settings (run_ACM.py / run_DBLP.py / run_IMDB.py)
CFG = {
    "ACM": dict(etypes=[[0, 1], [2, 3]], n_mp=2, n_et=4, src=0, out=3, drop=0.5, feats_opt="011", mb=True),
    "DBLP": dict(etypes=[[0, 1], [0, 2, 3, 1], [0, 4, 5, 1]], n_mp=3, n_et=6, src=1, out=4, drop=0.5, feats_opt="1011", mb=True),
    "IMDB": dict(etypes=[[[0, 1], [2, 3]], [[1, 0], [1, 2, 3, 0]], [[3, 2], [3, 0, 1, 2]]], n_mp=[2, 2, 2], n_et=4,
                 src=0, out=3, drop=0.2, feats_opt="011", mb=False),
}
COMMON = dict(hidden=64, heads=8, attn_vec=128, rnn="RotatE0", epochs=args.epochs, patience=5, batch=8, samples=100,
              feats_drop=0.7, lam=0.2, lr=0.005, wd=0.001, layers=2)

def load_official():
    loader = {"ACM": load_ACM_data, "DBLP": load_DBLP_data, "IMDB": load_IMDB_data}[ds]
    adjlists, idx_lists, feats, emb, adjM, type_mask, labels, tvt = loader(prefix=str(DATA / f"{ds}_processed"))
    feats = [np.asarray(f.todense() if hasattr(f, "todense") else f, dtype=np.float32) for f in feats]
    return dict(adjlists=adjlists, idx=idx_lists, feats=feats, emb=emb, adjM=adjM, type_mask=type_mask, labels=labels, tvt=tvt)


def train_eval(D, x_attr: np.ndarray, seed: int):
    """One official HGNN-AC run; returns validation loss / Macro-F1 and test Macro / Micro-F1 of the best checkpoint."""
    c, p = CFG[ds], COMMON
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed(seed)
    feats = [torch.FloatTensor(f).to(dev) for f in D["feats"]]
    feats[c["src"]] = torch.FloatTensor(x_attr).to(dev)
    in_dims = [f.shape[1] for f in feats]
    if "adjM_t" not in D:
        D["adjM_t"] = torch.FloatTensor(D["adjM"]).to(dev)
        D["emb_t"] = torch.FloatTensor(D["emb"]).to(dev)
    emb, adjM = D["emb_t"], D["adjM_t"]
    labels = torch.LongTensor(D["labels"]).to(dev)
    tvt = D["tvt"]
    train_idx, val_idx, test_idx = (np.sort(tvt[f"{k}_idx"]) for k in ("train", "val", "test"))
    type_mask = D["type_mask"]
    feats_opt = list(map(int, c["feats_opt"]))
    best, bad = np.inf, 0
    if c["mb"]:
        net = MAGNN_nc_mb_AC(c["n_mp"], c["n_et"], c["etypes"], in_dims, emb.shape[1], p["hidden"], c["out"], p["heads"],
                             p["attn_vec"], p["rnn"], c["drop"], True, feats_opt).to(dev)
        opt = torch.optim.Adam(net.parameters(), lr=p["lr"], weight_decay=p["wd"])
        tr_gen = index_generator(batch_size=p["batch"], indices=train_idx)

        def run_batch(b):
            g_list, ind_list, mapped = parse_minibatch(D["adjlists"], D["idx"], b, dev, p["samples"])
            mask_list, keep, drop = parse_mask(indices_list=ind_list, type_mask=type_mask, num_classes=c["out"],
                                               src_type=c["src"], rate=p["feats_drop"], device=dev)
            return net((adjM, feats, emb, mask_list, keep, drop, c["src"]), (g_list, type_mask, ind_list, mapped))

        def predict(idx):
            gen = index_generator(batch_size=p["batch"], indices=idx, shuffle=False)
            out = []
            with torch.no_grad():
                for _ in range(gen.num_iterations()):
                    out.append(run_batch(gen.next())[0])
            return torch.cat(out, 0)

        for _ in range(p["epochs"]):
            net.train()
            for _ in range(tr_gen.num_iterations()):
                b = tr_gen.next(); b.sort()
                logits, _, _, loss_ac = run_batch(b)
                loss = F.nll_loss(F.log_softmax(logits, 1), labels[b]) + p["lam"] * loss_ac
                opt.zero_grad(); loss.backward(); opt.step()
            net.eval()
            # official: mean over validation mini-batches of the batch NLL
            va_gen = index_generator(batch_size=p["batch"], indices=val_idx, shuffle=False)
            vl = 0.0
            with torch.no_grad():
                for _ in range(va_gen.num_iterations()):
                    b = va_gen.next()
                    vl += F.nll_loss(F.log_softmax(run_batch(b)[0], 1), labels[b]).item()
            vl /= va_gen.num_iterations()
            if vl < best:
                best, bad = vl, 0; torch.save(net.state_dict(), ckpt)
            else:
                bad += 1
                if bad >= p["patience"]:
                    break
        net.load_state_dict(torch.load(ckpt)); net.eval()
        val_pred, test_pred = predict(val_idx).argmax(1).cpu().numpy(), predict(test_idx).argmax(1).cpu().numpy()
    else:  # IMDB: full-batch MAGNN_nc_AC (run_IMDB.py)
        import dgl
        g_lists = []
        for nx_list in D["adjlists"]:
            g_lists.append([])
            for nx_G in nx_list:
                g = dgl.DGLGraph(multigraph=True)
                g.add_nodes(nx_G.number_of_nodes())
                g.add_edges(*list(zip(*sorted(map(lambda t: (int(t[0]), int(t[1])), nx_G.edges())))))
                g_lists[-1].append(g.to(dev))
        idx_lists = [[torch.LongTensor(i).to(dev) for i in il] for il in D["idx"]]
        mask_list = [torch.LongTensor(np.where(type_mask == i)[0]).to(dev) for i in range(c["out"])]
        from sklearn.model_selection import train_test_split
        keep, drop = train_test_split(np.arange(feats[0].shape[0]), test_size=p["feats_drop"])
        keep, drop = torch.LongTensor(keep).to(dev), torch.LongTensor(drop).to(dev)
        net = MAGNN_nc_AC(p["layers"], c["n_mp"], c["n_et"], c["etypes"], in_dims, emb.shape[1], p["hidden"], c["out"],
                          p["heads"], p["attn_vec"], p["rnn"], c["drop"], True, feats_opt).to(dev)
        opt = torch.optim.Adam(net.parameters(), lr=p["lr"], weight_decay=p["wd"])
        target = np.where(type_mask == 0)[0]
        inp1, inp2 = (adjM, feats, emb, mask_list, keep, drop, c["src"]), (g_lists, type_mask, idx_lists)
        for _ in range(p["epochs"]):
            net.train()
            logits, _, _, loss_ac = net(inp1, inp2, target)
            loss = F.nll_loss(F.log_softmax(logits, 1)[train_idx], labels[train_idx]) + p["lam"] * loss_ac
            opt.zero_grad(); loss.backward(); opt.step()
            net.eval()
            with torch.no_grad():
                logits, _, _, _ = net(inp1, inp2, target)
                vl = F.nll_loss(F.log_softmax(logits, 1)[val_idx], labels[val_idx]).item()
            if vl < best:
                best, bad = vl, 0; torch.save(net.state_dict(), ckpt)
            else:
                bad += 1
                if bad >= p["patience"]:
                    break
        net.load_state_dict(torch.load(ckpt)); net.eval()
        with torch.no_grad():
            logits, _, _, _ = net(inp1, inp2, target)
        val_pred, test_pred = logits[val_idx].argmax(1).cpu().numpy(), logits[test_idx].argmax(1).cpu().numpy()
    y = D["labels"]
    out = dict(val_loss=float(best),
               val_macro=float(f1_score(y[val_idx], val_pred, average="macro")),
               test_macro=float(f1_score(y[test_idx], test_pred, average="macro")),
               test_micro=float(f1_score(y[test_idx], test_pred, average="micro")))
    del net, opt
    torch.cuda.empty_cache()
    return out




logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                    handlers=[logging.FileHandler(O / "logs" / f"{args.worker}.log"), logging.StreamHandler(sys.stdout)])
log = logging.getLogger()
ckpt = str(O / "results" / f"ckpt_{args.worker}_{os.getpid()}.pt")
dev = torch.device("cuda")
ds = None  # dataset of the current run (train_eval / load_official read it)


def parse_cell(c):
    d, m, r = c.split("_")
    return d, m, float(r)


def cell_rows(c):
    """all finished runs of a cell; a run recorded twice counts once."""
    rows, seen = [], set()
    for r in read_jsonl(sorted((O / "results" / c).glob("*.jsonl"))):
        k = (r["stage"], r["method"], r["mask"], r["p1"], r["p2"])
        if k not in seen:
            seen.add(k)
            rows.append(r)
    return rows


def select(rows, method):
    """(p1, p2) with the best mean stage-1 val macro-F1 (tie: lower val loss); None until the grid is complete."""
    t = {}
    for r in rows:
        if r["stage"] == 1 and r["method"] == method and r["mask"] in args.sel_masks:
            t.setdefault((r["p1"], r["p2"]), {})[r["mask"]] = r
    full = {k: v for k, v in t.items() if len(v) == len(args.sel_masks)}
    if any(tuple(g) not in full for g in GRIDS[method]):
        return None, full
    best = max(full, key=lambda k: (np.mean([r["val_macro"] for r in full[k].values()]),
                                    -np.mean([r["val_loss"] for r in full[k].values()])))
    return best, full


CACHE = {}


def data_for(d):
    """official HGNN-AC data + our graph for one dataset (only one dataset kept in memory)."""
    global ds
    ds = d
    if CACHE.get("ds") != d:
        CACHE.clear()
        torch.cuda.empty_cache()
        D = load_official()
        g = load_hgnnac(d, root=str(DATA), lcc=False, same_type_directed=True)
        assert tuple(g.x.shape) == D["feats"][CFG[d]["src"]].shape
        CACHE.update(ds=d, D=D, g=g)
    return CACHE["D"], CACHE["g"]


def impute(g, obs, method, p1, p2, mask, state):
    x = g.x.to(dev)
    if method == "zero":
        return torch.where(obs, x, torch.zeros_like(x))
    if method == "mean":
        fill = (torch.where(obs, x, torch.zeros_like(x)).sum(0) / obs.float().sum(0).clamp_min(1.0)).expand_as(x)
        return torch.where(obs, x, fill)
    if method == "fp":
        return fp_orig(g, obs, 100, dev)
    if method == "pcfi":
        return pcfi_orig(g, obs, p1, p2, 100, dev)
    if method == "knn":
        return knn_impute(x, obs, int(p1))
    if method == "svd":
        return iterative_svd_impute(x, obs, x.shape[1] // 5 if p1 < 1 else x.shape[1] - 1)
    if state.get("rank") is None:  # edge-type ranking does not depend on (alpha, beta)
        state["rank"] = rank_edge_types(edge_type_homophily(preliminary_diffusion(g, obs, 100, dev), g, seed=mask))
    return hetgfd(g, obs, p1, p2, 100, dev, seed=mask, rank=state["rank"])[0]


def unit_jobs(c, stage):
    if stage == 1:
        return [(m, a, b) for m in SEL_METHODS for a, b in GRIDS[m]]
    rows = cell_rows(c)
    sel = {m: select(rows, m)[0] for m in SEL_METHODS}
    return [(m, *(sel[m] if m in sel else (0.0, 0.0))) for m in METHODS]


def summarize(c):
    d, mode, rate = parse_cell(c)
    rows = cell_rows(c)
    out = {"cell": c, "dataset": d, "missing": {"node": "structural", "entry": "uniform"}[mode], "rate": rate,
           "n_rows": len(rows), "paper_ab": PAPER_AB.get((d, mode, rate)),
           "paper_t18": PAPER_T18.get((d, mode)) if rate == 0.995 else None, "stage1": {}, "stage2": {}}
    for m in SEL_METHODS:
        sel, full = select(rows, m)
        if not full:
            continue
        mv = {k: np.mean([r["val_macro"] for r in v.values()]) for k, v in full.items()}
        mt = {k: np.mean([r["test_macro"] for r in v.values()]) for k, v in full.items()}
        info = {"n_complete": len(full), "selected": sel,
                "top5_by_val": [[k[0], k[1], round(mv[k] * 100, 2), round(mt[k] * 100, 2)] for k in sorted(mv, key=lambda k: -mv[k])[:5]],
                "best_test": max(([k[0], k[1], round(mt[k] * 100, 2)] for k in mt), key=lambda x: x[2])}
        pa = PAPER_AB.get((d, mode, rate))
        if m == "hetgfd" and pa in mt:
            info["paper_ab_test"] = round(mt[pa] * 100, 2)
        out["stage1"][m] = info
    lines = ["dataset,missing,rate,method,n_masks,p1,p2,test_macro_mean,test_macro_std,test_micro_mean,test_micro_std,paper_t18_macro"]
    for m in METHODS:
        rs = sorted([r for r in rows if r["stage"] == 2 and r["method"] == m], key=lambda r: r["mask"])
        if not rs:
            continue
        ma, mi = [r["test_macro"] * 100 for r in rs], [r["test_micro"] * 100 for r in rs]
        s = {"n": len(rs), "p": [rs[0]["p1"], rs[0]["p2"]], "test_macro": [round(float(np.mean(ma)), 2), round(float(np.std(ma)), 2)],
             "test_micro": [round(float(np.mean(mi)), 2), round(float(np.std(mi)), 2)],
             "per_mask": {r["mask"]: round(r["test_macro"] * 100, 2) for r in rs}}
        out["stage2"][m] = s
        pap = (out["paper_t18"] or {}).get(m, "")
        lines.append(",".join(map(str, [d, out["missing"], rate, m, s["n"], *s["p"], *s["test_macro"], *s["test_micro"], pap])))
    h, p = out["stage2"].get("hetgfd"), out["stage2"].get("pcfi")
    if h and p:
        cm = sorted(set(h["per_mask"]) & set(p["per_mask"]))
        out["hetgfd_minus_pcfi"] = {"n": len(cm), "wins": sum(h["per_mask"][k] > p["per_mask"][k] for k in cm),
                                    "mean_diff": round(float(np.mean([h["per_mask"][k] - p["per_mask"][k] for k in cm])), 2) if cm else None}
    atomic_write(O / "results" / f"summary_{c}.json", json.dumps(out, indent=1, ensure_ascii=False))
    atomic_write(O / "results" / f"summary_{c}.csv", "\n".join(lines) + "\n")


def run_unit(c, stage, mask):
    d, mode, rate = parse_cell(c)
    done = {(r["method"], r["p1"], r["p2"]) for r in cell_rows(c) if r["stage"] == stage and r["mask"] == mask}
    todo = [j for j in unit_jobs(c, stage) if tuple(j) not in done]
    log.info(f"---- {c} stage {stage} mask {mask}: {len(todo)} runs to do")
    if not todo:
        return
    D, g = data_for(d)
    obs = make_observed_mask_orig(*g.x.shape, rate, mode, seed=mask, device=dev)
    state = {}
    for n, (m, p1, p2) in enumerate(todo, 1):
        Q.status["current"] = f"{c} stage{stage} mask={mask} {m} p=({p1},{p2})"
        try:
            t0 = time.time()
            xh = impute(g, obs, m, p1, p2, mask, state)
            res = train_eval(D, xh.cpu().numpy(), 123 + mask)
        except Exception:
            log.error(f"error in {c} stage{stage} mask={mask} {m} ({p1},{p2}):\n{traceback.format_exc()}")
            torch.cuda.empty_cache()
            continue
        row = dict(cell=c, stage=stage, method=m, mask=mask, p1=p1, p2=p2, **res, sec=round(time.time() - t0, 1),
                   worker=args.worker, finished=time.strftime("%Y-%m-%d %H:%M:%S"))
        append_jsonl(O / "results" / c / f"stage{stage}_mask{mask}.jsonl", row)
        summarize(c)
        log.info(f"[{c} s{stage} mask={mask} {n}/{len(todo)}] {m} ({p1},{p2}) val={res['val_macro']*100:.2f} "
                 f"test macro={res['test_macro']*100:.2f} micro={res['test_micro']*100:.2f} ({row['sec']:.0f}s)")
        del xh
        torch.cuda.empty_cache()


def main():
    global Q
    Q = Queue(O / "state", args.launch, args.worker)
    log.info(f"==== worker {args.worker} (launch {args.launch}) | CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')} | "
             f"methods {METHODS} | sel masks {args.sel_masks} | final masks {args.final_masks} | torch {torch.__version__} | "
             f"GPU {torch.cuda.get_device_name(0)}")
    fm = args.final_masks
    units = [(c, 1, m) for c in CELLS for m in args.sel_masks]          # DBLP cells first (most expensive)
    for chunk in (fm[:5], fm[5:]):
        units += [(c, 2, m) for c in CELLS for m in chunk]
    uid = lambda u: f"{u[0]}_stage{u[1]}_mask{u[2]}"  # noqa: E731
    by_id = {uid(u): u for u in units}

    def is_ready(i):
        c, stage, _ = by_id[i]
        return stage == 1 or all(select(cell_rows(c), m)[0] is not None for m in SEL_METHODS)

    def is_done(i):
        c, stage, mask = by_id[i]
        if not is_ready(i):
            return False
        done = {(r["method"], r["p1"], r["p2"]) for r in cell_rows(c) if r["stage"] == stage and r["mask"] == mask}
        return all(tuple(j) in done for j in unit_jobs(c, stage))

    def deps(i):
        c, stage, _ = by_id[i]
        return [] if stage == 1 else [uid((c, 1, m)) for m in args.sel_masks]

    run_queue(Q, list(by_id), is_done, is_ready, lambda i: run_unit(*by_id[i]), deps, log)
    for c in CELLS:
        if cell_rows(c):
            summarize(c)
    if os.path.exists(ckpt):
        os.remove(ckpt)


if __name__ == "__main__":
    main()
