#!/usr/bin/env python3
"""Experiment~3 supplementary: linear vs shallow-MLP probe on two backbones.

Trains seven-way probes (same split as ``run_experiment3.py`` with
``--exclude-accents 闽南,山西``) for ECAPA-TDNN and HuBERT-ZH-B only, then
reports the same PLI / $\\Delta$ / $\\eta$ construction as the binary
probe-head ablation in ``run_experiment1_ablations.py`` (Section~IV-G), but
using **macro F1** and **UAR** from the multiclass test set (UAR equals macro
recall under sklearn's macro average).

Outputs under ``experiment3_supplement_mlp/results/`` (override with ``--output-root``):

  - ``exp3_mlp_probe_per_seed.csv`` — linear, mlp, and derived PLI rows per seed
  - ``exp3_mlp_probe_summary.csv`` — mean$\\pm$std aggregates + PLI row
  - ``exp3_mlp_probe_meta.json`` — protocol flags and class order

Example::

  python experiment3_supplement_mlp/run_exp3_mlp_probe.py \\
    --exclude-accents 闽南,山西 \\
    --device cpu

Figure for the paper (reads ``results/exp3_mlp_probe_summary.csv``)::

  python experiment3_supplement_mlp/plot_exp3_mlp_probe.py
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, TensorDataset

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import run_experiment2 as e2
import run_experiment3 as e3
from run_experiment1 import class_weights, discover_models, load_path_set
from run_experiment1_ablations import EPS_PLI, pli_eta_delta

HeadKind = Literal["linear", "mlp"]
DEFAULT_MODELS = ("ecapa-tdnn", "hubert-zh-b")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--feature-root", type=Path, default=_REPO / "features")
    p.add_argument("--output-root", type=Path, default=Path(__file__).resolve().parent / "results")
    p.add_argument("--dataset-root", type=Path, default=_REPO / "dataset")
    p.add_argument("--train-csv", type=Path, default=None)
    p.add_argument("--val-csv", type=Path, default=None)
    p.add_argument("--test-csv", type=Path, default=None)
    p.add_argument(
        "--mandarin-split-salt",
        type=str,
        default="exp1_mandarin_split_v1",
    )
    p.add_argument(
        "--exclude-accents",
        type=str,
        default="闽南,山西",
        help="Must match paper Experiment~3 seven-way setup.",
    )
    p.add_argument(
        "--models",
        type=str,
        default=",".join(DEFAULT_MODELS),
        help="Comma-separated model directory names under --feature-root.",
    )
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--warmup-frac", type=float, default=0.1)
    p.add_argument("--batch-size-transformer", type=int, default=16)
    p.add_argument("--batch-size-light", type=int, default=32)
    p.add_argument(
        "--light-models",
        type=str,
        default="ecapa-tdnn,resnet-voxceleb,rawnet3,xvect-voxceleb",
    )
    p.add_argument("--seeds", type=int, nargs="+", default=(42, 43, 44))
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--mlp-hidden", type=int, default=256)
    p.add_argument("--mlp-dropout", type=float, default=0.2)
    return p.parse_args()


def _resolve_csv(args: argparse.Namespace, name: str, default: Path) -> Path:
    v = getattr(args, name, None)
    if v is None:
        return default
    return v


def make_head(d_m: int, n_cls: int, kind: HeadKind, hidden: int, dropout: float) -> nn.Module:
    if kind == "linear":
        return nn.Linear(d_m, n_cls)
    return nn.Sequential(
        nn.Linear(d_m, hidden),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden, n_cls),
    )


def train_one_seed(
    X_tr: torch.Tensor,
    y_tr: torch.Tensor,
    X_va: torch.Tensor,
    y_va: torch.Tensor,
    X_te: torch.Tensor,
    y_te: torch.Tensor,
    seed: int,
    args: argparse.Namespace,
    d_m: int,
    batch_size: int,
    n_cls: int,
    head_kind: HeadKind,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device(args.device)

    crit = nn.CrossEntropyLoss(weight=class_weights(y_tr, n_cls).to(device))
    head = make_head(d_m, n_cls, head_kind, args.mlp_hidden, args.mlp_dropout).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    steps_per_epoch = max(1, math.ceil(len(X_tr) / batch_size))
    total_steps = args.epochs * steps_per_epoch
    warm = int(total_steps * args.warmup_frac)

    def lr_lambda(step: int) -> float:
        if step < warm:
            return (step + 1) / max(1, warm)
        prog = (step - warm) / max(1, total_steps - warm)
        return 0.5 * (1.0 + math.cos(math.pi * prog))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    tr_loader = DataLoader(
        TensorDataset(X_tr, y_tr),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )

    best_val_mf1 = -1.0
    bad = 0
    best_state: dict[str, torch.Tensor] | None = None

    for _epoch in range(args.epochs):
        head.train()
        for xb, yb in tr_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            loss = crit(head(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(head.parameters(), args.grad_clip)
            opt.step()
            sched.step()

        head.eval()
        with torch.no_grad():
            val_logits = head(X_va.to(device)).cpu()
            val_pred = val_logits.argmax(dim=1).numpy()
            val_mf1 = float(f1_score(y_va.numpy(), val_pred, average="macro", zero_division=0))
        if val_mf1 > best_val_mf1 + 1e-6:
            best_val_mf1 = val_mf1
            bad = 0
            best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                break

    if best_state is not None:
        head.load_state_dict(best_state)
    head.eval()
    with torch.no_grad():
        test_logits = head(X_te.to(device)).cpu()
        test_pred = test_logits.argmax(dim=1).numpy()
    y_true = y_te.numpy()
    metrics = e3.compute_metrics(y_true, test_pred)
    return {
        **metrics,
        "seed": seed,
        "best_val_mf1": best_val_mf1,
        "state_dict": {k: v.clone() for k, v in head.state_dict().items()},
    }


def aggregate_lines(runs: list[dict[str, Any]], keys: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for key in keys:
        vals = np.array([float(r[key]) for r in runs], dtype=np.float64)
        out[key] = f"{float(np.mean(vals)):.4f}±{float(np.std(vals)):.4f}"
    return out


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fieldnames})


def main() -> None:
    args = parse_args()
    train_csv = _resolve_csv(args, "train_csv", args.dataset_root / "train.csv")
    val_csv = _resolve_csv(args, "val_csv", args.dataset_root / "val.csv")
    test_csv = _resolve_csv(args, "test_csv", args.dataset_root / "test.csv")

    excluded = e3.parse_excluded_accents(args.exclude_accents)
    e3.configure_class_order(excluded)
    n_cls = e3.N_CLASSES

    light = {s.strip() for s in args.light_models.split(",") if s.strip()}
    train_paths = load_path_set(train_csv)
    val_paths = load_path_set(val_csv)
    test_paths = load_path_set(test_csv)

    want = [s.strip() for s in args.models.split(",") if s.strip()]
    discovered = discover_models(args.feature_root, want)
    by_name = {p.name: p for p in discovered}
    model_dirs = [by_name[n] for n in want if n in by_name]
    if len(model_dirs) != len(want):
        missing = [n for n in want if n not in by_name]
        raise RuntimeError(f"Missing model folders: {missing} under {args.feature_root}")

    per_fields = [
        "model",
        "head",
        "seed",
        "acc",
        "mprec",
        "mrec",
        "mf1",
        "wf1",
        "uar",
        "delta_mf1",
        "delta_uar",
        "eta_mf1",
        "eta_uar",
        "pli_mf1",
        "pli_uar",
    ]
    metric_keys = ["acc", "mprec", "mrec", "mf1", "wf1", "uar"]

    detail: list[dict[str, object]] = []
    summary: list[dict[str, str]] = []

    args.output_root.mkdir(parents=True, exist_ok=True)
    per_path = args.output_root / "exp3_mlp_probe_per_seed.csv"
    sum_path = args.output_root / "exp3_mlp_probe_summary.csv"

    print(
        f"[exp3-mlp] {len(model_dirs)} model(s), {len(args.seeds)} seeds, "
        f"n_classes={n_cls} → {args.output_root}",
        flush=True,
    )

    for md in model_dirs:
        manifest = e3.filter_manifest_rows(e2.load_manifest(md), excluded)
        tr_rows, va_rows, te_rows = e3.build_split_rows(
            manifest,
            train_paths,
            val_paths,
            test_paths,
            args.dataset_root,
            args.mandarin_split_salt,
        )
        if not tr_rows or not va_rows or not te_rows:
            raise RuntimeError(f"{md.name}: empty train/val/test after split")

        X_tr, y_tr = e3.rows_to_tensors(tr_rows)
        X_va, y_va = e3.rows_to_tensors(va_rows)
        X_te, y_te = e3.rows_to_tensors(te_rows)
        d_m = int(X_tr.shape[1])
        bs = args.batch_size_light if md.name in light else args.batch_size_transformer

        by_lp: dict[int, dict[str, Any]] = {}
        by_mlp: dict[int, dict[str, Any]] = {}

        for seed in args.seeds:
            by_lp[seed] = train_one_seed(
                X_tr, y_tr, X_va, y_va, X_te, y_te, seed, args, d_m, bs, n_cls, "linear"
            )
            by_mlp[seed] = train_one_seed(
                X_tr, y_tr, X_va, y_va, X_te, y_te, seed, args, d_m, bs, n_cls, "mlp"
            )
            ex = pli_eta_delta(
                float(by_lp[seed]["mf1"]),
                float(by_mlp[seed]["mf1"]),
                float(by_lp[seed]["uar"]),
                float(by_mlp[seed]["uar"]),
                eps=EPS_PLI,
            )
            detail.append(
                {
                    "model": md.name,
                    "head": "linear",
                    "seed": seed,
                    **{k: f"{float(by_lp[seed][k]):.6f}" for k in metric_keys},
                    **{k: "" for k in ("delta_mf1", "delta_uar", "eta_mf1", "eta_uar", "pli_mf1", "pli_uar")},
                }
            )
            detail.append(
                {
                    "model": md.name,
                    "head": "mlp",
                    "seed": seed,
                    **{k: f"{float(by_mlp[seed][k]):.6f}" for k in metric_keys},
                    **{k: "" for k in ("delta_mf1", "delta_uar", "eta_mf1", "eta_uar", "pli_mf1", "pli_uar")},
                }
            )
            detail.append(
                {
                    "model": md.name,
                    "head": "derived",
                    "seed": seed,
                    **{k: "" for k in metric_keys},
                    **{k: f"{float(v):.6f}" for k, v in ex.items()},
                }
            )
            print(
                f"[exp3-mlp] {md.name} seed={seed} "
                f"mF1_lin={by_lp[seed]['mf1']:.4f} mF1_mlp={by_mlp[seed]['mf1']:.4f} "
                f"PLI_mF1={ex['pli_mf1']:.4f}",
                flush=True,
            )

        for hk, store in ("linear", by_lp), ("mlp", by_mlp):
            runs = [store[s] for s in args.seeds]
            line = {"model": md.name, "head": hk, **aggregate_lines(runs, metric_keys)}
            for k in ("delta_mf1", "delta_uar", "eta_mf1", "eta_uar", "pli_mf1", "pli_uar"):
                line[k] = ""
            summary.append(line)

        pli_runs = [
            pli_eta_delta(
                float(by_lp[s]["mf1"]),
                float(by_mlp[s]["mf1"]),
                float(by_lp[s]["uar"]),
                float(by_mlp[s]["uar"]),
                eps=EPS_PLI,
            )
            for s in args.seeds
        ]
        pli_keys = list(pli_runs[0].keys())
        pli_agg: dict[str, str] = {"model": md.name, "head": "pli_derived"}
        for k in metric_keys:
            pli_agg[k] = ""
        for pk in pli_keys:
            vals = np.array([float(r[pk]) for r in pli_runs], dtype=np.float64)
            pli_agg[pk] = f"{float(np.mean(vals)):.4f}±{float(np.std(vals)):.4f}"
        summary.append(pli_agg)

        write_csv(per_path, per_fields, detail)
        write_csv(sum_path, per_fields, summary)

    meta = {
        "task": "experiment3_seven_way_mlp_probe_supplement",
        "feature_root": str(args.feature_root.resolve()),
        "output_root": str(args.output_root.resolve()),
        "exclude_accents": sorted(excluded),
        "class_order": list(e3.CLASS_ORDER),
        "n_classes": n_cls,
        "models": want,
        "seeds": list(args.seeds),
        "mlp_hidden": args.mlp_hidden,
        "mlp_dropout": args.mlp_dropout,
        "pli_note": "PLI_mf1 = mf1_lin^2 / max(mf1_lin, mf1_mlp); same eta/delta as run_experiment1_ablations.pli_eta_delta",
        "eps_pli": EPS_PLI,
    }
    (args.output_root / "exp3_mlp_probe_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[exp3-mlp] wrote {per_path.name}, {sum_path.name}, exp3_mlp_probe_meta.json", flush=True)


if __name__ == "__main__":
    main()
