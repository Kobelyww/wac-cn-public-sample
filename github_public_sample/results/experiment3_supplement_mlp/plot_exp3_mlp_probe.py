#!/usr/bin/env python3
"""Publication figure: seven-way linear vs MLP probe (ECAPA-TDNN, HuBERT-ZH-B).

Reads ``results/exp3_mlp_probe_summary.csv`` from ``run_exp3_mlp_probe.py`` and
writes ``figures/exp3_mlp_probe_bar.pdf`` (+ optional .png).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PIPE = Path(__file__).resolve().parent


def parse_pm(s: str) -> tuple[float, float]:
    s = str(s).strip()
    if "±" in s:
        a, b = s.split("±", 1)
        return float(a), float(b)
    return float(s), 0.0


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--summary-csv",
        type=Path,
        default=PIPE / "results" / "exp3_mlp_probe_summary.csv",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=PIPE / "figures",
    )
    args = p.parse_args()

    df = pd.read_csv(args.summary_csv)
    models = ["ecapa-tdnn", "hubert-zh-b"]

    # Per-head means/stds for bars
    means = np.zeros((4, 3))
    stds = np.zeros((4, 3))
    row_labels = [
        "ECAPA-TDNN\n(linear)",
        "ECAPA-TDNN\n(MLP)",
        "HuBERT-ZH-B\n(linear)",
        "HuBERT-ZH-B\n(MLP)",
    ]
    idx = 0
    for m in models:
        for head in ("linear", "mlp"):
            row = df[(df["model"] == m) & (df["head"] == head)].iloc[0]
            for j, key in enumerate(("acc", "mf1", "uar")):
                mu, sig = parse_pm(str(row[key]))
                means[idx, j] = mu * 100.0
                stds[idx, j] = sig * 100.0
            idx += 1

    pli_row = df[df["head"] == "pli_derived"].set_index("model")

    metric_titles = ["Acc.", "mF1", "UAR"]
    n_metrics = 3
    fig, ax = plt.subplots(figsize=(6.8, 2.75), constrained_layout=True)
    x0 = np.arange(n_metrics)
    bw = 0.18
    colors = ["#4C72B0", "#8CA9D8", "#55A868", "#A3D4B0"]
    bar_positions: list[np.ndarray] = []
    for i in range(4):
        pos = x0 + (i - 1.5) * bw
        bar_positions.append(pos)
        ax.bar(
            pos,
            means[i],
            bw,
            yerr=stds[i],
            capsize=2,
            label=row_labels[i].replace("\n", " "),
            color=colors[i],
            error_kw={"linewidth": 0.8},
        )

    # Numeric labels on accuracy bars (first metric, index 0) for quick reading
    for i in range(4):
        x_acc = float(bar_positions[i][0])
        m_acc, s_acc = float(means[i, 0]), float(stds[i, 0])
        y_txt = m_acc + s_acc + 2.5
        if y_txt > 98.0:
            y_txt = m_acc - 5.0
        ax.text(
            x_acc,
            y_txt,
            f"{m_acc:.1f}",
            ha="center",
            va="bottom",
            fontsize=5.5,
            rotation=0,
            color="0.15",
        )
    ax.set_xticks(x0)
    ax.set_xticklabels(metric_titles)
    ax.set_ylabel("Score (%)")
    ax.set_ylim(0, 100)
    ax.legend(ncol=2, fontsize=7, loc="lower right")
    ax.tick_params(labelsize=8)

    # Inset: accuracies + PLI / delta for both models
    # Left side, vertically centered in axes coordinates (x0, y0, width, height).
    ax_inset = ax.inset_axes((0.02, 0.28, 0.72, 0.40))
    ax_inset.set_facecolor("white")
    ax_inset.patch.set_edgecolor("black")
    ax_inset.patch.set_linewidth(0.7)
    ax_inset.axis("off")

    hdr = [
        "Model",
        "LP\nAcc.",
        "MLP\nAcc.",
        "\u0394mF1\n(pp)",
        "\u0394UAR\n(pp)",
        "\u03b7mF1",
        "PLI",
        "PLI_U",
    ]
    tbl_rows = [hdr]

    def scale_pm(s: str, mul: float) -> str:
        s = str(s).strip()
        if "±" in s:
            a, b = s.split("±", 1)
            return f"{float(a) * mul:.2f}±{float(b) * mul:.2f}"
        return f"{float(s) * mul:.4f}"

    def fmt_acc_cell(s: str) -> str:
        """CSV acc is in [0,1]; display as percentage with ±."""
        return scale_pm(s, 100.0)

    for m in models:
        r = pli_row.loc[m]
        short = "ECAPA" if m == "ecapa-tdnn" else "HB-ZH"
        lp = df[(df["model"] == m) & (df["head"] == "linear")].iloc[0]
        mp = df[(df["model"] == m) & (df["head"] == "mlp")].iloc[0]
        tbl_rows.append(
            [
                short,
                fmt_acc_cell(str(lp["acc"])),
                fmt_acc_cell(str(mp["acc"])),
                scale_pm(r["delta_mf1"], 100.0),
                scale_pm(r["delta_uar"], 100.0),
                str(r["eta_mf1"]),
                str(r["pli_mf1"]),
                str(r["pli_uar"]),
            ]
        )
    tbl = ax_inset.table(
        cellText=tbl_rows,
        loc="center",
        cellLoc="center",
        edges="closed",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(4.9)
    tbl.scale(0.96, 1.22)
    for (rr, cc), cell in tbl.get_celld().items():
        cell.set_edgecolor("black")
        cell.set_linewidth(0.55)
        if rr == 0:
            cell.set_facecolor("#ececec")
            cell.set_text_props(weight="bold", fontsize=5.4)
        else:
            cell.set_facecolor("white")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_pdf = args.output_dir / "exp3_mlp_probe_bar.pdf"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(args.output_dir / "exp3_mlp_probe_bar.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print("Wrote:", out_pdf)


if __name__ == "__main__":
    main()
