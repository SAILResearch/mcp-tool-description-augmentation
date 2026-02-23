#!/usr/bin/env python3
# Generate high-resolution Pareto PDFs per domain for LaTeX (no internal titles)

import argparse
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Data utilities
# -----------------------------
def load_and_reshape(csv_path: str) -> pd.DataFrame:
    """Read wide CSV and return long-form rows with columns:
       domainName, modelName, taskId, AE, AS."""
    df_wide = pd.read_csv(csv_path)
    ae_cols = [c for c in df_wide.columns if str(c).endswith("-AE")]
    models = [c[:-3] for c in ae_cols]

    records = []
    for m in models:
        ae_col = f"{m}-AE"
        as_col = f"{m}-AS"
        if ae_col in df_wide.columns and as_col in df_wide.columns:
            sub = df_wide[["domain", "task_name", ae_col, as_col]].copy()
            sub.columns = ["domainName", "taskId", "AE", "AS"]
            sub["modelName"] = m
            records.append(sub)

    if not records:
        raise ValueError("No model columns with -AE and -AS suffixes were found.")
    df_long = pd.concat(records, ignore_index=True).dropna(subset=["AE", "AS"])
    return df_long

def aggregate_domain_model(df_long: pd.DataFrame) -> pd.DataFrame:
    """Aggregate to domain-model rows with mean AE, mean AS, and count n."""
    return (
        df_long.groupby(["domainName", "modelName"], as_index=False)
               .agg(AE=("AE", "mean"), AS=("AS", "mean"), n=("taskId", "nunique"))
    )

def pareto_frontier_min_x_max_y(points_df: pd.DataFrame,
                                x: str = "AS",
                                y: str = "AE",
                                eps: float = 1e-9) -> pd.DataFrame:
    """Return non-dominated points that minimize x and maximize y."""
    pts = points_df.sort_values(x).reset_index(drop=True)
    mask = np.zeros(len(pts), dtype=bool)
    best_y = -np.inf
    for i, row in pts.iterrows():
        if row[y] > best_y + eps:
            mask[i] = True
            best_y = row[y]
    return pts[mask]

def safe_filename(name: str) -> str:
    """Make a filesystem-safe filename from a domain name."""
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name)).strip("_")
    return name or "figure"

# -----------------------------
# Label collision avoidance
# -----------------------------
def repel_texts(ax, texts, pad_px=2, iters=200):
    """Lightweight label repulsion to reduce overlaps."""
    if not texts:
        return
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    def bboxes():
        return [t.get_window_extent(renderer=renderer).expanded(1.05, 1.15) for t in texts]

    for _ in range(iters):
        moved = False
        boxes = bboxes()
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                b1, b2 = boxes[i], boxes[j]
                if b1.overlaps(b2):
                    dx = (b1.x1 + b1.x0 - b2.x1 - b2.x0) / 2.0
                    dy = (b1.y1 + b1.y0 - b2.y1 - b2.y0) / 2.0
                    if abs(dx) < pad_px:
                        dx = np.sign(dx) * pad_px if dx != 0 else pad_px
                    if abs(dy) < pad_px:
                        dy = np.sign(dy) * pad_px if dy != 0 else pad_px
                    inv = ax.transData.inverted()
                    dx_data, dy_data = inv.transform((dx, dy)) - inv.transform((0, 0))
                    for t, ddx, ddy in ((texts[i], dx_data, dy_data),
                                        (texts[j], -dx_data, -dy_data)):
                        x, y = t.get_position()
                        t.set_position((x + ddx, y + ddy))
                    moved = True
        if not moved:
            break
        fig.canvas.draw()


def push_labels_if_legend_overlap(ax, texts, legend=None, step_frac=0.02, max_iter=50):
    """Push labels downward only when they overlap the legend."""

    if not texts or legend is None:
        return
    fig = ax.figure
    inv = ax.transData.inverted()

    for _ in range(max_iter):
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        y_range = max(ax.get_ylim()[1] - ax.get_ylim()[0], 1e-9)
        step = step_frac * y_range

        legend_box = legend.get_window_extent(renderer=renderer)
        moved = False

        for t in texts:
            bbox = t.get_window_extent(renderer=renderer).expanded(1.05, 1.05)
            if legend_box and bbox.overlaps(legend_box):
                dx_data, dy_data = inv.transform((0, step)) - inv.transform((0, 0))
                x, y = t.get_position()
                t.set_position((x, y - dy_data))
                moved = True

        if not moved:
            break

# -----------------------------
# Plot configuration for LaTeX
# -----------------------------
def plot_one_domain_pdf(data: pd.DataFrame, domain: str, out_path: str,
                        label_strategy: str = "frontier_only"):
    """Write a single high-res PDF for LaTeX subfigure."""
    frontier = pareto_frontier_min_x_max_y(data, x="AS", y="AE")

    plt.rcParams.update({
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 11,
    })

    fig, ax = plt.subplots(figsize=(5.0, 4.2), dpi=400)

    # Scatter by model (color distinguishes models)
    for model, g in data.groupby("modelName"):
        ax.scatter(g["AS"], g["AE"], s=75, alpha=0.9, label=model, zorder=2)

    # Pareto frontier
    if len(frontier) >= 2:
        ax.plot(frontier["AS"], frontier["AE"], linewidth=2.5, zorder=3)
    ax.scatter(frontier["AS"], frontier["AE"], s=85, zorder=4)

    # Label placement
    texts = []
    x_range = max(data["AS"].max() - data["AS"].min(), 1.0)
    xpad = 0.02 * x_range
    y_range = max(data["AE"].max() - data["AE"].min(), 0.1)
    ypad = 0.02 * y_range
    place_below = domain == "financial_analysis"
    extra_down = 0.16 * y_range if place_below else 0.0
    extra_up = 0.16 * y_range if domain == "web_search" else 0.0
    x_shift = -0.02 * x_range if domain == "web_search" else xpad

    if label_strategy == "frontier_only":
        for _, r in frontier.iterrows():
            if place_below:
                t = ax.text(r["AS"], r["AE"] - ypad - extra_down, r["modelName"],
                            fontsize=11, va="top", ha="center", rotation=90)
            else:
                t = ax.text(r["AS"] + x_shift, r["AE"] + 0.008 + extra_up, r["modelName"],
                            fontsize=11, va="bottom", rotation=90)
            texts.append(t)
    else:
        for _, r in data.iterrows():
            if place_below:
                t = ax.text(r["AS"], r["AE"] - ypad - extra_down, r["modelName"],
                            fontsize=11, va="top", ha="center", rotation=90)
            else:
                t = ax.text(r["AS"] + x_shift, r["AE"] + 0.008 + extra_up, r["modelName"],
                            fontsize=11, va="bottom", rotation=90)
            texts.append(t)
        repel_texts(ax, texts, pad_px=2, iters=200)
    # Only push labels down if they collide with the legend box
    push_labels_if_legend_overlap(ax, texts, legend=ax.get_legend(), step_frac=0.015, max_iter=80)

    # Axes & layout (no title)
    ax.set_xlabel("AS (cost)")
    ax.set_ylabel("AE (accuracy)")
    ax.set_ylim(0, 1.0)
    ax.grid(alpha=0.35)
    ax.legend(
        title="Model",
        loc="upper right",
        frameon=True,
        fontsize=8,
        title_fontsize=9,
        labelspacing=0.3,
        borderpad=0.4,
    )

    fig.tight_layout()
    # fig.savefig(out_path, format="pdf", bbox_inches="tight", dpi=600)
    fig.savefig(out_path, format="pdf", dpi=600)
    plt.close(fig)

# -----------------------------
# Main entry
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Generate high-resolution Pareto PDFs for LaTeX (no chart titles).")
    parser.add_argument("--csv", default="scripts/AE-AS-Pareto.csv", help="Input CSV path")
    parser.add_argument("--outdir", default="pareto_domain_pdfs", help="Output directory")
    parser.add_argument("--labels", default="frontier_only",
                        choices=["frontier_only", "repel"],
                        help="Label strategy: frontier_only (clean) or repel (all points)")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df_long = load_and_reshape(args.csv)
    agg = aggregate_domain_model(df_long)

    for domain in agg["domainName"].unique():
        data = agg[agg["domainName"] == domain].copy()
        out_path = os.path.join(args.outdir, f"{safe_filename(domain)}.pdf")
        plot_one_domain_pdf(data, domain, out_path, label_strategy=args.labels)
        print(f"Saved clean high-res PDF: {out_path}")

if __name__ == "__main__":
    main()
