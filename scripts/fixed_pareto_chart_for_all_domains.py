#!/usr/bin/env python3
# Generate high-resolution Pareto PDFs per domain for LaTeX (no internal titles)
# FIXED: Ensures all charts have exactly the same size by using fixed axes positioning
# Legend only appears on the first chart and is inside the plot area

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
# Plot configuration for LaTeX
# -----------------------------
def plot_one_domain_pdf(data: pd.DataFrame, domain: str, out_path: str,
                        label_strategy: str = "frontier_only",
                        show_legend: bool = True):
    """Write a single high-res PDF for LaTeX subfigure.
    
    KEY FIX: Use fixed subplot position to ensure all PDFs have identical dimensions.
    """
    frontier = pareto_frontier_min_x_max_y(data, x="AS", y="AE")

    plt.rcParams.update({
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 11,
    })

    # Create figure with NO automatic layout adjustment
    fig = plt.figure(figsize=(5.0, 4.2), dpi=400)
    
    # FIX: Use fixed subplot position - same for ALL charts
    # [left, bottom, width, height] in figure coordinates (0-1)
    ax = fig.add_axes([0.14, 0.13, 0.82, 0.82])

    # Scatter by model (color distinguishes models)
    for model, g in data.groupby("modelName"):
        ax.scatter(g["AS"], g["AE"], s=75, alpha=0.9, label=model, zorder=2)

    # Pareto frontier
    if len(frontier) >= 2:
        ax.plot(frontier["AS"], frontier["AE"], linewidth=2.5, zorder=3)
    ax.scatter(frontier["AS"], frontier["AE"], s=85, zorder=4)

    # Set axis limits with consistent padding
    x_min, x_max = data["AS"].min(), data["AS"].max()
    x_range = x_max - x_min
    x_padding = 0.15 * x_range
    ax.set_xlim(x_min - x_padding, x_max + x_padding)
    ax.set_ylim(0, 1.05)

    # Label placement with clipping
    xpad = 0.02 * x_range

    if label_strategy == "frontier_only":
        for _, r in frontier.iterrows():
            t = ax.text(r["AS"] + xpad, r["AE"] + 0.008, r["modelName"],
                        fontsize=11, va="bottom",
                        clip_on=True)

    # Axes & layout (no title)
    ax.set_xlabel("AS (cost)")
    ax.set_ylabel("AE (accuracy)")
    ax.grid(alpha=0.35)
    
    # Only show legend on first chart, inside the plot area
    if show_legend:
        ax.legend(
            title="Model",
            loc="upper right",
            frameon=True,
            fontsize=8,
            title_fontsize=9,
            labelspacing=0.3,
            borderpad=0.4,
        )

    # Save without bbox_inches='tight' to preserve exact figure size
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

    domains = sorted(agg["domainName"].unique())  # Sort for consistent ordering
    
    for i, domain in enumerate(domains):
        data = agg[agg["domainName"] == domain].copy()
        out_path = os.path.join(args.outdir, f"{safe_filename(domain)}.pdf")
        # Only show legend on the first chart (i == 0)
        plot_one_domain_pdf(data, domain, out_path, 
                           label_strategy=args.labels,
                           show_legend=(i == 0))
        legend_status = "with legend" if i == 0 else "no legend"
        print(f"Saved: {out_path} ({legend_status})")

if __name__ == "__main__":
    main()