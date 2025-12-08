#!/usr/bin/env python3
# Combined Pareto chart with per-domain connecting lines
# Input: scripts/AE-AS-Pareto.csv
# Output: combined_pareto_with_domain_lines.pdf

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ======= Config =======
CSV_PATH = "scripts/AE-AS-Pareto.csv"
OUT_PDF  = "combined_pareto_with_domain_lines.pdf"
FIGSIZE_INCH = (7.0, 9.0)
POINT_SIZE = 85
POINT_ALPHA = 0.9
LINE_ALPHA = 0.35         # lighter lines to reduce clutter
LINE_WIDTH = 1.8
FONT_MAIN = 12
FONT_TITLE = 14
LABEL_DOMAINS = True      # label domain centroids on the plot
DRAW_GLOBAL_FRONTIER = False
# ======================

def load_and_reshape(csv_path: str) -> pd.DataFrame:
    """Convert wide CSV to long rows: domainName, modelName, taskId, AE, AS."""
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
    return pd.concat(records, ignore_index=True).dropna(subset=["AE", "AS"])

def aggregate_domain_model(df_long: pd.DataFrame) -> pd.DataFrame:
    return (
        df_long.groupby(["domainName", "modelName"], as_index=False)
               .agg(AE=("AE", "mean"), AS=("AS", "mean"), n=("taskId", "nunique"))
    )

def pareto_frontier_min_x_max_y(points_df, x="AS", y="AE", eps=1e-9):
    pts = points_df.sort_values(x).reset_index(drop=True)
    mask = np.zeros(len(pts), dtype=bool)
    best_y = -np.inf
    for i, row in pts.iterrows():
        if row[y] > best_y + eps:
            mask[i] = True
            best_y = row[y]
    return pts[mask]

def main():
    df_long = load_and_reshape(CSV_PATH)
    agg = aggregate_domain_model(df_long)

    plt.rcParams.update({
        "axes.labelsize": FONT_MAIN,
        "xtick.labelsize": FONT_MAIN,
        "ytick.labelsize": FONT_MAIN,
        "legend.fontsize": FONT_MAIN - 1,
    })

    fig, ax = plt.subplots(figsize=FIGSIZE_INCH)

    # 1) Scatter points, color encodes model (default cycle)
    model_handles = {}
    for (model, domain), g in agg.groupby(["modelName", "domainName"]):
        h = ax.scatter(
            g["AS"], g["AE"],
            s=POINT_SIZE,
            alpha=POINT_ALPHA,
            label=model
        )
        if model not in model_handles:
            model_handles[model] = h

    # 2) Draw per-domain connecting lines (sorted by AS)
    for domain, sub in agg.groupby("domainName"):
        sub_sorted = sub.sort_values("AS")
        if len(sub_sorted) >= 2:
            # default line style and color; only reduce alpha to keep it subtle
            ax.plot(sub_sorted["AS"], sub_sorted["AE"],
                    linewidth=LINE_WIDTH, alpha=LINE_ALPHA, zorder=1)

    # 3) Optional global frontier across all points
    if DRAW_GLOBAL_FRONTIER:
        frontier_all = pareto_frontier_min_x_max_y(agg, x="AS", y="AE")
        if len(frontier_all) >= 2:
            ax.plot(frontier_all["AS"], frontier_all["AE"], linewidth=2.2, zorder=3)

    # 4) Label each domain near its centroid to avoid a second legend
    if LABEL_DOMAINS:
        for domain, sub in agg.groupby("domainName"):
            x_mean, y_mean = sub["AS"].mean(), sub["AE"].mean()
            ax.text(
                x_mean, y_mean + 0.01,
                domain.replace("_", " ").title(),
                fontsize=11, weight="bold",
                ha="center", va="bottom",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.6)
            )

    # Axes and layout
    ax.set_xlabel("AS (cost)")
    ax.set_ylabel("AE (accuracy)")
    ax.set_ylim(0, 1.0)
    ax.grid(alpha=0.3)
    ax.set_title("Combined Pareto view with per-domain connecting lines", fontsize=FONT_TITLE)

    # Legend for models only
    ax.legend(
        handles=list(model_handles.values()),
        labels=list(model_handles.keys()),
        title="Model",
        loc="lower right",
        frameon=True
    )

    fig.tight_layout()
    fig.savefig(OUT_PDF, dpi=300, bbox_inches="tight")
    print(f"Saved: {OUT_PDF}")

if __name__ == "__main__":
    main()
