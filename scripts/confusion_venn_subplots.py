
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle

REQUIRED_COLS = {"domain", "Model", "task_name", "FR", "BC"}

FR_COLOR = "#FFD54F"   # Yellow
BC_COLOR = "#64B5F6"   # Blue

def load_and_summarize(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    df = df.copy()
    df["FR"] = df["FR"].astype(int)
    df["BC"] = df["BC"].astype(int)

    def bucket(fr, bc):
        if fr == 1 and bc == 1:
            return "both"
        if fr == 1 and bc == 0:
            return "FR_only"
        if fr == 0 and bc == 1:
            return "BC_only"
        return "none"
    df["bucket"] = [bucket(fr, bc) for fr, bc in zip(df["FR"], df["BC"])]

    grouped = (
        df.groupby(["domain", "Model", "bucket"], as_index=False)
          .size()
          .rename(columns={"size": "count"})
    )

    pairs = grouped[["domain", "Model"]].drop_duplicates().to_records(index=False).tolist()
    rows = []
    for d, m in pairs:
        sub = grouped[(grouped["domain"] == d) & (grouped["Model"] == m)]
        vals = {"both": 0, "FR_only": 0, "BC_only": 0, "none": 0}
        for _, r in sub.iterrows():
            vals[r["bucket"]] = int(r["count"])
        total = sum(vals.values())
        rows.append({"domain": d, "Model": m, "total": total, **vals})
    out = pd.DataFrame(rows).sort_values(["domain", "Model"]).reset_index(drop=True)
    return out

def draw_venn(ax, left_label, right_label, left_only, right_only, both, none_count, total_count):
    # Coordinates for a clean 2-circle Venn
    r = 1.0
    cx_left, cy_left = -0.5, 0.0
    cx_right, cy_right = 0.5, 0.0

    # Universal set rectangle around the circles
    # Provide padding so labels fit inside
    rect_x0, rect_y0 = -2.2, -1.8
    rect_w, rect_h = 4.4, 3.8
    rect = Rectangle((rect_x0, rect_y0), rect_w, rect_h,
                     fill=False, linewidth=2, edgecolor="black")
    ax.add_patch(rect)

    # Filled circles with alpha so overlap shows green visually
    ax.add_patch(Circle((cx_left, cy_left), r, facecolor=FR_COLOR, alpha=0.45, edgecolor="black", linewidth=1.5))
    ax.add_patch(Circle((cx_right, cy_right), r, facecolor=BC_COLOR, alpha=0.45, edgecolor="black", linewidth=1.5))

    # Region counts
    # FR only: left side, offset toward left
    ax.text(cx_left - 0.65, 0.0, str(left_only), ha="center", va="center", fontsize=14)
    # Both: center
    ax.text(0.0, 0.0, str(both), ha="center", va="center", fontsize=14, fontweight="bold", color="green")
    # BC only: right side
    ax.text(cx_right + 0.65, 0.0, str(right_only), ha="center", va="center", fontsize=14)

    # Labels for FR and BC on top
    ax.text(cx_left, r + 0.2, left_label, ha="center", va="bottom", fontsize=10)
    ax.text(cx_right, r + 0.2, right_label, ha="center", va="bottom", fontsize=10)

    # None count inside the rectangle at bottom center
    ax.text(0.0, rect_y0 + 0.2, f"None: {none_count}", ha="center", va="bottom", fontsize=10)

    # Total label in the rectangle top left
    ax.text(rect_x0 + 0.1, rect_y0 + rect_h - 0.1, f"Total: {total_count}",
            ha="left", va="top", fontsize=9)

    # Cosmetics
    ax.set_xlim(rect_x0 - 0.1, rect_x0 + rect_w + 0.1)
    ax.set_ylim(rect_y0 - 0.1, rect_y0 + rect_h + 0.1)
    ax.set_facecolor("white")
    ax.axis("off")

def plot_venn_subplots(summary_df: pd.DataFrame, title: str, out_path: str, cols: int = 3):
    pairs = summary_df[["domain", "Model"]].to_records(index=False).tolist()
    n = len(pairs)
    cols = max(1, int(cols))
    rows = int(np.ceil(n / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(5.8 * cols, 5.0 * rows))
    if isinstance(axes, np.ndarray):
        axes = axes.ravel()
    else:
        axes = [axes]

    for i, (domain, model) in enumerate(pairs):
        ax = axes[i]
        row = summary_df[(summary_df["domain"] == domain) & (summary_df["Model"] == model)].iloc[0]
        draw_venn(
            ax,
            left_label="FR",
            right_label="BC",
            left_only=int(row["FR_only"]),
            right_only=int(row["BC_only"]),
            both=int(row["both"]),
            none_count=int(row["none"]),
            total_count=int(row["total"]),
        )
        ax.set_title(f"{domain}\n{model}", fontsize=11)

    # Hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    # Legend-like color hints above the figure
    fig.suptitle(title, fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Generate FR vs BC Venn subplots with a universal set rectangle per domain-model.")
    parser.add_argument("--csv_path", required=True, help="Path to input CSV with columns: domain, Model, task_name, FR, BC")
    parser.add_argument("--title", default="FR vs BC Venn per Domain-Model", help="Figure title")
    parser.add_argument("--cols", type=int, default=3, help="Number of subplot columns")
    args = parser.parse_args()

    if not os.path.isfile(args.csv_path):
        raise FileNotFoundError(f"File not found: {args.csv_path}")

    summary = load_and_summarize(args.csv_path)

    base, _ = os.path.splitext(args.csv_path)
    counts_path = base + "_venn_counts.csv"
    fig_path = base + "_venn_subplots.png"

    summary.to_csv(counts_path, index=False)
    plot_venn_subplots(summary, args.title, fig_path, cols=args.cols)

    print("Done.")
    print(f"Wrote counts to: {counts_path}")
    print(f"Wrote venn subplots to: {fig_path}")

if __name__ == "__main__":
    main()
