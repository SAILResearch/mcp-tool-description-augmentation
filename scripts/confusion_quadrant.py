import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

REQUIRED_COLS = {"domain", "Model", "task_name", "FR", "BC"}

def build_counts(df: pd.DataFrame) -> pd.DataFrame:
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

    counts = (
        df.groupby(["domain", "Model", "bucket"], as_index=False)
          .size()
          .rename(columns={"size": "count"})
    )
    return counts


def plot_subplots(counts: pd.DataFrame, title: str, out_path: str):
    coords = {
        "both": (1, 1),
        "FR_only": (0, 1),
        "BC_only": (1, 0),
        "none": (0, 0),
    }

    pairs = counts[["domain", "Model"]].drop_duplicates().to_records(index=False).tolist()
    n = len(pairs)
    cols = 3
    rows = int(np.ceil(n / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    axes = np.ravel(axes)
    rng = np.random.default_rng(42)
    max_count = counts["count"].max() if len(counts) else 1
    size_scale = 800 if max_count <= 10 else 1200

    for i, (domain, model) in enumerate(pairs):
        ax = axes[i]
        subset = counts[(counts["domain"] == domain) & (counts["Model"] == model)]
        ax.set_title(f"{domain}\n{model}", fontsize=10)
        ax.set_xlim(-0.5, 1.5)
        ax.set_ylim(-0.5, 1.5)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.set_xlabel("BC", fontsize=8)
        ax.set_ylabel("FR", fontsize=8)

        for _, row in subset.iterrows():
            x, y = coords[row["bucket"]]
            jitter_x = float(rng.normal(0, 0.04))
            jitter_y = float(rng.normal(0, 0.04))
            c = row["count"]
            ax.scatter(x + jitter_x, y + jitter_y,
                       s=(c / max_count) * size_scale + 50,
                       alpha=0.7)
            ax.annotate(str(c), (x + jitter_x, y + jitter_y),
                        textcoords="offset points", xytext=(0, 6),
                        ha="center", fontsize=8)

    # Remove extra axes
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    fig.suptitle(title, fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Generate quadrant scatter subplots for each domain-model configuration.")
    parser.add_argument("--csv_path", required=True, help="Path to input CSV with columns: domain, Model, task_name, FR, BC")
    parser.add_argument("--title", default="FR vs BC Quadrants per Domain-Model", help="Title of the figure")
    args = parser.parse_args()

    csv_path = args.csv_path
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"File not found: {csv_path}")

    df = pd.read_csv(csv_path)
    counts = build_counts(df)

    base, _ = os.path.splitext(csv_path)
    out_counts = base + "_quadrant_counts.csv"
    out_fig = base + "_quadrant_subplots.png"

    pivot = (
        counts.pivot_table(index=["domain", "Model"],
                           columns="bucket",
                           values="count",
                           aggfunc="sum",
                           fill_value=0)
              .reindex(columns=["both", "FR_only", "BC_only", "none"], fill_value=0)
              .astype(int)
    )
    pivot.to_csv(out_counts)
    plot_subplots(counts, args.title, out_fig)

    print("✅ Done.")
    print(f"Wrote counts table: {out_counts}")
    print(f"Wrote subplot figure: {out_fig}")


if __name__ == "__main__":
    main()
