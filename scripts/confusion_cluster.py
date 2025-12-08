
import sys
import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt

REQUIRED_COLS = {"domain", "Model", "task_name", "FR", "BC"}

def build_confusion_cluster(df: pd.DataFrame) -> pd.DataFrame:
    # Validate columns
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    # Normalize FR and BC to 0-1 ints
    df["FR"] = df["FR"].astype(int)
    df["BC"] = df["BC"].astype(int)

    # Classify each task into four buckets
    def classify(row):
        fr, bc = row["FR"], row["BC"]
        if fr == 1 and bc == 1:
            return "both"
        if fr == 1 and bc == 0:
            return "FR_only"
        if fr == 0 and bc == 1:
            return "BC_only"
        return "none"

    df["bucket"] = df.apply(classify, axis=1)

    # Group and pivot to get counts per domain-model
    grouped = (
        df.groupby(["domain", "Model", "bucket"], as_index=False)
          .size()
          .rename(columns={"size": "count"})
    )

    pivot = (
        grouped.pivot_table(index=["domain", "Model"],
                            columns="bucket",
                            values="count",
                            aggfunc="sum",
                            fill_value=0)
            .reindex(columns=["both", "FR_only", "BC_only", "none"], fill_value=0)
            .astype(int)
    )

    # Ensure a conventional index and column order
    pivot = pivot.sort_index()
    pivot.columns.name = None
    return pivot

def plot_confusion_cluster(pivot: pd.DataFrame, title: str, out_path: str):
    # Friendly x labels
    labels = [f"{d}\n{m}" for d, m in pivot.index.to_list()]

    # Stacked bar chart for the four buckets
    ax = pivot.plot(kind="bar", stacked=True, figsize=(14, 7))
    ax.set_title(title)
    ax.set_xlabel("Domain and Model")
    ax.set_ylabel("Number of tasks")
    ax.set_xticklabels(labels, rotation=0, ha="center")
    ax.legend(title="Bucket", frameon=False)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Build a confusion matrix cluster for FR vs BC per domain-model.")
    parser.add_argument("--csv_path", default='/Users/mohammedmehedihasan/personal/codes/MCP-Universe/scripts/confusion_matrix_data.csv', help="Path to input CSV with columns: domain, Model, task_name, FR, BC")
    parser.add_argument("--title", default="Confusion matrix cluster by domain-model",
                        help="Chart title")
    args = parser.parse_args()

    csv_path = args.csv_path
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"File not found: {csv_path}")

    df = pd.read_csv(csv_path)
    pivot = build_confusion_cluster(df)

    # Save counts CSV next to input
    base, _ = os.path.splitext(csv_path)
    counts_path = base + "_confusion_cluster_counts.csv"
    fig_path = base + "_confusion_cluster.png"

    pivot.to_csv(counts_path)

    # Plot
    plot_confusion_cluster(pivot, args.title, fig_path)

    # Print small summary for CLI users
    total_rows = len(df)
    total_pairs = pivot.shape[0]
    print("Done.")
    print(f"Input rows: {total_rows}")
    print(f"Domain-model pairs: {total_pairs}")
    print(f"Wrote counts to: {counts_path}")
    print(f"Wrote figure to: {fig_path}")

if __name__ == "__main__":
    main()
