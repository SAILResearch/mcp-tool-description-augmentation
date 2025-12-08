import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# Step 1: Load data from CSV
# Save your table as "mcp_tool_quality_scores.csv"
# with columns:
# mcp_server_name,Type,Median_tool_quality_Score
csv_file = "/Users/mohammedmehedihasan/personal/codes/MCP-Universe/scripts/mcp_tool_quality_scores.csv"
df = pd.read_csv(csv_file)

# Step 2: Normalize column names to match the plotting code
df = df.rename(
    columns={
        "mcp_server_name": "mcp_servername",
        "Type": "integration_type",
        "Median_tool_quality_Score": "median_tool_quality_score",
    }
)

# Step 3: Group by server and integration_type
# (median over rows per server, in case there are repeats)
quality_df = (
    df.groupby(["mcp_servername", "integration_type"])["median_tool_quality_score"]
    .median()
    .reset_index()
)

# Step 4: Define desired integration type order and filter only these
ordered_types = ["official", "community"]
filtered_df = quality_df[quality_df["integration_type"].isin(ordered_types)]

# Calculate and print median tool quality score per integration type
median_quality_scores = (
    filtered_df.groupby("integration_type")["median_tool_quality_score"].median()
)
print("Median Tool Quality Score per Integration Type:")
print(median_quality_scores)

# Step 5: Define color map for consistent styling
palette = sns.color_palette("Set2", len(ordered_types))
color_map = dict(zip(ordered_types, palette))

# Step 6: Plot violin chart
plt.figure(figsize=(12, 7))
sns.violinplot(
    data=filtered_df,
    x="integration_type",
    y="median_tool_quality_score",
    palette=color_map,
    cut=0,
    order=ordered_types,
)

# Box plot (overlay)
sns.boxplot(
    data=filtered_df,
    x="integration_type",
    y="median_tool_quality_score",
    order=ordered_types,
    width=0.2,
    showcaps=True,
    boxprops={"facecolor": "none", "zorder": 10},
    showfliers=False,
    whiskerprops={"linewidth": 2},
    medianprops={"color": "white", "linewidth": 2},
    zorder=10,
)

plt.ylabel("Median Tool Quality Score", fontsize=22)
plt.xlabel("Type", fontsize=22)
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)

# Optional legend (matches sample style)
legend_labels = ordered_types
legend_handles = [
    Patch(color=color_map[label], label=label.capitalize()) for label in legend_labels
]

plt.legend(
    handles=legend_handles,
    title="Type",
    fontsize=18,
    title_fontsize=18,
    loc="upper left",
    frameon=False,
)

plt.tight_layout()
plt.savefig(
    "analysis_output/median_tool_quality_by_type_violin_plot.pdf",
    format="pdf",
    dpi=600,
    bbox_inches="tight",
)
plt.show()
