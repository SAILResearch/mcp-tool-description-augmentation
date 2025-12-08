#!/usr/bin/env python3
# Statistical comparison of median tool quality scores between MCP server types

import pandas as pd
from scipy.stats import kruskal, mannwhitneyu
from cliffs_delta import cliffs_delta

# -------------------------------------------
# 1. Load the data
# -------------------------------------------
df = pd.read_csv("scripts/tool-quality-score-by-server-purpose.csv")

# Ensure expected column names exist
print("Columns:", df.columns.tolist())

# Assuming the columns are something like:
# 'server_type' -> values: 'official', 'community'
# 'median_tool_quality_score' -> numeric
# Adjust if your column names differ
server_col = 'Type'
score_col = 'median_purpose_score'

# -------------------------------------------
# 2. Split groups by server type
# -------------------------------------------
groups = df.groupby(server_col)[score_col].apply(list)
print("\nGroup sizes:")
print(groups.apply(len))

# -------------------------------------------
# 3. Kruskal–Wallis H-test
# -------------------------------------------
h_stat, p_val = kruskal(*groups)
print("\nKruskal–Wallis H-test results:")
print(f"H = {h_stat:.4f}, p = {p_val:.6f}")

# -------------------------------------------
# 4. Post-hoc Mann–Whitney U test (pairwise)
# -------------------------------------------
unique_types = groups.index.tolist()

if len(unique_types) == 2:
    # Direct pairwise comparison for two groups
    g1, g2 = groups.iloc[0], groups.iloc[1]
    stat, p_mwu = mannwhitneyu(g1, g2, alternative="two-sided")
    delta, size = cliffs_delta(g1, g2)
    print("\nMann–Whitney U test results:")
    print(f"U = {stat:.4f}, p = {p_mwu:.6f}")
    print(f"Cliff’s Delta = {delta:.4f} ({size})")

else:
    # For more than 2 types, test all pairs
    from itertools import combinations
    print("\nPost-hoc pairwise Mann–Whitney U tests with Cliff’s Delta:")
    for (a, b) in combinations(unique_types, 2):
        g1, g2 = groups[a], groups[b]
        stat, p_mwu = mannwhitneyu(g1, g2, alternative="two-sided")
        delta, size = cliffs_delta(g1, g2)
        print(f"\n{a} vs {b}")
        print(f"U = {stat:.4f}, p = {p_mwu:.6f}, Cliff’s Delta = {delta:.4f} ({size})")

# -------------------------------------------
# 5. Optional summary interpretation
# -------------------------------------------
if p_val < 0.05:
    print("\n=> The Kruskal–Wallis test indicates a statistically significant difference among groups.")
else:
    print("\n=> No statistically significant difference detected among groups.")
