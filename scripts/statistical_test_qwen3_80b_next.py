import pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.contingency_tables import mcnemar
import numpy as np

# ---------------------------------------------------
# Load and rename columns
# ---------------------------------------------------
df = pd.read_csv("/Users/mohammedmehedihasan/personal/codes/MCP-Universe/scripts/qwen3-next-80b-statistical.csv")
# df = pd.read_csv("/Users/mohammedmehedihasan/personal/codes/MCP-Universe/scripts/qwen3-next-80b-statistical-old.csv")

rename_map = {
    "qwen3-next-80b-a3b-instruct-SR-optimized": "SR_optimized",
    "qwen3-next-80b-a3b-instruct-SR-baseline": "SR_baseline",
    "qwen3-next-80b-a3b-instruct-AE-optimized": "AE_optimized",
    "qwen3-next-80b-a3b-instruct-AE-baseline": "AE_baseline",
    "qwen3-next-80b-a3b-instruct-AS-optimized": "AS_optimized",
    "qwen3-next-80b-a3b-instruct-AS-baseline": "AS_baseline"
}

df = df.rename(columns=rename_map)

for col in [
    "SR_baseline", "SR_optimized",
    "AE_baseline", "AE_optimized",
    "AS_baseline", "AS_optimized"
]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# ---------------------------------------------------
# Helper functions
# ---------------------------------------------------

def signed_phi(df_subset):
    """McNemar effect size (Cohen's g), bounded in [-1, 1]."""

    contingency = pd.crosstab(df_subset["SR_baseline"], df_subset["SR_optimized"])
    contingency = contingency.reindex(index=[0, 1], columns=[0, 1], fill_value=0)
    b = contingency.loc[0, 1]
    c = contingency.loc[1, 0]
    denom = b + c
    if denom == 0:
        return 0.0
    return (b - c) / denom

def run_mcnemar_stats(df_subset):
    pair = df_subset[["SR_baseline", "SR_optimized"]].dropna()
    contingency = pd.crosstab(pair["SR_baseline"], pair["SR_optimized"])
    contingency = contingency.reindex(index=[0, 1], columns=[0, 1], fill_value=0)
    print("======= Contingency Table =======")
    result = mcnemar(contingency, exact=False, correction=True)
    phi = signed_phi(pair)
    return result.statistic, result.pvalue, phi, result.pvalue < 0.05

def run_wilcoxon_stats(df_subset, before_col, after_col):
    paired = df_subset[[before_col, after_col]].dropna()
    stat, p = wilcoxon(paired[before_col], paired[after_col])
    return stat, p, p < 0.05

# ---------------------------------------------------
# Build all three tables
# ---------------------------------------------------

domains = df["domain"].unique()

# Panel A: SR (McNemar)
panelA_rows = []
for d in domains:
    subset = df[df["domain"] == d]
    stat, p, phi, sig = run_mcnemar_stats(subset)
    panelA_rows.append([d, "ModelNameHere", p, phi, sig])

# Add ALL_DOMAINS
stat, p, phi, sig = run_mcnemar_stats(df)
panelA_rows.append(["ALL_DOMAINS", "ModelNameHere", p, phi, sig])

panelA = pd.DataFrame(
    panelA_rows,
    columns=["Domain", "Model", "p_value", "phi_signed", "Significant"]
)

# Panel B: AE (Wilcoxon)
panelB_rows = []
for d in domains:
    subset = df[df["domain"] == d]
    stat, p, sig = run_wilcoxon_stats(subset, "AE_baseline", "AE_optimized")
    panelB_rows.append([d, "ModelNameHere", stat, p, sig])

stat, p, sig = run_wilcoxon_stats(df, "AE_baseline", "AE_optimized")
panelB_rows.append(["ALL_DOMAINS", "ModelNameHere", stat, p, sig])

panelB = pd.DataFrame(
    panelB_rows,
    columns=["Domain", "Model", "Statistic", "p_value", "Significant"]
)

# Panel C: AS (Wilcoxon)
panelC_rows = []
for d in domains:
    subset = df[df["domain"] == d]
    stat, p, sig = run_wilcoxon_stats(subset, "AS_baseline", "AS_optimized")
    panelC_rows.append([d, "ModelNameHere", stat, p, sig])

stat, p, sig = run_wilcoxon_stats(df, "AS_baseline", "AS_optimized")
panelC_rows.append(["ALL_DOMAINS", "ModelNameHere", stat, p, sig])

panelC = pd.DataFrame(
    panelC_rows,
    columns=["Domain", "Model", "Statistic", "p_value", "Significant"]
)

# ---------------------------------------------------
# Print all tables
# ---------------------------------------------------
print("\n=== Panel A: Domain-level McNemar statistics (SR) ===")
print(panelA)

print("\n=== Panel B: Wilcoxon Signed Rank test (AE) ===")
print(panelB)

print("\n=== Panel C: Wilcoxon Signed Rank test (AS) ===")
print(panelC)
