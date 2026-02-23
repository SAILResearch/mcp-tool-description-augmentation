import os
import argparse
import pandas as pd
import numpy as np
from scipy.stats import chi2

def run_mcnemar(df, col1, col2):
    """Helper to run a single McNemar test between two columns"""
    b = ((df[col1] == 1) & (df[col2] == 0)).sum() # col1 passed, col2 failed
    c = ((df[col1] == 0) & (df[col2] == 1)).sum() # col2 passed, col1 failed
    
    total_discordant = b + c
    if total_discordant > 0:
        numerator = abs(b - c) - 1
        if numerator < 0: numerator = 0
        statistic = (numerator ** 2) / total_discordant
        p_value = chi2.sf(statistic, 1)
    else:
        statistic = 0.0
        p_value = 1.0
        
    # Odds Ratio (Effect Size): >1 means col1 is better, <1 means col2 is better
    if c > 0:
        odds_ratio = b / c
    else:
        odds_ratio = float('inf') if b > 0 else 1.0
        
    return p_value, odds_ratio

def run_cochrans_q(df, cols):
    """
    Manual implementation of Cochran's Q test.
    Q = (k-1) * (k * sum(C_j^2) - T^2) / (k * T - sum(R_i^2))
    k: number of treatments
    C_j: column totals
    R_i: row totals
    T: grand total
    """
    k = len(cols)
    data = df[cols].values
    
    # R_i: Row totals (sum of 1s for each task across models)
    R_i = data.sum(axis=1)
    
    # C_j: Column totals (total success for each model)
    C_j = data.sum(axis=0)
    
    # T: Grand total of successes
    T = data.sum()
    
    # Denominator components
    sum_Ri_sq = np.sum(R_i ** 2)
    numerator = (k - 1) * (k * np.sum(C_j ** 2) - (T ** 2))
    denominator = (k * T) - sum_Ri_sq
    
    if denominator == 0:
        return 0.0, 1.0 # No variance
        
    q_stat = numerator / denominator
    # Degrees of freedom = k - 1
    p_value = chi2.sf(q_stat, k - 1)
    
    return q_stat, p_value

def analyze_three_way(df):
    results = []
    grouped = df.groupby(["domain"]) # Analyze per domain
    
    # Columns to compare
    cols = ["FR", "BC", "WEx"]
    
    for domain, sub in grouped:
        # 1. Omnibus Test (Cochran's Q)
        q_stat, q_p_value = run_cochrans_q(sub, cols)
        
        row = {
            "domain": domain,
            "Cochrans_Q_p": q_p_value,
            "Significant_Difference_Exists": q_p_value < 0.05,
            # Placeholders for post-hoc
            "FR_vs_WEx_p": None,
            "BC_vs_WEx_p": None,
            "FR_vs_BC_p": None,
            "WEx_is_Worst": False
        }
        
        # 2. Post-hoc tests (Only if Q is significant)
        if row["Significant_Difference_Exists"]:
            # Bonferroni correction: alpha = 0.05 / 3 comparisons = 0.0167
            alpha_adj = 0.05 / 3
            
            # Pair 1: FR vs WEx
            p_fr_wex, or_fr_wex = run_mcnemar(sub, "FR", "WEx")
            row["FR_vs_WEx_p"] = p_fr_wex
            row["FR_vs_WEx_Sig"] = p_fr_wex < alpha_adj
            
            # Pair 2: BC vs WEx
            p_bc_wex, or_bc_wex = run_mcnemar(sub, "BC", "WEx")
            row["BC_vs_WEx_p"] = p_bc_wex
            row["BC_vs_WEx_Sig"] = p_bc_wex < alpha_adj

            # Pair 3: FR vs BC (for completeness)
            p_fr_bc, or_fr_bc = run_mcnemar(sub, "FR", "BC")
            row["FR_vs_BC_p"] = p_fr_bc
            
            # Conclusion Logic: Is WEx significantly worse than FR AND BC?
            # We check if FR>WEx and BC>WEx significantly
            fr_beats_wex = row["FR_vs_WEx_Sig"] and (sub["FR"].sum() > sub["WEx"].sum())
            bc_beats_wex = row["BC_vs_WEx_Sig"] and (sub["BC"].sum() > sub["WEx"].sum())
            
            if fr_beats_wex and bc_beats_wex:
                row["Conclusion"] = "WEx is significantly worse than both"
            elif fr_beats_wex:
                row["Conclusion"] = "WEx worse than FR only"
            elif bc_beats_wex:
                row["Conclusion"] = "WEx worse than BC only"
            else:
                row["Conclusion"] = "Differences exist but WEx is not strictly worst"
        else:
            row["Conclusion"] = "No significant difference among 3 methods"

        results.append(row)
        
    return pd.DataFrame(results)

def main():
    parser = argparse.ArgumentParser(description="Run Cochran's Q and Post-hoc McNemar for FR, BC, WEx.")
    parser.add_argument("--csv_path", required=True, help="CSV with columns: domain, FR, BC, WEx")
    args = parser.parse_args()

    if not os.path.isfile(args.csv_path):
        raise FileNotFoundError(f"File not found: {args.csv_path}")

    df = pd.read_csv(args.csv_path)
    # Ensure ints
    for c in ["FR", "BC", "WEx"]:
        df[c] = df[c].astype(int)

    results = analyze_three_way(df)

    base, _ = os.path.splitext(args.csv_path)
    out_path = base + "_cochran_results.csv"
    results.to_csv(out_path, index=False)

    print("\n✅ Analysis completed.")
    print(f"Results saved to: {out_path}")
    
    # Pretty print
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print("\nSummary:")
    cols_to_show = ["domain", "Cochrans_Q_p", "FR_vs_WEx_p", "BC_vs_WEx_p", "Conclusion"]
    print(results[cols_to_show])

if __name__ == "__main__":
    main()