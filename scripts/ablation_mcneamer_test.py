import os
import argparse
import pandas as pd
import numpy as np
from statsmodels.stats.contingency_tables import mcnemar

def run_actual_mcnemar_test(df):
    results = []
    grouped = df.groupby(["domain", "Model"])
    
    for (domain, model), sub in grouped:
        # Define the 2x2 contingency table for McNemar
        # a: Both pass
        # b: FR passed, BC failed (Discordant)
        # c: BC passed, FR failed (Discordant)
        # d: Both failed
        a = ((sub["FR"] == 1) & (sub["BC"] == 1)).sum()
        b = ((sub["FR"] == 1) & (sub["BC"] == 0)).sum()
        c = ((sub["FR"] == 0) & (sub["BC"] == 1)).sum()
        d = ((sub["FR"] == 0) & (sub["BC"] == 0)).sum()

        table = [[a, b], 
                 [c, d]]

        # McNemar's Test
        # exact=True uses the Binomial distribution (recommended for small samples/low discordant counts)
        # exact=False uses the Chi-square distribution
        mcnemar_result = mcnemar(table, exact=True)
        p_val = mcnemar_result.pvalue

        # Phi coefficient for effect size
        # (a*d - b*c) / sqrt((a+b)(c+d)(a+c)(b+d))
        denom = (a + b) * (c + d) * (a + c) * (b + d)
        phi_signed = ((a * d) - (b * c)) / (denom ** 0.5) if denom > 0 else 0.0

        results.append({
            "domain": domain,
            "Model": model,
            "FR_total_success": int(sub["FR"].sum()),
            "BC_total_success": int(sub["BC"].sum()),
            "FR_only_wins (b)": int(b),
            "BC_only_wins (c)": int(c),
            "Both_pass (a)": int(a),
            "Both_fail (d)": int(d),
            "p_value": p_val,
            "phi_signed": phi_signed,
            # Logic: BC is statistically better IF BC has more unique wins AND p < 0.05
            "BC_statistically_better": (c > b) and (p_val < 0.05)
        })

    return pd.DataFrame(results)

def main():
    parser = argparse.ArgumentParser(
        description="Proper McNemar's test to compare BC vs FR performance."
    )
    parser.add_argument("--csv_path", required=True, help="Path to input CSV")
    args = parser.parse_args()

    if not os.path.isfile(args.csv_path):
        print(f"Error: File {args.csv_path} not found.")
        return

    df = pd.read_csv(args.csv_path)
    
    # Ensure columns are integer
    df["FR"] = df["FR"].astype(int)
    df["BC"] = df["BC"].astype(int)

    results_df = run_actual_mcnemar_test(df)

    # Save output
    base, _ = os.path.splitext(args.csv_path)
    out_path = base + "_mcnemar_results.csv"
    results_df.to_csv(out_path, index=False)

    print("\n" + "="*80)
    print("MCNEMAR STATISTICAL TEST RESULTS")
    print("="*80)
    # Filter columns for a clean summary display
    summary_cols = ["domain", "Model", "FR_only_wins (b)", "BC_only_wins (c)", "p_value", "BC_statistically_better"]
    print(results_df[summary_cols].to_string(index=False))
    print("="*80)
    print(f"Full results saved to: {out_path}\n")

if __name__ == "__main__":
    main()