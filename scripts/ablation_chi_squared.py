import os
import argparse
import math
import pandas as pd
from scipy.stats import chi2_contingency

def run_mcnemar_test(df):
    results = []
    grouped = df.groupby(["domain", "Model"])
    for (domain, model), sub in grouped:
        # 2x2 paired table components
        both = ((sub["FR"] == 1) & (sub["BC"] == 1)).sum()
        fr_only = ((sub["FR"] == 1) & (sub["BC"] == 0)).sum()
        bc_only = ((sub["FR"] == 0) & (sub["BC"] == 1)).sum()
        none = ((sub["FR"] == 0) & (sub["BC"] == 0)).sum()

        # McNemar chi-squared (focuses on off-diagonals)
        table = [[both, fr_only], [bc_only, none]]
        chi2, p, _, _ = chi2_contingency(table, correction=True)

        # Signed phi for McNemar: use discordant N = fr_only + bc_only
        discordant = fr_only + bc_only
        if discordant > 0:
            phi_mag = math.sqrt(chi2 / discordant)
            sign = 1.0 if (bc_only - fr_only) > 0 else (-1.0 if (bc_only - fr_only) < 0 else 0.0)
            phi_signed = sign * phi_mag
        else:
            phi_signed = 0.0

        result = {
            "domain": domain,
            "Model": model,
            "FR_success": int(sub["FR"].sum()),
            "BC_success": int(sub["BC"].sum()),
            "FR_only": int(fr_only),
            "BC_only": int(bc_only),
            "Both": int(both),
            "None": int(none),
            "chi2": chi2,
            "p_value": p,
            "phi_signed": phi_signed,      # effect size: positive => BC > FR
            "BC_more_than_FR": (bc_only > fr_only) and (p < 0.05)
        }
        results.append(result)
    return pd.DataFrame(results)

def main():
    parser = argparse.ArgumentParser(
        description="Test whether BC solves significantly more tasks than FR using McNemar's chi-squared test, and report phi effect size."
    )
    parser.add_argument("--csv_path", required=True, help="Path to CSV with columns: domain, Model, FR, BC")
    args = parser.parse_args()

    if not os.path.isfile(args.csv_path):
        raise FileNotFoundError(f"File not found: {args.csv_path}")

    df = pd.read_csv(args.csv_path)
    df["FR"] = df["FR"].astype(int)
    df["BC"] = df["BC"].astype(int)

    results = run_mcnemar_test(df)

    base, _ = os.path.splitext(args.csv_path)
    out_path = base + "_chi2_results.csv"
    results.to_csv(out_path, index=False)

    print("✅ Statistical test completed.")
    print(f"Results saved to: {out_path}")
    print("\nSummary:")
    print(results[["domain", "Model", "p_value", "phi_signed", "BC_more_than_FR"]])

if __name__ == "__main__":
    main()
