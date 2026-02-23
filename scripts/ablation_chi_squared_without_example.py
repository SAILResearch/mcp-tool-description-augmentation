import os
import argparse
import pandas as pd
from scipy.stats import chi2_contingency

def run_pearson_test(df):
    results = []
    # Group by domain and model to analyze each pair separately
    grouped = df.groupby(["domain", "Model"])

    for (domain, model), sub in grouped:
        # 1. Calculate the 2x2 Contingency Table Components
        # We are looking for Agreement vs Disagreement
        both_success = ((sub["FR"] == 1) & (sub["WEx"] == 1)).sum()
        fr_only      = ((sub["FR"] == 1) & (sub["WEx"] == 0)).sum()
        wex_only     = ((sub["FR"] == 0) & (sub["WEx"] == 1)).sum()
        both_fail    = ((sub["FR"] == 0) & (sub["WEx"] == 0)).sum()

        # 2. Construct Table for Scipy
        # [[Both pass, FR passes/WEx fails],
        #  [WEx passes/FR fails, Both fail]]
        table = [[both_success, fr_only], [wex_only, both_fail]]

        # 3. Run Pearson's Chi-Squared Test
        # correction=True applies Yates' correction for continuity (standard for 2x2)
        chi2_stat, p_value, dof, expected = chi2_contingency(table, correction=True)

        # 4. Calculate Phi Coefficient (Effect Size for Correlation)
        # Phi = (ad - bc) / sqrt((a+b)(c+d)(a+c)(b+d))
        a, b = both_success, fr_only
        c, d = wex_only, both_fail
        
        numerator = (a * d) - (b * c)
        denominator = ((a + b) * (c + d) * (a + c) * (b + d)) ** 0.5
        
        if denominator > 0:
            phi = numerator / denominator
        else:
            phi = 0.0

        result = {
            "domain": domain,
            "Model": model,
            "Both_Success": int(both_success),
            "Both_Fail": int(both_fail),
            "Disagreement": int(fr_only + wex_only),
            "p_value": p_value,
            "phi_coefficient": phi,
            "Significant_Correlation": p_value < 0.05
        }
        
        # Add a text interpretation
        if result["Significant_Correlation"]:
            if phi > 0.7:
                result["Interpretation"] = "Very Strong Agreement"
            elif phi > 0.4:
                result["Interpretation"] = "Moderate Agreement"
            elif phi > 0:
                result["Interpretation"] = "Weak Agreement"
            else:
                result["Interpretation"] = "Negative Correlation (Inverse)"
        else:
            result["Interpretation"] = "Independent (No Correlation)"

        results.append(result)

    return pd.DataFrame(results)

def main():
    parser = argparse.ArgumentParser(
        description="Run Pearson's Chi-Squared test to check correlation/agreement between FR and WEx."
    )
    parser.add_argument("--csv_path", required=True, help="Path to CSV with columns: domain, Model, FR, WEx")
    args = parser.parse_args()

    if not os.path.isfile(args.csv_path):
        raise FileNotFoundError(f"File not found: {args.csv_path}")

    print(f"Reading data from: {args.csv_path}")
    df = pd.read_csv(args.csv_path)

    # Validate Columns
    required_cols = ["domain", "Model", "FR", "WEx"]
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"CSV is missing required columns. It must contain: {required_cols}")

    df["FR"] = df["FR"].astype(int)
    df["WEx"] = df["WEx"].astype(int)

    results = run_pearson_test(df)

    base, _ = os.path.splitext(args.csv_path)
    out_path = base + "_pearson_results.csv"
    results.to_csv(out_path, index=False)

    print("\n✅ Pearson Chi-Squared test completed.")
    print(f"Results saved to: {out_path}")
    print("\nSummary Table:")
    
    # Format display
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.float_format', '{:.4f}'.format)
    
    # Print key columns
    print(results[["domain", "Model", "p_value", "phi_coefficient", "Interpretation"]])

if __name__ == "__main__":
    main()