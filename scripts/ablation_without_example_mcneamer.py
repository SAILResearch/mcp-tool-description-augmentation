import os
import argparse
import pandas as pd
from scipy.stats import chi2

def run_ablation_significance(df):
    results = []
    # We are comparing FR (Baseline) vs WEx (Ablation)
    grouped = df.groupby(["domain", "Model"])
    
    for (domain, model), sub in grouped:
        # Construct the 2x2 components for FR vs WEx
        # b: FR Passed, WEx Failed (Example helped)
        fr_wins = ((sub["FR"] == 1) & (sub["WEx"] == 0)).sum()
        
        # c: WEx Passed, FR Failed (Example hurt)
        wex_wins = ((sub["FR"] == 0) & (sub["WEx"] == 1)).sum()
        
        total_discordant = fr_wins + wex_wins
        
        # McNemar Statistic Calculation with Continuity Correction
        # Formula: (|b - c| - 1)^2 / (b + c)
        if total_discordant > 0:
            numerator = abs(fr_wins - wex_wins) - 1
            if numerator < 0: numerator = 0
            statistic = (numerator ** 2) / total_discordant
            p_value = chi2.sf(statistic, 1) # 1 degree of freedom
        else:
            statistic = 0.0
            p_value = 1.0 # Perfect agreement, no impact
            
        result = {
            "domain": domain,
            "Model": model,
            "FR_only_wins": int(fr_wins),   # Times example was needed
            "WEx_only_wins": int(wex_wins), # Times example was bad
            "p_value": p_value,
            "Impact_Significant": p_value < 0.05,
            # Interpretation Helper
            "Conclusion": "No Impact"
        }
        
        # Logic to assign string conclusion
        if result["Impact_Significant"]:
            if fr_wins > wex_wins:
                result["Conclusion"] = "Significant Drop (Examples Critical)"
            else:
                result["Conclusion"] = "Significant Rise (Examples Harmful)"
        else:
            result["Conclusion"] = "No Significant Impact (Examples Redundant)"

        results.append(result)
        
    return pd.DataFrame(results)

def main():
    parser = argparse.ArgumentParser(
        description="Run McNemar's test to check if removing examples (WEx) significantly changes performance compared to Full Rubric (FR)."
    )
    parser.add_argument("--csv_path", required=True, help="Path to CSV with columns: domain, Model, FR, WEx")
    args = parser.parse_args()

    if not os.path.isfile(args.csv_path):
        raise FileNotFoundError(f"File not found: {args.csv_path}")

    # Load Data
    print(f"Reading data from: {args.csv_path}")
    df = pd.read_csv(args.csv_path)

    # Validate Columns
    required_cols = ["domain", "Model", "FR", "WEx"]
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"CSV is missing required columns. It must contain: {required_cols}")

    # Ensure integer types for logic comparisons
    df["FR"] = df["FR"].astype(int)
    df["WEx"] = df["WEx"].astype(int)

    # Run Analysis
    results = run_ablation_significance(df)

    # Save Results
    base, _ = os.path.splitext(args.csv_path)
    out_path = base + "_mcnemar_results.csv"
    results.to_csv(out_path, index=False)

    # Print Summary to Console
    print("\n✅ Statistical test completed.")
    print(f"Results saved to: {out_path}")
    print("\nSummary Table:")
    # Setting pandas display options to ensure columns don't get hidden
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print(results[["domain", "Model", "p_value", "Impact_Significant", "Conclusion"]])

if __name__ == "__main__":
    main()