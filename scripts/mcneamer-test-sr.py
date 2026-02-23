import pandas as pd
import numpy as np
from scipy import stats

def analyze_mcnemar(file_path):
    try:
        # Read the CSV file
        df = pd.read_csv(file_path)
        
        # Define column names
        col_optimized = 'qwen3-next-80b-a3b-instruct-SR-optimized'
        col_baseline = 'qwen3-next-80b-a3b-instruct-SR-baseline'
        
        # Verify columns exist
        if col_optimized not in df.columns or col_baseline not in df.columns:
            # Handle potential duplicate column names by checking if pandas mangled them (e.g., name.1)
            # or simply check indices if names are not exact.
            # For this script, we assume exact names or first occurrence.
            available_cols = df.columns.tolist()
            if col_optimized not in available_cols:
                raise ValueError(f"Column '{col_optimized}' not found.")
            if col_baseline not in available_cols:
                raise ValueError(f"Column '{col_baseline}' not found.")

        # Extract data and drop rows with NaN in these columns
        data = df[[col_optimized, col_baseline]].dropna()
        
        # Ensure data is binary (0 and 1)
        y_opt = data[col_optimized].astype(int)
        y_base = data[col_baseline].astype(int)
        
        # Calculate Contingency Table for McNemar
        # Table Layout:
        #                  Baseline
        #                  0      1
        # Optimized   0    a      b
        #             1    c      d
        
        # a: Both 0
        # b: Opt=0, Base=1 (Baseline better/Optimized failed)
        # c: Opt=1, Base=0 (Optimized better/Baseline failed)
        # d: Both 1
        
        a = ((y_opt == 0) & (y_base == 0)).sum()
        b = ((y_opt == 0) & (y_base == 1)).sum()
        c = ((y_opt == 1) & (y_base == 0)).sum()
        d = ((y_opt == 1) & (y_base == 1)).sum()
        
        total_n = a + b + c + d
        
        print(f"Contingency Table:")
        print(f"Both 0 (a): {a}")
        print(f"Opt=0, Base=1 (b - negative change): {b}")
        print(f"Opt=1, Base=0 (c - positive change): {c}")
        print(f"Both 1 (d): {d}")
        print("-" * 30)

        # McNemar's Test
        # Statistic = (b - c)^2 / (b + c)
        # We use a continuity correction if b + c is small, but standard request implies raw chi-squared usually.
        # Here we implement the standard calculation without continuity correction for the "chi squared" value requested,
        # but for p-value, exact binomial or corrected chi2 is often preferred for small samples.
        # We will output the standard Chi2 as requested.
        
        discordant_sum = b + c
        if discordant_sum == 0:
            print("No discordant pairs found. Chi-squared is 0, p-value is 1.0.")
            return

        chi2_stat = (abs(b - c) - 1)**2 / discordant_sum if discordant_sum < 25 else (b - c)**2 / discordant_sum
        # Note: The prompt asks for "chi squared", implying the approximate test. 
        # Standard definition: Chi2 = (b-c)^2 / (b+c).
        # We will use the uncorrected version for the statistic calculation to match standard formulas 
        # unless sample is very small, but sticking to (b-c)^2 / (b+c) is safest for general "McNemar Chi2".
        
        chi2_val = (b - c)**2 / (b + c)
        
        # P-value (1 degree of freedom)
        p_value = 1 - stats.chi2.cdf(chi2_val, df=1)
        
        # Signed Phi Calculation
        # Phi = sqrt(Chi2 / N)
        # Sign: Positive if c > b (Optimized > Baseline), Negative if b > c.
        
        phi_val = np.sqrt(chi2_val / total_n)
        if b > c:
            phi_val = -phi_val
            
        print(f"McNemar's Chi-squared statistic: {chi2_val:.4f}")
        print(f"P-value: {p_value:.4e}")
        print(f"Signed Phi (Effect Size): {phi_val:.4f}")
        
        if p_value < 0.05:
            print("Result: Statistically Significant Difference")
        else:
            print("Result: No Statistically Significant Difference")

    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage:
file_path = "/Users/mohammedmehedihasan/personal/codes/MCP-Universe/scripts/qwen3-next-80b-statistical-old.csv"
analyze_mcnemar(file_path)