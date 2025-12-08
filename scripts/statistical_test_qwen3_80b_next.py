import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.contingency_tables import mcnemar

# ==========================================
# 1. CONFIGURATION
# ==========================================
# Replace with your actual csv file path
file_path = "/Users/mohammedmehedihasan/personal/codes/MCP-Universe/scripts/qwen3-next-80b-statistical.csv"

# Define your specific column names here based on your CSV headers
# Based on your image/description, update these prefixes:
# Example: if column is 'qwen3-next-80b-a3b-instruct-SR-baseline'
model_name = "qwen3-next-80b" # Just for display in the table

cols = {
    'SR_base': 'qwen3-next-80b-a3b-instruct-SR-baseline',
    'SR_opt':  'qwen3-next-80b-a3b-instruct-SR-optimized',
    
    'AE_base': 'qwen3-next-80b-a3b-instruct-AE-baseline',
    'AE_opt':  'qwen3-next-80b-a3b-instruct-AE-optimized',
    
    'AS_base': 'qwen3-next-80b-a3b-instruct-AS-baseline',
    'AS_opt':  'qwen3-next-80b-a3b-instruct-AS-optimized',
    
    'domain':  'domain'
}

# ==========================================
# 2. STATISTICAL FUNCTIONS
# ==========================================

def get_effect_size_wilcoxon(stat, n):
    """Calculates r (Z / sqrt(N)) for Wilcoxon."""
    # Approximate Z score from W statistic
    # Mean W = n(n+1)/4, Sigma W = sqrt(n(n+1)(2n+1)/24)
    if n == 0: return 0
    mu = n * (n + 1) / 4
    sigma = np.sqrt(n * (n + 1) * (2 * n + 1) / 24)
    if sigma == 0: return 0
    z = (stat - mu) / sigma
    return z / np.sqrt(n)

def analyze_subset(df, subset_name):
    results = []
    
    # --- 1. SR (Success Rate) - McNemar's Test ---
    # Logic: SR is Binary (0 or 1). We check for frequency changes.
    # Effect Size: Phi equivalent (difference in proportions direction)
    try:
        # Create Contingency Table
        # 00: Fail->Fail, 01: Fail->Pass, 10: Pass->Fail, 11: Pass->Pass
        tbl = pd.crosstab(df[cols['SR_base']], df[cols['SR_opt']])
        
        # Ensure table is 2x2 for McNemar (fill missing categories with 0)
        tbl = tbl.reindex(index=[0, 1], columns=[0, 1], fill_value=0)
        
        # Calculate McNemar
        # exact=True is better for small sample sizes (<25 discordant pairs)
        mc_res = mcnemar(tbl, exact=True)
        p_val = mc_res.pvalue
        
        # Calculate Direction/Effect (Signed Phi-ish metric)
        # b = Fail->Pass (Improvement), c = Pass->Fail (Regression)
        b = tbl.loc[0, 1]
        c = tbl.loc[1, 0]
        total = b + c
        
        if total == 0:
            phi = 0.0
        else:
            # Simple improvement ratio scaled -1 to 1 for this context
            # +1 means all changes were improvements, -1 means all were regressions
            phi = (b - c) / total 
            
        results.append({
            'Domain': subset_name,
            'Metric': 'SR',
            'Model': model_name,
            'p_value': p_val,
            'phi_signed': round(phi, 4),
            'Significant': p_val < 0.05
        })
    except Exception as e:
        # Handle cases where column data might be missing or constant
        results.append({'Domain': subset_name, 'Metric': 'SR', 'Model': model_name, 'p_value': 1.0, 'phi_signed': 0, 'Significant': False})

    # --- 2. AE (Average Evaluator) - Wilcoxon Signed-Rank ---
    # Logic: Continuous bounded 0-1.
    try:
        diff = df[cols['AE_opt']] - df[cols['AE_base']]
        # Wilcoxon requires removing cases where difference is 0 (automatic in some libs, explicit here)
        diff = diff[diff != 0]
        n = len(diff)
        
        if n < 1:
            p_val = 1.0
            eff = 0.0
        else:
            w_stat, p_val = stats.wilcoxon(df[cols['AE_base']], df[cols['AE_opt']])
            # Direction: Positive means Opt > Base (Improvement)
            eff = get_effect_size_wilcoxon(w_stat, n)
            # Adjust sign based on mean difference
            if df[cols['AE_opt']].mean() < df[cols['AE_base']].mean():
                eff = -abs(eff)
            else:
                eff = abs(eff)

        results.append({
            'Domain': subset_name,
            'Metric': 'AE',
            'Model': model_name,
            'p_value': p_val,
            'phi_signed': round(eff, 4), # Reporting r as phi_signed equivalent
            'Significant': p_val < 0.05
        })
    except Exception as e:
        results.append({'Domain': subset_name, 'Metric': 'AE', 'Model': model_name, 'p_value': 1.0, 'phi_signed': 0, 'Significant': False})

    # --- 3. AS (Average Steps) - Wilcoxon Signed-Rank ---
    # Logic: Discrete Count.
    # NOTE: You asked for "Regression" to be significant.
    # Usually, Higher Steps = Regression (Worse). 
    try:
        diff = df[cols['AS_opt']] - df[cols['AS_base']]
        diff = diff[diff != 0]
        n = len(diff)
        
        if n < 1:
            p_val = 1.0
            eff = 0.0
        else:
            w_stat, p_val = stats.wilcoxon(df[cols['AS_base']], df[cols['AS_opt']])
            eff = get_effect_size_wilcoxon(w_stat, n)
            
            # Direction: 
            # If Opt > Base (More steps), that is a Regression. 
            # We will use Positive Phi to indicate Increase in Steps.
            if df[cols['AS_opt']].mean() < df[cols['AS_base']].mean():
                # Steps decreased (Improvement)
                eff = -abs(eff)
            else:
                # Steps increased (Regression)
                eff = abs(eff)

        results.append({
            'Domain': subset_name,
            'Metric': 'AS',
            'Model': model_name,
            'p_value': p_val,
            'phi_signed': round(eff, 4),
            'Significant': p_val < 0.05
        })
    except Exception as e:
        results.append({'Domain': subset_name, 'Metric': 'AS', 'Model': model_name, 'p_value': 1.0, 'phi_signed': 0, 'Significant': False})

    return results

# ==========================================
# 3. EXECUTION
# ==========================================

# Load Data
df = pd.read_csv(file_path)

all_results = []

# 1. Iterate over each domain
unique_domains = df[cols['domain']].unique()
for domain in unique_domains:
    domain_df = df[df[cols['domain']] == domain]
    all_results.extend(analyze_subset(domain_df, domain))

# 2. Run for the Whole Dataset
all_results.extend(analyze_subset(df, "OVERALL_DATASET"))

# 3. Create Final DataFrame
final_df = pd.DataFrame(all_results)

# Display
print(final_df)

# Save to CSV if needed
# final_df.to_csv('statistical_results.csv', index=False)