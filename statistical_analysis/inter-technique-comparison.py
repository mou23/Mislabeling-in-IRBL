from collections import defaultdict
from pathlib import Path
from scipy import stats
import numpy as np
from math import sqrt
import pandas as pd
from scipy.stats import rankdata

DATA_DIR = Path(__file__).resolve().parent / "rr-data"

def effect_size_interpretation(effect_size):
    effect_size = abs(effect_size)
    if effect_size < 0.1:
        return "trivial"
    elif effect_size < 0.3:
        return "small"
    elif effect_size < 0.5:
        return "medium"
    else:
        return "large"


def get_paired_values(df1, df2):
    # Align paired observations by ID so each pair belongs to the same bug report.
    p1 = df1.iloc[:, [0, 1]].copy()
    p1.columns = ['id', 'value1']
    p2 = df2.iloc[:, [0, 1]].copy()
    p2.columns = ['id', 'value2']
    paired = p1.merge(p2, on='id', how='inner')
    if paired.empty:
        raise ValueError("No common IDs between the two inputs")
    return paired['value1'].to_numpy(dtype=float), paired['value2'].to_numpy(dtype=float)


def holm_correction(p_values):
    p_values = np.asarray(p_values, dtype=float)
    m = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.empty(m)
    running_max = 0.0
    for rank, idx in enumerate(order):
        running_max = max(running_max, (m - rank) * p_values[idx])
        adjusted[idx] = min(running_max, 1.0)
    return adjusted


def calculate_statistic(arr1, arr2):
    # Step 1: Compute differences and remove zero-difference pairs
    differences = arr2 - arr1
    abs_differences = np.abs(differences)
    non_zero_mask = abs_differences != 0
    non_zero_count = np.sum(non_zero_mask)
    filtered_diffs = differences[non_zero_mask]
    filtered_abs_diffs = abs_differences[non_zero_mask]

    # Step 2: Rank the absolute differences
    ranks = rankdata(filtered_abs_diffs)
    signed_ranks = np.sign(filtered_diffs) * ranks

    # Step 3: Perform Wilcoxon signed-rank test
    statistic, p_value = stats.wilcoxon(arr1, arr2, zero_method='wilcox', correction=False)
    
    # Step 4a: Simple Rank-Biserial Correlation (Kerby, 2014)
    W_pos = np.sum(signed_ranks[signed_ranks > 0])
    W_neg = np.sum(np.abs(signed_ranks[signed_ranks < 0]))
    r_simple = (W_pos - W_neg) / (W_pos + W_neg)

    # Step 4b: Z-based effect size
    n_effect = len(filtered_diffs)
    expected_W = n_effect * (n_effect + 1) / 4
    std_W = sqrt(n_effect * (n_effect + 1) * (2 * n_effect + 1) / 24)
    Z = (statistic - expected_W) / std_W
    r_z_based = Z / sqrt(n_effect)

    # Step 4c: Rank-Biserial using T-formula (with total sample size and abs)
    n_total = len(arr1)  # includes zero-difference pair
    numerator_t = abs(4 * (min(W_pos, W_neg) - (W_pos + W_neg) / 2))
    denominator_t = n_total * (n_total + 1)
    r_t_formula = numerator_t / denominator_t
    
    return statistic, p_value, r_simple, r_z_based, r_t_formula, non_zero_count


if __name__ == "__main__":
    types = ["original", "test-cleaned"]
    metrics = ["rr"]
    techniques = ["vsm", "buglocator", "bluir", "brtracer", "dreamloc", "genloc"]

    print("\n" + "="*80)
    print("WILCOXON SIGNED-RANK TEST RESULTS")
    print("="*80)

    # Create separate tables for each type-metric combination
    for metric in metrics:
        for type_ in types:
            print(f"\n{type_.upper()} - {metric.upper()} COMPARISONS")
            results = []
            
            # Create all possible pairs of techniques
            for i, tech1 in enumerate(techniques):
                for tech2 in techniques[i+1:]:  # Only compare with techniques after current one
                    file1 = DATA_DIR / f"{tech1}-{type_}-{metric}.csv"
                    file2 = DATA_DIR / f"{tech2}-{type_}-{metric}.csv"
                    
                    try:
                        df1 = pd.read_csv(file1)
                        df2 = pd.read_csv(file2)

                        arr1, arr2 = get_paired_values(df1, df2)
                        statistic, p_value, r_simple, r_z_based, r_t_formula, non_zero_count = calculate_statistic(arr1, arr2)

                        results.append({
                            'Technique 1': tech1.upper(),
                            'Technique 2': tech2.upper(),
                            'Paired bugs': len(arr1),
                            'Statistic': f"{statistic:.3f}",
                            'p-value': p_value,
                            'r_simple': f"{r_simple:.3f}",
                            'r_simple_interpretation': effect_size_interpretation(r_simple),
                            # 'r_z_based': f"{r_z_based:.3f}",
                            # 'r_z_based_interpretation': effect_size_interpretation(r_z_based),
                            # 'r_t_formula': f"{r_t_formula:.3f}",
                            # 'r_t_formula_interpretation': effect_size_interpretation(r_t_formula),
                            # 'non_zero_count': non_zero_count,
                        })
                    except FileNotFoundError as e:
                        print(f"Warning: Could not find file for comparison {tech1} vs {tech2}: {e}")
                        continue

            # Convert results to DataFrame and print as table
            results_df = pd.DataFrame(results)
            if not results_df.empty:
                # Holm correction applied within each type (family of pairwise comparisons)
                p_holm = holm_correction(results_df['p-value'])
                results_df.insert(results_df.columns.get_loc('p-value') + 1, 'p-holm', [f"{p:.3f}" for p in p_holm])
                results_df.insert(results_df.columns.get_loc('p-holm') + 1, 'Significant', ['Yes' if p < 0.05 else 'No' for p in p_holm])
                results_df = results_df.assign(**{'p-value': results_df['p-value'].map(lambda p: f"{p:.3f}")})
                print(f"\nWilcoxon Signed-Rank Test Results for {type_.upper()} - {metric.upper()}:")
                print("="*100)
                print(results_df.to_string(index=False))
                print("="*100)
            else:
                print(f"\nNo valid comparisons found for {type_.upper()} - {metric.upper()}")
