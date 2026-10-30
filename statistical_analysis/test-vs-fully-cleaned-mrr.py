from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata

TECHNIQUES = ("buglocator", "brtracer", "dreamloc")
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


def wilcoxon_signed_rank(values1, values2):
    differences = values2 - values1
    differences = differences[differences != 0]
    n = differences.size
    if n == 0:
        raise ValueError("All paired differences are zero; Wilcoxon test is undefined")

    ranks = rankdata(np.abs(differences))
    positive_rank_sum = ranks[differences > 0].sum()
    negative_rank_sum = ranks[differences < 0].sum()
    statistic = min(positive_rank_sum, negative_rank_sum)

    expected = n * (n + 1) / 4
    _, tie_counts = np.unique(np.abs(differences), return_counts=True)
    tie_correction = (tie_counts**3 - tie_counts).sum() / 48
    variance = n * (n + 1) * (2 * n + 1) / 24 - tie_correction
    z = (statistic - expected) / np.sqrt(variance)
    p_value = 2 * norm.sf(abs(z))

    # Simple rank-biserial correlation (Kerby, 2014).
    r_simple = (positive_rank_sum - negative_rank_sum) / (
        positive_rank_sum + negative_rank_sum
    )
    return statistic, z, p_value, r_simple, n


def load_paired(technique):
    test_cleaned_path = DATA_DIR / f"{technique}-test-cleaned-rr.csv"
    fully_cleaned_path = DATA_DIR / f"{technique}-fully-cleaned-rr.csv"
    missing_files = [
        path.name
        for path in (test_cleaned_path, fully_cleaned_path)
        if not path.exists()
    ]
    if missing_files:
        raise FileNotFoundError(f"Missing input file(s): {', '.join(missing_files)}")

    test_cleaned = pd.read_csv(test_cleaned_path).iloc[:, [0, 1]]
    test_cleaned.columns = ["id", "test_cleaned_rr"]
    fully_cleaned = pd.read_csv(fully_cleaned_path).iloc[:, [0, 1]]
    fully_cleaned.columns = ["id", "fully_cleaned_rr"]

    paired = test_cleaned.merge(fully_cleaned, on="id", how="inner")
    if paired.empty:
        raise ValueError(f"No common Bug IDs for {technique}")
    return paired


def main():
    results = []
    for technique in TECHNIQUES:
        paired = load_paired(technique)
        test_cleaned_rr = paired["test_cleaned_rr"].to_numpy(dtype=float)
        fully_cleaned_rr = paired["fully_cleaned_rr"].to_numpy(dtype=float)

        test_cleaned_mrr = round(test_cleaned_rr.mean(), 3)
        fully_cleaned_mrr = round(fully_cleaned_rr.mean(), 3)
        difference = ((fully_cleaned_mrr - test_cleaned_mrr) / test_cleaned_mrr) * 100
        statistic, z, p_value, r_simple, non_zero_count = wilcoxon_signed_rank(
            test_cleaned_rr, fully_cleaned_rr
        )

        results.append(
            {
                "Technique": technique.upper(),
                "Paired bugs": len(paired),
                "Test-cleaned MRR": f"{test_cleaned_mrr:.3f}",
                "Fully-cleaned MRR": f"{fully_cleaned_mrr:.3f}",
                "Diff (%)": f"{difference:.2f}%",
                "Statistic": f"{statistic:.1f}",
                "z": f"{z:.3f}",
                "p-value": f"{p_value:.4f}",
                "Significant": "Yes" if p_value < 0.05 else "No",
                "r_simple": f"{r_simple:.3f}",
                "r_simple_interpretation": effect_size_interpretation(r_simple),
                "Non-zero pairs": non_zero_count,
            }
        )

    print("\n" + "=" * 80)
    print("TEST-CLEANED vs FULLY-CLEANED (MRR, WILCOXON SIGNED-RANK)")
    print("=" * 80)
    print(pd.DataFrame(results).to_string(index=False))
    print("=" * 80)


if __name__ == "__main__":
    main()
