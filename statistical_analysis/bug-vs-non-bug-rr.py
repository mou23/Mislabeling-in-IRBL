from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata

TECHNIQUES = ("vsm", "buglocator", "bluir", "brtracer", "dreamloc", "genloc")
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


def wilcoxon_rank_sum(bug_rr, non_bug_rr):
    n_bug = bug_rr.size
    n_non_bug = non_bug_rr.size
    if n_bug == 0 or n_non_bug == 0:
        raise ValueError("Both groups must contain at least one observation")

    combined = np.concatenate([bug_rr, non_bug_rr])
    n_total = combined.size
    ranks = rankdata(combined)
    rank_sum_bug = ranks[:n_bug].sum()

    expected = n_bug * (n_total + 1) / 2
    _, tie_counts = np.unique(combined, return_counts=True)
    tie_term = (tie_counts**3 - tie_counts).sum() / (n_total * (n_total - 1))
    variance = n_bug * n_non_bug / 12 * ((n_total + 1) - tie_term)
    if variance == 0:
        raise ValueError("All RR values are identical; rank-sum test is undefined")
    z = (rank_sum_bug - expected) / np.sqrt(variance)
    p_value = 2 * norm.sf(abs(z))

    # Rank-biserial correlation: positive means bugs tend to have higher RR.
    u_bug = rank_sum_bug - n_bug * (n_bug + 1) / 2
    r_rank_biserial = 2 * u_bug / (n_bug * n_non_bug) - 1
    return rank_sum_bug, u_bug, z, p_value, r_rank_biserial


def load_groups(technique):
    original_path = DATA_DIR / f"{technique}-original-rr.csv"
    test_cleaned_path = DATA_DIR / f"{technique}-test-cleaned-rr.csv"
    missing_files = [
        path.name for path in (original_path, test_cleaned_path) if not path.exists()
    ]
    if missing_files:
        raise FileNotFoundError(f"Missing input file(s): {', '.join(missing_files)}")

    original = pd.read_csv(original_path).iloc[:, [0, 1]]
    original.columns = ["id", "rr"]
    bug_ids = pd.read_csv(test_cleaned_path).iloc[:, 0]

    if original["id"].duplicated().any():
        raise ValueError(f"Duplicate Bug ID in {original_path.name}")
    if bug_ids.duplicated().any():
        raise ValueError(f"Duplicate Bug ID in {test_cleaned_path.name}")
    missing_ids = set(bug_ids) - set(original["id"])
    if missing_ids:
        raise ValueError(
            f"{test_cleaned_path.name} has {len(missing_ids)} IDs missing from "
            f"{original_path.name}"
        )

    is_bug = original["id"].isin(bug_ids)
    bug_rr = original.loc[is_bug, "rr"].to_numpy(dtype=float)
    non_bug_rr = original.loc[~is_bug, "rr"].to_numpy(dtype=float)
    return bug_rr, non_bug_rr


def main():
    results = []
    for technique in TECHNIQUES:
        bug_rr, non_bug_rr = load_groups(technique)
        rank_sum_bug, u_bug, z, p_value, r_rank_biserial = wilcoxon_rank_sum(
            bug_rr, non_bug_rr
        )

        results.append(
            {
                "Technique": technique.upper(),
                "Bugs": bug_rr.size,
                "Non-bugs": non_bug_rr.size,
                "Bug MRR": f"{bug_rr.mean():.3f}",
                "Non-bug MRR": f"{non_bug_rr.mean():.3f}",
                "W (bugs)": f"{rank_sum_bug:.1f}",
                "U (bugs)": f"{u_bug:.1f}",
                "z": f"{z:.3f}",
                "p-value": f"{p_value:.4f}",
                "Significant": "Yes" if p_value < 0.05 else "No",
                "r_rank_biserial": f"{r_rank_biserial:.3f}",
                "r_interpretation": effect_size_interpretation(r_rank_biserial),
            }
        )

    print("\n" + "=" * 80)
    print("BUGS vs NON-BUGS (RR, WILCOXON RANK-SUM)")
    print("=" * 80)
    print(pd.DataFrame(results).to_string(index=False))
    print("=" * 80)


if __name__ == "__main__":
    main()
