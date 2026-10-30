from pathlib import Path

import pandas as pd
from scipy import stats

from trnk import trnk1, trnk2

DATA_DIR = Path(__file__).resolve().parent / "rr-data"

TECHNIQUES = (
    ("VSM", ("vsm-original-rr.csv",)),
    ("BugLocator", ("buglocator-original-rr.csv",)),
    ("BLUIr", ("bluir-original-rr.csv",)),
    ("BRTracer", ("brtracer-original-rr.csv",)),
    ("DreamLoc", ("dreamloc-original-rr.csv",)),
    ("GenLoc", ("genloc-original-rr.csv",)),
)


def format_table(rows):
    headers = (
        ("Technique", ""),
        ("Original", "MRR"),
        ("Test-Cleaned", "MRR"),
        ("Diff (%)", ""),
        ("p-value", ""),
        ("Effect size", ""),
        ("Levene", "p / variance"),
        ("TRNK", "method"),
        ("Anderson-Darling", "O / C normality"),
    )
    formatted_rows = [
        (
            technique,
            f"{original_mrr:.3f}",
            f"{test_cleaned_mrr:.3f}",
            f"{difference:.2f}%",
            f"{p_value:.2f}",
            f"{effect_size:.2f}",
            levene_result,
            trnk_method,
            anderson_darling_result,
        )
        for (
            technique,
            original_mrr,
            test_cleaned_mrr,
            difference,
            p_value,
            effect_size,
            levene_result,
            trnk_method,
            anderson_darling_result,
        ) in rows
    ]
    widths = [
        max(*(len(header_line) for header_line in header), *(len(row[index]) for row in formatted_rows))
        for index, header in enumerate(headers)
    ]
    top_border = "┌" + "┬".join("─" * (width + 2) for width in widths) + "┐"
    middle_border = "├" + "┼".join("─" * (width + 2) for width in widths) + "┤"
    bottom_border = "└" + "┴".join("─" * (width + 2) for width in widths) + "┘"

    lines = [top_border]
    for header_line in range(2):
        lines.append(
            "│ "
            + " │ ".join(
                header[header_line].center(widths[index])
                for index, header in enumerate(headers)
            )
            + " │"
        )
    lines.append(middle_border)
    lines.extend(
        "│ "
        + " │ ".join(
            value.ljust(widths[index]) if index == 0 else value.center(widths[index])
            for index, value in enumerate(row)
        )
        + " │"
        for row in formatted_rows
    )
    lines.append(bottom_border)
    return "\n".join(lines)


def anderson_darling_result(values):
    result = stats.anderson(values, method="interpolate")
    return "Non-normal" if result.pvalue < 0.05 else "Normal"


def load_pair(original_filename):
    original_path = DATA_DIR / original_filename
    test_cleaned_path = DATA_DIR / original_filename.replace("-original-", "-test-cleaned-")
    missing_files = [
        path.name for path in (original_path, test_cleaned_path) if not path.exists()
    ]
    if missing_files:
        raise FileNotFoundError(f"Missing input file(s): {', '.join(missing_files)}")

    original = pd.read_csv(original_path)
    test_cleaned = pd.read_csv(test_cleaned_path)
    if list(original.columns) != list(test_cleaned.columns):
        raise ValueError(f"Input columns differ for {original_filename}")
    if original.shape[1] < 2:
        raise ValueError(f"Expected an ID column and MRR column for {original_filename}")
    return original, test_cleaned


def load_trials(file_stems):
    original_frames = []
    test_cleaned_frames = []
    for trial_number, original_filename in enumerate(file_stems, start=1):
        original, test_cleaned = load_pair(original_filename)
        original = original.copy()
        test_cleaned = test_cleaned.copy()

        # Distinguish repeated Bug IDs so each GenLoc trial remains a separate observation.
        original.iloc[:, 0] = original.iloc[:, 0].astype(str) + f"-trial-{trial_number}"
        test_cleaned.iloc[:, 0] = (
            test_cleaned.iloc[:, 0].astype(str) + f"-trial-{trial_number}"
        )
        original_frames.append(original)
        test_cleaned_frames.append(test_cleaned)

    return (
        pd.concat(original_frames, ignore_index=True),
        pd.concat(test_cleaned_frames, ignore_index=True),
    )


def main():
    rows = []
    for technique_name, file_stems in TECHNIQUES:
        original, test_cleaned = load_trials(file_stems)
        common_ids = list(set(original.iloc[:, 0]).intersection(test_cleaned.iloc[:, 0]))
        _, variance_p_value = stats.levene(
            original.iloc[:, 1], test_cleaned.iloc[:, 1], center="median"
        )

        if round(variance_p_value, 4) < 0.05:
            _, p_value, _, effect_size = trnk2(original, test_cleaned, common_ids)
            trnk_method = "TRNK2"
        else:
            _, p_value, _, effect_size = trnk1(original, test_cleaned, common_ids)
            trnk_method = "TRNK1"

        original_mrr = round(original.iloc[:, 1].mean(), 3)
        test_cleaned_mrr = round(test_cleaned.iloc[:, 1].mean(), 3)
        difference = (test_cleaned_mrr - original_mrr) / original_mrr * 100
        variance_result = "unequal" if variance_p_value <= 0.05 else "equal"
        levene_result = f"{variance_p_value:.4f} / {variance_result}"
        anderson_darling = (
            f"O: {anderson_darling_result(original.iloc[:, 1])}; "
            f"C: {anderson_darling_result(test_cleaned.iloc[:, 1])}"
        )
        rows.append(
            (
                technique_name,
                original_mrr,
                test_cleaned_mrr,
                difference,
                p_value,
                effect_size,
                levene_result,
                trnk_method,
                anderson_darling,
            )
        )

    table = format_table(rows)
    table_width = len(table.splitlines()[0])
    print("TABLE V".center(table_width))
    print("INTRA TECHNIQUE COMPARISON USING MRR".center(table_width))
    print(table)


if __name__ == "__main__":
    main()