import csv
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "rr-data"


def read_csv(path: Path) -> list[list[str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.reader(handle))


def main() -> None:
    for cleaned_path in sorted(DATA_DIR.glob("*-fully-cleaned-rr*.csv")):
        original_path = Path(
            str(cleaned_path).replace("-fully-cleaned-", "-original-")
        )
        target_path = Path(
            str(cleaned_path).replace("-fully-cleaned-", "-test-cleaned-")
        )

        if not original_path.exists():
            raise FileNotFoundError(
                f"Matching original file is missing: {original_path.name}"
            )
        if target_path.exists():
            raise FileExistsError(
                f"Refusing to overwrite existing output: {target_path.name}"
            )

        cleaned_rows = read_csv(cleaned_path)
        original_rows = read_csv(original_path)

        if not cleaned_rows or not original_rows:
            raise ValueError(f"Input file is empty: {cleaned_path.name}")
        if cleaned_rows[0] != original_rows[0]:
            raise ValueError(f"Headers differ for: {cleaned_path.name}")

        original_values = {row[0]: row[1:] for row in original_rows[1:]}
        cleaned_ids = [row[0] for row in cleaned_rows[1:]]

        if len(cleaned_ids) != len(set(cleaned_ids)):
            raise ValueError(f"Duplicate Bug ID in: {cleaned_path.name}")
        if len(original_values) != len(original_rows) - 1:
            raise ValueError(f"Duplicate Bug ID in: {original_path.name}")

        missing_ids = set(cleaned_ids) - original_values.keys()
        if missing_ids:
            raise ValueError(
                f"{cleaned_path.name} has {len(missing_ids)} IDs missing from "
                f"{original_path.name}"
            )

        output_rows = [cleaned_rows[0]] + [
            [bug_id, *original_values[bug_id]] for bug_id in cleaned_ids
        ]
        with target_path.open("w", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerows(output_rows)

        print(f"Created {target_path.name}")


if __name__ == "__main__":
    main()