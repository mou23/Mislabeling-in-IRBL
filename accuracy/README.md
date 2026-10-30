# accuracy

Computes Top-k accuracy (k = 1, 5, 10) of IRBL techniques before and after removing non-bug issues.

## Contents

- `<technique>/` – localized bug IDs per project for `bluir`, `brtracer`, `buglocator`, `dreamloc`, `genloc`, `vsm`
  - `localized_bugs_<project>.csv` – original data
  - `localized_bugs_<project>_clean.csv` – cleaned data
- `merger.py` – joins GPT predictions, train/test split and localization results into `merged_dataset.csv`
- `accuracy_calculator.py` – reports original, test-cleaned and fully-cleaned accuracy per technique

## Usage

```
python merger.py
python accuracy_calculator.py
```
