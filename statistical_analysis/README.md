# statistical_analysis

Statistical comparison of reciprocal ranks (RR / MRR) across IRBL techniques and dataset variants.

## Data

`rr-data/<technique>-<variant>-rr.csv`, where variant is:

- `original` – all issues
- `test-cleaned` – non-bug issues removed from the test set
- `fully-cleaned` – non-bug issues removed from train and test sets

## Scripts

| Script | Purpose |
|---|---|
| `create-test-cleaned.py` | Builds `test-cleaned` files from `original` and `fully-cleaned` |
| `intra-technique-mrr-comparison.py` | Original vs. test-cleaned MRR per technique |
| `test-vs-fully-cleaned-mrr.py` | Test-cleaned vs. fully-cleaned MRR (Wilcoxon signed-rank) |
| `inter-technique-comparison.py` | Pairwise technique comparison (Wilcoxon signed-rank, Holm correction) |
| `bug-vs-non-bug-rr.py` | RR of bug vs. non-bug issues (Wilcoxon rank-sum) |
| `trnk.py` | TRNK helper used by the intra-technique comparison |

## Usage

```
python intra-technique-mrr-comparison.py
python test-vs-fully-cleaned-mrr.py
python inter-technique-comparison.py
python bug-vs-non-bug-rr.py
```