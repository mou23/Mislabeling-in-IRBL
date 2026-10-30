# Mislabeling in IRBL

Studies how mislabeled (non-bug) issues in bug datasets affect Information Retrieval-based Bug Localization (IRBL) techniques.

## Structure

| Folder | Purpose |
|---|---|
| [gentag/](gentag/README.md) | Classify issues as bug / non-bug with GPT |
| [accuracy/](accuracy/README.md) | Top-k accuracy of IRBL techniques on original vs. cleaned data |
| [statistical_analysis/](statistical_analysis/README.md) | Statistical tests on reciprocal ranks (MRR) |

## Setup

```
python -m venv env
env\Scripts\activate
pip install -r requirements.txt
```

`gentag/gpt_classifier.py` also needs the `OPENAI_API_KEY` environment variable.
