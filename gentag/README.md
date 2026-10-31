# gentag

Classifies bug reports from IRBL datasets with GPT (`gpt-4o-mini`) to detect mislabeled issues.

## Setup

```
pip install -r requirements.txt
set OPENAI_API_KEY=your-key
```

## Running the classifier

```
python gpt_classifier.py <output_file> <dataset> <mode>
```

| Argument | Values |
|---|---|
| `output_file` | CSV to append results to (e.g. `outputs/ghrb-1.csv`) |
| `dataset` | `herzig`, `yeetal`, `bench4bl`, `zhou`, `ghrb`, `bettlebox` |
| `mode` | `sequential` or `parallel` (7 threads) |

Example:

```
python gpt_classifier.py outputs/ghrb-1.csv ghrb parallel
```

Each output row is `project, id, ground_truth_label, predicted_label, explanation`.

## Layout

- `dataset/` – input datasets, one folder per dataset
- `outputs/` – classifier runs (`<dataset>-1/2/3.csv`) and majority votes (`<dataset>-majority.csv`)
- `majority_vote*.py` – combine the three runs into a majority vote
- `result_process*.py` – compute metrics against ground truth
- `colavito/` – experiments with the Colavito et al. classifier
- `result on Herzig Dataset.txt` – the results of all three GenTag trials, together with the agreement analysis across trials.
- `result on IRBL Benchmarks.txt ` – dataset-specific precision, recall and F1-scores of the manually labeled issues

## Herzig dataset fields

- `label` – our binary classification of the issue
- `type` – the issue category assigned by the issue opener
- `classification` – the category assigned by Herzig et al.
