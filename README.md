# Mislabeling in IRBL

Studies how mislabeled (non-bug) issues in bug datasets affect Information Retrieval-based Bug Localization (IRBL) techniques.

## Selection of IRBL Techniques

To investigate the impact of mislabeled issues, we first defined selection criteria for choosing appropriate IRBL techniques. Our goal is to ensure that the selected approaches are reproducible, scalable and aligned with the scope of our study. In particular, we considered techniques that (1) have a publicly available and functional implementation, (2) complete the localization of each bug report within 80 seconds, as followed in prior studies, (3) perform file-level localization, and (4) support Java projects, since our benchmarks consist of large-scale Java systems and the majority of IRBL techniques and datasets in the literature are Java-based.

Based on these criteria, we selected six representative IRBL techniques that cover different methodological paradigms, including traditional information retrieval models, heuristic-enhanced approaches, deep learning-based techniques, and recent LLM-driven methods. Collectively, these techniques reflect the evolution of IRBL research. VSM and BugLocator represent foundational models; BLUiR and BRTracer incorporate structural and heuristic enhancements; DreamLoc exemplifies deep learning-based solutions; and GenLoc reflects the emerging paradigm of LLM-based bug localization. This diversity allows us to comprehensively assess how mislabeled issues affect techniques with varying dependencies on textual content, structural information and learning mechanisms. 

Several other state-of-the-art IRBL techniques were excluded because they did not satisfy our selection criteria. For example, DNNLOC and DeepLoc lack publicly available implementations. Although RLocator provides a replication package, it is incomplete due to missing source code. Similarly, AdaptiveBL could not be executed due to unresolved dependency issues, which also affects FLIM, as it builds on AdaptiveBL. Recent LLM-based techniques LLM-BL and BRaIn were also excluded due to scalability issues. Because their publicly available implementations only support sequential execution, processing large benchmark datasets is prohibitively time-consuming. Our preliminary experiments on 300 bug reports from the Ye et al. dataset showed that they far exceed the 80-second per-bug time limit. On average, LLM-BL required 3 min 53 s per bug, and BRaIn required 2 min 59 s. Scaling to the dataset used in our evaluation (9,097 bug reports) yields estimated single-pass runtimes of ≈ 588.78 hours (≈ 24.53 days) for LLM-BL and ≈ 452.32 hours (≈ 18.85 days) for BRaIn. Due to LLM non-determinism, we need at least three runs for stable results, which raises the totals to ≈ 1,766.33 hours (≈ 73.60 days) and ≈ 1,356.97 hours (≈ 56.54 days), respectively, making it infeasible to obtain comprehensive results for fair comparison in our empirical study. Consequently, despite their potential, these approaches were excluded from our study.

## Structure

| Folder | Purpose |
|---|---|
| [gentag/](gentag/README.md) | Classify issues as bug / non-bug with GenTag |
| [accuracy/](accuracy/README.md) | Top-k accuracy of IRBL techniques on original vs. cleaned data |
| [statistical_analysis/](statistical_analysis/README.md) | Statistical tests on reciprocal ranks (MRR) |

## Setup

```
python -m venv env
env\Scripts\activate
pip install -r requirements.txt
```

`gentag/gpt_classifier.py` also needs the `OPENAI_API_KEY` environment variable.
