# A Systematic Study of Recommendation Algorithms for E-commerce Platforms

**DATS 5990 – Spring 2026 | University of Pennsylvania**  
**Author:** Licheng Guo

**Code repository:** [https://github.com/Ellison097/ecommerce-recommendation-study](https://github.com/Ellison097/ecommerce-recommendation-study)

## Overview

A systematic comparison of **eight recommendation algorithms** across five paradigms on a **public 5-core implicit-feedback benchmark** (product-review interactions, single category).

| # | Model | Paradigm | Reference |
|---|-------|----------|-----------|
| 1 | Popularity | Heuristic baseline | — |
| 2 | ItemKNN | Neighbourhood CF | Sarwar et al. 2001 |
| 3 | BPR-MF | Matrix Factorisation | Rendle et al. UAI 2009 |
| 4 | NeuMF | Neural CF | He et al. WWW 2017 |
| 5 | LightGCN | Graph Neural Network | He et al. SIGIR 2020 |
| 6 | Multi-VAE | Variational Autoencoder | Liang et al. WWW 2018 |
| 7 | SASRec | Transformer / Sequential | Kang & McAuley ICDM 2018 |
| 8 | Ensemble | Weighted Blending | — |

## Quick Start

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python run_experiment.py
```

The data loader downloads cached benchmark ratings when missing, or falls back to synthetic data if offline.

## Project Structure

```
├── configs/config.yaml          # Hyper-parameters
├── data/ratings/                # Cached ratings CSV (or auto-download)
├── src/
│   ├── data_loader.py
│   ├── evaluator.py
│   ├── visualizer.py
│   └── models/                  # Eight models + ensemble
├── run_experiment.py
└── results/
    ├── figures/
    └── metrics/
```

## Evaluation Protocol

- **Split:** Leave-one-out (last interaction per user → test)
- **Ranking:** 1 positive + 99 sampled negatives
- **Metrics:** HR@K, NDCG@K, MRR@K (K ∈ {5, 10, 20})
- **Beyond-accuracy:** Catalogue Coverage, Novelty, Gini Coefficient

## Reports and poster (local only)

Practicum PDFs and poster PPT are generated with a **local script** not included in this repository. If you have `generate_report.py` in your private copy, run `python generate_report.py` after `run_experiment.py` to refresh outputs: `results/DATS5990_Report_Licheng_Guo.pdf` (final), `results/DATS5990_Interim_Report_Licheng_Guo.pdf` (Interim Report), and the poster PPTX.
