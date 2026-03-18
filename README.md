# DATS5990 Recommender Systems Project

This project implements a reproducible recommendation-system pipeline aligned with the proposal **"A Systematic Study of Recommendation Algorithms for E-commerce Platforms"**.

## What is included

### Phase I – Baseline Modeling
- Popularity-based recommendation
- User-based Collaborative Filtering
- Item-based Collaborative Filtering
- Matrix Factorization (Truncated SVD)

### Phase II – Advanced Modeling
- Neural Collaborative Filtering (PyTorch)

### Phase III – Hybrid + Fairness-aware Optimization
- Hybrid recommender combining matrix factorization and TF-IDF content similarity
- Diversity/fairness-aware re-ranking for long-tail exposure balancing

### Evaluation Metrics
- Precision@K
- Recall@K
- NDCG@K
- MAP@K
- Coverage
- Intra-list Diversity
- Fairness Gap (head vs long-tail exposure)

## Dataset
By default, the pipeline downloads **MovieLens 100K** automatically. It is used as a proxy benchmark dataset so the project is immediately runnable and reproducible. You can later replace it with a real e-commerce interaction dataset.

## Quick Start

```bash
cd recommender_project
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/run_experiment.py
```

Results are saved to:

- `results/metrics.csv`

## Suggested report framing
If you use this for your independent study write-up, position MovieLens as a **public benchmark for offline experimentation**, then discuss how the framework transfers to e-commerce scenarios involving sparse interactions, cold start, popularity bias, and fairness constraints.

## Project Structure

```text
recommender_project/
  configs/default.yaml
  data/
  results/
  scripts/run_experiment.py
  src/recommender/
    data/
    evaluation/
    models/
    utils/
```

## Next extensions you can add
- Wide & Deep model
- GRU4Rec / Transformer sequential recommendation
- Hyperparameter search
- Real e-commerce metadata and user features
- Online metric simulation (CTR / conversion proxy)

## Notes
- The current version is designed to be understandable, runnable, and easy to defend in a course setting.
- If you want, the next step is to add notebooks, visualizations, and a final report template.
