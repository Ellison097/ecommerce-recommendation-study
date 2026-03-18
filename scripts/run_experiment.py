from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from recommender.data.dataset import load_dataset
from recommender.models import (
    PopularityRecommender,
    UserCFRecommender,
    ItemCFRecommender,
    MatrixFactorizationRecommender,
    NCFRecommender,
    HybridRecommender,
)
from recommender.evaluation import evaluate_model
from recommender.utils.config import load_config


MODEL_REGISTRY = {
    'popularity': lambda cfg: PopularityRecommender(),
    'user_cf': lambda cfg: UserCFRecommender(),
    'item_cf': lambda cfg: ItemCFRecommender(),
    'matrix_factorization': lambda cfg: MatrixFactorizationRecommender(n_factors=cfg['training']['n_factors']),
    'ncf': lambda cfg: NCFRecommender(
        n_epochs=cfg['training']['n_epochs'],
        batch_size=cfg['training']['batch_size'],
        learning_rate=cfg['training']['learning_rate'],
        weight_decay=cfg['training']['weight_decay'],
        negative_samples=cfg['training']['negative_samples'],
        device=cfg['training']['device'],
    ),
    'hybrid': lambda cfg: HybridRecommender(n_factors=cfg['training']['n_factors']),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/default.yaml')
    args = parser.parse_args()

    config = load_config(ROOT / args.config)
    dataset = load_dataset(config)

    results = []
    for model_name in config['models']['run']:
        model = MODEL_REGISTRY[model_name](config)
        model.fit(dataset)
        scores = evaluate_model(model, dataset, config)
        scores['model'] = model_name
        results.append(scores)
        print(f'Finished {model_name}: {scores}')

    df = pd.DataFrame(results)
    out_path = ROOT / 'results' / 'metrics.csv'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f'\nSaved metrics to {out_path}')
    print(df.sort_values('ndcg@k', ascending=False).to_string(index=False))


if __name__ == '__main__':
    main()
