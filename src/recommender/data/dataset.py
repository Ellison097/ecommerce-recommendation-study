from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

from .download import download_movielens_100k, download_online_retail


@dataclass
class DatasetBundle:
    interactions: pd.DataFrame
    users: pd.DataFrame
    items: pd.DataFrame
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    train_matrix: csr_matrix
    user_encoder: dict
    item_encoder: dict
    user_decoder: dict
    item_decoder: dict
    item_features: np.ndarray
    item_group: dict
    dataset_name: str


def _leave_last_split(df: pd.DataFrame, val_ratio: float, test_ratio: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_parts, val_parts, test_parts = [], [], []
    for _, g in df.groupby('user_id'):
        g = g.sort_values('timestamp')
        n = len(g)
        if n < 3:
            continue
        test_n = max(1, int(round(n * test_ratio)))
        val_n = max(1, int(round(n * val_ratio))) if n >= 5 else 0
        if test_n + val_n >= n:
            test_n = 1
            val_n = 1 if n > 2 else 0
        test_parts.append(g.iloc[-test_n:])
        if val_n > 0:
            val_parts.append(g.iloc[-(test_n + val_n):-test_n])
            train_parts.append(g.iloc[:-(test_n + val_n)])
        else:
            train_parts.append(g.iloc[:-test_n])
    train = pd.concat(train_parts).reset_index(drop=True)
    val = pd.concat(val_parts).reset_index(drop=True) if val_parts else pd.DataFrame(columns=df.columns)
    test = pd.concat(test_parts).reset_index(drop=True)
    return train, val, test


def _finalize_bundle(ratings: pd.DataFrame, users: pd.DataFrame, items: pd.DataFrame, config: dict, dataset_name: str, text_col: str) -> DatasetBundle:
    train, val, test = _leave_last_split(ratings, config['dataset']['val_ratio'], config['dataset']['test_ratio'])

    user_ids = sorted(ratings['user_id'].unique().tolist())
    item_ids = sorted(ratings['item_id'].unique().tolist())
    user_encoder = {u: i for i, u in enumerate(user_ids)}
    item_encoder = {it: i for i, it in enumerate(item_ids)}
    user_decoder = {i: u for u, i in user_encoder.items()}
    item_decoder = {i: it for it, i in item_encoder.items()}

    def encode(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out['user_idx'] = out['user_id'].map(user_encoder)
        out['item_idx'] = out['item_id'].map(item_encoder)
        return out.dropna(subset=['user_idx', 'item_idx']).astype({'user_idx': int, 'item_idx': int})

    train = encode(train)
    val = encode(val)
    test = encode(test)

    train_matrix = csr_matrix(
        (np.ones(len(train)), (train['user_idx'], train['item_idx'])),
        shape=(len(user_ids), len(item_ids))
    )

    item_text = items.set_index('item_id').reindex(item_ids)[text_col].fillna('')
    tfidf = TfidfVectorizer(stop_words='english', max_features=800)
    item_features = tfidf.fit_transform(item_text).toarray()

    popularity = train.groupby('item_id').size().sort_values()
    threshold = popularity.quantile(config['reranking'].get('protected_group_quantile', 0.5))
    item_group = {item_id: ('long_tail' if popularity.get(item_id, 0) <= threshold else 'head') for item_id in item_ids}

    return DatasetBundle(
        interactions=ratings,
        users=users,
        items=items,
        train=train,
        val=val,
        test=test,
        train_matrix=train_matrix,
        user_encoder=user_encoder,
        item_encoder=item_encoder,
        user_decoder=user_decoder,
        item_decoder=item_decoder,
        item_features=item_features,
        item_group=item_group,
        dataset_name=dataset_name,
    )


def _load_movielens(config: dict) -> DatasetBundle:
    raw_dir = Path(config['dataset']['raw_dir'])
    ml_dir = download_movielens_100k(raw_dir)
    ratings = pd.read_csv(ml_dir / 'u.data', sep='\t', names=['user_id', 'item_id', 'rating', 'timestamp'], engine='python')
    items = pd.read_csv(
        ml_dir / 'u.item', sep='|', encoding='latin-1', header=None,
        names=['item_id', 'title', 'release_date', 'video_release_date', 'imdb_url'] + [f'genre_{i}' for i in range(19)]
    )
    users = pd.read_csv(ml_dir / 'u.user', sep='|', names=['user_id', 'age', 'gender', 'occupation', 'zip_code'])
    min_u = config['dataset']['min_user_interactions']
    min_i = config['dataset']['min_item_interactions']
    ratings = ratings.groupby('user_id').filter(lambda x: len(x) >= min_u)
    ratings = ratings.groupby('item_id').filter(lambda x: len(x) >= min_i)
    return _finalize_bundle(ratings, users, items[['item_id', 'title']].rename(columns={'title': 'description'}), config, 'MovieLens 100K', 'description')


def _load_online_retail(config: dict) -> DatasetBundle:
    raw_dir = Path(config['dataset']['raw_dir'])
    xlsx_path = download_online_retail(raw_dir)
    df = pd.read_excel(xlsx_path)
    df = df.rename(columns={
        'CustomerID': 'user_id',
        'StockCode': 'item_id',
        'Description': 'description',
        'InvoiceDate': 'timestamp',
        'Quantity': 'quantity',
        'Country': 'country',
        'UnitPrice': 'unit_price',
        'InvoiceNo': 'invoice_no',
    })
    df = df.dropna(subset=['user_id', 'item_id', 'description', 'timestamp'])
    df = df[~df['invoice_no'].astype(str).str.startswith('C')]
    df = df[(df['quantity'] > 0) & (df['unit_price'] > 0)]
    df['user_id'] = df['user_id'].astype(int)
    df['item_id'] = df['item_id'].astype(str)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['rating'] = np.log1p(df['quantity'])
    df = df[['user_id', 'item_id', 'rating', 'timestamp', 'description', 'country', 'quantity', 'unit_price']].sort_values('timestamp')

    # aggregate repeated purchases at same timestamp granularity to stabilize interactions
    ratings = (df.groupby(['user_id', 'item_id', 'timestamp'], as_index=False)
                 .agg({'rating': 'sum', 'description': 'first', 'country': 'first', 'quantity': 'sum', 'unit_price': 'mean'}))

    min_u = config['dataset']['min_user_interactions']
    min_i = config['dataset']['min_item_interactions']
    ratings = ratings.groupby('user_id').filter(lambda x: len(x) >= min_u)
    ratings = ratings.groupby('item_id').filter(lambda x: len(x) >= min_i)

    users = ratings.groupby('user_id', as_index=False).agg(country=('country', 'first'), interactions=('item_id', 'count'))
    items = ratings.groupby('item_id', as_index=False).agg(description=('description', 'first'), purchases=('quantity', 'sum'))
    return _finalize_bundle(ratings[['user_id', 'item_id', 'rating', 'timestamp']], users, items, config, 'Online Retail (UCI E-commerce)', 'description')


def load_dataset(config: dict) -> DatasetBundle:
    name = config['dataset']['name']
    if name == 'movielens_100k':
        return _load_movielens(config)
    if name == 'online_retail':
        return _load_online_retail(config)
    raise ValueError(f'Unsupported dataset: {name}')
