"""
Precompute product-level signals from historical order data.

Run once before using the inventory proxy:
    python scripts/build_product_signals.py

Output: models/product_signals.json
"""

import json
import argparse
import pandas as pd
from pathlib import Path


def build_product_signals(data_dir: str, output_dir: str) -> None:
    prior_path = Path(data_dir) / 'order_products__prior.csv'
    print(f"Loading {prior_path} ...")
    prior = pd.read_csv(prior_path)

    # Total purchase count per product
    popularity = prior.groupby('product_id')['order_id'].count().rename('count')

    # Fraction of purchases that were reorders
    reorder_rate = prior.groupby('product_id')['reordered'].mean().rename('reorder_rate')

    signals = pd.concat([popularity, reorder_rate], axis=1).fillna(0.0)

    # Percentile rank: 0.0 = least popular, 1.0 = most popular
    signals['popularity_pct'] = signals['count'].rank(pct=True)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_path = Path(output_dir) / 'product_signals.json'

    out = {
        str(pid): {
            'popularity_pct': round(float(row['popularity_pct']), 6),
            'reorder_rate':   round(float(row['reorder_rate']),   6),
        }
        for pid, row in signals.iterrows()
    }

    with open(output_path, 'w') as f:
        json.dump(out, f, indent=2)

    print(f"Saved signals for {len(out):,} products -> {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build product signals for inventory proxy')
    parser.add_argument('--data_dir',   default='./data',   help='Path to data directory')
    parser.add_argument('--output_dir', default='./models', help='Path to output directory')
    args = parser.parse_args()
    build_product_signals(args.data_dir, args.output_dir)
