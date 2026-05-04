"""
Compute per-subject z-scores for all base features and save to parquet.
This is a one-time precomputation that the main experiment scripts can reuse.
"""

import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))
from config import TARGETS, DATA_PROCESSED

META_COLS = {"subject_id", "lifelog_date", "sleep_date", "date"}
TARGET_COLS = TARGETS


def get_feature_cols(df):
    return [c for c in df.columns
            if c not in META_COLS | set(TARGET_COLS)
            and df[c].dtype in [np.float64, np.int64, float, int, bool, np.bool_]]


def main():
    log.info("Computing z-score personalization for all features...")

    feat = pd.read_parquet(DATA_PROCESSED / "features_v11.parquet")
    log.info(f"Loaded: {feat.shape}")

    base_cols = get_feature_cols(feat)
    log.info(f"Base features: {len(base_cols)}")

    # Compute z-score stats
    stats = {}
    for col in base_cols:
        col_filled = feat[col].fillna(0)
        subj_stats = col_filled.groupby(feat['subject_id']).agg(['mean', 'std'])
        subj_stats.columns = [f'{col}_subj_mean', f'{col}_subj_std']
        stats[col] = subj_stats

    log.info("Computing z-scores...")

    # Compute z-scores in batches (not merging, just computing from stats)
    zscore_cols = []
    for col in base_cols:
        ss = stats[col]
        if ss[f'{col}_subj_std'].min() == 0:
            # All zero std — skip
            log.debug(f"  Skipping {col} (zero std)")
            continue

        # Join stats to full data and compute z-score
        merged = feat.merge(ss, on='subject_id', how='left')
        mask_std_zero = merged[f'{col}_subj_std'] == 0
        mask_null = feat[col].isnull()
        zscore = np.where(
            mask_std_zero | mask_null, 0.0,
            (merged[col] - merged[f'{col}_subj_mean']) / merged[f'{col}_subj_std']
        )
        zscore_cols.append(pd.Series(zscore, name=f'{col}_zscore', index=feat.index))

    # Build zscore dataframe
    zscore_df = pd.concat(zscore_cols, axis=1)
    log.info(f"Z-score columns: {zscore_df.shape}")

    # Merge with original
    feat_perm = pd.concat([feat, zscore_df], axis=1)
    log.info(f"Personalized: {feat_perm.shape}")

    # Save
    out_path = DATA_PROCESSED / "features_v11_personalized.parquet"
    feat_perm.to_parquet(out_path, index=False)
    log.info(f"Saved: {out_path}")

    # Save stats too (for reference)
    stats_path = DATA_PROCESSED / "zscore_stats.json"
    # Convert stats dict to serializable form
    stats_data = {}
    for col, ss in stats.items():
        stats_data[col] = {
            'mean': ss[f'{col}_subj_mean'].to_dict(),
            'std': ss[f'{col}_subj_std'].to_dict(),
        }
    import json
    with open(stats_path, 'w') as f:
        json.dump(stats_data, f)
    log.info(f"Saved stats: {stats_path}")

    return feat_perm


if __name__ == "__main__":
    main()
