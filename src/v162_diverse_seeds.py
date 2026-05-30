"""
V162 — More Diverse Seeds (Random Range)

Hypothesis: V160 used seeds spaced by 7 (42,49,56,...,144). These may be
too correlated. Using random seeds from a wider range increases diversity,
which should improve ensemble quality.

Method:
- Generate 15 random seeds from range [42, 500) (no repeated digits)
- Use same V160 architecture
- Compare OOF vs V160

Risk: Low — same architecture, only seed selection changes
Expected: OOF improvement 0.001-0.003 over V160

Why this time: V160 proved more seeds help. But regular spacing (step=7)
may not provide maximum diversity. Random seeds from wider range should
produce more independent models.
"""
import sys, gc, logging, json, re, time, warnings, random
from pathlib import Path
from datetime import datetime
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
import lightgbm as lgb

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", stream=sys.stdout)
log = logging.getLogger(__name__)

ROOT = Path('/home/mwoo423/.openclaw/workspace')
DATA = ROOT / 'data_processed'
SUBMIT = ROOT / 'submissions'

TARGETS = ['Q1','Q2','Q3','S1','S2','S3','S4']
META_COLS = {'subject_id', 'lifelog_date', 'sleep_date', 'date'}

LEAK_S = {
    'wLight_w_light_mean','wLight_w_light_std','wLight_w_light_min',
    'wLight_w_light_max','wLight_w_light_count',
    'wHr_hr_mean','wHr_hr_std','wHr_hr_min','wHr_hr_max',
    'wHr_hr_median','wHr_hr_count',
    'wPedo_pedo_step_mean','wPedo_pedo_step_sum',
    'wPedo_pedo_step_frequency_mean','wPedo_pedo_step_frequency_sum',
    'wPedo_pedo_running_step_mean','wPedo_pedo_running_step_sum',
    'wPedo_pedo_walking_step_mean','wPedo_pedo_walking_step_sum',
    'wPedo_pedo_distance_mean','wPedo_pedo_distance_sum',
    'wPedo_pedo_speed_mean','wPedo_pedo_speed_sum',
    'wPedo_pedo_burned_calories_mean','wPedo_pedo_burned_calories_sum',
}
LEAK_Q = {
    'wHr_hr_mean','wHr_hr_std','wHr_hr_min','wHr_hr_max',
    'wHr_hr_median','wHr_hr_count',
}

CFGS = {
    'wide':   {'num_leaves': 30, 'max_depth': 3, 'learning_rate': 0.05, 'n_estimators': 300,
               'subsample': 0.8, 'colsample_bytree': 0.8, 'reg_alpha': 2.0, 'reg_lambda': 5.0, 'min_child_samples': 5},
    'deep':   {'num_leaves': 20, 'max_depth': 5, 'learning_rate': 0.02, 'n_estimators': 1000,
               'subsample': 0.7, 'colsample_bytree': 0.6, 'reg_alpha': 0.5, 'reg_lambda': 2.0, 'min_child_samples': 15},
    'v48':    {'num_leaves': 15, 'max_depth': 4, 'learning_rate': 0.03, 'n_estimators': 500,
               'subsample': 0.7, 'colsample_bytree': 0.7, 'reg_alpha': 1.0, 'reg_lambda': 3.0, 'min_child_samples': 10},
    'safety': {'num_leaves': 10, 'max_depth': 3, 'learning_rate': 0.02, 'n_estimators': 1000,
               'subsample': 0.6, 'colsample_bytree': 0.6, 'reg_alpha': 3.0, 'reg_lambda': 10.0, 'min_child_samples': 20},
}

V53_SWEEP = {
    'Q1':  {'cfg': 'deep',   'n_feat': 19},
    'Q2':  {'cfg': 'deep',   'n_feat': 14},
    'Q3':  {'cfg': 'v48',    'n_feat': 11},
    'S1':  {'cfg': 'wide',   'n_feat': 21},
    'S2':  {'cfg': 'deep',   'n_feat': 19},
    'S3':  {'cfg': 'safety', 'n_feat': 23},
    'S4':  {'cfg': 'wide',   'n_feat': 20},
}

SEED = 42
N_FOLDS = 5
N_SEEDS = 15
META_C = 10.0


def generate_diverse_seeds(n, rng_seed=123):
    """Generate diverse seeds by avoiding patterns in binary representation."""
    random.seed(rng_seed)
    candidates = []
    for s in range(42, 300):
        # Avoid seeds with repeated binary patterns
        bin_s = bin(s)
        if '0000' not in bin_s and '1111' not in bin_s:
            candidates.append(s)
    return random.sample(candidates, n)


def sanitize_col(n):
    return re.sub(r'[^a-zA-Z0-9_]', '_', n)

def get_feature_cols(df):
    return [c for c in df.columns
            if c not in META_COLS | set(TARGETS)
            and np.issubdtype(df[c].dtype, np.number)]

def remove_leak(cols, target):
    if target.startswith('S'):
        return [c for c in cols if c not in LEAK_S]
    elif target.startswith('Q'):
        return [c for c in cols if c not in LEAK_Q]
    return cols

def rank_features(feat_df, feat_cols, target, seed=SEED):
    y = feat_df[target].values.astype(np.float64)
    X = feat_df[feat_cols].fillna(0).values.astype(np.float64)
    spw = max(((y == 0).sum()) / max((y == 1).sum(), 1), 0.1)
    cfg_name = V53_SWEEP[target]['cfg']
    base = CFGS[cfg_name]
    params = {**{k: base[k] for k in ['num_leaves', 'max_depth', 'n_estimators']},
              'learning_rate': 0.05, 'scale_pos_weight': spw,
              'random_state': seed, 'force_row_wise': True, 'n_jobs': 1, 'verbose': -1}
    sn = [sanitize_col(c) for c in feat_cols]
    ds = lgb.Dataset(X, label=y, feature_name=sn)
    m = lgb.train(params, ds, num_boost_round=50)
    imp = m.feature_importance(importance_type='gain')
    ranked = sorted(zip(feat_cols, imp), key=lambda x: -x[1])
    del m, ds, X
    gc.collect()
    return [r[0] for r in ranked]

def run_target(t, train_df, test_df, feat_cols, cfg, n_feat, seeds, group, n_folds=5):
    """Run all seeds for a single target, return OOF + test preds."""
    y = train_df[t].values.astype(np.float64)
    feat_cols_clean = remove_leak(feat_cols, t)
    ranked = rank_features(train_df, feat_cols_clean, t)
    sel_cols = ranked[:n_feat]
    
    n_train = len(train_df)
    n_test = len(test_df)
    
    per_seed_oofs = []
    test_preds = np.zeros((n_test, len(seeds)))
    
    for si, seed in enumerate(seeds):
        seed_oof = np.zeros(n_train)
        seed_test = np.zeros(n_test)
        
        for fold, (tr_idx, va_idx) in enumerate(GroupKFold(n_splits=n_folds).split(train_df, y, group)):
            X_tr = train_df[sel_cols].iloc[tr_idx].fillna(0).values.astype(np.float64)
            X_va = train_df[sel_cols].iloc[va_idx].fillna(0).values.astype(np.float64)
            y_tr = y[tr_idx]
            spw = max(((y_tr == 0).sum()) / max((y_tr == 1).sum(), 1), 0.1)
            params = {**cfg, 'scale_pos_weight': spw, 'random_state': seed,
                      'force_row_wise': True, 'n_jobs': 1, 'verbose': -1}
            sn = [sanitize_col(c) for c in sel_cols]
            ds = lgb.Dataset(X_tr, label=y_tr, feature_name=sn)
            m = lgb.train(params, ds, num_boost_round=cfg['n_estimators'])
            seed_oof[va_idx] = m.predict(X_va)
            seed_test += m.predict(test_df[sel_cols].fillna(0).values.astype(np.float64))
        
        seed_oof = np.clip(seed_oof, 0.001, 0.999)
        seed_test /= n_folds
        per_seed_oofs.append(seed_oof)
        test_preds[:, si] = seed_test
    
    return per_seed_oofs, test_preds, sel_cols


def main():
    t_start = time.time()
    
    log.info("=" * 70)
    log.info("V162 — More Diverse Seeds (Random Range)")
    log.info("=" * 70)
    
    train_df = pd.read_parquet(DATA / "features.parquet")
    test_df = pd.read_parquet(DATA / "test_features.parquet")
    
    for df in [train_df, test_df]:
        for c in ['sleep_date', 'lifelog_date', 'date']:
            if c in df.columns:
                df[c] = pd.to_datetime(df[c]).dt.strftime('%Y-%m-%d')
    
    feat_cols = get_feature_cols(train_df)
    group = train_df['subject_id'].values
    
    # Generate diverse seeds
    diverse_seeds = generate_diverse_seeds(N_SEEDS, rng_seed=42)
    regular_seeds = [SEED + si * 7 for si in range(N_SEEDS)]
    
    log.info(f"Regular seeds:  {regular_seeds}")
    log.info(f"Diverse seeds:  {diverse_seeds}")
    
    # Run with both seed strategies for comparison
    for seed_strategy, seeds_name in [(diverse_seeds, "Diverse"), (regular_seeds, "Regular")]:
        log.info(f"\n{'='*70}")
        log.info(f"Seed strategy: {seeds_name}")
        log.info(f"{'='*70}")
        
        train_oof = {}
        test_seed = {}
        all_sel_cols = {}
        
        for t in TARGETS:
            n_feat = V53_SWEEP[t]['n_feat']
            cfg = CFGS[V53_SWEEP[t]['cfg']]
            
            per_seed_oofs, test_preds, sel_cols = run_target(
                t, train_df, test_df, feat_cols, cfg, n_feat, seeds_name == "Diverse" and diverse_seeds or regular_seeds, group
            )
            
            # Fix: use correct seeds variable
            per_seed_oofs, test_preds, sel_cols = run_target(
                t, train_df, test_df, feat_cols, cfg, n_feat, seeds_name == "Diverse" and diverse_seeds or regular_seeds, group
            )
            
            train_oof[t] = np.column_stack(per_seed_oofs)
            test_seed[t] = test_preds
            all_sel_cols[t] = sel_cols
        
        # Meta learners for each strategy
        avg_oofs = {}
        for t in TARGETS:
            y = train_df[t].values.astype(np.float64)
            stacked = train_oof[t]
            meta = LogisticRegression(C=META_C, max_iter=1000, random_state=SEED)
            meta.fit(stacked, y)
            train_oof[t] = meta.predict_proba(stacked)[:, 1]
            oof_val = log_loss(y, np.clip(train_oof[t], 0.001, 0.999))
            avg_oofs[t] = oof_val
            log.info(f"  {t}: OOF={oof_val:.5f}")
        
        avg = np.mean(list(avg_oofs.values()))
        log.info(f"  AVG OOF: {avg:.5f}")
        log.info(f"  Δ vs V146 (0.63169): {avg - 0.63169:+.5f}")
        log.info(f"  Δ vs V160 (0.62240): {avg - 0.62240:+.5f}")
    
    log.info(f"\nTotal time: {time.time()-t_start:.0f}s")


if __name__ == '__main__':
    main()
