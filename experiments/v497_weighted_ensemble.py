"""
V497 — Per-Subject Norm + Weighted Ensemble (NO RANKING, FAST)

Skip feature ranking entirely. Use first K features from base set.
Per-target weight optimization (LGBM:XGB:CB).
V496 showed K=20-30 optimal → test K=10,20,30,40.
"""
import sys, gc, logging, time, json, warnings
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import GroupKFold
import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
    import xgboost as xgb
    import catboost as cb
except ImportError:
    print("ERROR: Required packages")
    sys.exit(1)

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", stream=sys.stdout)
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'data_processed'
SUBMIT = ROOT / 'submissions'
TARGETS = ['Q1', 'Q2', 'Q3', 'S1', 'S2', 'S3', 'S4']
LEAK_REMOVE = {'wHr_hr_median', 'wLight_w_light_sum', 'mActivity_m_activity_sum'}

def logloss(y_true, y_pred):
    eps = 1e-15
    return -np.mean(y_true * np.log(np.clip(y_pred, eps, 1-eps)) + (1-y_true) * np.log(np.clip(1-y_pred, eps, 1-eps)))

def per_subject_zscore(df, feature_cols):
    result = df[feature_cols].copy()
    for col in feature_cols:
        grouped = df.groupby('subject_id')[col]
        mean = grouped.transform('mean')
        std = grouped.transform('std').replace(0, 1e-8)
        result[col] = (df[col].fillna(0) - mean) / std
    return result

def make_model_lgb(seed, spw):
    return lgb.LGBMClassifier(num_leaves=15, max_depth=3, learning_rate=0.02,
        n_estimators=500, subsample=0.7, colsample_bytree=0.7,
        reg_alpha=2.0, reg_lambda=5.0, min_child_samples=10,
        scale_pos_weight=spw, random_state=seed, verbose=-1)

def make_model_xgb(seed, spw):
    return xgb.XGBClassifier(max_depth=3, learning_rate=0.02, n_estimators=500,
        subsample=0.7, colsample_bytree=0.7, reg_alpha=2.0, reg_lambda=5.0,
        min_child_weight=3, random_state=seed, eval_metric='logloss')

def make_model_cb(seed, spw):
    return cb.CatBoostClassifier(iterations=500, learning_rate=0.02, depth=3,
        subsample=0.7, colsample_bylevel=0.7, l2_leaf_reg=5.0,
        min_data_in_leaf=10, random_state=seed, loss_function='Logloss')

def main():
    t_start = time.time()
    log.info("=" * 70)
    log.info("V497 — Per-Subject Norm + Weighted Ensemble (FAST)")
    log.info("=" * 70)

    # Load
    train = pd.read_parquet(DATA / "features.parquet")
    test = pd.read_parquet(DATA / "test_features.parquet")

    target_cols_set = set(TARGETS)
    meta_cols = {'subject_id', 'lifelog_date', 'sleep_date', 'date'}
    feature_cols_all = [c for c in train.columns
                        if c not in target_cols_set and c not in meta_cols
                        and np.issubdtype(train[c].dtype, np.number)]
    test_numeric = [c for c in test.columns if np.issubdtype(test[c].dtype, np.number)]
    common_cols = [c for c in feature_cols_all if c in test_numeric]
    leak_cols = [c for c in common_cols if c not in LEAK_REMOVE]
    
    train = train[leak_cols + list(target_cols_set) + ['subject_id']]
    test = test[['subject_id'] + [c for c in leak_cols if c in test.columns]]
    log.info(f"  Train: {train.shape}, Test: {test.shape}, Feat: {len(leak_cols)}")

    # Z-score
    train_orig = train[leak_cols].fillna(0).values.astype(np.float64)
    test_orig = test[leak_cols].fillna(0).values.astype(np.float64)
    train_z = per_subject_zscore(train, leak_cols).fillna(0).values.astype(np.float64)
    test_z = per_subject_zscore(test, leak_cols).fillna(0).values.astype(np.float64)
    
    X_train = np.hstack([train_orig, train_z])
    X_test = np.hstack([test_orig, test_z])
    
    groups = train['subject_id'].values
    gkf = GroupKFold(n_splits=5)

    predictions = {}
    target_results = {}

    for target in TARGETS:
        log.info(f"\n--- {target} (rate={train[target].mean():.3f}) ---")
        y = train[target].values.astype(np.float64)
        spw = max(((y == 0).sum()) / max((y == 1).sum(), 1), 0.1)

        # Quick ranking: LGBM on base features ONLY (141, not 282), 10 rounds
        sn_base = [c.replace(' ','').replace(',','_').replace('(','').replace(')','').replace('/','').replace('-','_') for c in leak_cols]
        ds = lgb.Dataset(train_orig, label=y, feature_name=sn_base)
        m_rank = lgb.train({
            'objective': 'binary', 'metric': 'binary_logloss', 'verbose': -1,
            'num_leaves': 8, 'max_depth': 2, 'learning_rate': 0.05,
            'n_estimators': 10, 'subsample': 0.8, 'colsample_bytree': 0.8,
            'random_state': 42, 'min_child_samples': 20,
        }, ds, num_boost_round=10)
        imp = m_rank.feature_importance(importance_type='gain')
        ranked_base_idx = np.argsort(-imp)
        
        # Map: ranked_base → interleaved (base0, base1, ..., base140, zbase0, zbase1, ...)
        ranked_combined = []
        for bi in ranked_base_idx:
            ranked_combined.append(bi)          # base feature
            ranked_combined.append(bi + 141)    # zscore feature
        ranked_combined = np.array(ranked_combined)

        # Try K=10, 20, 30, 40
        best_k = 20
        best_avg_oof = float('inf')
        best_w = (1/3, 1/3, 1/3)
        
        for n_feat in [10, 20, 30, 40]:
            n_feat = min(n_feat, len(ranked_combined))
            top_idx = ranked_combined[:n_feat]
            X_top = X_train[:, top_idx]
            X_test_top = X_test[:, top_idx]

            oof_lgb = np.zeros(len(y))
            oof_xgb = np.zeros(len(y))
            oof_cb = np.zeros(len(y))

            for fold, (tr, va) in enumerate(gkf.split(X_top, y, groups)):
                m = make_model_lgb(42, spw); m.fit(X_top[tr], y[tr])
                oof_lgb[va] = m.predict_proba(X_top[va])[:, 1]
                
                m = make_model_xgb(42, spw); m.fit(X_top[tr], y[tr])
                oof_xgb[va] = m.predict_proba(X_top[va])[:, 1]
                
                m = make_model_cb(42, spw); m.fit(X_top[tr], y[tr], eval_set=(X_top[va], y[va]), use_best_model=True)
                oof_cb[va] = m.predict_proba(X_top[va])[:, 1]

            # Weight sweep
            best_wkf = float('inf')
            best_wkf_w = (1/3, 1/3, 1/3)
            for wl in np.arange(0, 0.81, 0.1):
                for wx in np.arange(0, 0.81 - wl, 0.1):
                    wc = round(1.0 - wl - wx, 2)
                    if wc < 0: continue
                    oof_w = wl * oof_lgb + wx * oof_xgb + wc * oof_cb
                    wf = logloss(y, oof_w)
                    if wf < best_wkf:
                        best_wkf = wf
                        best_wkf_w = (wl, wx, wc)
            
            log.info(f"    K={n_feat}: LGB={logloss(y, oof_lgb):.4f}, XGB={logloss(y, oof_xgb):.4f}, CB={logloss(y, oof_cb):.4f}, Best=({best_wkf_w[0]:.1f},{best_wkf_w[1]:.1f},{best_wkf_w[2]:.1f})={best_wkf:.4f}")
            
            if best_wkf < best_avg_oof:
                best_avg_oof = best_wkf
                best_k = n_feat
                best_w = best_wkf_w
                best_oof_lgb = oof_lgb.copy()
                best_oof_xgb = oof_xgb.copy()
                best_oof_cb = oof_cb.copy()

        log.info(f"  → K={best_k}, OOF={best_avg_oof:.4f}, W={best_w}")

        # Final predictions: 4 seeds
        top_idx = ranked_combined[:best_k]
        X_top_all = X_train[:, top_idx]
        X_test_top_all = X_test[:, top_idx]

        final_lgb = np.zeros(len(X_test_top_all))
        final_xgb = np.zeros(len(X_test_top_all))
        final_cb = np.zeros(len(X_test_top_all))
        
        count = 0
        for seed in [42, 123, 456, 789]:
            for fold, (tr, va) in enumerate(gkf.split(X_top_all, y, groups)):
                for fn, cfg in [('lgb', lambda: make_model_lgb(seed, spw)),
                                ('xgb', lambda: make_model_xgb(seed, spw)),
                                ('cb', lambda: make_model_cb(seed, spw))]:
                    m = cfg()
                    m.fit(X_top_all[tr], y[tr])
                    preds = m.predict_proba(X_test_top_all)[:, 1]
                    if fn == 'lgb': final_lgb += preds
                    elif fn == 'xgb': final_xgb += preds
                    else: final_cb += preds
                    count += 1

        final_lgb /= count
        final_xgb /= count
        final_cb /= count

        wl, wx, wc = best_w
        test_preds = wl * final_lgb + wx * final_xgb + wc * final_cb
        predictions[target] = np.clip(test_preds, 0.0001, 0.9999)

        target_results[target] = {
            'best_n_feat': best_k, 'best_avg_oof': float(best_avg_oof),
            'best_weights': [float(wl), float(wx), float(wc)],
            'lgb_oof': float(logloss(y, best_oof_lgb)),
            'xgb_oof': float(logloss(y, best_oof_xgb)),
            'cb_oof': float(logloss(y, best_oof_cb)),
        }
        log.info(f"  Final: LGB={target_results[target]['lgb_oof']:.4f}")

        del X_top_all, X_test_top_all, final_lgb, final_xgb, final_cb
        gc.collect()

    # Summary
    avg_oof = np.mean([v['best_avg_oof'] for v in target_results.values()])
    log.info(f"\n{'='*70}")
    log.info("V497 RESULTS")
    for t in TARGETS:
        r = target_results[t]; w = r['best_weights']
        log.info(f"  {t}: K={r['best_n_feat']}, W=({w[0]:.1f},{w[1]:.1f},{w[2]:.1f}), OOF={r['best_avg_oof']:.4f}")
    log.info(f"  AVG OOF: {avg_oof:.4f}  |  Time: {time.time()-t_start:.0f}s")

    # Save
    sample = pd.read_csv(ROOT / "data_raw" / "ch2026_submission_sample.csv")
    sub = sample.copy()
    for t in TARGETS: sub[t] = predictions[t]
    sub_path = SUBMIT / f"submission_v497_weighted_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    sub.to_csv(sub_path, index=False)
    log.info(f"  Submission: {sub_path}")

    meta = {'version': 'V497_weighted', 'avg_oof': float(avg_oof), 'target_results': target_results,
            'submission_file': str(sub_path), 'timestamp': datetime.now().isoformat(),
            'total_time': f"{time.time()-t_start:.0f}s"}
    with open(SUBMIT / f'meta_v497_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json', 'w') as f:
        json.dump(meta, f, indent=2)
    log.info("  DONE.")

if __name__ == "__main__":
    main()
