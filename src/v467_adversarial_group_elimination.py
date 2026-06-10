"""
V467 — Feature Distribution Normalization + Adversarial Group Elimination

Hypothesis: V461/V466에서 adversarial features가 train-test distribution mismatch를 일으킴.
단순 removal(V461)이나 scaling(V465)은 imperfect. 
새로운 접근: (1) distribution mismatch가 큰 group(예: wifi, ble, gps rssi)은 아예 제거
           (2) 남은 group 내에서는 feature-wise distribution normalization (train+test joint z-score)
           (3) Cross-group statistics: 각 group의 global mean/std을 meta feature로 추가

핵심 아이디어: 같은 sensor type의 features끼리는 상호보완적 → group-level pattern을 capturing하는 meta feature
"""
import sys, gc, logging, json, re, time, warnings
from pathlib import Path
from datetime import datetime
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupKFold
import numpy as np
import pandas as pd
import lightgbm as lgb

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", stream=sys.stdout)
log = logging.getLogger(__name__)

ROOT = Path('/home/mwoo423/.openclaw/workspace')
DATA = ROOT / 'data_processed'
SUBMIT = ROOT / 'submissions'
EXPERIMENTS = ROOT / 'experiments'
SUBMIT.mkdir(exist_ok=True)
EXPERIMENTS.mkdir(exist_ok=True)

TARGETS = ['Q1','Q2','Q3','S1','S2','S3','S4']
META_COLS = {'subject_id', 'lifelog_date', 'sleep_date', 'date'}
Q_TARGETS = ['Q1','Q2','Q3']
S_TARGETS = ['S1','S2','S3','S4']

# Adversarial group definitions — features that had high adversarial importance
# These were the top features in V461/V465/V466 that showed train-test distribution shift
ADV_GROUPS = {
    'mGps': ['gps_count_mean', 'gps_count_std'],
    'wLight': ['w_light_count'],
    'wHr': ['hr_mean'],
    'mWifi': ['wifi_avg_rssi_mean', 'wifi_avg_rssi_max', 'wifi_max_rssi_max'],
    'mAmbience': ['ambience_inside,_large_room_or_hall_sum'],
    'mACStatus': ['hour_evening', 'm_charging_count'],
    'mBle': ['ble_rssi_std_max'],
    'wPedo': ['pedo_burned_calories_mean'],
    'mUsageStats': ['usage_total_time_min'],
}

LEAK_S = {'wLight_w_light_mean','wLight_w_light_std','wLight_w_light_min','wLight_w_light_max','wLight_w_light_count',
    'wHr_hr_mean','wHr_hr_std','wHr_hr_min','wHr_hr_max','wHr_hr_median','wHr_hr_count',
    'wPedo_pedo_step_mean','wPedo_pedo_step_sum','wPedo_pedo_step_frequency_mean','wPedo_pedo_step_frequency_sum',
    'wPedo_pedo_running_step_mean','wPedo_pedo_running_step_sum','wPedo_pedo_walking_step_mean','wPedo_pedo_walking_step_sum',
    'wPedo_pedo_distance_mean','wPedo_pedo_distance_sum','wPedo_pedo_speed_mean','wPedo_pedo_speed_sum',
    'wPedo_pedo_burned_calories_mean','wPedo_pedo_burned_calories_sum'}
LEAK_Q = {'wHr_hr_mean','wHr_hr_std','wHr_hr_min','wHr_hr_max','wHr_hr_median','wHr_hr_count'}

SEED = 42
N_FOLDS = 5
N_SEEDS = 15

CFGS = [
    {'num_leaves': 10, 'max_depth': 2, 'learning_rate': 0.01, 'n_estimators': 3000,
     'subsample': 0.4, 'colsample_bytree': 0.4, 'reg_alpha': 10.0, 'reg_lambda': 50.0,
     'min_child_samples': 40},
    {'num_leaves': 20, 'max_depth': 4, 'learning_rate': 0.01, 'n_estimators': 2500,
     'subsample': 0.5, 'colsample_bytree': 0.5, 'reg_alpha': 5.0, 'reg_lambda': 20.0,
     'min_child_samples': 25},
    {'num_leaves': 50, 'max_depth': 7, 'learning_rate': 0.03, 'n_estimators': 1000,
     'subsample': 0.8, 'colsample_bytree': 0.8, 'reg_alpha': 0.1, 'reg_lambda': 1.0,
     'min_child_samples': 8},
]


def sanitize_col(n):
    return re.sub(r'[^a-zA-Z0-9_]', '_', n)

def remove_leak(cols, target):
    if target.startswith('S'):
        return [c for c in cols if c not in LEAK_S]
    elif target.startswith('Q'):
        return [c for c in cols if c not in LEAK_Q]
    return cols


def build_adversarial_groups(train_df, test_df, num_cols, n_adv_folds=3):
    """Build adversarial importance per group using GroupKFold on train."""
    adv_imp = {g: [] for g in ADV_GROUPS}
    
    skf = GroupKFold(n_splits=n_adv_folds)
    groups_arr = train_df['subject_id'].values
    
    for adv_fold, (tr_idx, va_idx) in enumerate(skf.split(train_df, train_df['Q1'], groups_arr)):
        tr_fold = train_df.iloc[tr_idx]
        
        for group_name, cols in ADV_GROUPS.items():
            matching_cols = [c for c in cols if c in num_cols]
            if not matching_cols:
                continue
            
            tr_data = tr_fold[matching_cols].fillna(0)
            te_data = test_df[matching_cols].fillna(0)
            
            adv_X = pd.concat([tr_data, te_data], axis=0)
            adv_y = np.array([0]*len(tr_data) + [1]*len(te_data))
            
            if adv_X.shape[0] < 10:
                continue
                
            sn = [sanitize_col(c) for c in matching_cols]
            ds = lgb.Dataset(adv_X.values, label=adv_y, feature_name=sn)
            params = {'objective': 'binary', 'metric': 'binary_logloss', 'verbose': -1,
                      'num_leaves': 15, 'max_depth': 2, 'learning_rate': 0.05, 'n_estimators': 50,
                      'random_state': SEED + adv_fold, 'force_row_wise': True, 'n_jobs': 1}
            m = lgb.train(params, ds, num_boost_round=50)
            imp = m.feature_importance(importance_type='gain')
            for i, c in enumerate(matching_cols):
                adv_imp[group_name].append(imp[i])
    
    # Average per group
    adv_group_importance = {}
    for g, vals in adv_imp.items():
        if len(vals) > 0:
            adv_group_importance[g] = np.mean(vals)
        else:
            adv_group_importance[g] = 0.0
    
    return adv_group_importance


def main():
    global t_start
    t_start = time.time()

    log.info("=" * 70)
    log.info("V467 — Feature Distribution Normalization + Adversarial Group Elimination")
    log.info("=" * 70)

    train_df = pd.read_parquet(DATA / "features.parquet")
    test_df = pd.read_parquet(DATA / "test_features.parquet")
    groups_arr = train_df['subject_id'].values

    num_cols = [c for c in train_df.columns
                if c not in META_COLS | set(TARGETS)
                and np.issubdtype(train_df.dtypes[c], np.number)]

    # ===== Step 1: Adversarial Group Assessment =====
    log.info("  Step 1: Adversarial Group Assessment")
    adv_group_imp = build_adversarial_groups(train_df, test_df, num_cols)
    sorted_groups = sorted(adv_group_imp.items(), key=lambda x: -x[1])
    log.info("  Group adversarial importance:")
    for g, v in sorted_groups:
        log.info(f"    {g}: {v:.4f}")
    
    # Remove top adversarial groups (if importance > median)
    median_imp = np.median(list(adv_group_imp.values()))
    groups_to_remove = set()
    for g, v in adv_group_imp.items():
        if v > median_imp and v > 0.02:
            groups_to_remove.add(g)
            for col in ADV_GROUPS.get(g, []):
                log.info(f"    Removing group {g} (adv_imp={v:.4f} > median {median_imp:.4f})")
    
    # ===== Step 2: Build feature set without adv groups =====
    log.info(f"\n  Step 2: Build feature set (removed {len(groups_to_remove)} groups)")
    
    # Determine which columns belong to adv groups
    adv_group_cols = set()
    for g in groups_to_remove:
        adv_group_cols.update(ADV_GROUPS.get(g, []))
    
    safe_cols = [c for c in num_cols if c not in adv_group_cols]
    log.info(f"  Safe features: {len(safe_cols)} (from {len(num_cols)} total)")
    log.info(f"  Removed features: {len(adv_group_cols)}")
    
    # ===== Z-Score per subject =====
    log.info("  Step 3: Z-Score + Baseline")
    zscore_train = pd.DataFrame(index=train_df.index)
    zscore_test = pd.DataFrame(index=test_df.index)
    
    # Joint z-score (train + test for normalization)
    for col in safe_cols:
        tr_vals = train_df[col].fillna(0)
        te_vals = test_df[col].fillna(0)
        
        tr_mean = train_df.groupby('subject_id')[col].transform('mean')
        tr_std = train_df.groupby('subject_id')[col].transform('std').fillna(0).replace(0, 1)
        te_mean = test_df.groupby('subject_id')[col].transform('mean')
        te_std = test_df.groupby('subject_id')[col].transform('std').fillna(0).replace(0, 1)
        
        zscore_train[f'z_{col}'] = (tr_vals - tr_mean) / tr_std
        zscore_test[f'z_{col}'] = (te_vals - te_mean) / te_std
    
    # ===== Per-group global stats as meta features =====
    log.info("  Step 4: Per-group global stats as meta features")
    group_stats_train = pd.DataFrame(index=train_df.index)
    group_stats_test = pd.DataFrame(index=test_df.index)
    
    for group_name, cols in ADV_GROUPS.items():
        if group_name in groups_to_remove:
            continue
        matching = [c for c in cols if c in safe_cols]
        if not matching:
            continue
        for t in TARGETS:
            tr_vals = train_df[matching].fillna(0).mean(axis=1)
            te_vals = test_df[matching].fillna(0).mean(axis=1)
            group_stats_train[f'gstat_{t}_{group_name}'] = tr_vals
            group_stats_test[f'gstat_{t}_{group_name}'] = te_vals
    
    # ===== Baseline =====
    subject_ids = np.unique(groups_arr)
    baselines = {}
    for t in TARGETS:
        y_t = train_df[t].values
        bl = {}
        for sid in subject_ids:
            mask = groups_arr == sid
            s_y = y_t[mask]; n_samples = mask.sum(); global_rate = y_t.mean()
            subj_rate = s_y.mean() if n_samples > 0 else global_rate
            bl[sid] = 0.7 * subj_rate + 0.3 * global_rate
        baselines[t] = bl

    # ===== Interaction features =====
    log.info("  Step 5: Interaction features")
    for t in TARGETS:
        train_bl = np.array([baselines[t][sid] for sid in groups_arr])
        test_bl = np.array([baselines[t][sid] for sid in test_df['subject_id'].values])
        for col in safe_cols:
            z_col = f'z_{col}'
            if z_col in zscore_train.columns:
                zscore_train[f'zb_{t}_{col}'] = zscore_train[z_col] * train_bl
                zscore_test[f'zb_{t}_{col}'] = zscore_test[z_col] * test_bl
                zscore_train[f'z3_{t}_{col}'] = zscore_train[z_col].values ** 3
                zscore_test[f'z3_{t}_{col}'] = zscore_test[z_col].values ** 3

    all_train_features = pd.concat([train_df[safe_cols], zscore_train, group_stats_train], axis=1)
    all_test_features = pd.concat([test_df[safe_cols], zscore_test, group_stats_test], axis=1)
    log.info(f"  Features: {all_train_features.shape[1]}")

    feat_cols_all = [c for c in all_train_features.columns
                     if c not in META_COLS | set(TARGETS)
                     and np.issubdtype(all_train_features.dtypes[c], np.number)]
    log.info(f"  Number features: {len(feat_cols_all)}")

    # ===== Phase 1: 3-Model Stacking =====
    log.info("\n=== Phase 1: 3-Model Stacking ===")
    model_oof_seeds = [{} for _ in CFGS]
    model_test_seeds = [{} for _ in CFGS]
    NFEAT = {'Q1': 30, 'Q2': 30, 'Q3': 28, 'S1': 32, 'S2': 30, 'S3': 32, 'S4': 30}

    for cfg_idx in range(len(CFGS)):
        cfg = CFGS[cfg_idx]
        log.info(f"\n  --- Config {cfg_idx+1}/3 ---")
        for t_idx, target in enumerate(TARGETS):
            n_feat = NFEAT[target]
            y = train_df[target].values.astype(np.float64)
            feat_cols = remove_leak(feat_cols_all, target)

            # Feature ranking via importance
            train_with_target = all_train_features.copy()
            train_with_target[target] = train_df[target]
            
            spw = max(((y == 0).sum()) / max((y == 1).sum(), 1), 0.1)
            params = {'objective': 'binary', 'metric': 'binary_logloss', 'verbose': -1,
                      'num_leaves': 20, 'max_depth': 5, 'learning_rate': 0.05, 'n_estimators': 100,
                      'scale_pos_weight': spw, 'random_state': SEED, 'force_row_wise': True, 'n_jobs': 1}
            
            X_rank = train_with_target[feat_cols].fillna(0).values.astype(np.float64)
            sn = [sanitize_col(c) for c in feat_cols]
            ds = lgb.Dataset(X_rank, label=y, feature_name=sn)
            m_rank = lgb.train(params, ds, num_boost_round=100)
            imp = m_rank.feature_importance(importance_type='gain')
            ranked = sorted(zip(feat_cols, imp), key=lambda x: -x[1])
            top_features = [r[0] for r in ranked[:n_feat]]
            del m_rank, ds, X_rank
            gc.collect()

            X_base = train_with_target[top_features].fillna(0).values.astype(np.float64)
            X_test_base = all_test_features[top_features].fillna(0).values.astype(np.float64)
            train_baselines = np.array([baselines[target][sid] for sid in groups_arr]).reshape(-1, 1)
            test_baselines = np.array([baselines[target][sid] for sid in test_df['subject_id'].values]).reshape(-1, 1)
            X_all = np.hstack([X_base, train_baselines])
            X_test_all = np.hstack([X_test_base, test_baselines])

            oof_seed_arr = np.zeros((len(train_df), N_SEEDS))
            test_seed_arr = np.zeros((len(test_df), N_SEEDS))
            skf_inner = GroupKFold(n_splits=N_FOLDS)

            for s in range(N_SEEDS):
                if (s + 1) % 5 == 0:
                    log.info(f"    {target}: seed {s+1}/{N_SEEDS}")
                sk = SEED + s * 7 + t_idx + cfg_idx * 300
                seed_oof = np.zeros(len(train_df))
                seed_test = np.zeros(len(test_df))
                for fold, (tr_idx, va_idx) in enumerate(skf_inner.split(X_all, y, groups_arr)):
                    x_train, y_train = X_all[tr_idx], y[tr_idx]
                    x_val = X_all[va_idx]
                    spw = max(((y_train == 0).sum()) / max((y_train == 1).sum(), 1), 0.1)
                    params = {'objective': 'binary', 'metric': 'binary_logloss', 'verbose': -1,
                        **cfg, 'scale_pos_weight': spw, 'random_state': sk,
                        'force_row_wise': True, 'n_jobs': 1}
                    ds_train = lgb.Dataset(x_train, label=y_train,
                        feature_name=[sanitize_col(c) for c in top_features + ['baseline']])
                    ds_val = lgb.Dataset(x_val, label=y[va_idx],
                        feature_name=[sanitize_col(c) for c in top_features + ['baseline']], reference=ds_train)
                    model = lgb.train(params, ds_train, num_boost_round=cfg['n_estimators'],
                        valid_sets=[ds_val],
                        callbacks=[lgb.early_stopping(150, verbose=False), lgb.log_evaluation(period=0)])
                    seed_oof[va_idx] = model.predict(x_val)
                    seed_test += model.predict(X_test_all) / N_FOLDS
                    del model, ds_train, ds_val
                    gc.collect()
                oof_seed_arr[:, s] = np.clip(seed_oof, 0.001, 0.999)
                test_seed_arr[:, s] = np.clip(seed_test, 0.001, 0.999)

            model_oof_seeds[cfg_idx][target] = oof_seed_arr
            model_test_seeds[cfg_idx][target] = test_seed_arr
            avg_oof = log_loss(y, oof_seed_arr.mean(axis=1))
            log.info(f"  {target} [{cfg_idx+1}]: oof={avg_oof:.5f}")

    # ===== Phase 2: Meta =====
    log.info("\n=== Phase 2: Meta ===")
    student_oofs = {}; test_preds = {}; meta_oofs = {}

    for t_idx, target in enumerate(TARGETS):
        y = train_df[target].values
        all_oof = np.mean([model_oof_seeds[i][target].mean(axis=1) for i in range(3)], axis=0)
        all_test = np.mean([model_test_seeds[i][target].mean(axis=1) for i in range(3)], axis=0)
        student_oofs[target] = log_loss(y, all_oof)

        group = 'Q' if target.startswith('Q') else 'S'
        group_targets = Q_TARGETS if group == 'Q' else S_TARGETS
        other_group = S_TARGETS if group == 'Q' else Q_TARGETS
        
        from xgboost import XGBClassifier
        cross_oof_list = []
        cross_test_list = []
        for t_cross in group_targets:
            if t_cross == target: continue
            cross_oof_list.append(np.mean([model_oof_seeds[i][t_cross].mean(axis=1) for i in range(3)], axis=0))
            cross_test_list.append(np.mean([model_test_seeds[i][t_cross].mean(axis=1) for i in range(3)], axis=0))
        for t_cross in other_group:
            cross_oof_list.append(np.mean([model_oof_seeds[i][t_cross].mean(axis=1) for i in range(3)], axis=0) * 0.5)
            cross_test_list.append(np.mean([model_test_seeds[i][t_cross].mean(axis=1) for i in range(3)], axis=0) * 0.5)
        cross_arr = np.column_stack(cross_oof_list)
        cross_arr_test = np.column_stack(cross_test_list)

        X_meta = np.hstack([all_oof.reshape(-1, 1), cross_arr])
        X_test = np.hstack([all_test.reshape(-1, 1), cross_arr_test])

        mm = XGBClassifier(n_estimators=15, max_depth=3, reg_alpha=0.01, reg_lambda=0.0,
            gamma=0.0, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
            random_state=SEED, n_jobs=1, min_child_weight=5, verbosity=0)
        mm.fit(X_meta, y)
        meta_oofs[target] = log_loss(y, mm.predict_proba(X_meta)[:, 1])
        test_preds[target] = mm.predict_proba(X_test)[:, 1]
        log.info(f"  {target}: meta={meta_oofs[target]:.5f}, student={student_oofs[target]:.5f}")

    avg_meta = np.mean(list(meta_oofs.values()))
    avg_student = np.mean(list(student_oofs.values()))
    gap = avg_student - avg_meta
    v339 = avg_meta + gap * 0.85

    log.info(f"\n{'='*70}")
    log.info("V467 Results:")
    log.info(f"  AVG Meta OOF: {avg_meta:.5f}")
    log.info(f"  AVG Student OOF: {avg_student:.5f}")
    log.info(f"  Gap: {gap:.5f} ({gap/0.070:.2f}x)")
    log.info(f"  V339 LB: {v339:.5f}")
    log.info(f"  Groups removed: {groups_to_remove}")
    log.info(f"  Safe features: {len(feat_cols_all)}")
    log.info(f"{'='*70}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    sub = pd.DataFrame()
    sub['subject_id'] = test_df['subject_id'].values
    sub['sleep_date'] = test_df['sleep_date'].values
    sub['lifelog_date'] = test_df['lifelog_date'].values
    for t in TARGETS: sub[t] = test_preds[t]
    sub_path = SUBMIT / f"submission_v467_group_elimination_{ts}.csv"
    sub.to_csv(sub_path, index=False)
    meta_data = {'version': 'V467', 'name': 'Adversarial Group Elimination + Distribution Normalization',
        'avg_meta_oof': round(float(avg_meta), 5), 'avg_student_oof': round(float(avg_student), 5),
        'v308_lb': 0.63893, 'estimated_lb_v339_pattern': round(float(v339), 5),
        'student_meta_gap': round(float(gap), 5), 'n_models': 3, 'n_seeds': N_SEEDS,
        'submission_file': str(sub_path), 'timestamp': ts,
        'total_time_s': round(time.time() - t_start, 0),
        'groups_removed': list(groups_to_remove), 'safe_features': len(feat_cols_all)}
    meta_path = EXPERIMENTS / f'v467_{ts}.json'
    with open(meta_path, 'w') as f: json.dump(meta_data, f, indent=2)
    log.info(f"Saved: {sub_path}, Total: {time.time()-t_start:.0f}s")

if __name__ == '__main__':
    main()
