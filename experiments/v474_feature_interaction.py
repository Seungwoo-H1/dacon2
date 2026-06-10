#!/usr/bin/env python3
"""
V474 — Feature Engineering: Interaction Terms
가설: top predictive features 간의 product/ratio interactions이 추가 signal 제공.
      각 target별로 top-K features를 찾고, 그들 간의 2-way interactions 생성.
"""

import warnings
warnings.filterwarnings('ignore')

import json
import time
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path('/home/mwoo423/.openclaw/workspace')
DATA_PROC = ROOT / 'data_processed'
EXPERIMENTS = ROOT / 'experiments'
DATA_RAW = ROOT / 'data_raw'

TARGETS = ['Q1', 'Q2', 'Q3', 'S1', 'S2', 'S3', 'S4']

def main():
    start_time = time.time()
    print("=" * 60)
    print("V474 — Feature Engineering: Interaction Terms")
    print("=" * 60)
    
    # Load
    train_feat = pd.read_parquet(DATA_PROC / 'train_features_clean_v60.parquet')
    test_feat = pd.read_parquet(DATA_PROC / 'test_features_clean_v60.parquet')
    train_csv = pd.read_csv(DATA_RAW / 'ch2026_metrics_train.csv')
    
    # Merge targets
    train_feat_cp = train_feat.copy()
    train_feat_cp['date_parsed'] = pd.to_datetime(train_feat_cp['date'])
    train_csv_cp = train_csv[['subject_id', 'lifelog_date', 'Q1','Q2','Q3','S1','S2','S3','S4']].copy()
    train_csv_cp['lf_parsed'] = pd.to_datetime(train_csv_cp['lifelog_date'])
    
    train = train_feat_cp.merge(
        train_csv_cp,
        left_on=['subject_id', 'date_parsed'],
        right_on=['subject_id', 'lf_parsed'],
        how='inner'
    ).drop(columns=['date_parsed', 'lf_parsed'])
    
    print(f"Train: {train.shape}")
    
    exclude_cols = ['subject_id', 'date', 'lifelog_date'] + TARGETS
    feature_cols = [c for c in train.columns if c not in exclude_cols
                    and train[c].dtype in ['float64', 'int64', 'float32', 'int32', 'float16']
                    and train[c].nunique() > 1]
    
    print(f"Base features: {len(feature_cols)}")
    
    X_base = train[feature_cols].fillna(0).replace([np.inf, -np.inf], 0).values
    
    # Feature importance for each target (using a single model)
    import lightgbm as lgb
    from sklearn.model_selection import StratifiedKFold
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Get average feature importance across folds for each target
    target_importances = {}
    for t in TARGETS:
        y_t = (train[t] > 0.5).astype(int).values
        imp_list = []
        for fold, (tr_idx, val_idx) in enumerate(skf.split(X_base, y_t)):
            X_tr = X_base[tr_idx]
            m = lgb.LGBMClassifier(
                n_estimators=200, learning_rate=0.1, num_leaves=31,
                max_depth=5, random_state=42 + fold, verbose=-1, n_jobs=-1
            )
            m.fit(X_tr, y_t[tr_idx])
            imp_list.append(m.feature_importances_)
        target_importances[t] = np.mean(imp_list, axis=0)
    
    # For each target, find top-10 features
    for t in TARGETS:
        top10_idx = np.argsort(target_importances[t])[-10:][::-1]
        print(f"\n{t} top-10 features:")
        for i in top10_idx:
            print(f"  {feature_cols[i]}: {target_importances[t][i]:.0f}")
    
    # Generate interactions: product of top-5 feature pairs per target
    print("\n\nGenerating interactions...")
    
    # For now, generate interactions across ALL targets' top features
    # Use global top features (average importance across all targets)
    global_imp = np.mean([v for v in target_importances.values()], axis=0)
    global_top5 = np.argsort(global_imp)[-5:][::-1]
    global_top10 = np.argsort(global_imp)[-10:][::-1]
    
    print(f"\nGlobal top-5 features:")
    for i in global_top5:
        print(f"  {feature_cols[i]}: {global_imp[i]:.0f}")
    
    # Generate pairwise interactions for top-10
    interactions = {}
    interaction_names = []
    
    # Product interactions
    for i in range(len(global_top10)):
        for j in range(i+1, len(global_top10)):
            f1_idx = global_top10[i]
            f2_idx = global_top10[j]
            name = f"{feature_cols[f1_idx]}×{feature_cols[f2_idx]}"
            interactions[name] = X_base[:, f1_idx] * X_base[:, f2_idx]
            interaction_names.append(name)
    
    # Ratio interactions (avoid div by zero)
    for i in range(len(global_top10)):
        for j in range(i+1, len(global_top10)):
            f1_idx = global_top10[i]
            f2_idx = global_top10[j]
            name = f"{feature_cols[f1_idx]}/{feature_cols[f2_idx]}"
            denom = np.abs(X_base[:, f2_idx]) + 1e-10
            interactions[name] = X_base[:, f1_idx] / denom
            interaction_names.append(name)
    
    print(f"\nGenerated {len(interactions)} interactions ({len(interaction_names)//2} product + {len(interaction_names)//2} ratio)")
    
    # Add interactions to features
    interaction_array = np.column_stack([interactions[n] for n in interaction_names])
    # Normalize interactions
    for c in range(interaction_array.shape[1]):
        col = interaction_array[:, c]
        col = (col - col.mean()) / (col.std() + 1e-10)
        interaction_array[:, c] = col
    
    X_aug = np.column_stack([X_base, interaction_array])
    all_feature_names = feature_cols + interaction_names
    
    print(f"Total features: {X_aug.shape[1]} ({len(feature_cols)} base + {len(interaction_names)} interactions)")
    
    # Evaluate
    y_avg = train[TARGETS].mean(axis=1).values
    y_bin = (y_avg > 0.5).astype(int)
    
    print("\n" + "=" * 60)
    print("5-fold CV with interactions")
    print("=" * 60)
    
    all_meta_preds = np.zeros(len(X_aug))
    all_student_preds = np.zeros(len(X_aug))
    
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_aug, y_bin)):
        X_tr, X_val = X_aug[tr_idx], X_aug[val_idx]
        y_tr_avg = y_avg[tr_idx]
        
        # Student
        student_tr = np.zeros(len(X_tr))
        student_val = np.zeros(len(X_val))
        
        for t in TARGETS:
            y_t_tr = (train.iloc[tr_idx][t] > 0.5).astype(int).values
            m = lgb.LGBMClassifier(
                n_estimators=500, learning_rate=0.05, num_leaves=31,
                max_depth=5, min_child_samples=30,
                subsample=0.8, colsample_bytree=0.8,
                reg_alpha=0.5, reg_lambda=1.0,
                random_state=42 + fold, verbose=-1, n_jobs=-1
            )
            m.fit(X_tr, y_t_tr)
            student_tr += m.predict_proba(X_tr)[:, 1] / len(TARGETS)
            student_val += m.predict_proba(X_val)[:, 1] / len(TARGETS)
        
        all_student_preds[val_idx] += student_val
        
        # Meta
        meta_m = lgb.LGBMRegressor(
            n_estimators=500, learning_rate=0.03, num_leaves=25,
            max_depth=5, min_child_samples=30,
            subsample=0.8, colsample_bytree=0.7,
            reg_alpha=1.0, reg_lambda=3.0,
            random_state=99 + fold, verbose=-1, n_jobs=-1
        )
        meta_m.fit(X_tr, y_tr_avg)
        all_meta_preds[val_idx] += meta_m.predict(X_val)
    
    student_oof = 1 - np.mean(np.abs(all_student_preds - y_avg))
    meta_oof = 1 - np.mean(np.abs(all_meta_preds - y_avg))
    gap = np.mean(all_meta_preds - all_student_preds)
    
    print(f"\nMeta OOF:      {meta_oof:.5f}")
    print(f"Student OOF:   {student_oof:.5f}")
    print(f"Gap:           {gap:.5f}")
    print(f"Δ student vs base (0.786): {0.786 - student_oof:+.5f}")
    
    elapsed = time.time() - start_time
    
    # Save
    ts = time.strftime("%Y%m%d_%H%M%S")
    result = {
        "version": "V474",
        "name": "Feature Engineering: Interaction Terms",
        "n_base_features": len(feature_cols),
        "n_interactions": len(interaction_names),
        "total_features": X_aug.shape[1],
        "meta_oof": round(meta_oof, 5),
        "student_oof": round(student_oof, 5),
        "gap": round(gap, 5),
        "improvement": round(0.786 - student_oof, 5),
        "timestamp": ts,
        "total_time_s": round(elapsed),
        "interaction_names": interaction_names
    }
    
    EXPERIMENTS.mkdir(parents=True, exist_ok=True)
    with open(EXPERIMENTS / f"v474_{ts}.json", 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"Saved: {EXPERIMENTS}/v474_{ts}.json")

if __name__ == '__main__':
    main()
