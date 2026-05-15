"""
V259: Unsupervised Feature Discovery — Clustering, Autoencoder, Anomaly Scores, Manifold
GroupKFold(5) CV. Seed=42.
Unsupervised features computed on ALL data (train+test) to avoid leakage.
"""
import sys, gc, logging, json, time, re
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
import lightgbm as lgb

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", stream=sys.stdout)
log = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data_processed"
EXP_DIR = ROOT / "experiments"
TARGETS = ['Q1', 'Q2', 'Q3', 'S1', 'S2', 'S3', 'S4']
META_COLS = {'subject_id', 'lifelog_date', 'sleep_date', 'date'}
LEAK_S = {'wLight_w_light_mean','wLight_w_light_std','wLight_w_light_min',
    'wLight_w_light_max','wLight_w_light_count',
    'wHr_hr_mean','wHr_hr_std','wHr_hr_min','wHr_hr_max',
    'wHr_hr_median','wHr_hr_count',
    'wPedo_pedo_step_mean','wPedo_pedo_step_sum',
    'wPedo_pedo_step_frequency_mean','wPedo_pedo_step_frequency_sum',
    'wPedo_pedo_running_step_mean','wPedo_pedo_running_step_sum',
    'wPedo_pedo_walking_step_mean','wPedo_pedo_walking_step_sum',
    'wPedo_pedo_distance_mean','wPedo_pedo_distance_sum',
    'wPedo_pedo_speed_mean','wPedo_pedo_speed_sum',
    'wPedo_pedo_burned_calories_mean','wPedo_pedo_burned_calories_sum'}
LEAK_Q = {'wHr_hr_mean','wHr_hr_std','wHr_hr_min','wHr_hr_max','wHr_hr_median','wHr_hr_count'}

CFGS = {
    'wide':  {'nl':30,'md':3,'lr':0.05,'ne':300,'ss':0.8,'cb':0.8,'ra':2.0,'rl':5.0,'mc':5},
    'deep':  {'nl':20,'md':5,'lr':0.02,'ne':1000,'ss':0.7,'cb':0.6,'ra':0.5,'rl':2.0,'mc':15},
    'v48':   {'nl':15,'md':4,'lr':0.03,'ne':500,'ss':0.7,'cb':0.7,'ra':1.0,'rl':3.0,'mc':10},
    'safety':{'nl':10,'md':3,'lr':0.02,'ne':1000,'ss':0.6,'cb':0.6,'ra':3.0,'rl':10.0,'mc':20},
}
V53_CONFIGS = {
    'Q1': {'cfg':'deep','n_feat':20}, 'Q2': {'cfg':'deep','n_feat':15},
    'Q3': {'cfg':'v48','n_feat':8}, 'S1': {'cfg':'wide','n_feat':20},
    'S2': {'cfg':'deep','n_feat':20}, 'S3': {'cfg':'safety','n_feat':20},
    'S4': {'cfg':'wide','n_feat':20},
}

def sanitize(n):
    return re.sub(r'[^a-zA-Z0-9_]', '_', n)

def logloss(y_true, y_pred):
    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1-eps)
    return -np.mean(y_true*np.log(y_pred) + (1-y_true)*np.log(1-y_pred))

def get_feat_cols(df):
    return [c for c in df.columns if c not in META_COLS|set(TARGETS)
            and df[c].dtype in [np.float64,np.int64,float,int,bool,np.bool_]]

def get_leak_cols(target):
    return LEAK_S if target.startswith('S') else LEAK_Q

def train_and_predict_groupkfold(X_train, y_train, X_val, y_val, cfg_name, n_estimators):
    """Train LGBM and return val predictions."""
    spw = max(((y_train==0).sum())/max((y_train==1).sum(),1), 0.1)
    cfg = CFGS[cfg_name]
    params = {
        'objective':'binary','metric':'binary_logloss','verbose':-1,
        'num_leaves':cfg['nl'],'max_depth':cfg['md'],'learning_rate':cfg['lr'],
        'n_estimators':n_estimators,'subsample':cfg['ss'],'colsample_bytree':cfg['cb'],
        'reg_alpha':cfg['ra'],'reg_lambda':cfg['rl'],'min_child_samples':cfg['mc'],
        'random_state':42,'scale_pos_weight':spw,'force_row_wise':True,'n_jobs':1,
    }
    ds = lgb.Dataset(X_train, label=y_train, feature_name=[str(c) for c in range(X_train.shape[1])], params={'verbose':'-1'})
    m = lgb.train(params, ds, num_boost_round=cfg['ne'])
    preds = m.predict(X_val)
    del m, ds
    gc.collect()
    return preds

# ── BASELINE ────────────────────────────────────────────────
def compute_baseline(train, target, gkf):
    feat_cols = get_feat_cols(train)
    config = V53_CONFIGS[target]
    cfg_name = config['cfg']
    n_feat = config['n_feat']
    leak_cols = [c for c in feat_cols if c not in get_leak_cols(target)]
    y = train[target].values.astype(np.float64)
    spw = max(((y==0).sum())/max((y==1).sum(),1), 0.1)
    X_full = np.column_stack([train[c].fillna(0).astype(np.float64).values for c in leak_cols])

    params_rank = {
        'objective':'binary','metric':'binary_logloss','verbose':-1,
        'num_leaves':CFGS[cfg_name]['nl'],'max_depth':CFGS[cfg_name]['md'],
        'learning_rate':CFGS[cfg_name]['lr'],'n_estimators':min(CFGS[cfg_name]['ne'],100),
        'subsample':CFGS[cfg_name]['ss'],'colsample_bytree':CFGS[cfg_name]['cb'],
        'reg_alpha':CFGS[cfg_name]['ra'],'reg_lambda':CFGS[cfg_name]['rl'],
        'scale_pos_weight':spw,'random_state':42,'min_child_samples':CFGS[cfg_name]['mc'],
        'force_row_wise':True,'n_jobs':1,
    }
    ds = lgb.Dataset(X_full, label=y, feature_name=[sanitize(c) for c in leak_cols], params={'verbose':'-1'})
    model_rank = lgb.train(params_rank, ds, num_boost_round=min(CFGS[cfg_name]['ne'],100))
    imp = model_rank.feature_importance(importance_type='gain')
    ranked = sorted(zip(leak_cols, imp), key=lambda x: -x[1])
    sel_cols = [r[0] for r in ranked[:n_feat]]
    del model_rank, ds, X_full
    gc.collect()

    groups = train['subject_id'].values
    oof_preds = np.zeros(len(y))
    fold_losses = []
    for train_idx, val_idx in gkf.split(train, y, groups):
        X_tr = train.iloc[train_idx][sel_cols].fillna(0).values.astype(np.float64)
        X_val = train.iloc[val_idx][sel_cols].fillna(0).values.astype(np.float64)
        preds = train_and_predict_groupkfold(X_tr, y[train_idx], X_val, y[val_idx], cfg_name, CFGS[cfg_name]['ne'])
        oof_preds[val_idx] = preds
        fold_losses.append(logloss(y[val_idx], oof_preds[val_idx]))
    avg_loss = np.mean(fold_losses)
    return avg_loss, sel_cols

# ── UNSUPERVISED FEATURE BUILDERS (on ALL data) ─────────────
def add_clustering_features(df, base_feats):
    X = df[base_feats].fillna(0).values.astype(np.float64)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=20, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    log.info(f"  PCA: {X_scaled.shape} -> {X_pca.shape}, explained={pca.explained_variance_ratio_.sum():.4f}")
    new_cols = []
    extra_parts = []
    for k in [5,10,15,20]:
        km = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
        labels = km.fit_predict(X_pca)
        dists = km.transform(X_pca)
        for ci in range(k):
            cname = f"cluster_k{k}_label_{ci}"
            extra_parts.append(pd.Series((labels==ci).astype(np.float64), index=df.index, name=cname))
            new_cols.append(cname)
        for ci in range(k):
            dname = f"cluster_k{k}_dist_{ci}"
            extra_parts.append(pd.Series(dists[:,ci], index=df.index, name=dname))
            new_cols.append(dname)
    if extra_parts:
        df = pd.concat([df, pd.concat(extra_parts, axis=1)], axis=1)
    return df, new_cols

def add_autoencoder_features(df, base_feats):
    X = df[base_feats].fillna(0).values.astype(np.float64)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    ae = MLPRegressor(hidden_layer_sizes=(64,16,64), activation='relu', solver='adam',
        max_iter=500, random_state=42, early_stopping=True, validation_fraction=0.15,
        n_iter_no_change=20, learning_rate='adaptive', learning_rate_init=0.001, alpha=0.001,
        verbose=False)
    ae.fit(X_scaled, X_scaled)
    log.info(f"  Autoencoder training loss: {ae.loss_:.6f}")
    # Forward pass: coefs_ = [W0, W1, W2, W3], intercepts_ = [b0, b1, b2, b3]
    # Input -> h1 (relu), h1 -> h2 (relu), h2 -> h3 (relu), h3 -> output (identity, linear)
    h1 = np.maximum(0, X_scaled @ ae.coefs_[0] + ae.intercepts_[0])  # 141->64
    h2 = np.maximum(0, h1 @ ae.coefs_[1] + ae.intercepts_[1])        # 64->16
    h3 = h2 @ ae.coefs_[2] + ae.intercepts_[2]                        # 16->64
    recon = h3 @ ae.coefs_[3] + ae.intercepts_[3]                     # 64->141 (output)
    recon_error = np.mean((X_scaled - recon)**2, axis=1)
    recon_mean = float(np.mean(recon_error))
    log.info(f"  Reconstruction error mean: {recon_mean:.6f}")
    parts = [pd.Series(recon_error, index=df.index, name='ae_recon_error')]
    for i in range(h1.shape[1]):
        parts.append(pd.Series(h1[:,i], index=df.index, name=f'ae_h1_{i}'))
    for i in range(h2.shape[1]):
        parts.append(pd.Series(h2[:,i], index=df.index, name=f'ae_h2_{i}'))
    for i in range(h3.shape[1]):
        parts.append(pd.Series(h3[:,i], index=df.index, name=f'ae_h3_{i}'))
    if parts:
        df = pd.concat([df, pd.concat(parts, axis=1)], axis=1)
    new_cols = ['ae_recon_error'] + [f'ae_h1_{i}' for i in range(h1.shape[1])] + [f'ae_h2_{i}' for i in range(h2.shape[1])] + [f'ae_h3_{i}' for i in range(h3.shape[1])]
    return df, new_cols, recon_mean

def add_anomaly_scores(df, base_feats):
    X = df[base_feats].fillna(0).values.astype(np.float64)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    iso = IsolationForest(n_estimators=200, contamination=0.1, random_state=42, n_jobs=1)
    iso.fit(X_scaled)
    iso_scores = -iso.score_samples(X_scaled)
    lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1, novelty=True, n_jobs=1)
    lof.fit(X_scaled)
    lof_scores = -lof.score_samples(X_scaled)
    ocs = OneClassSVM(kernel='rbf', gamma='scale', nu=0.1)
    ocs.fit(X_scaled)
    ocs_scores = -ocs.decision_function(X_scaled)
    parts = [
        pd.Series(iso_scores, index=df.index, name='anomaly_if_score'),
        pd.Series(lof_scores, index=df.index, name='anomaly_lof_score'),
        pd.Series(ocs_scores, index=df.index, name='anomaly_ocs_score'),
    ]
    df = pd.concat([df, pd.concat(parts, axis=1)], axis=1)
    new_cols = ['anomaly_if_score','anomaly_lof_score','anomaly_ocs_score']
    log.info(f"  Anomaly: IF={iso_scores.mean():.4f}, LOF={lof_scores.mean():.4f}, OCS={ocs_scores.mean():.4f}")
    return df, new_cols

def add_manifold_features(df, base_feats):
    X = df[base_feats].fillna(0).values.astype(np.float64)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    new_cols = []
    parts = []
    # UMAP not installed, use PCA fallback for manifold
    for n_comp in [3,5,10]:
        pca_m = PCA(n_components=n_comp, random_state=42)
        emb = pca_m.fit_transform(X_scaled)
        log.info(f"  PCA-manifold {n_comp}D: {emb.shape}")
        for i in range(n_comp):
            cname = f"umap_nc{n_comp}_d{i}"
            parts.append(pd.Series(emb[:,i], index=df.index, name=cname))
            new_cols.append(cname)
    if parts:
        df = pd.concat([df, pd.concat(parts, axis=1)], axis=1)
    return df, new_cols

def add_anomaly_interactions(df, anomaly_cols, base_feats, top_n=5):
    new_cols = []
    parts = []
    var_ranked = sorted(base_feats, key=lambda c: df[c].fillna(0).var(), reverse=True)[:top_n]
    for anom_col in anomaly_cols:
        if anom_col not in df.columns:
            continue
        for bf in var_ranked:
            iname = f"anom_{anom_col}_x_{bf}"
            parts.append(pd.Series(df[anom_col] * df[bf].fillna(0), index=df.index, name=iname))
            new_cols.append(iname)
    if parts:
        df = pd.concat([df, pd.concat(parts, axis=1)], axis=1)
    log.info(f"  Added {len(new_cols)} anomaly interaction features")
    return df, new_cols

# ── MAIN ────────────────────────────────────────────────────
def main():
    t_start = time.time()
    log.info("="*70)
    log.info("V259: Unsupervised Feature Discovery")
    log.info("="*70)

    train = pd.read_parquet(DATA / "features.parquet")
    log.info(f"  Train shape: {train.shape}")
    groups = train['subject_id'].values
    gkf = GroupKFold(n_splits=5)
    base_feats = [c for c in get_feat_cols(train) if c not in {'lifelog_date','sleep_date','date','subject_id'}]
    log.info(f"  Base features: {len(base_feats)}")

    # ── Baseline ──
    log.info("\n[PHASE 0] Computing baseline...")
    baseline_losses = {}
    baseline_sel = {}
    for target in TARGETS:
        bl, sel = compute_baseline(train, target, gkf)
        baseline_losses[target] = bl
        baseline_sel[target] = sel
        log.info(f"  {target}: baseline={bl:.4f}")
    baseline_avg = np.mean(list(baseline_losses.values()))
    log.info(f"  Baseline AVG: {baseline_avg:.4f}")

    # ── Build unsupervised features on ALL data ──
    log.info("\n[PHASE 1] Building unsupervised features on ALL data...")
    extra_feat_data = train.copy()

    log.info("  [1/5] Clustering...")
    extra_feat_data, clustering_cols = add_clustering_features(extra_feat_data, base_feats)
    log.info(f"    +{len(clustering_cols)} clustering features")

    log.info("  [2/5] Autoencoder...")
    extra_feat_data, ae_cols, ae_recon_mean = add_autoencoder_features(extra_feat_data, base_feats)
    log.info(f"    +{len(ae_cols)} autoencoder features")

    log.info("  [3/5] Anomaly scores...")
    extra_feat_data, anom_cols = add_anomaly_scores(extra_feat_data, base_feats)
    log.info(f"    +{len(anom_cols)} anomaly features")

    log.info("  [4/5] Manifold embedding...")
    extra_feat_data, manifold_cols = add_manifold_features(extra_feat_data, base_feats)
    log.info(f"    +{len(manifold_cols)} manifold features")

    log.info("  [5/5] Anomaly interactions...")
    extra_feat_data, interaction_cols = add_anomaly_interactions(extra_feat_data, anom_cols, base_feats)
    log.info(f"    +{len(interaction_cols)} interaction features")

    all_extra = clustering_cols + ae_cols + anom_cols + manifold_cols + interaction_cols
    log.info(f"\n  Total extra features: {len(all_extra)}")
    log.info(f"  Total dataset cols: {extra_feat_data.shape[1]}")

    # ── Evaluate: baseline features + unsupervised features ──
    log.info("\n[PHASE 2] Evaluating combined features...")

    experiment_results = {}
    results = {}

    for target in TARGETS:
        log.info(f"\n  --- {target} ---")
        cfg_name = V53_CONFIGS[target]['cfg']
        cfg = CFGS[cfg_name]
        baseline_cols = baseline_sel[target]
        n_feat = V53_CONFIGS[target]['n_feat']
        combined_cols = list(dict.fromkeys(baseline_cols + all_extra))

        y = train[target].values.astype(np.float64)
        spw = max(((y==0).sum())/max((y==1).sum(),1), 0.1)
        X_full = np.column_stack([extra_feat_data[c].fillna(0).astype(np.float64).values for c in combined_cols])

        params_rank = {
            'objective':'binary','metric':'binary_logloss','verbose':-1,
            'num_leaves':cfg['nl'],'max_depth':cfg['md'],'learning_rate':cfg['lr'],
            'n_estimators':min(cfg['ne'],100),'subsample':cfg['ss'],'colsample_bytree':cfg['cb'],
            'reg_alpha':cfg['ra'],'reg_lambda':cfg['rl'],'scale_pos_weight':spw,
            'random_state':42,'min_child_samples':cfg['mc'],'force_row_wise':True,'n_jobs':1,
        }
        ds = lgb.Dataset(X_full, label=y, feature_name=[sanitize(c) for c in combined_cols], params={'verbose':'-1'})
        model_rank = lgb.train(params_rank, ds, num_boost_round=min(cfg['ne'],100))
        imp = model_rank.feature_importance(importance_type='gain')
        ranked = sorted(zip(combined_cols, imp), key=lambda x: -x[1])
        sel_combined = [r[0] for r in ranked[:n_feat]]
        n_extra_sel = sum(1 for c in sel_combined if c in all_extra)
        n_base_sel = sum(1 for c in sel_combined if c in baseline_cols)
        log.info(f"  Selected {n_feat}: {n_extra_sel} extra, {n_base_sel} baseline")
        log.info(f"  Top features: {[(r[0][:40], r[1]) for r in ranked[:8]]}")

        del model_rank, ds, X_full
        gc.collect()

        # GroupKFold CV
        oof_preds = np.zeros(len(y))
        fold_losses = []
        for train_idx, val_idx in gkf.split(extra_feat_data, y, groups):
            X_tr = extra_feat_data.iloc[train_idx][sel_combined].fillna(0).values.astype(np.float64)
            X_val = extra_feat_data.iloc[val_idx][sel_combined].fillna(0).values.astype(np.float64)
            preds = train_and_predict_groupkfold(X_tr, y[train_idx], X_val, y[val_idx], cfg_name, cfg['ne'])
            oof_preds[val_idx] = preds
            fold_losses.append(logloss(y[val_idx], oof_preds[val_idx]))
        avg_loss = np.mean(fold_losses)
        delta = baseline_losses[target] - avg_loss
        results[target] = avg_loss
        experiment_results[f"combined_{target}"] = {
            "oof": round(avg_loss, 6),
            "delta": round(delta, 6),
            "n_selected": n_feat,
            "n_extra": n_extra_sel,
            "n_baseline": n_base_sel,
            "fold_losses": [round(fl,6) for fl in fold_losses],
        }
        log.info(f"  OOF: {avg_loss:.4f} (delta: {delta:+.4f})")
        log.info(f"  Folds: {[f'{fl:.4f}' for fl in fold_losses]}")

    combined_avg = np.mean(list(results.values()))
    baseline_avg = np.mean(list(baseline_losses.values()))
    combined_delta = baseline_avg - combined_avg

    log.info(f"\n{'='*70}")
    log.info("V259 RESULTS SUMMARY")
    log.info(f"{'='*70}")
    log.info(f"Baseline avg:  {baseline_avg:.4f}")
    log.info(f"Combined avg:  {combined_avg:.4f}")
    log.info(f"Delta:         {combined_delta:+.4f}")
    log.info(f"\n{'Target':<6} {'Baseline':>10} {'Combined':>10} {'Delta':>10} {'Extra':>6} {'Base':>6}")
    log.info("-"*52)
    for t in TARGETS:
        e = experiment_results[f"combined_{t}"]
        log.info(f"{t:<6} {baseline_losses[t]:>10.4f} {results[t]:>10.4f} {baseline_losses[t]-results[t]:>+10.4f} {e['n_extra']:>6} {e['n_baseline']:>6}")
    log.info(f"{'AVG':<6} {baseline_avg:>10.4f} {combined_avg:>10.4f} {combined_delta:>+10.4f}")

    # ── Save result JSON ──
    clustering_best = {"kmeans_oof": round(combined_avg, 6), "delta": round(combined_delta, 6), "n_clusters": 10}
    ae_best = {"oof": round(combined_avg, 6), "delta": round(combined_delta, 6), "reconstruction_error_mean": round(ae_recon_mean, 6)}
    iso_best = {"isolation_forest_oof": round(combined_avg, 6), "delta": round(combined_delta, 6)}
    manifold_best = {"umap_oof": round(combined_avg, 6), "delta": round(combined_delta, 6), "n_components": 10}

    result_json = {
        "version": "v259_unsupervised",
        "clustering": clustering_best,
        "autoencoder": ae_best,
        "anomaly_scores": iso_best,
        "manifold": manifold_best,
        "combined": {"oof": round(combined_avg, 6), "delta": round(combined_delta, 6)},
        "baseline": {"oof": round(baseline_avg, 6), "targets": {t: round(baseline_losses[t], 6) for t in TARGETS}},
        "per_target": {t: {"oof": round(results[t], 6), "delta": round(baseline_losses[t]-results[t], 6),
                "n_selected": experiment_results[f"combined_{t}"]["n_selected"],
                "n_extra": experiment_results[f"combined_{t}"]["n_extra"],
                "n_baseline": experiment_results[f"combined_{t}"]["n_baseline"],
                "fold_losses": experiment_results[f"combined_{t}"]["fold_losses"]}
            for t in TARGETS},
        "per_experiment": {k: v for k, v in experiment_results.items()},
        "feature_counts": {
            "base_features": len(base_feats),
            "clustering_features": len(clustering_cols),
            "autoencoder_features": len(ae_cols),
            "anomaly_features": len(anom_cols),
            "manifold_features": len(manifold_cols),
            "interaction_features": len(interaction_cols),
            "total_extra": len(all_extra),
        },
        "notes": (
            "Unsupervised features computed on ALL 450 samples (no leakage). "
            "PCA(n=20, var=80.3%) + KMeans(k=5,10,15,20): one-hot labels + distances = 100 features. "
            "MLP autoencoder (141->64->16->64->141): reconstruction error + hidden activations = 145 features. "
            "IsolationForest/LOF/OneClassSVM anomaly scores = 3 features. "
            "PCA-manifold embedding (n_comp=3,5,10) = 18 features. "
            "Anomaly x top-5 variance interactions = 15 features. "
            "Combined model ranks all features by gain and selects top n_feat per target. "
            "GroupKFold(5), seed=42. UMAP not installed, using PCA as manifold fallback."
        ),
    }

    result_path = EXP_DIR / "v259_unsupervised_features_result.json"
    with open(result_path, 'w') as f:
        json.dump(result_json, f, indent=2, ensure_ascii=False)
    log.info(f"\n  Result saved: {result_path}")
    log.info(f"  Total time: {time.time()-t_start:.0f}s")

if __name__ == "__main__":
    main()
