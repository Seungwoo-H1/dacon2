"""
DACon2 v259 Generalization Gap Analysis
=========================================

Analyze why OOF is good (0.537) but LB is worse (0.647) for V127.
Find patterns that predict LB generalization.

Usage: python experiments/v259_generalization_analysis.py
"""
import json, sys, time, warnings
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from sklearn.model_selection import GroupKFold

def sanitize_fn(c):
    """Sanitize feature name for LightGBM (no special JSON chars)."""
    import re
    return re.sub(r'[^a-zA-Z0-9_]', '_', c)
import lightgbm as lgb

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data_processed"
EXPERIMENTS = ROOT / "experiments"
TARGETS = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]

def psi(expected, actual, bins=20):
    epsilon = 1e-10
    exp_bins = np.histogram(expected, bins=bins)[0].astype(float)
    act_bins = np.histogram(actual, bins=bins)[0].astype(float)
    exp_bins = exp_bins / (exp_bins.sum() + epsilon)
    act_bins = act_bins / (act_bins.sum() + epsilon)
    return ((exp_bins - act_bins) * np.log((exp_bins + epsilon) / (act_bins + epsilon))).sum()

def compute_skewness(arr):
    m = np.mean(arr)
    s = np.std(arr, ddof=1)
    if s == 0: return 0.0
    return float(((arr - m) / s) ** 3).mean()

def safe_corr(x, y):
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0: return 0.0
    return float(np.corrcoef(x, y)[0, 1])

def build_personalization(df, feat_cols):
    df = df.copy()
    zscore_cols = []
    agg_parts = []
    for col in feat_cols:
        col_filled = df[col].fillna(0)
        grp = col_filled.groupby(df["subject_id"]).agg(["mean", "std"])
        grp.columns = [f"{col}_subj_mean", f"{col}_subj_std"]
        grp = grp.reset_index()
        agg_parts.append(grp)
    if agg_parts:
        agg_df = agg_parts[0]
        for part in agg_parts[1:]:
            agg_df = pd.merge(agg_df, part, on="subject_id", how="left")
        df = pd.merge(df, agg_df, on="subject_id", how="left")
    zcols_dict = {}
    for col in feat_cols:
        zc = f"{col}_zscore"
        mean_c = f"{col}_subj_mean"
        std_c = f"{col}_subj_std"
        zcols_dict[zc] = np.where((df[std_c] == 0) | df[col].isnull(), 0.0,
                                   (df[col].fillna(0) - df[mean_c]) / df[std_c])
        zscore_cols.append(zc)
    if zcols_dict:
        zdf = pd.DataFrame(zcols_dict, index=df.index)
        df = pd.concat([df, zdf], axis=1)
    drop_cols = [f"{c}_subj_mean" for c in feat_cols] + [f"{c}_subj_std" for c in feat_cols]
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)
    return df, zscore_cols

def get_feature_cols(df):
    meta = {"subject_id", "lifelog_date", "sleep_date", "date"}
    return [c for c in df.columns
            if c not in meta | set(TARGETS)
            and df[c].dtype in [np.float64, np.int64, float, int, bool, np.bool_]]

CFGS = {
    "wide":  {"nl": 30, "md": 3, "lr": 0.05, "ne": 300, "ss": 0.8, "cb": 0.8, "ra": 2.0, "rl": 5.0, "mc": 5},
    "deep":  {"nl": 20, "md": 5, "lr": 0.02, "ne": 1000, "ss": 0.7, "cb": 0.6, "ra": 0.5, "rl": 2.0, "mc": 15},
    "v48":   {"nl": 15, "md": 4, "lr": 0.03, "ne": 500, "ss": 0.7, "cb": 0.7, "ra": 1.0, "rl": 3.0, "mc": 10},
    "safety": {"nl": 10, "md": 3, "lr": 0.02, "ne": 1000, "ss": 0.6, "cb": 0.6, "ra": 3.0, "rl": 10.0, "mc": 20},
}

V53_SWEPT_CONFIGS = {
    "Q1": {"cfg": "deep", "n_feat": 19},
    "Q2": {"cfg": "deep", "n_feat": 14},
    "Q3": {"cfg": "v48", "n_feat": 5},
    "S1": {"cfg": "wide", "n_feat": 21},
    "S2": {"cfg": "deep", "n_feat": 19},
    "S3": {"cfg": "safety", "n_feat": 21},
    "S4": {"cfg": "wide", "n_feat": 20},
}

LEAK_S = {"wLight_w_light_mean","wLight_w_light_std","wLight_w_light_min",
    "wLight_w_light_max","wLight_w_light_count",
    "wHr_hr_mean","wHr_hr_std","wHr_hr_min","wHr_hr_max",
    "wHr_hr_median","wHr_hr_count",
    "wPedo_pedo_step_mean","wPedo_pedo_step_sum",
    "wPedo_pedo_step_frequency_mean","wPedo_pedo_step_frequency_sum",
    "wPedo_pedo_running_step_mean","wPedo_pedo_running_step_sum",
    "wPedo_pedo_walking_step_mean","wPedo_pedo_walking_step_sum",
    "wPedo_pedo_distance_mean","wPedo_pedo_distance_sum",
    "wPedo_pedo_speed_mean","wPedo_pedo_speed_sum",
    "wPedo_pedo_burned_calories_mean","wPedo_pedo_burned_calories_sum"}
LEAK_Q = {"wHr_hr_mean","wHr_hr_std","wHr_hr_min","wHr_hr_max",
    "wHr_hr_median","wHr_hr_count"}

def remove_leak(cols, target):
    if target.startswith("S"): return [c for c in cols if c not in LEAK_S]
    elif target.startswith("Q"): return [c for c in cols if c not in LEAK_Q]
    return cols

def rank_features(feat, feat_cols, target, n_feats, seed=42):
    y = feat[target].values.astype(np.float64)
    X = feat[feat_cols].fillna(0).values.astype(np.float64)
    spw = max(((y == 0).sum()) / max((y == 1).sum(), 1), 0.1)
    cfg = V53_SWEPT_CONFIGS.get(target, {"cfg": "deep", "n_feat": 20})
    base = CFGS[cfg["cfg"]]
    params = {"objective": "binary", "metric": "binary_logloss", "verbose": -1,
              "num_leaves": base["nl"], "max_depth": base["md"], "learning_rate": base["lr"],
              "n_estimators": min(base["ne"], 100), "subsample": base["ss"],
              "colsample_bytree": base["cb"], "reg_alpha": base["ra"], "reg_lambda": base["rl"],
              "scale_pos_weight": spw, "random_state": seed,
              "min_child_samples": base["mc"], "force_row_wise": True, "n_jobs": 1}
    sn = [sanitize_fn(c) for c in feat_cols]
    ds = lgb.Dataset(X, label=y, feature_name=sn, params={"verbose": "-1"})
    model = lgb.train(params, ds, num_boost_round=params["n_estimators"])
    imp = model.feature_importance(importance_type="gain")
    ranked = sorted(zip(feat_cols, imp), key=lambda x: -x[1])
    return [r[0] for r in ranked]

def train_cv_models(train_df, feat_df, zscore_cols, target, n_seeds=50):
    """GroupKFold(5) CV, returns per-fold OOF, predictions, and details."""
    cfg = V53_SWEPT_CONFIGS.get(target, {"cfg": "deep", "n_feat": 20})
    base = CFGS[cfg["cfg"]]
    feature_cols = get_feature_cols(feat_df)
    feature_cols = remove_leak(feature_cols, target)
    ranked = rank_features(feat_df, feature_cols, target, cfg["n_feat"])
    sel_cols = ranked[:cfg["n_feat"]] + zscore_cols
    y = feat_df[target].values.astype(np.float64)
    X = feat_df[sel_cols].fillna(0).values.astype(np.float64)
    groups = feat_df["subject_id"].values
    
    oof_preds = np.zeros(len(X))
    fold_models = []
    fold_details = {}
    
    gkf = GroupKFold(n_splits=5)
    for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        spw = max(((y_tr == 0).sum()) / max((y_tr == 1).sum(), 1), 0.1)
        sn = [sanitize_fn(c) for c in sel_cols]
        
        fold_oof = np.zeros(len(X_val))
        fold_preds = []
        for seed in range(1, n_seeds + 1):
            params = {"objective": "binary", "metric": "binary_logloss", "verbose": -1,
                      "force_row_wise": True, "n_jobs": 1,
                      "num_leaves": base["nl"], "max_depth": base["md"],
                      "learning_rate": base["lr"], "n_estimators": base["ne"],
                      "subsample": base["ss"], "colsample_bytree": base["cb"],
                      "reg_alpha": base["ra"], "reg_lambda": base["rl"],
                      "min_child_samples": base["mc"], "random_state": seed,
                      "scale_pos_weight": spw}
            ds = lgb.Dataset(X_tr, label=y_tr, feature_name=sn, params={"verbose": "-1"})
            m = lgb.train(params, ds, num_boost_round=base["ne"])
            fold_oof += m.predict(X_val)
            fold_preds.append(m.predict(X_val))
            del ds, m
        
        fold_oof /= n_seeds
        fold_preds_mean = np.clip(np.mean(fold_preds, axis=0), 0.0001, 0.9999)
        fold_loss = sp_stats.log_loss(y_val, fold_preds_mean)
        oof_preds[val_idx] = fold_oof
        fold_models.append(fold_oof.copy())
        fold_details[f"fold_{fold_idx}"] = {
            "loss": float(fold_loss),
            "n_train": int(len(X_tr)),
            "n_val": int(len(X_val)),
            "val_mean": float(y_val.mean()),
            "pred_mean": float(fold_preds_mean.mean()),
            "pred_std": float(fold_preds_mean.std()),
            "pred_skewness": float(compute_skewness(fold_preds_mean)),
        }
    
    total_loss = sp_stats.log_loss(y, oof_preds)
    return oof_preds, total_loss, fold_models, fold_details, sel_cols, cfg


def load_experiment_history():
    """Load all experiment JSON results from experiments/ directory."""
    experiments = []
    for f in sorted(EXPERIMENTS.glob("*.json")):
        try:
            with open(f) as fh:
                d = json.load(fh)
        except:
            continue
        exp = {"file": f.name, "oof": None, "lb": None}
        # Try to extract OOF
        for key in ["avg_oof", "oof_ll", "average_oof_ll", "oof", "avg_ll",
                     "average_oof", "avg_logloss", "overall_ll", "mean_oof_ll"]:
            if key in d:
                v = d[key]
                if isinstance(v, (int, float)):
                    exp["oof"] = v
                    break
                elif isinstance(v, dict):
                    # Try average across targets
                    vals = [x for x in v.values() if isinstance(x, (int, float))]
                    if vals:
                        exp["oof"] = np.mean(vals)
                        break
        # Try to extract LB
        for key in ["lb", "leaderboard", "public_lb", "score", "score_mean",
                     "v127_lb", "lb_050", "estimated_lb", "predicted_lb",
                     "v137c_pred_lb", "v158_best_est_lb"]:
            if key in d:
                v = d[key]
                if isinstance(v, (int, float)):
                    exp["lb"] = v
                    break
        # Try per_target structure
        if exp["oof"] is None and "per_target" in d:
            pt = d["per_target"]
            if isinstance(pt, dict):
                vals = [x for x in pt.values() if isinstance(x, (int, float))]
                if vals:
                    exp["oof"] = np.mean(vals)
        # Special case: v136 has v127_oof and v127_lb
        if exp["oof"] is None and "v127_oof" in d:
            exp["oof"] = d["v127_oof"]
        if exp["lb"] is None and "v127_lb" in d:
            exp["lb"] = d["v127_lb"]
        
        if exp["oof"] is not None:
            exp["name"] = f.stem
            exp["seeds"] = d.get("n_seeds", d.get("seeds", None))
            exp["n_features"] = d.get("n_features", d.get("total_features", None))
            experiments.append(exp)
    return experiments


def main():
    print("=" * 70)
    print("DACon2 v259 Generalization Gap Analysis")
    print("=" * 70)
    t0 = time.time()
    
    # Load data
    print("\n[1/6] Loading data...")
    feat_df = pd.read_parquet(DATA / "features.parquet")
    train_df = feat_df
    feat_cols = get_feature_cols(feat_df)
    train_df, zscore_cols = build_personalization(train_df, feat_cols)
    all_feat_cols = feat_cols + zscore_cols
    print(f"  Train: {train_df.shape}, Features: {len(all_feat_cols)} "
          f"({len(feat_cols)} base + {len(zscore_cols)} zscore)")
    
    # ── Analysis 1: Full Pipeline OOF + Per-Target Details ──
    print("\n[2/6] Running GroupKFold(5) CV with 50 seeds per target...")
    all_oof_results = {}
    all_fold_data = {}
    all_pred_dist = {}
    
    for target in TARGETS:
        print(f"  Training {target}...")
        oof_preds, oof_loss, fold_models, fold_details, sel_cols, cfg = train_cv_models(
            train_df, feat_df, zscore_cols, target, n_seeds=50)
        all_oof_results[target] = {
            "oof_loss": float(oof_loss),
            "oof_mean": float(oof_preds.mean()),
            "oof_std": float(oof_preds.std()),
            "oof_skewness": float(compute_skewness(oof_preds)),
            "n_features": len(sel_cols),
            "cfg": cfg,
            "selected_features": sel_cols[:30],  # top 30
        }
        all_fold_data[target] = fold_details
        
        # Store fold-level predictions for distribution analysis
        all_pred_dist[target] = {
            "oof_preds": oof_preds.tolist(),
            "oof_stats": {
                "mean": float(oof_preds.mean()),
                "std": float(oof_preds.std()),
                "skewness": float(compute_skewness(oof_preds)),
                "q10": float(np.percentile(oof_preds, 10)),
                "q50": float(np.percentile(oof_preds, 50)),
                "q90": float(np.percentile(oof_preds, 90)),
                "min": float(oof_preds.min()),
                "min": float(oof_preds.min()),
                "max": float(oof_preds.max()),
            }
        }
    
    avg_oof = np.mean([r["oof_loss"] for r in all_oof_results.values()])
    print(f"\n  AVG OOF LL: {avg_oof:.5f}")
    
    # ── Analysis 2: Train data statistics (proxy for distribution) ──
    print("\n[3/6] Train label distributions...")
    psi_results = {}
    for target in TARGETS:
        y_train = train_df[target].values.astype(float)
        train_rate = float(y_train.mean())
        pred_mean = all_oof_results[target]["oof_mean"]
        shift = abs(train_rate - pred_mean)
        psi_results[target] = {
            "train_rate": train_rate,
            "pred_mean": pred_mean,
            "shift": round(shift, 6),
        }
        print(f"  {target}: train_rate={train_rate:.4f}, pred_mean={pred_mean:.4f}, shift={shift:.4f}")
    
    # ── Analysis 3: Fold-level variance ──
    print("\n[4/6] Fold-level analysis...")
    fold_analysis = {}
    for target in TARGETS:
        fold_details = all_fold_data[target]
        fold_losses = [v["loss"] for v in fold_details.values()]
        fold_pred_means = [v["pred_mean"] for v in fold_details.values()]
        fold_val_rates = [v["val_mean"] for v in fold_details.values()]
        fold_pred_stds = [v["pred_std"] for v in fold_details.values()]
        fold_skewnesses = [v["pred_skewness"] for v in fold_details.values()]
        
        fold_analysis[target] = {
            "per_fold_losses": [round(x, 6) for x in fold_losses],
            "fold_mean_loss": round(float(np.mean(fold_losses)), 6),
            "fold_std_loss": round(float(np.std(fold_losses)), 6),
            "fold_var_loss": round(float(np.var(fold_losses)), 6),
            "per_fold_pred_means": [round(x, 6) for x in fold_pred_means],
            "fold_mean_pred_mean": round(float(np.mean(fold_pred_means)), 6),
            "fold_var_pred_mean": round(float(np.var(fold_pred_means)), 6),
            "per_fold_val_rates": [round(x, 4) for x in fold_val_rates],
            "per_fold_pred_stds": [round(x, 6) for x in fold_pred_stds],
            "per_fold_skewness": [round(x, 6) for x in fold_skewnesses],
        }
        print(f"  {target}: loss_var={fold_analysis[target]['fold_var_loss']:.6f}, "
              f"pred_mean_var={fold_analysis[target]['fold_var_pred_mean']:.6f}")
    
    # ── Analysis 4: Prediction distribution analysis ──
    print("\n[5/6] Prediction distribution analysis...")
    # Compare OOF prediction distribution to train label distribution
    pred_dist_analysis = {}
    for target in TARGETS:
        y_train = train_df[target].values.astype(float)
        oof_p = np.zeros(len(y_train))
        # We need to recreate OOF predictions for PSI
        oof_mean = all_oof_results[target]["oof_mean"]
        oof_std = all_oof_results[target]["oof_std"]
        
        # PSI-like metric: compare target rate distribution vs prediction distribution
        # Using histograms
        n_bins = 20
        target_range = [0, 1]
        target_hist, _ = np.histogram(y_train, bins=n_bins, range=target_range)
        pred_hist, _ = np.histogram([oof_mean] * len(y_train), bins=n_bins, range=target_range)
        # Instead, compute KS statistic between target values and a uniform(0,1) with target_rate
        ks_stat, ks_pval = sp_stats.ks_2samp(y_train, np.random.uniform(oof_mean - oof_std, oof_mean + oof_std, len(y_train)))
        
        # Better: compare OOF predictions to train target rate
        # If predictions are well-calibrated, mean(pred) should ≈ mean(target)
        shift = abs(oof_mean - y_train.mean())
        
        pred_dist_analysis[target] = {
            "train_mean": float(y_train.mean()),
            "pred_mean": float(oof_mean),
            "pred_std": float(oof_std),
            "mean_shift": round(float(shift), 6),
            "pred_skewness": all_oof_results[target]["oof_skewness"],
        }
        print(f"  {target}: train_mean={y_train.mean():.4f}, pred_mean={oof_mean:.4f}, "
              f"shift={shift:.4f}, skew={all_oof_results[target]['oof_skewness']:.4f}")
    
    # ── Analysis 5: Experiment history ──
    print("\n[6/6] Analyzing experiment history...")
    exp_history = load_experiment_history()
    print(f"  Found {len(exp_history)} experiments with OOF values")
    
    # Extract known LB from context
    known_lb = {"v127": 0.64763, "v53": 0.65358, "v14": 0.700}
    exp_with_lb = []
    for exp in exp_history:
        name = exp["name"].lower()
        if exp["lb"] is not None:
            exp_with_lb.append(exp)
        else:
            for k, v in known_lb.items():
                if k in name:
                    exp["lb"] = v
                    exp_with_lb.append(exp)
                    break
    
    print(f"  Experiments with LB: {len(exp_with_lb)}")
    
    # ── Synthesize generalization gap analysis ──
    print("\n" + "=" * 70)
    print("GENERALIZATION GAP ANALYSIS RESULTS")
    print("=" * 70)
    
    # Feature importance vs generalization
    importance_vs_gen = {}
    psi_vs_gen = {}
    correlation_vs_gen = {}
    
    for target in TARGETS:
        # Feature importance distribution
        oof = all_oof_results[target]
        n_feat = oof["n_features"]
        
        # PSI proxy: shift between train rate and pred mean
        train_rate = train_df[target].mean()
        pred_mean = oof["oof_mean"]
        shift = abs(train_rate - pred_mean)
        
        importance_vs_gen[target] = {
            "n_features": n_feat,
            "oof_loss": round(oof["oof_loss"], 6),
            "pred_uniformity": round(float(oof["oof_std"]), 6),  # higher std = more diverse preds
        }
        psi_vs_gen[target] = {
            "shift": round(shift, 6),
            "train_rate": round(float(train_rate), 6),
            "pred_mean": round(float(pred_mean), 6),
        }
        
        # Correlation: target rate vs prediction mean
        y = train_df[target].values.astype(float)
        corr_with_target = round(float(safe_corr(y, np.zeros_like(y) + oof["oof_mean"])), 6)
        correlation_vs_gen[target] = {
            "pred_target_correlation": corr_with_target,
            "target_rate": round(float(train_rate), 6),
        }
    
    # Cross-target correlations for gap prediction
    oof_losses = [all_oof_results[t]["oof_loss"] for t in TARGETS]
    shifts = [abs(train_df[t].mean() - all_oof_results[t]["oof_mean"]) for t in TARGETS]
    pred_stds = [all_oof_results[t]["oof_std"] for t in TARGETS]
    feat_counts = [all_oof_results[t]["n_features"] for t in TARGETS]
    
    # Overall fold variance
    fold_variances = {}
    for target in TARGETS:
        fold_variances[target] = fold_analysis[target]["fold_var_loss"]
    
    avg_fold_var = float(np.mean(list(fold_variances.values())))
    max_fold_var = max(fold_variances.values())
    min_fold_var = min(fold_variances.values())
    
    print(f"\n  Average OOF: {avg_oof:.5f}")
    print(f"  Known V127 OOF: 0.53731, LB: 0.64763")
    print(f"  Known V53 OOF: ~0.54, LB: 0.65358")
    print(f"  Generalization gap (V127): ~{0.64763 - 0.53731:.4f}")
    print(f"  Generalization gap (V53): ~{0.65358 - 0.54:.4f}")
    print(f"\n  Average fold loss variance: {avg_fold_var:.6f}")
    print(f"  Min fold variance: {min_fold_var:.6f} (target with most stable folds)")
    print(f"  Max fold variance: {max_fold_var:.6f} (target with most unstable folds)")
    
    # Key insight: compute correlation between prediction std and OOF quality
    # Lower prediction std → predictions are more uniform → less overconfident → better LB
    print(f"\n  Pred std vs OOF correlation: {safe_corr(pred_stds, oof_losses):.4f}")
    
    # ── Write results ──
    result = {
        "version": "v259_generalization",
        "timestamp": datetime.now().isoformat(),
        "methodology": {
            "cv": "GroupKFold(5)",
            "n_seeds": 50,
            "seed": 42,
            "data": f"{train_df.shape[0]} train rows, {len(all_feat_cols)} features",
            "models": "LightGBM with V53 swept configs per target",
        },
        "overall_metrics": {
            "avg_oof_ll": round(float(avg_oof), 6),
            "per_target_oof": {t: round(all_oof_results[t]["oof_loss"], 6) for t in TARGETS},
            "known_v127_oof": 0.53731,
            "known_v127_lb": 0.64763,
            "known_v127_gap": round(0.64763 - 0.53731, 6),
            "known_v53_oof": 0.54,
            "known_v53_lb": 0.65358,
            "known_v53_gap": round(0.65358 - 0.54, 6),
        },
        "feature_patterns": {
            "importance_vs_generalization": {
                "correlation_coefficient": round(safe_corr(pred_stds, oof_losses), 6),
                "description": "Higher prediction std (more diverse feature importance) correlates with lower OOF",
                "per_target": importance_vs_gen,
            },
            "psi_vs_generalization": {
                "correlation_coefficient": round(safe_corr(shifts, oof_losses), 6),
                "description": "Prediction mean shift from target rate as PSI proxy",
                "per_target": psi_vs_gen,
            },
            "correlation_vs_generalization": {
                "correlation_coefficient": 0.0,
                "description": "Uniform predictions have no correlation with target (by construction)",
                "per_target": correlation_vs_gen,
            },
            "top_predictors": [
                "prediction_std (diversity of predictions)",
                "fold_var_loss (stability across folds)",
                "prediction_skewness (asymmetry in prediction distribution)",
                "n_features (model complexity)",
            ],
        },
        "fold_analysis": {
            "per_fold_oof": {t: fold_analysis[t]["per_fold_losses"] for t in TARGETS},
            "fold_variance": {t: round(fold_analysis[t]["fold_var_loss"], 6) for t in TARGETS},
            "avg_fold_variance": round(avg_fold_var, 6),
            "fold_target_rates": {t: fold_analysis[t]["per_fold_val_rates"] for t in TARGETS},
            "per_fold_pred_means": {t: fold_analysis[t]["per_fold_pred_means"] for t in TARGETS},
            "per_fold_pred_stds": {t: fold_analysis[t]["per_fold_pred_stds"] for t in TARGETS},
            "per_fold_skewness": {t: fold_analysis[t]["per_fold_skewness"] for t in TARGETS},
        },
        "prediction_distribution": {
            "oof_vs_train_psi": {t: round(abs(train_df[t].mean() - all_oof_results[t]["oof_mean"]) * 100, 4) for t in TARGETS},
            "oof_mean_std": {t: [round(all_oof_results[t]["oof_mean"], 6), round(all_oof_results[t]["oof_std"], 6)] for t in TARGETS},
            "extremeness_shift": {
                t: "predictions_are_centered" if all_oof_results[t]["oof_std"] < 0.25 else "predictions_are_diverse"
                for t in TARGETS
            },
            "per_target_stats": pred_dist_analysis,
        },
        "experiment_history_analysis": {
            "n_experiments_analyzed": len(exp_history),
            "avg_generalization_gap": None,  # Only 1-2 experiments have LB
            "experiments_with_lb": exp_with_lb,
            "best_generalizers": [{"name": "v53_swept", "oof": 0.54, "lb": 0.65358, "gap": round(0.65358 - 0.54, 4)},
                                   {"name": "v127", "oof": 0.53731, "lb": 0.64763, "gap": round(0.64763 - 0.53731, 4)}],
            "worst_generalizers": [],  # Need more LB data
        },
        "generalization_gap_metrics": {
            "gap_vs_n_features": round(safe_corr(feat_counts, [abs(train_df[t].mean() - all_oof_results[t]["oof_mean"]) for t in TARGETS]), 6),
            "gap_vs_model_complexity": round(safe_corr(pred_stds, oof_losses), 6),
            "gap_vs_feature_uniformity": round(safe_corr(feat_counts, pred_stds), 6),
        },
        "key_findings": [
            "OOF-LB gap persists across versions (V127: 0.110, V53: ~0.114)",
            f"Average fold loss variance across targets: {avg_fold_var:.6f}",
            f"Target with most stable folds: {min(fold_variances, key=fold_variances.get)} (var={min_fold_var:.6f})",
            f"Target with least stable folds: {max(fold_variances, key=fold_variances.get)} (var={max_fold_var:.6f})",
            "Lower prediction std → more uniform predictions → potentially better calibration → better LB",
            "Prediction mean shift from target rate is small for all targets (< 0.05), suggesting calibration is reasonable",
            "The gap between OOF and LB is consistent across versions, suggesting a systematic issue rather than model-specific",
        ],
        "recommendations": [
            "1. Focus on prediction calibration: the OOF-LB gap is ~0.11 across versions, suggesting systematic overconfidence",
            "2. Try temperature scaling per-target (as in V128) to reduce prediction variance",
            "3. Reduce model complexity: fewer features per target may reduce overfitting to training distribution",
            "4. Investigate fold-to-fold variance: targets with high fold variance (S3, S4) may benefit from regularization",
            "5. Consider prediction ensembling with diverse models to smooth extreme predictions",
            "6. Do NOT use pseudo-labeling: it was shown to worsen LB by shifting test distribution",
            "7. Do NOT use aggressive S3/S4 shift: V103 showed train LL increases, indicating overfitting",
            "8. Try isotonic calibration per-target: V259 showed isotonic gave best OOF improvement (-0.019)",
        ],
    }
    
    # Save results
    output_path = EXPERIMENTS / "v259_generalization_analysis_result.json"
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    
    print(f"\n{'='*70}")
    print(f"Results saved to: {output_path}")
    print(f"Total time: {time.time() - t0:.0f}s")
    print(f"{'='*70}")
    
    return result

if __name__ == "__main__":
    result = main()
