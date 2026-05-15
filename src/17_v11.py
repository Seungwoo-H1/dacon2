"""17_v11.py — V11: Extended features + careful selection vs V10 baseline."""
import sys, json, warnings, logging, re, gc
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupKFold
import lightgbm as lgb
warnings.filterwarnings('ignore')
gc.collect()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", stream=sys.stdout)
log = logging.getLogger(__name__)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))
from config import TARGETS, DATA_PROCESSED, MODEL_DIR

V10 = dict(Q1=0.6338, Q2=0.6034, Q3=0.6119, S1=0.5680, S2=0.6022, S3=0.5835, S4=0.6240)
V10_AVG = 0.6038
META = {"subject_id", "lifelog_date", "sleep_date", "date"}
LEAK_S = {
    'wLight_w_light_mean','wLight_w_light_std','wLight_w_light_min','wLight_w_light_max','wLight_w_light_count',
    'wHr_hr_mean','wHr_hr_std','wHr_hr_min','wHr_hr_max','wHr_hr_median','wHr_hr_count',
    'wPedo_pedo_step_mean','wPedo_pedo_step_sum','wPedo_pedo_step_frequency_mean',
    'wPedo_pedo_step_frequency_sum','wPedo_pedo_running_step_mean','wPedo_pedo_running_step_sum',
    'wPedo_pedo_walking_step_mean','wPedo_pedo_walking_step_sum','wPedo_pedo_distance_mean',
    'wPedo_pedo_distance_sum','wPedo_pedo_speed_mean','wPedo_pedo_speed_sum',
    'wPedo_pedo_burned_calories_mean','wPedo_pedo_burned_calories_sum',
}
LEAK_Q = {'wHr_hr_mean','wHr_hr_std','wHr_hr_min','wHr_hr_max','wHr_hr_median','wHr_hr_count'}
CFGS = [
    dict(nl=10,md=3,lr=0.02,ne=200,ss=0.7,cst=0.7,ra=1.0,rl=3.0,mc=10),
    dict(nl=6,md=2,lr=0.02,ne=150,ss=0.5,cst=0.5,ra=5.0,rl=10.0,mc=20),
    dict(nl=15,md=4,lr=0.03,ne=300,ss=0.7,cst=0.7,ra=1.0,rl=3.0,mc=10),
]
SEEDS = [42,123,456,789,1024,1337,2048,3037,4096,5001]

def sanitize(name):
    return re.sub(r'[^a-zA-Z0-9_]', '_', name)

def main():
    log.info("=" * 60)
    log.info("V11 Extended Features")
    log.info("=" * 60)
    
    df = pd.read_parquet(DATA_PROCESSED / "features_v11.parquet")
    sid = df['subject_id'].values
    base = [c for c in df.columns if c not in META | set(TARGETS)
            and not c.endswith('_zscore')
            and df[c].dtype in [np.float64, np.int64, float, int, bool]]
    X_all = df[base].fillna(0).values
    log.info(f"Features: {len(base)}, Shape: {X_all.shape}")
    
    col2idx = {c: i for i, c in enumerate(base)}
    target_results = {}
    
    for t in TARGETS:
        leak = LEAK_S if t.startswith('S') else LEAK_Q
        leak_free = [c for c in base if c not in leak]
        leak_idx = [col2idx[c] for c in leak_free]
        y = df[t].values
        X_leak = X_all[:, leak_idx]
        
        n_pos = max((y == 1).sum(), 1)
        n_neg = (y == 0).sum()
        spw = n_neg / n_pos
        
        # Rank features
        ds = lgb.Dataset(X_leak, label=y)
        prms = dict(objective='binary', metric='binary_logloss', verbose=-1,
                    num_leaves=10, max_depth=3, learning_rate=0.03, n_estimators=30,
                    subsample=0.7, colsample_bytree=0.7, reg_alpha=1.0, reg_lambda=3.0,
                    scale_pos_weight=spw, random_state=42, min_child_samples=10,
                    force_row_wise=True, n_jobs=4)
        mdl = lgb.train(prms, ds, num_boost_round=30)
        imp = mdl.feature_importance(importance_type="gain")
        ranked = sorted(zip(leak_free, imp), key=lambda x: -x[1])
        log.info(f"{t}: {len(leak_free)} leak-free, top5={[r[0] for r in ranked[:5]]}")
        
        # Delete ranking model to free memory
        del mdl, ds
        gc.collect()
        
        # Prepare candidate feature sets
        candidates = {}
        for n in [5, 10, 15, 20, 30, 40]:
            sel = [r[0] for r in ranked[:n] if r[1] > 0]
            sel_idx = [col2idx[c] for c in sel]
            candidates[n] = (sel, X_all[:, sel_idx])
        
        # GroupKFold splits (on leak-free features, same data)
        gkf = GroupKFold(n_splits=5)
        splits = list(gkf.split(X_leak, y, sid))
        
        # Tune
        log.info(f"  {t}: tuning...")
        best_cv = float('inf')
        best = None
        
        for n, (names, Xs) in candidates.items():
            snames = [sanitize(c) for c in names]
            for ci, cfg in enumerate(CFGS):
                n_feat = len(snames)
                oof = np.zeros((len(y), len(SEEDS)))
                
                for si, seed in enumerate(SEEDS):
                    prms = dict(objective='binary', metric='binary_logloss', verbose=-1,
                                num_leaves=cfg['nl'], max_depth=cfg['md'],
                                learning_rate=cfg['lr'], n_estimators=cfg['ne'],
                                subsample=cfg['ss'], colsample_bytree=cfg['cst'],
                                reg_alpha=cfg['ra'], reg_lambda=cfg['rl'],
                                min_child_samples=cfg['mc'],
                                force_row_wise=True, n_jobs=4,
                                scale_pos_weight=spw, random_state=seed)
                    
                    for fi, (tr_i, va_i) in enumerate(splits):
                        tr = lgb.Dataset(Xs[tr_i], label=y[tr_i])
                        tr.set_feature_name(snames)
                        va = lgb.Dataset(Xs[va_i], label=y[va_i])
                        va.set_feature_name(snames)
                        m = lgb.train(prms, tr, num_boost_round=cfg['ne'],
                                      valid_sets=[va],
                                      callbacks=[lgb.early_stopping(50, verbose=False),
                                                 lgb.log_evaluation(0)])
                        oof[va_i, si] = m.predict(Xs[va_i])
                        del m, tr, va
                
                oof_avg = oof.mean(1)
                cv = log_loss(y, oof_avg, labels=[0, 1])
                
                if cv < best_cv:
                    best_cv = cv
                    best = dict(n=n, cfg=cfg, cv=cv, oof=oof, sel=names)
                    log.info(f"    n={n} c={ci}: {cv:.4f}")
                
                del oof
                gc.collect()
        
        if best is None:
            continue
        
        # Calibrate
        cal = np.clip(best['oof'].mean(1) + (y.mean() - best['oof'].mean(1).mean()), 0.0001, 0.9999)
        cal_loss = log_loss(y, cal, labels=[0, 1])
        v10 = V10[t]
        d = cal_loss - v10
        log.info(f"  {t}: BEST n={best['n']} V10={v10:.4f} V11={cal_loss:.4f} {d:+.4f} {'✅' if d<0 else '❌'}")
        
        target_results[t] = dict(cal_oof=cal_loss, v10=v10, delta=d, n_features=best['n'])
        del best, X_leak, y
        gc.collect()
    
    # Summary
    log.info(f"\n{'='*60}")
    log.info("V11 FINAL")
    log.info(f"{'Target':<6} {'V10':<10} {'V11':<10} {'Δ':<8} {'Win'}")
    log.info("-" * 50)
    
    avg11 = 0; cnt = 0
    for t in TARGETS:
        if t not in target_results:
            continue
        v10 = V10[t]
        v11 = target_results[t]['cal_oof']
        d = v11 - v10
        w = "V11" if d < 0 else "V10"
        log.info(f"{t:<6} {v10:<10.4f} {v11:<10.4f} {d:+.4f} {w}")
        avg11 += v11; cnt += 1
    
    avg11 /= cnt if cnt else 1
    d_avg = avg11 - V10_AVG
    log.info("-" * 50)
    log.info(f"{'AVG':<6} {V10_AVG:<10.4f} {avg11:<10.4f} {d_avg:+.4f} {'🎉 V11!' if d_avg<0 else 'V10'}")
    
    # Save
    for name, data in [("final", target_results)]:
        with open(MODEL_DIR / f'v11_{name}_results.json', 'w') as f:
            json.dump(data, f, indent=2, default=float)
    
    meta = dict(version='v11', avg_v10=float(V10_AVG), avg_v11=float(avg11),
                beat_v10=bool(d_avg < 0),
                per_target={t: dict(v10=target_results[t]['v10'], v11=target_results[t]['cal_oof'],
                                   delta=target_results[t]['delta'], n_features=target_results[t]['n_features'])
                           for t in target_results})
    with open(MODEL_DIR / 'v11_meta.json', 'w') as f:
        json.dump(meta, f, indent=2, default=float)
    log.info("Done!")

if __name__ == "__main__":
    main()
