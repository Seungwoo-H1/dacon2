"""V257: Ensemble Weight Optimization + Distribution Shift Analysis

Hypothesis 1: Optimal blend of base/pairwise/transformed feature sets per target
can improve over V127's fixed 0.35/0.25/0.40 weights.

Hypothesis 2: Train/test distribution analysis (PSI, adversarial validation)
will reveal leaky/drift features that, if removed, improve LB generalization.

Data: features_clean_v60.parquet (275 base features + zscore personalization)
"""
import logging, sys, gc, re, json, warnings, time, copy
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from scipy.optimize import minimize
import lightgbm as lgb

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", stream=sys.stdout)
log = logging.getLogger(__name__)

ROOT = Path('/home/mwoo423/projects/dacon2')
DATA = ROOT / 'data_processed'
EXPERIMENTS = ROOT / 'experiments'
SUBMIT = ROOT / 'submissions'
TARGETS = ['Q1','Q2','Q3','S1','S2','S3','S4']
META = {'subject_id','lifelog_date','sleep_date','date'}
SEEDS = [42, 7, 999, 777]

def sanitize(n): return re.sub(r'[^a-zA-Z0-9_]','_',n)
def get_feat_cols(df):
    return [c for c in df.columns if c not in META | set(TARGETS) 
            and c not in ['subject_id','lifelog_date','sleep_date','date']
            and df[c].dtype in [np.float64,np.int64,float,int,bool,np.bool_]]

def add_zscore(df, feat_cols, stats=None, for_test=False):
    df = df.copy()
    all_stats = {}
    zcols = []
    for c in feat_cols:
        vals = df[c].fillna(0)
        grp = vals.groupby(df['subject_id']).agg(mean='mean', std='std').reset_index()
        grp.columns = ['subject_id', f'{c}_subj_mean', f'{c}_subj_std']
        df = df.merge(grp, on='subject_id', how='left')
        sm = df[f'{c}_subj_mean']; ss = df[f'{c}_subj_std']
        if not for_test: all_stats[c] = {'mean': sm, 'std': ss}
        mask = (ss == 0) | df[c].isnull()
        df[f'{c}_z'] = np.where(mask, 0.0, (df[c].fillna(0) - sm) / np.maximum(ss, 1e-8))
        zcols.append(f'{c}_z')
        gc.collect()
    return df, zcols, all_stats

def mean_match(pred, target_mean):
    shift = target_mean - pred.mean()
    return np.clip(pred + shift, 0.0001, 0.9999)

def rank_features(df, feat_cols, target, seed=42):
    y = df[target].values.astype(np.float64)
    X = df[feat_cols].fillna(0).values.astype(np.float64)
    spw = max(((y==0).sum()) / max((y==1).sum(), 1), 0.1)
    params = {'objective':'binary','metric':'binary_logloss','verbose':-1,
              'num_leaves':15,'max_depth':4,'learning_rate':0.03,'n_estimators':50,
              'subsample':0.7,'colsample_bytree':0.7,'reg_alpha':1.0,'reg_lambda':3.0,
              'scale_pos_weight':spw,'random_state':seed,'min_child_samples':10,'force_row_wise':True,'n_jobs':1}
    sn = [sanitize(c) for c in feat_cols]
    ds = lgb.Dataset(X, label=y, feature_name=sn, params={'verbose':'-1'})
    model = lgb.train(params, ds, num_boost_round=50)
    imp = model.feature_importance(importance_type='gain')
    ranked = sorted(zip(feat_cols, imp), key=lambda x:-x[1])
    del model, ds; gc.collect()
    return [r[0] for r in ranked]

def build_base(df, stats=None, for_test=False):
    fc = get_feat_cols(df)
    df, zcols, all_stats = add_zscore(df, fc, stats, for_test)
    return df, fc + zcols, all_stats if not for_test else None

def build_pairwise(df):
    df = df.copy()
    fc = get_feat_cols(df)
    # Add pairwise interactions
    added = []
    for i in range(min(len(fc), 8)):
        for j in range(i+1, min(len(fc), 8)):
            f1, f2 = fc[i], fc[j]
            v1, v2 = df[f1].fillna(0).values, df[f2].fillna(0).values
            df[f'{f1}_x_{f2}'] = v1 * v2
            df[f'{f1}_diff_{f2}'] = np.abs(v1 - v2)
            added.extend([f'{f1}_x_{f2}', f'{f1}_diff_{f2}'])
    return df, fc + added

def build_transformed(df):
    df = df.copy()
    fc = get_feat_cols(df)
    added = []
    for f in fc[:20]:
        vals = df[f].fillna(0).values
        df[f'{f}_log'] = np.sign(vals) * np.log1p(np.abs(vals))
        df[f'{f}_sqrt'] = np.sign(vals) * np.sqrt(np.abs(vals))
        df[f'{f}_abs'] = np.abs(vals)
        added.extend([f'{f}_log', f'{f}_sqrt', f'{f}_abs'])
    return df, fc + added

def train_cv(df, df_test, sel_cols, y, seeds, cfg, n_folds=5):
    gkf = GroupKFold(n_splits=n_folds)
    oof = np.zeros((len(y), len(seeds)))
    tp = np.zeros((len(df_test), len(seeds)))
    spw = max(((y==0).sum()) / max((y==1).sum(), 1), 0.1)
    X_full = df[sel_cols].fillna(0).values.astype(np.float64)
    X_test = df_test[sel_cols].fillna(0).values.astype(np.float64)
    sn = [sanitize(c) for c in sel_cols]
    for si, seed in enumerate(seeds):
        cfg_full = {
            'objective':'binary','metric':'binary_logloss','verbose':-1,'force_row_wise':True,'n_jobs':1,
            'num_leaves':cfg['nl'],'max_depth':cfg['md'],'learning_rate':cfg['lr'],'n_estimators':cfg['ne'],
            'subsample':cfg['ss'],'colsample_bytree':cfg['cb'],'reg_alpha':cfg['ra'],'reg_lambda':cfg['rl'],
            'min_child_samples':cfg['mc'],'random_state':seed,'scale_pos_weight':spw,
        }
        for tr_i, va_i in gkf.split(df, y, df['subject_id']):
            ds = lgb.Dataset(X_full[tr_i], label=y[tr_i], feature_name=sn, params={'verbose':'-1'})
            vd = lgb.Dataset(X_full[va_i], label=y[va_i], feature_name=sn, reference=ds, params={'verbose':'-1'})
            m = lgb.train(cfg_full, ds, num_boost_round=cfg['ne'], valid_sets=[vd],
                         callbacks=[lgb.early_stopping(50,verbose=False), lgb.log_evaluation(0)])
            oof[va_i, si] = m.predict(X_full[va_i])
            tp[:, si] = m.predict(X_test)
            del ds, vd, m; gc.collect()
    return np.clip(oof, 0.0001, 0.9999), np.clip(tp, 0.0001, 0.9999)

# V127 ensemble structure
CFGS = {
    'wide':   {'nl':30,'md':3,'lr':0.05,'ne':300,'ss':0.8,'cb':0.8,'ra':2.0,'rl':5.0,'mc':5},
    'deep':   {'nl':20,'md':5,'lr':0.02,'ne':1000,'ss':0.7,'cb':0.6,'ra':0.5,'rl':2.0,'mc':15},
    'v48':    {'nl':15,'md':4,'lr':0.03,'ne':500,'ss':0.7,'cb':0.7,'ra':1.0,'rl':3.0,'mc':10},
    'safety': {'nl':10,'md':3,'lr':0.02,'ne':1000,'ss':0.6,'cb':0.6,'ra':3.0,'rl':10.0,'mc':20},
}

V53_SWEEP = {
    'Q1': {'cfg': 'deep', 'n_feat': 19},
    'Q2': {'cfg': 'deep', 'n_feat': 14},
    'Q3': {'cfg': 'v48', 'n_feat': 11},
    'S1': {'cfg': 'wide', 'n_feat': 21},
    'S2': {'cfg': 'deep', 'n_feat': 19},
    'S3': {'cfg': 'safety','n_feat': 23},
    'S4': {'cfg': 'wide', 'n_feat': 20},
}

# ============================================================
# Load data
# ============================================================
log.info("Loading data...")
feat = pd.read_parquet(DATA / "features_clean_v60.parquet")
feat_test = pd.read_parquet(DATA / "test_features_clean_v60.parquet")
for df in [feat, feat_test]:
    for c in ['sleep_date','lifelog_date','date']:
        if c in df.columns: df[c] = pd.to_datetime(df[c]).dt.strftime('%Y-%m-%d')

y_train = {t: feat[t].values for t in TARGETS}
train_rates = {t: feat[t].mean() for t in TARGETS}

# ============================================================
# Part 1: Ensemble Weight Optimization
# ============================================================
log.info("\n" + "="*70)
log.info("PART 1: ENSEMBLE WEIGHT OPTIMIZATION")
log.info("="*70)

# Build 3 feature pools
log.info("Building feature pools...")
feat_base, base_cols, stats_base = build_base(feat)
feat_test_base, _, _ = build_base(feat_test, stats_base, for_test=True)
feat_pair, pair_cols = build_pairwise(feat.copy())
feat_test_pair, _ = build_pairwise(feat_test.copy())
feat_trans, trans_cols = build_transformed(feat.copy())
feat_test_trans, _ = build_transformed(feat_test.copy())

# For each target, build 3 model pools (base/pair/trans × cfg) and get OOF
all_model_oofs = {}  # target -> model_name -> oof_array
all_model_test = {}  # target -> model_name -> test_array

# V127 configs per target
for target in TARGETS:
    log.info(f"\n--- {target} ---")
    t_cfg = V53_SWEEP[target]
    cfg_name = t_cfg['cfg']
    n_feat = t_cfg['n_feat']
    cfg = CFGS[cfg_name]
    y = y_train[target]
    
    # Feature ranking for each pool
    log.info("  Ranking features...")
    base_ranked = rank_features(feat_base, base_cols, target)
    pair_ranked = rank_features(feat_pair, pair_cols, target)
    trans_ranked = rank_features(feat_trans, trans_cols, target)
    sel_base = base_ranked[:n_feat]
    sel_pair = pair_ranked[:n_feat]
    sel_trans = trans_ranked[:n_feat]
    log.info(f"    top-{n_feat} base: {sel_base[:3]}...")
    log.info(f"    top-{n_feat} pair: {sel_pair[:3]}...")
    log.info(f"    top-{n_feat} trans: {sel_trans[:3]}...")
    
    # Train models
    log.info("  Training base models...")
    oof_b, tp_b = train_cv(feat_base, feat_test_base, sel_base, y, SEEDS, cfg)
    oof_b = oof_b.mean(axis=1); tp_b = tp_b.mean(axis=1)
    
    log.info("  Training pair models...")
    oof_p, tp_p = train_cv(feat_pair, feat_test_pair, sel_pair, y, SEEDS, cfg)
    oof_p = oof_p.mean(axis=1); tp_p = tp_p.mean(axis=1)
    
    log.info("  Training trans models...")
    oof_t, tp_t = train_cv(feat_trans, feat_test_trans, sel_trans, y, SEEDS, cfg)
    oof_t = oof_t.mean(axis=1); tp_t = tp_t.mean(axis=1)
    
    # Calibrate
    oof_b_c = mean_match(oof_b, train_rates[target])
    oof_p_c = mean_match(oof_p, train_rates[target])
    oof_t_c = mean_match(oof_t, train_rates[target])
    
    tp_b_c = mean_match(tp_b, train_rates[target])
    tp_p_c = mean_match(tp_p, train_rates[target])
    tp_t_c = mean_match(tp_t, train_rates[target])
    
    ll_b = log_loss(y, oof_b_c, labels=[0,1])
    ll_p = log_loss(y, oof_p_c, labels=[0,1])
    ll_t = log_loss(y, oof_t_c, labels=[0,1])
    
    log.info(f"    base: {ll_b:.5f}, pair: {ll_p:.5f}, trans: {ll_t:.5f}")
    
    all_model_oofs[target] = {'base': oof_b_c, 'pair': oof_p_c, 'trans': oof_t_c}
    all_model_test[target] = {'base': tp_b_c, 'pair': tp_p_c, 'trans': tp_t_c}
    
    # Now try 2-model and 3-model ensembles
    # V127 style: 0.35*pair + 0.25*base + 0.40*trans (but we don't have V115 separate)
    # Try all combos
    def opt_blend(oof1, oof2, w1_range=(0.05, 0.95)):
        def obj(w):
            blended = w[0]*oof1 + (1-w[0])*oof2
            blended = mean_match(blended, train_rates[target])
            return log_loss(y, blended, labels=[0,1])
        res = minimize(obj, [0.5], method='Nelder-Mead', options={'maxiter':200})
        w = np.clip(res.x[0], w1_range[0], w1_range[1])
        blended = mean_match(w*oof1 + (1-w)*oof2, train_rates[target])
        return blended, log_loss(y, blended, labels=[0,1])
    
    # 2-model combos
    best_2 = float('inf'); best_2_blend = None; best_2_w = None
    for name1, o1 in [('base',oof_b_c),('pair',oof_p_c),('trans',oof_t_c)]:
        for name2, o2 in [('base',oof_b_c),('pair',oof_p_c),('trans',oof_t_c)]:
            if name1 >= name2: continue
            blended, ll = opt_blend(o1, o2)
            if ll < best_2:
                best_2 = ll; best_2_blend = blended; best_2_w = f"{name1}:{name2}"
    
    # 3-model blend
    def opt_blend3(w0, w1, w2):
        w = np.array([w0, w1, w2])
        w = w / w.sum()
        blended = w[0]*oof_b_c + w[1]*oof_p_c + w[2]*oof_t_c
        blended = mean_match(blended, train_rates[target])
        return log_loss(y, blended, labels=[0,1])
    
    best_3 = float('inf')
    for i in range(0, 100):
        for j in range(0, 100-i):
            k = 100 - i - j
            w = np.array([i,j,k])/100.0
            blended = mean_match(w[0]*oof_b_c + w[1]*oof_p_c + w[2]*oof_t_c, train_rates[target])
            ll = log_loss(y, blended, labels=[0,1])
            if ll < best_3:
                best_3 = ll; best_3_w = w.copy()
                best_3_blend = blended
    
    log.info(f"    V127 weights (0.35p+0.25b+0.40t): {log_loss(y, mean_match(0.35*oof_p_c+0.25*oof_b_c+0.40*oof_t_c, train_rates[target]), labels=[0,1]):.5f}")
    log.info(f"    Best 2-model: {best_2:.5f} ({best_2_w})")
    log.info(f"    Best 3-model: {best_3:.5f} (w={best_3_w.round(3)})")

# ============================================================
# Part 2: Distribution Shift Analysis
# ============================================================
log.info("\n" + "="*70)
log.info("PART 2: DISTRIBUTION SHIFT ANALYSIS")
log.info("="*70)

# PSI: Population Stability Index per feature
def compute_psi(expected, actual, bins=10):
    """Compute PSI between two distributions."""
    eps = 1e-10
    breakpoints = np.linspace(expected.min(), expected.max(), bins+1)
    breakpoints[0] -= eps
    breakpoints[-1] += eps
    
    def pct(arr):
        counts = np.histogram(arr, bins=breakpoints)[0]
        return (counts + eps) / (len(arr) + bins*eps)
    
    p_expected = pct(expected)
    p_actual = pct(actual)
    
    psi = np.sum((p_expected - p_actual) * np.log(p_expected / p_actual))
    return psi

# Compute PSI on base features (without zscore)
base_feat_cols = get_feat_cols(feat)
# Sample a few features to check
log.info("Computing PSI on key features (sampling 30 random features)...")
import random
random.seed(42)
sample_feats = random.sample(base_feat_cols, min(30, len(base_feat_cols)))

psi_results = {}
for f in sample_feats:
    fe = feat[f].fillna(0).values
    ft = feat_test[f].fillna(0).values
    psi = compute_psi(fe, ft)
    psi_results[f] = psi

high_psi = {k:v for k,v in psi_results.items() if v > 0.1}  # PSI > 0.1 = significant drift
log.info(f"Features with PSI > 0.1: {len(high_psi)}")
if high_psi:
    log.info(f"  Top 10: {sorted(high_psi.items(), key=lambda x:-x[1])[:10]}")

# Adversarial validation: train/test classifier
log.info("\nAdversarial validation...")
all_feats = get_feat_cols(feat)
X = feat[all_feats].fillna(0).values
# Create labels: 0=train, 1=test
train_mask = np.zeros(len(feat))
X_full = np.vstack([X, feat_test[all_feats].fillna(0).values])
y_adv = np.array([0]*len(feat) + [1]*len(feat_test))

# Use a simple model to find discriminating features
spw = max((y_adv==0).sum() / max((y_adv==1).sum(), 1), 0.1)
params_adv = {'objective':'binary','metric':'binary_logloss','verbose':-1,
              'num_leaves':15,'max_depth':4,'learning_rate':0.03,'n_estimators':100,
              'subsample':0.8,'colsample_bytree':0.8,'reg_alpha':1.0,'reg_lambda':3.0,
              'scale_pos_weight':spw,'random_state':42,'min_child_samples':10,'force_row_wise':True,'n_jobs':1}
sn = [sanitize(c) for c in all_feats]
ds_adv = lgb.Dataset(X_full, label=y_adv, feature_name=sn, params={'verbose':'-1'})
model_adv = lgb.train(params_adv, ds_adv, num_boost_round=100)
adv_imp = model_adv.feature_importance(importance_type='gain')
adv_ranked = sorted(zip(all_feats, adv_imp), key=lambda x:-x[1])

log.info("Top 20 discriminating features (train vs test):")
for name, imp in adv_ranked[:20]:
    log.info(f"  {name}: {imp}")

# Remove top discriminating features and re-evaluate
top_disc = [x[0] for x in adv_ranked[:20]]
log.info(f"\nRemoving top {len(top_disc)} discriminating features...")

feat_no_disc = feat.drop(columns=[c for c in top_disc if c in feat.columns], errors='ignore')
feat_test_no_disc = feat_test.drop(columns=[c for c in top_disc if c in feat_test.columns], errors='ignore')
no_disc_cols = get_feat_cols(feat_no_disc)
feat_no_disc, z_no_disc, stats_no_disc = add_zscore(feat_no_disc, no_disc_cols)
feat_test_no_disc, _, _ = add_zscore(feat_test_no_disc, get_feat_cols(feat_test_no_disc), stats_no_disc, for_test=True)
all_no_disc = no_disc_cols + z_no_disc

log.info("Re-training with reduced features...")
no_disc_results = {}
for target in TARGETS:
    t_cfg = V53_SWEEP[target]
    cfg = CFGS[t_cfg['cfg']]
    y = y_train[target]
    
    ranked = rank_features(feat_no_disc, all_no_disc, target)
    sel = ranked[:t_cfg['n_feat']]
    
    oof, tp = train_cv(feat_no_disc, feat_test_no_disc, sel, y, SEEDS, cfg)
    oof = oof.mean(axis=1); tp = tp.mean(axis=1)
    oof_c = mean_match(oof, train_rates[target])
    ll = log_loss(y, oof_c, labels=[0,1])
    no_disc_results[target] = ll
    log.info(f"  {target}: {ll:.5f}")

avg_no_disc = np.mean(list(no_disc_results.values()))
log.info(f"\nAVG_OOF without top-20 adv features: {avg_no_disc:.5f}")
log.info(f"V127 baseline: 0.53731")
log.info(f"Delta: {avg_no_disc - 0.53731:+.5f}")

# ============================================================
# Summary
# ============================================================
log.info(f"\n{'='*70}")
log.info("V257 SUMMARY")
log.info(f"{'='*70}")

# Print PSI results
psi_avg = np.mean(list(psi_results.values())) if psi_results else 0
psi_max = max(psi_results.values()) if psi_results else 0
log.info(f"PSI: avg={psi_avg:.4f}, max={psi_max:.4f}, high-drift features: {len(high_psi)}")
log.info(f"Adversarial validation top feature: {adv_ranked[0][0] if adv_ranked else 'N/A'} (imp={adv_ranked[0][1] if adv_ranked else 0})")

# Write experiment log
exp_log = {
    'version': 'V257',
    'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
    'description': 'Ensemble weight optimization + Distribution shift analysis',
    'ensemble': {
        'v127_weights': [0.35, 0.25, 0.40],
        'per_target_results': {t: {'base_ll': float(log_loss(y_train[t], all_model_oofs[t]['base'], labels=[0,1])),
                                   'pair_ll': float(log_loss(y_train[t], all_model_oofs[t]['pair'], labels=[0,1])),
                                   'trans_ll': float(log_loss(y_train[t], all_model_oofs[t]['trans'], labels=[0,1]))}
                               for t in TARGETS},
    },
    'distribution_shift': {
        'psi_avg': round(float(psi_avg), 4),
        'psi_max': round(float(psi_max), 4),
        'high_drift_features': len(high_psi),
        'top_discriminating': [f for f,_ in adv_ranked[:10]],
        'removing_top20_avg_oof': round(float(avg_no_disc), 5),
        'delta_vs_v127': round(float(avg_no_disc - 0.53731), 5),
    },
    'v257_no_disc_avg_oof': round(float(avg_no_disc), 5),
    'v257_delta': round(float(avg_no_disc - 0.53731), 5),
}
exp_path = EXPERIMENTS / f'v257_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
with open(exp_path, 'w') as f: json.dump(exp_log, f, indent=2, default=str)
log.info(f"Saved: {exp_path}")
