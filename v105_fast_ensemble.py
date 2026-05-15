"""V105: Fast OOF Ensemble — scipy.optimize + fewer trials."""
import sys, json, time, logging, warnings, os
from pathlib import Path
from datetime import datetime
from sklearn.metrics import log_loss
import numpy as np
import pandas as pd
from scipy.optimize import minimize

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", stream=sys.stdout)
log = logging.getLogger(__name__)

ROOT = Path('/home/mwoo423/projects/dacon2')
DATA = ROOT / "data_processed"
SUBMIT = ROOT / "submissions"
EXPERIMENTS = ROOT / "experiments"
TARGETS = ['Q1','Q2','Q3','S1','S2','S3','S4']

def align(oof_df):
    oof_df = oof_df.copy()
    oof_df['sleep_date'] = pd.to_datetime(oof_df['sleep_date']).dt.strftime('%Y-%m-%d')
    oof_df['lifelog_date'] = pd.to_datetime(oof_df['lifelog_date']).dt.strftime('%Y-%m-%d')
    return oof_df.sort_values(['subject_id','sleep_date']).reset_index(drop=True)

t_start = time.time()

feat = pd.read_parquet(DATA / "features.parquet")
feat['sleep_date'] = feat['sleep_date'].astype(str)
feat['lifelog_date'] = feat['lifelog_date'].astype(str)
feat_s = feat.sort_values(['subject_id','sleep_date']).reset_index(drop=True)

leaked = {'oof_v45a.csv', 'oof_v46.csv', 'oof_v55_re2_S4.csv'}

# Phase 1: Load OOFs
log.info("[Phase 1] Loading OOF files...")
oof_data = {}
for fname in sorted(os.listdir(DATA)):
    if not fname.startswith('oof_') or not fname.endswith('.csv') or fname in leaked:
        continue
    oof_df = pd.read_csv(DATA / fname)
    if not all(t in oof_df.columns for t in TARGETS):
        continue
    oof_df = oof_df.copy()
    oof_df['sleep_date'] = pd.to_datetime(oof_df['sleep_date']).dt.strftime('%Y-%m-%d')
    oof_df['lifelog_date'] = pd.to_datetime(oof_df['lifelog_date']).dt.strftime('%Y-%m-%d')
    oof_s = oof_df.sort_values(['subject_id','sleep_date']).reset_index(drop=True)
    lls = []
    for t in TARGETS:
        p = np.clip(oof_s[t].to_numpy().astype(float), 1e-15, 1-1e-15)
        y = feat_s[t].to_numpy().astype(float)
        lls.append(log_loss(y, p, labels=[0,1]))
    oof_data[fname] = {'ll': np.mean(lls), 'per_target': lls, 'data': oof_s}
    log.info(f"  {fname}: LL={np.mean(lls):.5f}")

sorted_models = sorted(oof_data.items(), key=lambda x: x[1]['ll'])
for name, info in sorted_models:
    log.info(f"  {name}: {info['ll']:.5f}")

# Pre-extract prediction arrays: preds[t][model_idx] = 450-array
preds = {t: [] for t in TARGETS}
all_names = []
for name, info in sorted_models:
    d = info['data']
    all_names.append(name)
    for t in TARGETS:
        preds[t].append(d[t].to_numpy().astype(float))
preds_arr = {t: np.column_stack(preds[t]) for t in TARGETS}

# Pre-extract y arrays
y_dict = {t: feat_s[t].to_numpy().astype(float) for t in TARGETS}

def ensemble_ll(w, top_k):
    """Compute ensemble log_loss for given weights and top-k models."""
    w = w[:top_k]
    w_norm = w / w.sum()
    avg_ll = 0
    for j, t in enumerate(TARGETS):
        vals = preds_arr[t][:, :top_k] @ w_norm
        p = np.clip(vals, 1e-15, 1-1e-15)
        avg_ll += log_loss(y_dict[t], p, labels=[0,1])
    return avg_ll / 7

# Phase 2: Search
log.info("\n[Phase 2] Ensemble optimization...")
best_ll = 999
best_cfg = None

np.random.seed(42)
for k in [2, 3, 4, 5]:
    t0 = time.time()
    
    # Coarse random search first
    random_trials = 2000
    random_scores = []
    for _ in range(random_trials):
        w = np.random.dirichlet(np.ones(k))
        score = ensemble_ll(w, k)
        random_scores.append((score, w.copy()))
    
    random_scores.sort()
    best_random_w = random_scores[0][1]
    best_random_ll = random_scores[0][0]
    log.info(f"  Top-{k}: random best={best_random_ll:.5f} ({time.time()-t0:.0f}s, {random_trials} trials)")
    
    # Fine optimization with scipy (Nelder-Mead, no gradients needed)
    def obj(w_raw):
        w = np.maximum(w_raw, 1e-6)
        return ensemble_ll(w, k)
    
    result = minimize(obj, best_random_w, method='Nelder-Mead',
                      options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-7})
    
    if result.fun < best_random_ll:
        fine_w = np.maximum(result.x, 1e-6)
        fine_ll = ensemble_ll(fine_w, k)
    else:
        fine_ll = best_random_ll
        fine_w = best_random_w
    
    log.info(f"  Top-{k}: fine={fine_ll:.5f}, w={np.round(fine_w/fine_w.sum(),3)} ({time.time()-t0:.0f}s)")
    
    if fine_ll < best_ll:
        best_ll = fine_ll
        best_cfg = {'k': k, 'names': all_names[:k], 'weights': (fine_w/fine_w.sum()).tolist()}

log.info(f"\nBest: LL={best_ll:.5f}, K={best_cfg['k']}")
log.info(f"Models: {best_cfg['names']}")
log.info(f"Weights: {np.round(best_cfg['weights'],3)}")
log.info(f"V53=0.54793, delta={0.54793-best_ll:+.5f}")

# Phase 3: Test predictions
log.info("\n[Phase 3] Building test predictions...")
base_test = pd.read_csv(SUBMIT / 'submission_v53_swept_20260510_215247.csv')

test_ens = base_test[['subject_id']].copy()
found = []; missing = []

for i, name in enumerate(best_cfg['names']):
    w = best_cfg['weights'][i]
    v_num = name.split('v')[1].split('_')[0]
    for fname in sorted(os.listdir(SUBMIT)):
        if fname.endswith('.csv') and f'v{v_num}' in fname.lower():
            test_df = pd.read_csv(SUBMIT / fname)
            if len(test_df) == 250 and all(t in test_df.columns for t in TARGETS):
                for t in TARGETS:
                    test_ens[t] += w * test_df[t].values
                found.append((name, fname))
                break
    else:
        missing.append(name)

log.info(f"Found: {[n for n,_ in found]}")
log.info(f"Missing: {missing}")

for t in TARGETS:
    test_ens[t] = np.clip(test_ens[t], 1e-15, 1-1e-15)

ts = datetime.now().strftime("%Y%m%d_%H%M%S")
test_path = SUBMIT / f'v105_ensemble_{ts}.csv'
test_ens.to_csv(test_path, index=False)
log.info(f"Saved: {test_path}")
log.info(f"Means: { {t: f'{test_ens[t].mean():.4f}' for t in TARGETS} }")

# Phase 4: OOF validation
log.info("\n[Phase 4] OOF validation...")
w_norm = np.array(best_cfg['weights'])
w_norm = w_norm / w_norm.sum()
oof_ll = sum(log_loss(y_dict[t], np.clip(preds_arr[t][:, :best_cfg['k']] @ w_norm, 1e-15, 1-1e-15), labels=[0,1])
             for t in TARGETS) / 7
log.info(f"OOF ensemble LL: {oof_ll:.5f}")
log.info(f"OOF vs V53: {0.54793-oof_ll:+.5f}")

exp_log = {
    'version': 'V105', 'best_oof_ll': best_ll, 'oof_ensemble_ll': oof_ll,
    'weights': [round(w, 3) for w in best_cfg['weights']],
    'best_k': best_cfg['k'], 'best_models': best_cfg['names'],
    'test_submission': str(test_path.name),
    'found': [n for n,_ in found], 'missing': missing,
    'all_oof_lls': {n: round(info['ll'],5) for n,info in sorted_models},
    'total_time_s': time.time() - t_start,
}
with open(EXPERIMENTS / f'v105_{ts}.json', 'w') as f:
    json.dump(exp_log, f, indent=2)
log.info(f"Log: {EXPERIMENTS / f'v105_{ts}.json'}")
log.info(f"Done in {time.time()-t_start:.0f}s")
