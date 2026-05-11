"""V104: Multi-Model OOF Weighted Ensemble + Test Submission
Finds optimal ensemble weights for top-k OOF models.
"""
import sys, json, time, logging, warnings, os
from pathlib import Path
from datetime import datetime
from sklearn.metrics import log_loss
import numpy as np
import pandas as pd

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

# Load train labels
feat = pd.read_parquet(DATA / "features.parquet")
feat['sleep_date'] = feat['sleep_date'].astype(str)
feat['lifelog_date'] = feat['lifelog_date'].astype(str)
feat_s = feat.sort_values(['subject_id','sleep_date']).reset_index(drop=True)

leaked = {'oof_v45a.csv', 'oof_v46.csv', 'oof_v55_re2_S4.csv'}

# Phase 1: Load all OOFs
log.info("[Phase 1] Loading OOF files...")
oof_data = {}
for fname in sorted(os.listdir(DATA)):
    if not fname.startswith('oof_') or not fname.endswith('.csv') or fname in leaked:
        continue
    oof_df = pd.read_csv(DATA / fname)
    if not all(t in oof_df.columns for t in TARGETS):
        continue
    oof_s = align(oof_df)
    lls = []
    for t in TARGETS:
        p = np.clip(oof_s[t].to_numpy().astype(float), 1e-15, 1-1e-15)
        y = feat_s[t].to_numpy().astype(float)
        lls.append(log_loss(y, p, labels=[0,1]))
    avg = np.mean(lls)
    oof_data[fname] = {'ll': avg, 'per_target': lls, 'data': oof_s}
    log.info(f"  {fname}: LL={avg:.5f}")

sorted_models = sorted(oof_data.items(), key=lambda x: x[1]['ll'])
log.info(f"\nAll models sorted:")
for name, info in sorted_models:
    log.info(f"  {name}: {info['ll']:.5f}")

# Phase 2: Ensemble search
log.info("\n[Phase 2] Ensemble weight optimization...")
np.random.seed(42)
best_ll = 999
best_cfg = None

# Pre-extract y arrays
y_dict = {t: feat_s[t].to_numpy().astype(float) for t in TARGETS}

for k in [2, 3, 4, 5]:
    subset_names = [sorted_models[i][0] for i in range(k)]
    subset_oofs = [oof_data[n]['data'] for n in subset_names]
    k_best = 999
    k_best_w = None
    
    for trial in range(8000):
        w = np.random.dirichlet(np.ones(k))
        w = np.round(w, 1)
        w = w / w.sum()
        
        avg_ll = 0
        for j, t in enumerate(TARGETS):
            vals = np.zeros(450)
            for i in range(k):
                vals += w[i] * subset_oofs[i][t].to_numpy()
            p = np.clip(vals, 1e-15, 1-1e-15)
            y = y_dict[t]
            avg_ll += log_loss(y, p, labels=[0,1])
        avg_ll /= 7
        
        if avg_ll < k_best:
            k_best = avg_ll
            k_best_w = w.copy()
    
    log.info(f"  Top-{k}: best={k_best:.5f}, w={np.round(k_best_w,2)}")
    if k_best < best_ll:
        best_ll = k_best
        best_cfg = {'k': k, 'names': subset_names, 'weights': k_best_w}

log.info(f"\nBest: LL={best_ll:.5f}, K={best_cfg['k']}")
log.info(f"Models: {best_cfg['names']}")
log.info(f"Weights: {np.round(best_cfg['weights'],2)}")
log.info(f"V53=0.54793, delta={0.54793-best_ll:+.5f}")

# Phase 3: Build test predictions from ensemble
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
test_path = SUBMIT / f'v104_ensemble_{ts}.csv'
test_ens.to_csv(test_path, index=False)
log.info(f"Saved: {test_path}")
log.info(f"Means: { {t: f'{test_ens[t].mean():.4f}' for t in TARGETS} }")

# Phase 4: OOF validation
log.info("\n[Phase 4] OOF validation...")
oof_ens = np.zeros((450, 7))
for j, t in enumerate(TARGETS):
    for i, name in enumerate(best_cfg['names']):
        oof_ens[:, j] += best_cfg['weights'][i] * oof_data[name]['data'][t].to_numpy()

oof_ll = sum(log_loss(y_dict[t], np.clip(oof_ens[:,j],1e-15,1-1e-15), labels=[0,1]) 
             for j,t in enumerate(TARGETS)) / 7
log.info(f"OOF ensemble LL: {oof_ll:.5f}")
log.info(f"OOF improvement over V53: {0.54793-oof_ll:+.5f}")

# Save
exp_log = {
    'version': 'V104', 'best_oof_ll': best_ll, 'oof_ensemble_ll': oof_ll,
    'weights': np.round(best_cfg['weights'].tolist(),3),
    'best_k': best_cfg['k'], 'best_models': best_cfg['names'],
    'test_submission': str(test_path.name),
    'found': [n for n,_ in found], 'missing': missing,
    'all_oof_lls': {n: round(info['ll'],5) for n,info in sorted_models},
}
with open(EXPERIMENTS / f'v104_{ts}.json', 'w') as f:
    json.dump(exp_log, f, indent=2)
log.info(f"Log: {EXPERIMENTS / f'v104_{ts}.json'}")
log.info(f"Total time: {time.time()-t_start:.0f}s")
