"""Recompute LB estimates for all submission files.

Approach:
- 250-row test submissions: use gap model (V127 anchor)
  est_LB = est_oof + gap, where gap ≈ 0.10-0.11 calibrated from V127
- 450-row OOF submissions: compute actual OOF, then est_LB = OOF + gap
- Gap estimated from calibration error + entropy
"""
import sys, gc, json, re, time, warnings, logging
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", stream=sys.stdout)
log = logging.getLogger(__name__)

ROOT = Path('/home/mwoo423/projects/dacon2')
DATA = ROOT / "data_processed"
SUBMIT = ROOT / "submissions"
EXPERIMENTS = ROOT / "experiments"
TARGETS = ['Q1','Q2','Q3','S1','S2','S3','S4']

def mean_match(pred, target_mean):
    return np.clip(pred + (target_mean - pred.mean()), 0.0001, 0.9999)

def entropy_bernoulli(pred, eps=1e-10):
    p = np.clip(pred, eps, 1-eps)
    return -(p * np.log(p) + (1-p) * np.log(1-p)).mean()

def compute_oof(sub, train_df):
    """Compute OOF log-loss. Returns (avg_oof, per_target_losses, n_rows)."""
    n_rows = len(sub)
    if n_rows != 450:
        return None, {}, n_rows
    
    oof_losses = {}
    for t in TARGETS:
        if t not in sub.columns:
            continue
        ll = log_loss(train_df[t].values, sub[t].values, labels=[0,1])
        oof_losses[t] = ll
    
    if oof_losses:
        avg_oof = np.mean(list(oof_losses.values()))
    else:
        avg_oof = None
    return avg_oof, oof_losses, n_rows

def extract_features(sub):
    """Extract prediction features from a submission."""
    feats = {}
    for t in TARGETS:
        if t not in sub.columns:
            continue
        p = sub[t].values
        feats[f'{t}_mean'] = float(p.mean())
        feats[f'{t}_std'] = float(p.std())
        feats[f'{t}_min'] = float(p.min())
        feats[f'{t}_max'] = float(p.max())
        feats[f'{t}_entropy'] = float(entropy_bernoulli(p))
    return feats

def estimate_est_lb(sub, train_means, oof=None):
    """Estimate LB from submission or OOF.
    
    Uses V127 anchor: OOF=0.53731, LB=0.64763, gap=0.11032.
    Gap increases with calibration error.
    """
    cal_errors = []
    entropies = []
    for t in TARGETS:
        if t not in sub.columns:
            continue
        p = sub[t].values
        cal_errors.append(abs(p.mean() - train_means[t]))
        entropies.append(entropy_bernoulli(p))
    
    if not cal_errors:
        return None
    
    avg_cal_err = np.mean(cal_errors)
    avg_entropy = np.mean(entropies)
    
    # Gap = base_gap + calibration_error_penalty + entropy_penalty
    # base_gap ≈ 0.10 (from V127)
    # cal_err penalty: poorly calibrated models generalize worse
    # entropy penalty: too confident (low entropy) → worse generalization
    base_gap = 0.10
    cal_penalty = avg_cal_err * 3.0
    entropy_penalty = max(0, 0.693 - avg_entropy) * 2.0  # 0.693 = Bernoulli(0.5) entropy
    
    gap = base_gap + cal_penalty + entropy_penalty
    gap = max(0.06, min(0.18, gap))  # bound gap
    
    if oof is not None:
        est_lb = oof + gap
    else:
        # For 250-row test submissions, estimate OOF from features
        # Approximate: test OOF ≈ train OOF (mean-matched)
        avg_pred = np.mean([sub[t].mean() for t in TARGETS if t in sub.columns])
        avg_train = np.mean(list(train_means.values()))
        # Estimate OOF as: use V127-style relationship
        # Since we don't know true OOF, use a heuristic based on prediction characteristics
        # Well-calibrated + high entropy → OOF around 0.55-0.60
        avg_std = np.mean([sub[t].std() for t in TARGETS if t in sub.columns])
        est_oof = 0.55 + max(0, 0.693 - avg_entropy) * 0.5 + abs(avg_pred - avg_train) * 2.0
        est_oof = max(0.50, min(0.75, est_oof))
        est_lb = est_oof + gap
    
    return est_lb

def main():
    log.info("Loading training data...")
    train_df = pd.read_parquet(DATA / "features.parquet")
    train_means = {t: train_df[t].mean() for t in TARGETS}
    
    # Known LB points from history
    known_versions = {
        'v53': {'lb': 0.65358, 'oof': 0.54793, 'note': 'Best LB historical (manual submit)'},
        'v127': {'lb': 0.64763, 'oof': 0.53731, 'note': 'Best OOF (3-way ensemble)'},
        'v102': {'pred_lb': 0.61998, 'note': 'Shift amplification, predicted'},
        'v99': {'pred_lb': 0.73, 'note': '100 seeds blending, predicted'},
        'v260': {'oof': 0.6565, 'note': 'Quantile+PSI, submitted file'},
        'v262': {'oof': 0.58819, 'note': 'Isotonic, OOF only'},
        'v58': {'note': '53-swept ensemble'},
    }
    
    sub_files = sorted(SUBMIT.glob("*.csv"))
    log.info(f"Found {len(sub_files)} submission files")
    
    results = []
    for i, sf in enumerate(sub_files):
        try:
            sub = pd.read_csv(sf)
        except Exception:
            continue
        
        if not all(t in sub.columns for t in ['Q1','Q2','Q3','S1','S2','S3','S4']):
            continue
        
        sub_name = sf.stem
        
        # Compute OOF (for 450-row files)
        avg_oof, oof_losses, n_rows = compute_oof(sub, train_df)
        
        # Extract features
        feats = extract_features(sub)
        
        # Estimate LB
        est_lb = estimate_est_lb(sub, train_means, oof=avg_oof)
        
        # Find known version match
        known_info = None
        for k, vd in known_versions.items():
            if k in sub_name.lower() or sub_name.lower().startswith(k):
                known_info = vd
                break
        
        results.append({
            'name': sub_name,
            'file': str(sf.relative_to(ROOT)),
            'n_rows': n_rows,
            'avg_oof': round(avg_oof, 6) if avg_oof is not None else None,
            'oof_per_target': {t: round(v, 6) for t, v in oof_losses.items()},
            'avg_cal_err': round(np.mean([feats.get(f'{t}_mean', 0) - train_means.get(t, 0) for t in TARGETS]), 6),
            'avg_entropy': round(feats.get('avg_entropy', 0), 6),
            'est_lb': round(est_lb, 6) if est_lb is not None else None,
            'known_lb': known_info.get('lb', None) if known_info else None,
            'known_oof': known_info.get('oof', None) if known_info else None,
            'known_note': known_info.get('note', '') if known_info else '',
            'features': feats,
        })
        
        if (i+1) % 50 == 0:
            log.info(f"Processed {i+1}/{len(sub_files)} submissions...")
    
    # Sort by est_lb
    with_lb = [r for r in results if r['est_lb'] is not None]
    with_lb.sort(key=lambda x: x['est_lb'])
    
    # Group by version prefix
    groups = {}
    for r in results:
        m = re.match(r'(v\d+)', r['name'], re.IGNORECASE)
        if m:
            v = m.group(1).lower()
            if v not in groups:
                groups[v] = []
            groups[v].append(r)
    
    # Build group summary
    group_summary = []
    for v in sorted(groups.keys(), key=lambda x: int(x[1:]) if x[1:].isdigit() else 9999):
        items = groups[v]
        oofs = [i['avg_oof'] for i in items if i['avg_oof'] is not None]
        lbs = [i['est_lb'] for i in items if i['est_lb'] is not None]
        
        known = None
        for k, vd in known_versions.items():
            if k in v or v in k:
                known = vd
                break
        
        group_summary.append({
            'version': v,
            'n_files': len(items),
            'n_with_oof': len(oofs),
            'avg_oof': round(np.mean(oofs), 5) if oofs else None,
            'best_oof': round(min(oofs), 5) if oofs else None,
            'avg_est_lb': round(np.mean(lbs), 5) if lbs else None,
            'best_est_lb': round(min(lbs), 5) if lbs else None,
            'known_lb': known.get('lb', None) if known else None,
            'known_note': known.get('note', '') if known else '',
        })
    
    # Print results
    log.info(f"\n{'='*70}")
    log.info("TOP 30 SUBMISSIONS BY ESTIMATED LB:")
    log.info(f"{'='*70}")
    for i, r in enumerate(with_lb[:30], 1):
        oof_str = f"{r['avg_oof']:.5f}" if r['avg_oof'] else 'N/A(250)'
        lb_str = f"{r['est_lb']:.5f}" if r['est_lb'] else 'N/A'
        known_flag = ' [KNOWN LB]' if r['known_lb'] else ''
        note = f" [{r['known_note']}]" if r['known_note'] else ''
        log.info(f"{i:2d}. {r['name']:<60s} OOF={oof_str}  est_LB={lb_str}{known_flag}{note}")
    
    log.info(f"\n{'='*70}")
    log.info("GROUP SUMMARY:")
    log.info(f"{'='*70}")
    for g in group_summary:
        oof_s = f"{g['avg_oof']:.5f}" if g['avg_oof'] else 'N/A'
        best_oof_s = f"{g['best_oof']:.5f}" if g['best_oof'] else 'N/A'
        lb_s = f"{g['avg_est_lb']:.5f}" if g['avg_est_lb'] else 'N/A'
        best_lb_s = f"{g['best_est_lb']:.5f}" if g['best_est_lb'] else 'N/A'
        known_s = f" LB={g['known_lb']}" if g['known_lb'] else ''
        note = f" [{g['known_note']}]" if g['known_note'] else ''
        log.info(f"{g['version']:<8s} n={g['n_files']:2d}  "
                 f"avg_OOF={oof_s}  best_OOF={best_oof_s}  "
                 f"avg_est_LB={lb_s}  best_est_LB={best_lb_s}"
                 f"{known_s}{note}")
    
    # Save
    output = {
        'all_submissions': [
            {k: v for k, v in r.items() if k != 'features'}
            for r in results
        ],
        'group_summary': group_summary,
        'known_versions': {k: {'lb': vd.get('lb'), 'oof': vd.get('oof'), 'note': vd.get('note','')}
                          for k, vd in known_versions.items()},
        'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
    }
    
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = EXPERIMENTS / f'recompute_lb_all_{ts}.json'
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    log.info(f"\nSaved: {out_path}")
    
    # Return for inspection
    return results, group_summary, with_lb

if __name__ == '__main__':
    results, groups, with_lb = main()
