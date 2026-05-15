"""
V103: Aggressive S3/S4 Shift Amplification

Strategy:
- The LB predictor formula shows S3_shift and S4_shift have positive coefficients (0.4262, 0.2194)
- But S3 and S4 have NEGATIVE shifts in V53 (test mean < OOF mean)
- Amplifying these negative shifts further → more negative shift → LOWER predicted LB
- Key insight: If the LB predictor is roughly correct, pushing S3/S4 predictions
  closer to OOF mean (i.e., making the shift LESS negative, or even positive)
  should improve LB.
  
BUT the task says: "S3/S4 negative shifts → lower LB (key insight)"
This means the predictor captures a pattern where negative S3/S4 shifts correlate with lower LB.
So we should AMPLIFY (make more negative) the S3/S4 shifts.

Approach:
- Load V53 submission predictions
- For each amplification factor, shift S3/S4 predictions toward (or past) OOF means
  by: shifted = oof_mean + factor * (sub_mean - oof_mean)
  → At factor=1.0: original
  → At factor=1.3: amplified negative shift
  → At factor=2.0+: even more aggressive

Also try the REVERSE approach: shift predictions AWAY from OOF (factor < 0)
to see if the relationship holds.
"""

import sys, gc, logging, json, time, warnings, math
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.stats import entropy, skew as skew_func
from sklearn.metrics import log_loss

np.random.seed(42)
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", stream=sys.stdout)
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data_processed"
SUBMIT = ROOT / "submissions"
EXPERIMENTS = ROOT / "experiments"
SUBMIT.mkdir(exist_ok=True)
EXPERIMENTS.mkdir(exist_ok=True)

TARGETS = ['Q1', 'Q2', 'Q3', 'S1', 'S2', 'S3', 'S4']
S34_TARGETS = ['S3', 'S4']


def compute_lb(preds_df, oof_df):
    """
    LB prediction using the RMSE=0.0017 formula.
    
    LB = 0.0896*entropy - 0.4205*max_shift + 0.1877*skew + 0.4262*S3_shift + 0.2194*S4_shift + 0.7740
    """
    entropies = []
    shifts = []
    skews = []
    for t in TARGETS:
        p = np.clip(preds_df[t].values, 0.005, 0.995)
        entropies.append(entropy([p.mean(), 1 - p.mean()], base=2))
        shifts.append(preds_df[t].mean() - oof_df[t].mean())
        skews.append(skew_func(p))
    
    S3_shift = shifts[5]
    S4_shift = shifts[6]
    avg_entropy = np.mean(entropies)
    avg_shift = np.mean(shifts)
    avg_skew = np.mean(skews)
    
    lb = (0.0896 * avg_entropy 
          - 0.4205 * avg_shift 
          + 0.1877 * avg_skew 
          + 0.4262 * S3_shift 
          + 0.2194 * S4_shift 
          + 0.7740)
    return lb


def compute_train_ll(preds_df, train_labels):
    """Compute average log_loss per target.
    
    preds_df and train_labels must have matching rows (same subjects and dates).
    We align by subject_id + sleep_date.
    """
    # preds_df should have subject_id, sleep_date, lifelog_date + pred columns (e.g., Q1_pred)
    merged = preds_df.merge(train_labels, on=['subject_id', 'sleep_date', 'lifelog_date'], )
    losses = []
    for t in TARGETS:
        p = np.clip(merged[f'{t}_pred'].values, 0.005, 0.995)
        y = merged[f'{t}'].values.astype(float)
        ll = log_loss(y, p, labels=[0, 1])
        losses.append(ll)
    return np.mean(losses), losses


def apply_shift_amplification(sub, oof, train_labels, factor, targets_to_shift=S34_TARGETS):
    """
    Apply amplification to specified targets.
    
    shifted_value = sub_mean + factor * (sub_mean - oof_mean)
    This shifts the target mean further from OOF mean by `factor` times the original shift.
    factor=1.0: original prediction
    factor=-1.0: predict exactly OOF mean (no shift from train)
    factor=1.3: 30% more shift than original
    factor=2.0: 2x the original shift
    """
    shifted = sub.copy()
    for t in targets_to_shift:
        oof_mean = oof[t].mean()
        sub_mean = sub[t].mean()
        original_shift = sub_mean - oof_mean
        new_shift = factor * original_shift
        original_pred = sub[t].values
        original_mean = original_pred.mean()
        new_mean = oof_mean + new_shift
        shifted_values = original_pred - original_mean + new_mean
        shifted[t] = np.clip(shifted_values, 0.005, 0.995)
    
    # Compute metrics
    stats = {'factor': factor, 'targets_shifted': targets_to_shift}
    
    # Per-target stats
    for t in TARGETS:
        p = shifted[t].values
        stats[f'{t}_mean'] = float(p.mean())
        stats[f'{t}_std'] = float(p.std())
        p_clip = np.clip(p, 0.005, 0.995)
        stats[f'{t}_entropy'] = float(entropy([p_clip.mean(), 1 - p_clip.mean()], base=2))
    
    # LB prediction
    stats['predicted_lb'] = float(compute_lb(shifted, oof))
    
    # Train LL proxy: apply the same shift to OOF predictions and compare with train labels
    oof_shifted = oof.copy()
    for t in targets_to_shift:
        oof_mean = oof[t].mean()
        sub_mean = sub[t].mean()
        original_shift = sub_mean - oof_mean
        new_shift = factor * original_shift
        original_oof = oof[t].values
        original_oof_mean = original_oof.mean()
        new_mean = oof_mean + new_shift
        shifted_oof_values = original_oof - original_oof_mean + new_mean
        oof_shifted[t] = np.clip(shifted_oof_values, 0.005, 0.995)
    
    # Rename columns for merging with train labels
    oof_shifted = oof_shifted.rename(
        columns={c: f'{c}_pred' for c in TARGETS if c in oof_shifted.columns}
    )
    train_ll, per_target_ll = compute_train_ll(oof_shifted, train_labels)
    stats['train_ll_proxy'] = float(train_ll)
    for i, t in enumerate(TARGETS):
        stats[f'{t}_train_ll'] = float(per_target_ll[i])
    
    # Validity checks on test submission
    p_all = shifted[TARGETS].values
    valid_count = ((p_all >= 0.005) & (p_all <= 0.995)).sum()
    total = p_all.size
    stats['out_of_range_count'] = int(total - valid_count)
    stats['out_of_range_pct'] = float((total - valid_count) / total * 100)
    
    return stats


def main():
    t_start = time.time()
    log.info("=" * 80)
    log.info("V103: Aggressive S3/S4 Shift Amplification")
    log.info("=" * 80)
    
    # Load data
    oof = pd.read_csv(DATA / "oof_v53.csv")
    sub = pd.read_csv(SUBMIT / "submission_v53_swept_20260510_215247.csv")
    train_labels = pd.read_csv(ROOT / "data_raw" / "ch2026_metrics_train.csv")
    
    # Baseline metrics on test submission
    baseline_lb = compute_lb(sub, oof)
    # For train LL: use OOF as is (no shift)
    oof_for_ll = oof.rename(columns={c: f'{c}_pred' for c in TARGETS})
    baseline_train_ll, baseline_per_target_ll = compute_train_ll(oof_for_ll, train_labels)
    
    log.info(f"\nBaseline V53:")
    log.info(f"  Predicted LB: {baseline_lb:.5f}")
    log.info(f"  Train LL proxy: {baseline_train_ll:.5f}")
    for t in TARGETS:
        oof_m = oof[t].mean()
        sub_m = sub[t].mean()
        log.info(f"  {t}: OOF={oof_m:.4f}, sub={sub_m:.4f}, shift={sub_m-oof_m:+.4f}")
    
    # Try amplification factors
    factors = [0.0, 0.3, 0.5, 0.7, 1.0, 1.2, 1.3, 1.5, 1.8, 2.0, 2.5, 3.0, -0.5, -1.0]
    all_results = []
    
    log.info(f"\n--- Testing {len(factors)} amplification factors ---\n")
    
    for factor in factors:
        r = apply_shift_amplification(sub, oof, train_labels, factor)
        
        indicator = ""
        if r['predicted_lb'] < baseline_lb:
            indicator = " ↓ IMPROVES"
        if r['predicted_lb'] < 0.62:
            indicator += " ★ STRONG"
        if r['predicted_lb'] < 0.60:
            indicator += " ★★ EXCEPTIONAL"
        
        log.info(f"  factor={factor:6.1f}: LB={r['predicted_lb']:.5f} "
                 f"(delta={r['predicted_lb']-baseline_lb:+.5f}) "
                 f"train_ll={r['train_ll_proxy']:.5f} "
                 f"(delta={r['train_ll_proxy']-baseline_train_ll:+.5f})"
                 f" oor={r['out_of_range_pct']:.1f}%{indicator}")
        
        all_results.append(r)
    
    # Sort by predicted LB (lower is better)
    all_results.sort(key=lambda x: x['predicted_lb'])
    
    log.info(f"\n{'='*80}")
    log.info("TOP 5 Results (by predicted LB)")
    log.info(f"{'='*80}")
    for i, r in enumerate(all_results[:5]):
        log.info(f"  #{i+1}: factor={r['factor']:.1f}, LB={r['predicted_lb']:.5f}, "
                 f"train_ll={r['train_ll_proxy']:.5f}, oor={r['out_of_range_pct']:.1f}%")
    
    # Select TOP 3 factors that balance LB improvement with valid predictions
    candidates = []
    for r in all_results:
        oor_penalty = r['out_of_range_pct'] * 0.5  # penalty for out-of-range
        score = r['predicted_lb'] + oor_penalty
        candidates.append((score, r))
    
    candidates.sort(key=lambda x: x[0])
    
    top_factors = []
    for score, r in candidates:
        if r['factor'] not in [x['factor'] for x in top_factors]:
            top_factors.append(r)
            if len(top_factors) >= 3:
                break
    
    log.info(f"\n{'='*80}")
    log.info("TOP 3 Factors for Submission Generation")
    log.info(f"{'='*80}")
    for i, r in enumerate(top_factors):
        log.info(f"  #{i+1}: factor={r['factor']:.1f}, LB={r['predicted_lb']:.5f}, "
                 f"train_ll={r['train_ll_proxy']:.5f}, oor={r['out_of_range_pct']:.1f}%")
    
    # Generate submissions for TOP 3
    for i, top_r in enumerate(top_factors):
        factor = top_r['factor']
        
        # Apply shift to submission
        shifted = sub.copy()
        for t in S34_TARGETS:
            oof_mean = oof[t].mean()
            sub_mean = sub[t].mean()
            original_shift = sub_mean - oof_mean
            new_shift = factor * original_shift
            original_pred = sub[t].values
            original_mean = original_pred.mean()
            new_mean = oof_mean + new_shift
            shifted_values = original_pred - original_mean + new_mean
            shifted[t] = np.clip(shifted_values, 0.005, 0.995)
        
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        sub_name = f"v103_amp_s3s4_f{factor}_{ts}"
        sub_path = SUBMIT / f"{sub_name}.csv"
        shifted.to_csv(sub_path, index=False)
        
        log.info(f"\n  ✓ Submission saved: {sub_path}")
        for t in TARGETS:
            log.info(f"    {t}: mean={shifted[t].mean():.4f}, std={shifted[t].std():.4f}")
    
    # Save all results
    output = {
        'version': 'V103_aggressive_shift',
        'timestamp': datetime.now().isoformat(),
        'baseline': {
            'predicted_lb': float(baseline_lb),
            'train_ll': float(baseline_train_ll),
        },
        'all_results': all_results,
        'top_factors': top_factors,
        'oof_means': {t: float(oof[t].mean()) for t in TARGETS},
        'submissions_generated': [f"v103_amp_s3s4_f{r['factor']}_{ts}" for r in top_factors],
    }
    
    result_path = EXPERIMENTS / "v103_results.json"
    with open(result_path, 'w') as f:
        json.dump(output, f, indent=2)
    log.info(f"\n✅ Results saved: {result_path}")
    log.info(f"  Total time: {time.time()-t_start:.0f}s")
    
    # Final summary
    log.info(f"\n{'='*80}")
    log.info("V103 SUMMARY")
    log.info(f"{'='*80}")
    for r in top_factors:
        log.info(f"  Factor={r['factor']:.1f}: LB={r['predicted_lb']:.5f}, "
                 f"train_ll={r['train_ll_proxy']:.5f}, "
                 f"improvement={baseline_lb-r['predicted_lb']:.5f}")


if __name__ == "__main__":
    main()
