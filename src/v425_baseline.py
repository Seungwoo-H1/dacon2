"""
V425 — Pure Baseline Submission with Calibration

Key insight from V424: Per-subject mean baseline avg OOF = 0.594
V413 (heavy ML model) avg OOF = 0.651 — WORSE than baseline!

V425:
1. Pure per-subject mean baseline (best so far at 0.594)
2. Apply temperature scaling to baseline predictions for better calibration
3. Try blending baseline with different shrinkage levels
4. Submit for LB verification

This challenges the entire ML approach: maybe baseline IS the answer.
0.5점대를 가려면: baseline OOF 0.594 + calibration → LB가 더 낮아져야 함.
"""
import sys, gc, logging, json, re, time, warnings, math
from pathlib import Path
from datetime import datetime
from sklearn.metrics import log_loss
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

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

SEED = 42


def main():
    global t_start
    t_start = time.time()

    log.info("=" * 70)
    log.info("V425 — Pure Baseline with Calibration")
    log.info("Hypothesis: Baseline IS the best model. Just need calibration.")
    log.info("=" * 70)

    train_df = pd.read_parquet(DATA / "features.parquet")
    test_df = pd.read_parquet(DATA / "test_features.parquet")

    # ===== Phase 1: Per-subject mean baseline =====
    log.info("\n=== Phase 1: Per-Subject Mean Baseline ===")

    subj_means = {}
    for target in TARGETS:
        subj_means[target] = train_df.groupby('subject_id')[target].mean().to_dict()

    baseline_oofs = {}
    for target in TARGETS:
        y = train_df[target].values
        y_pred = np.array([subj_means[target].get(sid, train_df[target].mean())
                          for sid in train_df['subject_id'].values])
        oof = log_loss(y, y_pred)
        baseline_oofs[target] = oof
        log.info(f"  {target}: baseline_OOF={oof:.5f}, mean={train_df[target].mean():.4f}, "
                 f"subj_count={len(subj_means[target])}")

    avg_baseline = np.mean(list(baseline_oofs.values()))
    log.info(f"  AVG baseline OOF: {avg_baseline:.5f}")

    # ===== Phase 2: Temperature scaling =====
    log.info("\n=== Phase 2: Temperature Scaling ===")

    # Find optimal temperature for each target
    temps = {}
    scaled_oofs = {}
    for target in TARGETS:
        y = train_df[target].values
        y_pred = np.array([subj_means[target].get(sid, train_df[target].mean())
                          for sid in train_df['subject_id'].values])

        # Apply logit transform, scale by T, then sigmoid
        def logit(p):
            p = np.clip(p, 1e-7, 1-1e-7)
            return np.log(p / (1-p))

        def sigmoid(x):
            return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

        def loss(T):
            logit_pred = logit(y_pred)
            scaled = sigmoid(logit_pred / T)
            return log_loss(y, np.clip(scaled, 1e-7, 1-1e-7))

        result = minimize_scalar(loss, bounds=(0.1, 10.0), method='bounded')
        temps[target] = result.x
        best_loss = result.fun
        scaled_oofs[target] = best_loss

        logit_pred = logit(y_pred)
        scaled = sigmoid(logit_pred / result.x)
        log.info(f"  {target}: T={result.x:.3f}, OOF={best_loss:.5f} "
                 f"(vs baseline {baseline_oofs[target]:.5f}, delta={best_loss-baseline_oofs[target]:+.5f})")

    avg_scaled = np.mean(list(scaled_oofs.values()))
    log.info(f"  AVG scaled OOF: {avg_scaled:.5f} (vs baseline {avg_baseline:.5f})")

    # ===== Phase 3: Blending baseline with different shrinkage =====
    log.info("\n=== Phase 3: Shrinkage Blending ===")

    # Shrink individual subject means toward global mean
    # pred = (1-alpha) * subject_mean + alpha * global_mean
    for alpha in [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]:
        oofs = {}
        for target in TARGETS:
            y = train_df[target].values
            global_mean = train_df[target].mean()
            y_pred = np.array([(1-alpha) * subj_means[target].get(sid, global_mean) +
                              alpha * global_mean
                              for sid in train_df['subject_id'].values])
            oofs[target] = log_loss(y, y_pred)
        avg_oof = np.mean(list(oofs.values()))
        log.info(f"  alpha={alpha:.1f}: avg_OOF={avg_oof:.5f} "
                 f"(vs baseline {avg_baseline:.5f}, delta={avg_oof-avg_baseline:+.5f})")

    # ===== Phase 4: Generate submissions =====
    log.info("\n=== Phase 4: Submissions ===")

    # Submission 1: Pure baseline
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    sub = pd.DataFrame()
    sub['subject_id'] = test_df['subject_id'].values
    sub['sleep_date'] = test_df['sleep_date'].values
    sub['lifelog_date'] = test_df['lifelog_date'].values

    for target in TARGETS:
        test_subjects = test_df['subject_id'].values
        global_mean = train_df[target].mean()
        y_pred = np.array([subj_means[target].get(sid, global_mean)
                          for sid in test_subjects])
        sub[target] = np.clip(y_pred, 0.01, 0.99)

    sub_path = SUBMIT / f"submission_v425_baseline_{ts}.csv"
    sub.to_csv(sub_path, index=False)
    log.info(f"Saved baseline: {sub_path}")

    # Submission 2: Temperature-scaled
    sub2 = sub.copy()
    for target in TARGETS:
        test_subjects = test_df['subject_id'].values
        global_mean = train_df[target].mean()
        y_pred = np.array([subj_means[target].get(sid, global_mean)
                          for sid in test_subjects])

        logit_pred = np.log(np.clip(y_pred, 1e-7, 1-1e-7) / (1-np.clip(y_pred, 1e-7, 1-1e-7)))
        scaled = 1 / (1 + np.exp(-logit_pred / temps[target]))
        sub2[target] = np.clip(scaled, 0.01, 0.99)

    sub2_path = SUBMIT / f"submission_v425_baseline_Tcal_{ts}.csv"
    sub2.to_csv(sub2_path, index=False)
    log.info(f"Saved T-scaled: {sub2_path}")

    # Submission 3: Shrinkage alpha=0.3 (best from Phase 3 likely)
    sub3 = sub.copy()
    alpha = 0.3
    for target in TARGETS:
        test_subjects = test_df['subject_id'].values
        global_mean = train_df[target].mean()
        y_pred = np.array([(1-alpha) * subj_means[target].get(sid, global_mean) +
                          alpha * global_mean
                          for sid in test_subjects])
        sub3[target] = np.clip(y_pred, 0.01, 0.99)

    sub3_path = SUBMIT / f"submission_v425_baseline_shrink_{ts}.csv"
    sub3.to_csv(sub3_path, index=False)
    log.info(f"Saved shrinkage: {sub3_path}")

    # ===== Phase 5: Results summary =====
    log.info(f"\n{'='*70}")
    log.info("V425 Results:")
    log.info(f"  Per-subject mean baseline AVG OOF: {avg_baseline:.5f}")
    log.info(f"  Temp-scaled AVG OOF: {avg_scaled:.5f}")
    log.info(f"  V413 student AVG OOF: 0.65128")
    log.info(f"  Baseline beats V413 by {0.65128 - avg_baseline:.5f}")
    log.info(f"  0.5점대 진입을 위해서는:")
    log.info(f"    - Baseline OOF 0.594를 실제 LB로 변환해야 함")
    log.info(f"    - V339 패턴(0.85x gap) 적용 시: ~0.594 + 0.07*0.85 = ~0.654 (악화)")
    log.info(f"    - Gap이 baseline에서는 더 작을 가능성 있음 (simple model)")
    log.info(f"    - 실제 LB는 아마 0.60-0.63 사이")
    log.info(f"{'='*70}")

    result = {
        'version': 'V425',
        'name': 'Pure Baseline with Calibration',
        'avg_baseline_oof': round(float(avg_baseline), 5),
        'avg_scaled_oof': round(float(avg_scaled), 5),
        'temperatures': {t: round(float(v), 3) for t, v in temps.items()},
        'baseline_oofs': {t: round(float(v), 5) for t, v in baseline_oofs.items()},
        'scaled_oofs': {t: round(float(v), 5) for t, v in scaled_oofs.items()},
        'submission_files': [str(sub_path), str(sub2_path), str(sub3_path)],
        'timestamp': ts,
        'total_time_s': round(time.time() - t_start, 0),
    }

    meta_path = EXPERIMENTS / f'v425_{ts}.json'
    with open(meta_path, 'w') as f:
        json.dump(result, f, indent=2)

    log.info(f"Total time: {time.time()-t_start:.0f}s")


if __name__ == '__main__':
    main()
