"""
V313 OOF-LB Gap Analysis

V313: Meta OOF=0.59512, Student OOF≈0.77~0.78
Student-Meta gap ≈ 0.18

V312: Meta OOF=0.61448, Student OOF≈0.76~0.79
Student-Meta gap ≈ 0.15

V308: Meta OOF=0.62235, Student OOF≈0.75~0.78
Student-Meta gap ≈ 0.13

V146: Meta OOF=0.63169, Student OOF≈0.63~0.80
Student-Meta gap ≈ 0.15

Question: Is the gap stable or does it grow with more seeds/C=500?

If V313 gap=0.18, predicted LB=0.59512+0.18=0.77 (WORSE than V308!)
If V313 gap=0.13, predicted LB=0.59512+0.13=0.725 (still worse than V308!)
If V313 gap=0.15, predicted LB=0.59512+0.15=0.745 (worse than V308!)

Wait - the gap is much larger. Let me recalculate with actual student OOFs.
"""
import pandas as pd
import json
from pathlib import Path

exp_dir = Path('/home/mwoo423/.openclaw/workspace/experiments')

# Load V313 meta
v313_meta = json.loads((exp_dir / 'v313_20260602_025815.json').read_text())

# Load V312 meta
v312_meta = json.loads((exp_dir / 'v312_20260602_024509.json').read_text())

print("=== V313 vs V312 Comparison ===")
print(f"\nV313: Meta OOF = {v313_meta['avg_oof']}, Seeds = {v313_meta['n_seeds']}, C = {v313_meta['meta_c']}")
print(f"V312: Meta OOF = {v312_meta['avg_oof']}, Seeds = 15, C = 500")
print(f"\nΔ OOF: {v313_meta['avg_oof'] - v312_meta['avg_oof']:+.5f}")

print(f"\nPer-target comparison:")
for t in ['Q1','Q2','Q3','S1','S2','S3','S4']:
    v313_oof = v313_meta['per_target_oof'][t]
    v312_oof = v313_meta['v312_per_target_oof'][t]  # From V313 meta
    delta = v313_oof - v312_oof
    print(f"  {t}: V313={v313_oof:.5f}, V312={v312_oof:.5f}, Δ={delta:+.5f}")

# Key insight: with 30 seeds and C=500, meta OOF drops significantly
# because meta can leverage more student diversity.
# BUT: the student-meta gap also increases.
# V313 student OOF ≈ 0.775, meta OOF = 0.595
# Gap = 0.180 → This is the overfitting concern.

# V308: OOF-LB gap = 0.01658 (from OOF 0.62235 to LB 0.63893)
# But V308 is a DIFFERENT metric — LB uses log_loss on predictions, not OOF-mean.
# The OOF-LB gap for V308 is: LB - OOF = 0.63893 - 0.62235 = +0.01658
# This means LB is WORSE than OOF.

# V313: If gap remains +0.01658 → LB = 0.59512 + 0.01658 = 0.61170
# This would BEAT V308's LB 0.63893 by 0.027!

# However, with 30 seeds and C=500, the gap may be larger.
# Let's check: V308 had 15 seeds, C=10 → gap 0.01658
# V312 had 15 seeds, C=500 → we don't know LB gap
# V313 has 30 seeds, C=500 → gap could be 0.015-0.020

# The student-meta gap is not the OOF-LB gap!
# OOF-LB gap is about calibration/test distribution, not train overfitting.
# The student-meta gap is about ensemble variance.
# V313's large student-meta gap means students disagree more,
# but meta learns to weight them well → low OOF.
# This does NOT necessarily mean large OOF-LB gap.

print(f"\n=== OOF-LB Gap Analysis ===")
print(f"V308: OOF=0.62235, LB=0.63893, gap=+0.01658")
print(f"V313: OOF=0.59512, LB=?")
print(f"If gap same as V308: LB=0.59512+0.01658=0.61170")
print(f"This would BEAT V308 by 0.027")
print(f"\n⚠️ But gap may be different. Submit to check.")
