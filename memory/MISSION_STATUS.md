# DACon2 연구 루프 — V308 초월 모델 탐색

## 🔴 MISSION
V308(LB=0.63893)을 초과하는 모델 발견. LB 예측 점수가 V308보다 개선되지 않으면 보고 금지.

## 📊 Current Status (2026-06-02)

### ✅ Verified Best (LB 제출 확인)
- **V308**: OOF=0.62235, LB=**0.63893** | 15 seeds, C=10, z-score

### ⏳ Pending LB Submission
- **V312**: OOF=0.61448 | 15 seeds, C=500 | Δ vs V308: -0.00787
- **V313**: OOF=0.59512 | 30 seeds, C=500 | Δ vs V308: -0.02723

### 🔬 V313 Analysis
- 30 seeds + C=500이 V312(15 seeds + C=500) 대비 -0.01936 추가 개선
- All 7 targets improved vs V312
- Expected LB: 0.612 (if OOF-LB gap = V308's gap)
- Improvement vs V308: **-0.027 예상**

### ⚠️ Critical Risk
- V313은 30 seeds + C=500으로 overfitting risk ↑
- Student OOF: 0.77~0.78, Meta OOF: 0.595 → gap ~0.18
- OOF-LB gap이 V308보다 커질 가능성 있음
- 실제 LB 결과는 승우 수동 제출 필요

## 🎯 Next Steps
1. V312, V313 LB 제출
2. V313이 V308을 넘으면 연구 종료
3. 넘지 않으면 새로운 가설 탐색
