# MEMORY.md — DaCon2 장기 기억

## 승우 (Seungwoo Hong)
- Telegram 사용, 한국어
- 시간대: KST (Asia/Seoul)
- 2026-05-08 ~ DaCon2 경진대회 지속 진행 중

## 대회 개요
- ETRI DaCon2 — 사물인터넷/라이프로그 기반 건강 예측 경진대회
- 7개 타겟: Q1, Q2, Q3, S1, S2, S3, S4
- 학습 450 rows, 테스트 250 rows
- Feature: 141 base + 141 zscore per-person = 282 columns

## MISSION (2026-06-20 업데이트)
- 현재 최고 모델: **V540_Cfgs+V542_α** (avg_gap=-0.05915, expected LB~0.682) ⭐
- V308 verified LB 0.63893 대비 OOF 상의 avg_gap 개선
- **0.5점대 진입**까지 무한 연구 루프 계속

## ⭐ 현재 BEST (OOF 기준): V540_Cfgs + V542 Optimal α
- **OOF: avg_gap=-0.05915** (V308 gap 0.070 대비 -0.129!)
- Config: Q1_s_strong+heavy_lgb_n3, Q2_q_narrow+heavy_lgb_n10, Q3_heavy_reg+light_lgb_n7, S1_s_strong+heavy_lgb_n3, S2_light_reg+heavy_lgb_n7, S3_s_strong+heavy_lgb_n23, S4_s_strong+heavy_lgb_n20
- Meta: Ridge per-target α: Q1=0.0001, Q2=10.0, Q3=0.0001, S1=0.0001, S2=0.03, S3=10.0, S4=0.0001
- **Expected LB: ~0.682** (아직 제출 안됨)
- **Seed: 13 (42,53,64,75,86,97,108,119,130,141,152,163,174)**

## 🔑 결정적 인사이트 (V545, 2026-06-20)
- **Configs와 α는 반드시 짝지어야 함!**
- V534 base configs + V542 α → avg_gap=-0.01438 (**V537보다 나쁨!**)
- V540 configs + V542 α → avg_gap=-0.05915 (**V537보다 +0.02899 개선!**)
- V540 configs는 V542 α에 최적화, V534 configs는 V537 α에 최적화
- 즉 "good configs + bad α"는 "bad configs + good α"보다 나쁨

## V540 Configs (Best Found)
| Target | XGB Config | LGBM Config | n_feat | n_est |
|--------|-----------|-------------|--------|-------|
| Q1 | s_strong | heavy_lgb | 3 | 600 |
| Q2 | q_narrow | heavy_lgb | 10 | 800 |
| Q3 | heavy_reg | light_lgb | 7 | 500 |
| S1 | s_strong | heavy_lgb | 3 | 500 |
| S2 | light_reg | heavy_lgb | 7 | 500 |
| S3 | s_strong | heavy_lgb | 23 | 1000 |
| S4 | s_strong | heavy_lgb | 20 | 300 |

## V542 Optimal Alphas (Per-Target)
| Target | α |
|--------|------|
| Q1 | 0.0001 |
| Q2 | 10.0 |
| Q3 | 0.0001 |
| S1 | 0.0001 |
| S2 | 0.03 |
| S3 | 10.0 |
| S4 | 0.0001 |

## 📊 Version Comparison Table (2026-06-20)
| Version | avg_gap | Δ V537 | vs308 | Exp LB | Key Innovation |
|---------|---------|--------|-------|--------|----------------|
| V537 | -0.03016 | baseline | 7/7 | 0.65251 | fine per-target α |
| V540+V541 | -0.05915 | +0.02899 | 7/7 | 0.68150 | V540 configs + Ridge_opt |
| V542 per-targ | -0.05915 | +0.02899 | 7/7 | 0.68150 | per-target α sweep on V540 |
| V543 nfeat=3 | -0.04443 | +0.01427 | 7/7 | 0.66678 | feature ranking ensemble |
| V534+V542α | -0.01438 | -0.01578 | 7/7 | 0.63673 | V534 configs + V542 α (**WORSE!**) |
| EN_0.5_0.5 | +0.00893 | -0.03909 | 6/7 | 0.61342 | ElasticNet meta (**Q3 붕괴**) |

## Gap Trajectory
```
V308:       +0.07000  (verified LB 0.63893)
V537:       -0.03016  (per-target α, OOF)
V540+V542:  -0.05915  (V540 configs + per-target α) ← BEST
```

## Key Learnings
1. **Configs-α pairing이 핵심**: V534 configs는 V537 α에, V540 configs는 V542 α에 최적화
2. **V540 configs (s_strong+heavy_lgb) > V534 configs**: heavy regularization이 더 좋음
3. **α=0.0001이 α=0.001보다 Q1,Q3,S1,S4에서 좋음**
4. **α=10.0이 Q2,S3에서 α=0.06보다 훨씬 좋음** (edge에서 멈춤 → 더 커질 수 있음)
5. **EN meta는 Q3에서 붕괴** (+0.166 gap) → Ridge가 더 안정적
6. **Feature ranking ensemble은 single seed와 동일** → averaging이 큰 의미 없음
7. **n_feat=3이 Q1,S1에서 최적** (V543에서 확인)

## Next Hypotheses (2026-06-20)
1. **S3 α > 10 exploration**: S3 α=10이 edge → α=[20, 50, 100] 탐색
2. **Per-target n_feat sweep with V540 configs**: n_feat=1~30까지 sweep
3. **More seeds**: 13→26 seeds로 OOF stability 검증
4. **Cross-validation LB estimation**: OOF-LB gap 분석
5. **Feature interactions**: 타겟별 feature interaction engineering

## Submission Files
- `submission_v541_Ridgeopt_20260620_051325.csv` (V541, Ridge_optimal)
- `submission_v542_per_target_alpha_20260620_051855.csv` (V542)
- `submission_v545_v540_configs_20260620_054835.csv` (V545, BEST)
- **승우의 수동 제출 필요** (API 제출 금지)

## Experiment Results Directory
- `/home/mwoo423/.openclaw/workspace/experiments/v541_*.json`
- `/home/mwoo423/.openclaw/workspace/experiments/v542_*.json`
- `/home/mwoo423/.openclaw/workspace/experiments/v543_*.json`
- `/home/mwoo423/.openclaw/workspace/experiments/v545_*.json`

## Next Steps
1. **V545 submission 승우에게 수동 업로드 요청** (가장 좋은 OOF 모델)
2. LB 결과가 나오면 MEMORY.md 업데이트
3. V546: α > 10 exploration (S3) + Q2 α > 10 exploration
4. V547: Per-target n_feat sweep with V540 configs (n_feat=1~30)
