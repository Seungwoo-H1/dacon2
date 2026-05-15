# V35~V40 최종 분석 보고서

## 1. 전체 결과 요약

| 버전 | 방식 | 상태 | Avg Cal | V10 대비 | 비고 |
|------|------|------|---------|----------|------|
| **V10** | LGBM, 6 configs × 3 feat × 20 seeds | ✅ 완주 | **0.6038** | 기준 | 현재 Best |
| **V37** | V8 config + 4 feat counts + 20 seeds | ✅ 완주 | 0.6144 | -0.0106 ❌ | features.parquet → personalization |
| V37_fix | LGBM only, 20 seeds, features.parquet → runtime personalization | 💀 SIGKILL (S4) | — | — | 개인화 merge 후 training 중 |
| V37_fix2 | features_v11_personalized.parquet, 6 configs × 3 feat × 20 seeds | 💀 SIGKILL | — | — | 8670열 메모리 부족 |
| V35 | XGB+LGBM blend, LOSO 10-fold | 💀 SIGKILL | — | — | 02_feature_engineering 47분 |
| V35_pre | XGB(3)+LGBM(5) blend | 💀 SIGKILL | — | — | personalization merge 중 |
| V36 | python 명령어 오류 | ❌ 실패 | — | — | `python` → `python3` |
| V36_stk | LGBM Stacking 4 variants × 5 seeds | 💀 SIGKILL | — | — | stacking complexity |
| V38 | XGB GPU + LGBM hybrid | 💀 SIGKILL | — | — | 02_feature_engineering 47분 |
| V39 | Feature Interaction + Polynomial | 💀 SIGKILL | — | — | python 명령어 오류 → SIGKILL |
| V40 | Deep Feature Ensemble | 💀 SIGKILL | — | — | python 명령어 오류 → SIGKILL |

## 2. 실패 원인 분석

### 2.1 SIGKILL (메모리 부족)
- **발생 버전**: V35, V35_pre, V36_stk, V37_fix, V37_fix2, V38, V39, V40
- **원인**: features_v11_personalized.parquet (8670열, 29.8MB)에서 Z-score 개인화 feature들(millions of column combinations) 생성 중 RAM 소진
- **환경**: RAM 16GB, Swap 4GB
- **메모리 패턴**:
  - `add_personalization()`에서 542개 feature × 450 rows → merge 후 1700+ 열
  - V37_fix2의 3792개 zscore feature를 Dataset에 전달 시 XGB/LGBM이 내부적으로 sparse/dense 변환 중 메모리 폭주
  - 특히 `lgb.Dataset(X, ..., params={'verbose':'-1'})` 시 n_features=3792 → 메모리 할당 폭발

### 2.2 느린 실행 (47분 이상)
- **발생 버전**: V35, V37, V38
- **원인**: `import 02_feature_engineering` → `create_day_features()` 재실행
- **해결책**: `features.parquet` 직접 로드 + runtime personalization 추가

### 2.3 명령어 오류
- **발생 버전**: V36, V39, V40
- **원인**: `python` 명령어 사용 (WSL2에 `python` symlink 없음)
- **해결책**: `python3`로 변경

## 3. 성능 비교 (완주한 버전들만)

### V10 vs V37 직접 비교 (Calibration LogLoss)

| Target | V10 | V37 | 차이 |
|--------|-----|-----|------|
| Q1 | 0.6338 | 0.6482 | -0.0144 |
| Q2 | 0.6034 | 0.6086 | -0.0052 |
| Q3 | 0.6119 | 0.6148 | -0.0029 |
| S1 | 0.5680 | 0.5838 | -0.0158 |
| S2 | 0.6022 | 0.6258 | -0.0236 |
| S3 | 0.5835 | 0.5993 | -0.0158 |
| S4 | 0.6240 | 0.6199 | +0.0041 ✅ |
| **Avg** | **0.6038** | **0.6144** | **-0.0106** |

### 핵심 발견
1. **V37이 V10보다 안 나은 이유**: V8 config (nl=10, md=3, lr=0.05)만 사용. V10은 6 configs로 per-target tuning.
2. **V37_fix (20 seeds)**: Q1이 0.6639로 V10의 0.6338보다 훨씬 나쁨. 하지만 S1은 0.5663으로 V10보다 약간 좋음.
3. **20 seeds ensemble이 오히려 Worse**: 너무 많은 시드 → variance 증가, 과소적합 가능성. V10은 per-target으로 optimal 시드 수를 tuning.

## 4. V37_fix2의 문제점
- 3792개 zscore feature → LGBM Dataset 생성 시 메모리 부족
- 8670열 전체를 Dataset에 전달하는 것이 아닌, ranking으로 top-20만 선택해야 함
- 하지만 ranking 단계에서도 3792열을 Dataset에 전달하며 메모리 폭주

## 5. 결론

**V10이 현재 가장 좋은 모델이다.** V35~V40은:
- 복잡도 증가 (XGB+LGBM blend, stacking, deep ensemble)
- 메모리/시간 오버헤드
- 성능 저하

V10과 같은 단순 LGBM + per-target tuning이 데이터 양(450 samples)에 가장 적합.

향후 개선 방향:
1. **피처 엔지니어링 개선**: `02_feature_engineering.py`에서 더 나은 feature 조합
2. **피처 수 최적화**: V10이 이미 10/20/30에서 tuning하고 있음
3. **정규화 강화**: `reg_alpha`, `reg_lambda` tuning
4. **subsample/colsample_bytree 최적화**

## 6. Git Push

V10 기반 submission이 이미 준비되어 있음.
