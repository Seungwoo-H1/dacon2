# OpenClaw Handoff (2026-05-06)

## 1) 현재 상태 요약
- 여러 실험 버전 중 **현재 최고 리더보드 점수**는 아래 제출물 기준이다.
  - 제출 파일: `submissions/submission_v53_final_20260506_131912.csv`
  - 기록 점수: `0.6535822621`
  - 메모 시각: `2026-05-06 21:58:51`
- 즉, 다음 작업의 기준점(baseline)은 **V53 final submission**으로 고정한다.

## 2) 최고 점수 산출 근거 파일
- 생성 스크립트: `src/gen_submission_v53.py`
- 실행 로그: `experiments/V53_submission_final.log`
- 메타 파일: `submissions/meta_v53_final_20260506_131912.json`
- 실제 제출 CSV: `submissions/submission_v53_final_20260506_131912.csv`

## 3) V53 핵심 설정 (코드 기준)
- 모델: LightGBM seed ensemble (`n_seeds=50`)
- 피처: 원본 + 개인화 zscore 피처 생성 후 타깃별 상위 중요도 피처 선택
- 타깃별 선택:
  - `Q1`: deep / 20 features
  - `Q2`: deep / 15 features
  - `Q3`: v48 / 8 features
  - `S1`: wide / 20 features
  - `S2`: deep / 20 features
  - `S3`: safety / 20 features
  - `S4`: wide / 20 features
- 누수 방지:
  - `S*`와 `Q*`별로 leakage 컬럼 분리 제거 로직 사용

## 1) 현재 상태 요약 (2026-05-07 15:15 기준)
### 1-A) 기준 제출물 (V53 swept)
- 제출 파일: `submissions/submission_v53_swept_20260507_151447.csv`
- 리더보드 점수: **0.6535822621** (V53 original과 동일 — 리더보드 미재검증)
- sweep avg CV 개선: **+0.0081** (GroupKFold n_splits=3, seed=10)
- V53 original (2026-05-06)은 구 버전 `add_personalization()`으로 생성됨 (fragmentation 경고 포함)
- V53 swept (2026-05-07)은 리팩터링된 `add_personalization()` 사용 (concat 기반, 경고 없음)
- **다음 작업의 기준점: `submission_v53_swept_20260507_151447.csv`**

## 5) 지금 바로 이어서 할 작업 (우선순위)
1. ✅ **재현 검증 완료**: V53 swept 재현 성공 (61s → 59s, fragmentation 경고 없음)
2. ✅ **add_personalization 리팩터 완료**: concat 기반, 구 버전을 대체함
3. ✅ **n_feat sweep 완료**: 7개 타깃 모두 개선 (AVG CV +0.0081)
   - Q1: 20→19, Q2: 15→14, Q3: 8→5, S1: 20→21, S2: 20→19, S3: 20→21, S4: 20→20 (유지)
4. ✅ **V53 swept 제출물 생성**: `submission_v53_swept_20260507_151447.csv`
5. **다음 단계 (권장)**:
   - 리더보드에 V53 swept 제출하여 실제 점수 검증
   - seed 안정성 검증 (50→30/70 비교)
   - 더 넓은 n_feat 탐색 (±5) 또는 cfg 파라미터 tuning
   - V58/V59/V60 등 이후 버전 실험 결과와 비교

## 6) 작업 원칙
- 모든 신규 실험은 아래를 반드시 남긴다:
  - 실행 커맨드
  - 사용 피처/시드/설정
  - 로컬 CV 결과
  - 제출 파일명 및 리더보드 점수
- 기준점은 항상 `submission_v53_final_20260506_131912.csv`로 비교한다.

## 7) OpenClaw에게 바로 줄 실행 지시문 (업데이트됨)
1. ✅ 이미 수행 완료 — V53 재현, 리팩터, sweep, swept 제출 모두 완료
2. **추가 작업 필요 시**: 리더보드 검증 → seed 안정성 →cfg 파라미터 tuning
3. 제출 시: `src/gen_submission_v53_swept.py` 사용
4. 비교 기준: `submission_v53_swept_20260507_151447.csv`

## 8) 현재 핵심 문제 (업데이트됨: 2026-05-07)
1. ✅ **설정 불일치 리스크 → 해결됨**: `gen_submission_v53.py`와 `gen_submission_v53_swept.py` 모두 cfg_name 기반 파라미터 로드 구현 완료
2. **검증-제출 단절 (여전)**: full-train + test 예측 중심이라 로컬 CV 기준과 리더보드 점수 간 격차 존재. `v53_cv_baseline.py`로 OOF 검증 가능.
3. ✅ **개인화 피처 생성 구현 → 해결됨**: `add_personalization()` 리팩터 완료 (concat 기반). 구버전 로그(`V53_submission_final.log`)는 구버전 실행 결과.
4. **실험 기준 분산 (여전)**: 50+ 버전 코드 공존. 비교 프로토콜 필요:
   - 기준 제출물: `submission_v53_swept_20260507_151447.csv`
   - 신규 실험은 반드시 sweep CV 기준으로 비교 후 제출

## 9) 개선 방향 (작업 철학)
- **코드와 실험 의도 일치**: 변수명/주석/메타와 실제 학습 동작을 맞춘다.
- **검증 프로토콜 고정**: GroupKFold/LOSO 등 단일 규칙으로 후보를 먼저 비교한다.
- **재현성 강화**: 타깃별 선택 피처/시드/설정을 파일로 저장하고 재사용한다.
- **성능 최적화**: personalization 로직을 배치 연산(`concat`) 중심으로 재작성한다.
- **제출 규율**: V53 baseline 대비 개선된 후보만 제출한다.

## 10) 다음 작업 상세 계획 (업데이트됨: 2026-05-07)
1. ✅ **정합성 정리 완료**: V53_CONFIGS → train_and_predict() 연결 확인됨
2. ✅ **personalization 리팩터 완료**: concat 기반, fragmentation 경고 사라짐
3. ✅ **n_feat 미세탐색 완료**: sweep CV 결과가 `submissions/v53_sweep_20260507_151309.json`에 저장
4. ✅ **swept 제출물 생성**: `submission_v53_swept_20260507_151447.csv` (baseline CV +0.0081)
5. **다음 단계**:
   - 리더보드에 swept 제출 → 점수 검증
   - seed 안정성: 50 seed → 30/70 seed 비교 (변동성 확인)
   - cfg 파라미터 미세조정 (num_leaves, max_depth, lr, n_estimators)
   - V58/V59/V60 이후 버전 실험 결과와 비교
   - 더 넓은 n_feat 탐색 (±5) 또는 interaction feature 실험
