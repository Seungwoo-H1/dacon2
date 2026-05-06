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

## 4) 재현 방법 (즉시 실행)
1. 프로젝트 루트에서 실행:
   - `python src/gen_submission_v53.py`
2. 산출물 확인:
   - `submissions/submission_v53_final_YYYYMMDD_HHMMSS.csv`
   - `submissions/meta_v53_final_YYYYMMDD_HHMMSS.json`
3. 로그 확인:
   - `experiments/V53_submission_final.log` (기존 참고)

## 5) 지금 바로 이어서 할 작업 (우선순위)
1. **재현 검증**: 동일 스크립트 재실행 후 새 CSV 생성/기본 통계 비교
2. **안정성 점검**: seed 수 50 고정 vs 30/70 비교 (점수 변동성 확인)
3. **성능 개선 포인트**:
   - `add_personalization()`의 DataFrame fragmentation 경고 해결
   - zscore 생성 로직을 `concat` 기반으로 바꿔 속도/메모리 개선
4. **피처 선택 미세튜닝**:
   - 타깃별 `n_feat` 주변 탐색 (예: +-3)
   - `Q3` 저차원(8개) 설정의 민감도 검증
5. **제출 전략**:
   - 신규 후보는 항상 V53와 A/B 비교 후 제출
   - 개선이 없으면 V53 유지

## 6) 작업 원칙
- 모든 신규 실험은 아래를 반드시 남긴다:
  - 실행 커맨드
  - 사용 피처/시드/설정
  - 로컬 CV 결과
  - 제출 파일명 및 리더보드 점수
- 기준점은 항상 `submission_v53_final_20260506_131912.csv`로 비교한다.

## 7) OpenClaw에게 바로 줄 실행 지시문
아래 순서로 바로 진행:
1. `src/gen_submission_v53.py`와 `experiments/V53_submission_final.log`를 먼저 읽어 V53 재현 파이프라인을 이해한다.
2. V53를 1회 재생성해 산출물 무결성(행수/컬럼/예측범위)을 확인한다.
3. 성능 영향 없는 속도 개선(특히 personalization 경고 구간)부터 수정한다.
4. 그 다음 타깃별 `n_feat` 미세탐색을 수행하고, V53 대비 개선된 후보만 제출 후보로 남긴다.

## 8) 현재 핵심 문제 (반드시 먼저 인지)
1. **설정 불일치 리스크**
   - `src/gen_submission_v53.py`에 `V53_CONFIGS`, `CFGS`가 정의되어 있으나 실제 학습 함수에서 타깃별 설정이 온전히 반영되지 않을 수 있다.
   - 즉, "best config"라는 이름과 실제 동작의 불일치 가능성이 있다.
2. **검증-제출 단절**
   - 제출 스크립트는 full-train 학습 + test 예측 중심이라, 동일 파일 내부에서 안정적인 OOF/CV 비교 근거가 약하다.
   - `avg_cal_loss_v53` 등 메타 수치는 하드코드일 가능성이 있어 신뢰도 점검 필요.
3. **개인화 피처 생성 구현 비효율**
   - `add_personalization()`에서 DataFrame fragmentation 경고가 대량 발생.
   - 반복 실험 시 속도/메모리/재현성 관리에 부담.
4. **실험 기준 분산**
   - 여러 버전 코드가 공존하므로, 비교 프로토콜이 통일되지 않으면 개선 여부 판단이 흔들린다.

## 9) 개선 방향 (작업 철학)
- **코드와 실험 의도 일치**: 변수명/주석/메타와 실제 학습 동작을 맞춘다.
- **검증 프로토콜 고정**: GroupKFold/LOSO 등 단일 규칙으로 후보를 먼저 비교한다.
- **재현성 강화**: 타깃별 선택 피처/시드/설정을 파일로 저장하고 재사용한다.
- **성능 최적화**: personalization 로직을 배치 연산(`concat`) 중심으로 재작성한다.
- **제출 규율**: V53 baseline 대비 개선된 후보만 제출한다.

## 10) 다음 작업 상세 계획 (즉시 실행용)
1. **정합성 정리 (가장 먼저)**
   - `src/gen_submission_v53.py`에서 타깃별 config가 실제 모델 파라미터에 반영되는지 점검.
   - 반영 안 되면 연결 구현 또는 미사용 설정 제거로 혼동 제거.
2. **CV 기준선 확보**
   - V53 동일 피처/시드 기준으로 OOF/CV 평균 log-loss를 먼저 산출.
   - 이후 모든 실험은 동일 프로토콜로만 비교.
3. **personalization 리팩터**
   - fragmentation 경고 구간을 `concat` 기반으로 변경.
   - 변경 전/후 실행시간, 메모리, 점수 차이 기록.
4. **피처 선택 안정화**
   - 타깃별 최종 피처 목록을 JSON으로 저장/로딩.
   - 재실행 시 동일 피처 사용으로 재현성 확보.
5. **미세 탐색**
   - 타깃별 `n_feat` 주변값(예: +-3) 탐색.
   - 개선 폭이 있는 타깃만 파고들고, 악화 타깃은 즉시 롤백.
6. **최종 제출 판단**
   - 로컬 기준 개선 + 예측 분포 이상 없음 + 누수 규칙 준수 시에만 제출.
   - 기준 미달이면 `submission_v53_final_20260506_131912.csv` 유지.
