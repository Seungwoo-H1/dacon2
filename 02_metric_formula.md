# 평가 지표(산식) 정리

## 공식 표기
- 리더보드 평가 산식: **Average Log-Loss**
- Public Score: 테스트의 사전 샘플링 44%
- Private Score: 테스트 100%

## Log Loss 기본식
이진 분류 기준 샘플별 Log Loss:

\[
\text{LogLoss}_i = -\left(y_i \log(p_i) + (1-y_i)\log(1-p_i)\right)
\]

- \(y_i\): 정답 라벨(0 또는 1)
- \(p_i\): 정답 클래스에 대한 예측 확률

N개 샘플 평균:

\[
\text{LogLoss} = \frac{1}{N}\sum_{i=1}^{N}\text{LogLoss}_i
\]

## Average Log-Loss 해석(대회형 멀티타깃)
본 대회는 7개 지표(Q1~Q3, S1~S4)를 예측하므로 보통 다음처럼 계산됨:
1. 지표별 Log Loss 계산
2. 지표별 점수 평균(또는 플랫폼 정의 집계)으로 최종 Average Log-Loss 산출

정확한 내부 집계 구현은 플랫폼 평가 서버 정의를 따르므로, 실제 제출 기준은 데이콘 채점 결과를 우선해야 함.

## 모델링 시 실무 포인트
- 확률 캘리브레이션(Platt/Isotonic) 검토
- 지표별 불균형 대응(가중치/샘플링)
- CV 폴드별 분포 보존 및 누수 차단
- 추론 시 확률 범위 클리핑(\(1e^{-15}\) ~ \(1-1e^{-15}\)) 검토

## 검증 전략 제안
- 지표별 OOF Log Loss 산출
- 전체 Average Log-Loss와 지표별 편차 동시 모니터링
- Public shake-up 대비해 CV 안정성(표준편차) 관리

## 출처
- https://dacon.io/competitions/official/236690/overview/evaluation
