# ETRI 라이프로그 2024 — 수면·웰빙 예측 (DACON ch2026)

스마트폰·스마트워치 패시브 센서로 하루 단위 **7개 이진 타깃**(Q1–Q3 주관적 피로/스트레스/수면질, S1–S4 객관적 수면지표 준수)을 예측한다.

| 항목 | 내용 |
|---|---|
| 원천 데이터 | ETRI Lifelog 2024 (10명 · 700일 · 2024.5–12, arXiv:2508.03698) |
| 분할 | train 450 / test 250 — **동일 10명**, 일부는 기간 내 교차(interleaved), 일부는 미래 연장(extension) |
| 지표 | 7개 타깃 평균 binary log-loss (낮을수록 좋음) |
| 최종 성적 | **평균 log-loss 0.5988** (`submissions/submission_best.csv`) |

## 핵심 아이디어

**일반화되는 신호는 피처가 아니라 "피험자별 시간 개인화"다.** 타깃은 개인 평균 대비 편차이므로 전역 모델은 무의미하고, 유일한 단서는 같은 사람의 **하루 단위 자기상관**이다.

```
pred = shrink_to_0.5( w·recency(개인 최근 라벨 시간가중평균) + (1−w)·정규화 LightGBM )
```

- **recency(P)** — 같은 피험자의 과거 라벨을 `0.5^(gap/halflife)`로 시간가중 평균. 핵심·전이 가능.
- **LightGBM(G)** — `[subject_rate, 계절성, 주간(06–22h) 센서 집계]`에 강한 정규화. S 타깃에 소폭 기여.
- **(w,s)** — 타깃별로 **전방(시간블록) CV 3분할·5분할 둘 다**에서 순수 recency를 이겨야만 채택(과적합 게이트).
- **Q1–Q3** — 짧은 halflife recency로 덮어써 교차 구간의 자기상관을 활용(R53 개선).
- **S1** — 폰 야간센서로 복원한 수면시간(TST) 피처를 외과적으로 추가.

## 모델 및 학습 방법

### 학습 절차 (타깃별 독립, 7회 반복)
1. **recency(P)** — 학습 데이터만으로 같은 피험자의 라벨을 시간가중 평균. 학습이 필요 없는 비모수 추정.
2. **LightGBM(G)** — `[subject_rate, 계절성(sin/cos), 주간 센서 mean/std/count]`로 학습. 약한 일 단위 신호가 쉽게 과적합하므로 의도적으로 작고 강하게 정규화: `num_leaves=8, lr=0.02, n_estimators=250, reg_lambda=5, reg_alpha=1, subsample=0.8, colsample=0.6`.
3. **블렌드 선택** — 전방 시간블록 CV(3·5분할)로 OOF의 P·G를 만들고, `w·P+(1−w)·G`를 shrink. **3분할·5분할 둘 다에서 순수 recency를 이길 때만** `(w,s)` 채택, 아니면 `w=1`로 폴백(과적합 방지 게이트).
4. **전체 재학습 후 추론** — 채택된 설정으로 train 전체에 G 재적합, test 250행 예측.
5. **Q1–Q3 덮어쓰기** — 짧은 halflife recency로 교체(자기상관 활용).

`subject_rate`·recency 이웃은 **항상 학습 행만** 사용하고, 폴드는 피험자별 시간순 블록으로 구성해 누수를 차단한다.

### 시도한 모델 클래스와 결과

| 모델 | 결과 / 이유 |
|---|---|
| **recency 개인화** | 핵심. LB로 전이되는 유일한 신호. 다른 모든 것의 기준선. |
| **LightGBM (정규화)** | 과적합 게이트 통과 시 S 타깃에 소폭 기여. 최종 G 컴포넌트. |
| XGBoost · CatBoost · HistGBM | ≈ LightGBM. 전방 CV에서 다양성 이득 없음. |
| TabPFN | 중첩 CV에선 Q에 −0.017로 유망했으나 **실제 LB에서 악화** → 과적합. |
| FT-Transformer / RTDL / 1D-CNN | 기준율 수준, 전이 0. 한계는 모델 용량이 아니라 **신호**. |
| 스태킹 / winner-stack / hill-climb 앙상블 | 랜덤 CV 환상 이득, 전방 홀드아웃에서 전부 기각. |
| **GNN / 그래프 신경망** | **시도하지 않음(부적합).** 의미 있는 그래프 구조가 없다 — 노드는 피험자×날짜 700개뿐이고, 핵심 신호는 메시지 패싱이 아니라 **같은 피험자 내부의 시간 자기상관**이라 recency가 이를 직접·정확히 포착한다. 피험자 그래프(10노드)나 kNN 그래프를 만들어도 정규화 GBM/recency 대비 이득이 기대되지 않아 우선순위에서 제외. |

**왜 트리/개인화가 딥·그래프 계열을 이기나:** test 절반이 미래 연장 구간이라 가까운 라벨 이웃이 없고, 일 단위 피처는 시간을 넘어 일반화되지 않는다. 표현학습 용량을 키우면 이 약한 신호에 과적합할 뿐이다. 자기상관은 실재·전이 가능하므로 그것만 잡는 recency가 가장 견고하다. (전체 ~140개 실험은 `docs/research_summary.md`.)

## 저장소 구조

```
run.py                  진입점 — 최적 제출 재현
src/                    파이프라인 모듈
  config.py             경로·하이퍼파라미터 일원화
  data.py               라벨/템플릿/원시 센서 로드
  features.py           계절성·센서 집계·subject_rate
  recency.py            시간 개인화(핵심 신호)
  sleep.py              TST·수면질 피처 접근
  model.py              LightGBM + 블렌드 + shrink
  validation.py         전방 시간블록 교차검증
  pipeline.py           전체 오케스트레이션 → 제출
scripts/
  build_sleep_features.py  분 단위 수면 검출 → sleep_v3.parquet (1회)
  test_pipeline.py         재현 결과가 최적 제출과 일치하는지 스모크 테스트
docs/                   architecture(mermaid)·research_summary·future_work·external_data_analysis
```

> `data_raw/`(원천 데이터)와 `submissions/`(결과물)는 용량·라이선스 때문에 git에서 제외(로컬 보관). 실행하려면 `data_raw/`에 대회 데이터를 두면 된다. `archive/`에는 전체 연구 이력(실험 스크립트·로그·구 제출)이 보존돼 있다.

## 실행

```bash
pip install -r requirements.txt          # numpy, pandas, pyarrow, scikit-learn, lightgbm
python scripts/build_sleep_features.py   # 1회: data_processed/sleep_v3.parquet 생성
python run.py                            # submissions/submission_reproduced.csv 생성
python scripts/test_pipeline.py          # (선택) 최적 제출과 일치 검증
```

시드 고정(`SEED=42`), 네트워크·GPU 불필요, 결정론적 재현. 재현 결과는 Q 타깃 동일·S1 근사로 최적 제출과 지표 4째 자리까지 일치.

## 검증 전략 (가장 중요한 교훈)

- **랜덤 K-fold는 +0.02 낙관 편향** — 모델이 쓰는 시간 인접성을 흩뜨림.
- 신뢰 가능한 검증기는 **전방 시간블록 CV**(피험자별 앞쪽 날짜로 학습→뒤쪽 검증)뿐 — test의 extension 구간을 모사.
- **날짜 단위 피처는 시간으로, 피험자 단위 피처는 피험자로** 분할 검증해야 한다. 이를 어겨 날씨/TabPFN 개선이 내부 검증을 통과하고도 실제 LB에서 실패함.
- 결론: 내부 점수보다 **LB 앵커**를 신뢰한다.

## 실험 타임라인 / 한계

```
개인평균 베이스라인 → R8(recency+LGBM, LB 0.603) → R53(Q 단기 recency, LB 0.59915)
→ R86(시드 robust-mean, LB 0.5988, 최종) → TabPFN/날씨/스태킹(모두 LB 악화, 기각)
```

- **Q1–Q3**는 편차라 개인 기준율이 ≈0.5 → 자기상관만 유효(이웃 라벨이 가까운 test 행에서만).
- **S2–S4**는 원 연구의 매트리스 수면센서로 측정 — 공개 폰/워치 센서엔 없어 물리적으로 복원 불가.
- 따라서 공개 센서만으론 honest 한계 ≈ **0.59**. 자세한 내용은 `docs/research_summary.md`.
