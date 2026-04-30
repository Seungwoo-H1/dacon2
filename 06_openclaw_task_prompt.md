# OpenClaw 실행용 작업 프롬프트

아래 프롬프트를 OpenClaw에 그대로 넣어 실행:

---
너는 대회 전담 AI 리서치 에이전트다.
작업 디렉터리 `dacon2/` 안의 md 문서들을 먼저 모두 읽고, 그 내용을 근거로 대회 분석을 수행하라.

필수 입력 문서:
- `dacon2/00_overview.md`
- `dacon2/01_rules_and_constraints.md`
- `dacon2/02_metric_formula.md`
- `dacon2/03_submission_spec.md`
- `dacon2/04_data_inventory.md`
- `dacon2/05_strategy_plan.md`

작업 목표:
1. 평가 지표(Average Log-Loss)를 기준으로 모델링 전략을 구체화
2. 데이터 수집 이후 즉시 실행할 EDA 체크리스트 작성
3. 재현성 검증을 통과할 수 있는 코드/환경/문서 제출 계획 수립
4. 논문 제출(IEEE 6-page)까지 고려한 실험 로그 체계 설계

출력 형식:
- `dacon2/10_execution_plan.md` 생성
- `dacon2/11_eda_checklist.md` 생성
- `dacon2/12_model_experiment_table.md` 생성
- `dacon2/13_reproducibility_checklist.md` 생성
- 각 문서에 "즉시 실행 가능한 To-do" 항목 포함

주의:
- 규칙 문서와 충돌하는 제안 금지
- 데이터가 아직 없으면 "데이터 수신 대기 단계"와 "수신 즉시 단계"를 분리해서 작성
- 불확실한 내용은 가정으로 표시하고 검증 방법을 제시
---
