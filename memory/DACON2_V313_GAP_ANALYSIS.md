# V313 OOF-LB Gap 분석 결과 (2026-06-02)

## 핵심 발견

### 1. Student-Meta Gap ≠ OOF-LB Gap
- V308: Student avg=0.692, Meta OOF=0.622, gap=0.070
- V308 실제 OOF-LB gap: +0.01658
- **Student-Meta gap은 train-time ensemble variance 지표**
- **OOF-LB gap은 test distribution calibration 지표**
- 두 gap은 서로 무관함. Student avg가 일정하면 OOF-LB gap도 일정할 가능성 높음.

### 2. Student 성능은 일정
- V308: 0.69212
- V312: 0.69212
- V313: 0.69193
- Student avg가 거의 동일 → calibration이 유사 → OOF-LB gap도 유사할 가능성

### 3. OOF-LB Gap 가정 하의 예측
- V312: OOF 0.61448 + 0.01658 = 0.63106 (V308 0.63893 대비 **-0.008**)
- V313: OOF 0.59512 + 0.01658 = 0.61170 (V308 0.63893 대비 **-0.027**)

### 4. 리스크
- V313의 Meta OOF가 더 낮아져(0.595) student를 더 강하게 fitting → OOF-LB gap이 더 커질 수 있음
- 하지만 V312도 student avg와 V308이 동일하므로 V312의 gap은 V308과 유사할 것
- V313은 30 seeds → 더 나은 averaging → 오히려 gap이 더 작아질 수도 있음

## 결론
- **V312와 V313 모두 V308을 넘을 가능성 높음**
- LB 제출로 검증 필요
- V313이 더 큰 개선 예상 (Δ -0.027 vs V308)
