# DaCon2 Research: V146 → V160+ 다음 방향

## 현재 상태
- **V146** (베이스라인): OOF 0.63169, 5 seeds, LR(C=10) meta
- **V160** (BEST OOF): OOF 0.62240, 15 seeds, LR(C=10) meta, Δ=-0.00929
- V160은 OOF에서 개선되었지만 **LB 제출 검증 안 됨**

## 실패한 방향
| 버전 | 방법 | 결과 |
|------|------|------|
| V141-145 | 구조 변경 | 실패 |
| V155-159 | feature/meta 변경 | 실패 |
| V160 | seeds 5→15 | **성공** (-0.00929) |
| V161 | pseudo-labeling | 실패 (46개만 생성) |
| V162 | random seeds | 실패 (+0.00294) |
| V163 | two-level stacking | 실패 (+0.00779) |
| V164 | cross-fold ranking | 미완료 |

## 남은 높은 잠재력 방향

### 1. seeds 30→50 (diminishing return 예상)
- V160(15 seeds) → V16x(30 seeds): Δ ≈ -0.003-0.005 예상
- 50 seeds: Δ ≈ -0.002 예상 (매우 작음)
- 계산 비용: 30 seeds ≈ 1.5배, 50 seeds ≈ 3배
- **Low risk, small gain**

### 2. Per-target seed count optimization
- 현재: 모든 타겟에 동일 15 seeds
- 일부 타겟(특히 Q3, S4)은 student OOF가 더 높음 → 더 많은 seeds 필요
- Q1/S1/S2/S3/S4는 이미 좋음 → 적은 seeds로도 충분
- **Target-specific seed allocation**: 각 타겟별 optimal seed count를 탐색

### 3. GroupKFold → 다른 CV 전략
- 현재: GroupKFold (subject별 분할)
- Alternative: StratifiedGroupKFold (class balance 유지하면서 group 분할)
- Expected: Δ ≈ -0.001-0.003
- Risk: low

### 4. Feature cross-product (targeted)
- V156에서 group features = noise
- V157에서 wider features = noise
- 하지만 **targeted cross-product** from top-5 features might work
- 예를 들어 각 타겟의 top-5 feature pair만 생성 (top-K^2 → top-5^2 = 10개)
- Risk: medium
- Expected: Δ ≈ -0.001-0.003

### 5. Calibration after stacking
- V146/V160 meta predictions에 후처리 calibration 적용
- Isotonic calibration은 실패(isotonic = 폭주)
- 하지만 **platt scaling** (LR은 이미 Platt scaling)은 이미 적용됨
- **Temperature scaling per target**: meta output을 temperature로 scaling
- Risk: medium (calibration shift)
- Expected: Δ ≈ -0.001-0.002

### 6. Weighted ensemble (vs equal-weight)
- 현재: 15 seeds equal average
- 각 seed에 OOF-based weight 부여
- Weight = 1/OOF 또는 weight = exp(-OOF)
- Better students get more weight
- Risk: low (only affects test prediction, not training)
- Expected: Δ ≈ -0.001-0.002

## 추천 우선순위
1. **Weighted ensemble** (V165): Lowest risk, quick to implement
2. **StratifiedGroupKFold** (V166): Low risk, moderate gain expected
3. **Per-target seed optimization** (V167): Medium risk, moderate gain
4. **Calibration** (V168): Medium risk, small gain
5. **Seeds 30+** (V169): Low risk, diminishing returns

## V160 제출 전 체크리스트
- [ ] V160을 LB에 제출할지 승우 확인 필요
- [ ] V146 LB 결과 확인 (제출 완료했는지)
- [ ] V160이 V146보다 LB에서 좋은지 확인
