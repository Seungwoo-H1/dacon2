# MEMORY.md - Long-Term Memory

## 승우 (Seungwoo Hong)
- Telegram 사용, 한국어
- 시간대: KST (Asia/Seoul)
- 주요用途: 코드 분석, Dacon2 경진대회

## 집가헤응 (🏠)
- AI 펫, 차분하고 정확한 스타일
- 멀티에이전트 코드 분석 파이프라인 구축

## Dacon2 ETRI 경진대회
### 베이스라인
- **LGBM V10**: cal OOF log_loss = **0.6038** (7 targets avg)
  - Q1: 0.6338, Q2: 0.6034, Q3: 0.6119, S1: 0.5680, S2: 0.6022, S3: 0.5835, S4: 0.6240
- **LGBM V13** (better calibration): **0.6385** (Worse!)
  - 개인별 z-score 적용했으나 overfit. V10이 더 좋음.

### FT-Transformer (GPU RTX 4060 Laptop 8GB)
#### V1 (d=64, layers=4, fs=141)
- AVG AUC: **0.5225** (-0.0813 vs V10)
- 데이터 부족, 과대적합

#### V2 (d=32, layers=2, fs=20, drop=0.4) — BEST DL
- Config: d_token=32, n_heads=2, dropout=0.4, feature_select=20, lr=8e-4
- AVG AUC: **0.5847** (-0.0191 vs V10)
  - Q1: 0.6022 (-0.0016), Q2: 0.5444 (-0.0594), Q3: 0.6181 (+0.0143)✅
  - S1: 0.6693 (+0.0655)✅✅, S2: 0.5437 (-0.0601)
  - S3: 0.5686 (-0.0352), S4: 0.5467 (-0.0571)

### Ensemble (LGBM V10 + FT-V2, 50/50 blend)
- AVG AUC: **0.5943** (-0.0095 vs V10) — 아쉽게도 미달
- Best weights: FT=0.50, LGBM=0.50

### V35~V40 실험 (2026-05-05 완료)
- **V37 완주** (V8 config + 4 feat counts + 20 seeds): Avg Cal 0.6144, V10보다 -0.0106
- **V37_fix**: S4에서 SIGKILL (20 seeds training 중 메모리 폭주)
- **V37_fix2**: SIGKILL (features_v11_personalized 3792 zscore feature → Dataset 메모리 폭발)
- **V35/V35_pre/V36_stk/V38**: SIGKILL
- **V36/V39/V40**: `python` 명령어 오류
- **근본 원인**: 3000+ zscore feature를 LGBM Dataset에 전달 시 메모리 할당 폭발 (RAM 16GB, Swap 4GB 부족)
- **결론**: V10이 현재 최고. V35~V40은 복잡도만 증가.
- 향후 방향: 피처 엔지니어링 개선 또는 정규화 파라미터 tuning

### V53 — Deep Feature Engineering (2026-05-06)
#### 핵심 발견 & 버그 픽스
- **설정 불일치 버그**: `V53_CONFIGS`/`CFGS` 정의 있었으나 `train_and_predict()`/`rank_features_importance()`에서 하드코딩 파라미터만 사용
  - 수정: 두 함수 모두 `cfgs`, `v53_cfgs` 인자 추가 → cfg_name 기반 파라미터 로드
- **personalization fragmentation fix**: `add_personalization()`에서 df.insert → batch concat
  - old: 576 cols (mean/std 누수), new: 294 cols (base 141 + zscore 141 + meta 12)
  - 속도 개선, 메moire 절감

#### 리더보드 제출물
- **V53 final**: Score **0.6535822621** (target별 cfg: deep/wide/v48/safety)
- **V53 Swept**: optimized n_feat 적용, avg CV 0.6674→0.6500

#### V53 Feature Sweep 결과
- n_feat ±3 탐색: Q1:20→17, Q2:15→17, Q3:8→11, S1:20→17, S2:20→20, S3:20→23, S4:20→23
- **AVG CV: 0.6674 → 0.6500 (Δ=-0.0174) ✅ 모든 타깃 개선**
  - Q1: -0.0333, Q3: -0.0303, S4: -0.0224가 가장 큼
  - S2: 최적점 baseline과 동일(20), S3: 20→23 미미한 개선

### Key Findings
- 450 샘플에 Transformer는 과대
- S1은 DL이 LGBM을 넘었음 (+0.0655)
- LGBM이 전체적으로 더 안정적
- **LGBM V10 CV 0.6038, V53 리더보드 0.65358**
- 피처 선택 미세탐색으로 CV 개선 확인
- **향후**: V53 Swept 리더보드 제출, 더 넓은 탐색(±5), seed 안정성 검증

### WSL2 GPU 설정
- RTX 4060 Laptop 8GB, Windows Driver 581.83 (CUDA 13.0)
- PyTorch 2.6.0+cu124, CUDA 12.4
- LGBM 훈련 시 CPU 코어 다 사용 (1600%+)
- FT-Transformer에서 `LD_LIBRARY_PATH` 설정 시 CPU 폴백 주의

### 프로젝트 구조
- `/home/mwoo423/.openclaw/workspace/dl_project/` — DL 프로젝트
- `/home/mwoo423/projects/dacon2/` — LGBM 프로젝트
- `/home/mwoo423/.openclaw/workspace/dl_project/results/ft_v2_gpu/` — FT-V2 OOF
- `/home/mwoo423/.openclaw/workspace/dl_project/results/ensemble_lgbm_ftv2/` — Ensemble 결과

### WSL2 GPU 설정
- RTX 4060 Laptop 8GB, Windows Driver 581.83 (CUDA 13.0)
- `/usr/lib/wsl/lib/libcuda.so.1` 브리지 사용
- PyTorch 2.6.0+cu124, CUDA 12.4
- **주의**: miniconda 환경에서 LGBM 훈련 시 CPU 코어 다 사용 (1600%+)
- **주의**: FT-Transformer 스크립트에서 `LD_LIBRARY_PATH` 설정하면 CPU로 폴백

### 프로젝트 구조
- `/home/mwoo423/.openclaw/workspace/dl_project/` — DL 프로젝트
- `/home/mwoo423/projects/dacon2/` — LGBM 프로젝트 (V13 훈련 중)
- `/home/mwoo423/.openclaw/workspace/dl_project/results/ft_v2_gpu/` — FT-V2 OOF
- `/home/mwoo423/.openclaw/workspace/dl_project/results/ensemble_lgbm_ftv2/` — Ensemble 결과
