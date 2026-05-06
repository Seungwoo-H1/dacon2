# 🏠 Deep Learning Tabular — Dacon2 / ETRI

기존 LightGBM V10 (cal OOF: 0.6038)을 딥러닝으로 재도전.

## 프로젝트 구조

```
dl_project/
├── src/
│   ├── 00_prepare_data.py    # 데이터 로드/정제
│   ├── 01_baseline_lgbm.py   # V10 재현 (베이스라인)
│   ├── 02_ft_transformer.py  # FT-Transformer training
│   └── 03_ensemble.py        # LGBM + FT blend
├── configs/
│   └── ft_config.yaml        # FT-Hyperparams
├── data/                     # raw data
├── data_processed/           # processed parquet
├── models/                   # saved models
├── results/                  # OOF predictions
├── requirements.txt
└── README.md
```

## 사용법

### 1. 베이스라인 재현 (LGBM V10)
```bash
python src/01_baseline_lgbm.py --target <target_name>
```

### 2. FT-Transformer
```bash
python src/02_ft_transformer.py --target <target_name> \
    --n-layers 4 --n-heads 4 --batch-size 256 --lr 1e-3
```

### 3. Ensemble
```bash
python src/03_ensemble.py --targets t1 t2 t3
```

## 핵심 가정
- FT-Transformer는 Tabular에서 attention mechanism을 통해 feature interaction 학습
- 기존 V10의 personalization (z-score)은 FT-Transformer에서도 적용
- Subject-aware validation (GroupKFold)으로 leakage 방지

## 참고
- FT-Transformer: [gustavo-bernuardi/Tabular-Datasets](https://github.com/gustavo-bernuardi/Tabular-Datasets)
- pytabkit: [pytabkit](https://github.com/PyTabKit)
