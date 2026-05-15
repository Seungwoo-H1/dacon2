# 데이터 인벤토리 및 확보 상태

## 공식 안내 데이터 구조

```text
data/
├── ch2025_data_items/
│   ├── ch2025_mACStatus.parquet
│   ├── ch2025_mActivity.parquet
│   ├── ch2025_mAmbience.parquet
│   ├── ch2025_mBle.parquet
│   ├── ch2025_mGps.parquet
│   ├── ch2025_mLight.parquet
│   ├── ch2025_mScreenStatus.parquet
│   ├── ch2025_mUsageStats.parquet
│   ├── ch2025_mWifi.parquet
│   ├── ch2025_wHr.parquet
│   ├── ch2025_wLight.parquet
│   └── ch2025_wPedo.parquet
├── ch2026_metrics_description.pdf
├── ch2026_metrics_train.csv
└── ch2026_submission_sample.csv
```

## 제공 범위
- 라이프로그 12개 항목, 700일분
- 레이블(7개 지표), 450일분

## 현재 자동 수집 결과
- 대회 데이터 페이지 메타 정보 수집: 완료
- 실제 데이터 파일 다운로드: 미완료(로그인 필요)

## 로그인 필요 사항
- `ch2025_data_items.zip (122.0MB)` 다운로드는 데이콘 로그인/참가 권한 필요
- 브라우저 세션 인증 또는 사용자 수동 다운로드 후 `dacon2/data_raw/` 배치 필요

## 권장 로컬 폴더 구조

```text
dacon2/
├── data_raw/
│   ├── ch2025_data_items.zip
│   ├── ch2026_metrics_train.csv
│   ├── ch2026_submission_sample.csv
│   └── ch2026_metrics_description.pdf
└── data_unpacked/
    └── ch2025_data_items/*.parquet
```

## 데이터 수신 후 즉시 수행할 작업
1. 파일 무결성 체크(크기/행수/컬럼)
2. 타임스탬프 정렬 및 키 정합성 확인
3. 학습/검증 분할 전략 설계(시간 누수 방지)
4. 샘플 제출 파일 컬럼명/순서 완전 일치 검증

## 출처
- https://dacon.io/competitions/official/236690/data
