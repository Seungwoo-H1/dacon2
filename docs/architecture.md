# Architecture

All diagrams are GitHub-flavored Mermaid and render directly in the web UI.

## 1. Overall pipeline

```mermaid
flowchart TD
    RAW["data_raw/<br/>labels + 12 sensor parquets"] --> SLEEP["scripts/build_sleep_features.py<br/>minute-grid sleep detection"]
    SLEEP --> SV["data_processed/sleep_v3.parquet<br/>TST, efficiency, WASO, ..."]
    RAW --> FEAT["src/features.py<br/>seasonal · sensor aggs · subject_rate"]
    RAW --> REC["src/recency.py<br/>temporal personalization"]
    FEAT --> PIPE["src/pipeline.py"]
    REC --> PIPE
    SV --> PIPE
    PIPE --> SUB["submissions/submission_reproduced.csv<br/>avg log-loss ~0.599"]
```

## 2. Per-target training & inference

```mermaid
flowchart LR
    subgraph PerTarget["for each of the 7 targets"]
        Y["labels y"] --> OOF["out-of-fold P (recency) and G (LightGBM)<br/>on 3-block & 5-block forward CV"]
        OOF --> GATE{"blend beats pure-P<br/>on BOTH schemes?"}
        GATE -- yes --> WS["adopt (w, s)"]
        GATE -- no --> PUREP["w=1 (pure recency)"]
        WS --> REFIT["refit on all train -> predict 250 test rows"]
        PUREP --> REFIT
    end
    REFIT --> QO{"target in Q1/Q2/Q3?"}
    QO -- yes --> OVR["override with SHORT-halflife recency (R53)"]
    QO -- no --> KEEP["keep blended prediction"]
    OVR --> OUT["submission column"]
    KEEP --> OUT
```

## 3. Data flow (train vs test, leakage boundaries)

```mermaid
flowchart TD
    L["train labels (450)"] -->|train-only| SR["subject_rate (shrunk)"]
    L -->|train-only| RECN["recency neighbors"]
    SENS["daytime sensor aggregates"] --> XG["GBM design matrix"]
    SR --> XG
    SEAS["seasonal encodings"] --> XG
    TST["TST features (S1 only)"] --> XG
    XG --> GBM["LightGBM (regularized)"]
    RECN --> P["recency P"]
    GBM --> G["GBM G"]
    P --> BL["blend + shrink"]
    G --> BL
    BL --> TESTP["test predictions (250)"]
```

## 4. Forward (time-blocked) validation

```mermaid
flowchart LR
    subgraph subj["each subject's chronological days"]
        direction LR
        E["earliest --- TRAIN window ---"] --> V["--- VALIDATION block ---"] --> F["future"]
    end
    note["3-block: val = [0.60-0.74],[0.74-0.87],[0.87-1.0]<br/>5-block: val = [0.50-0.60]...[0.90-1.0]<br/>train = everything strictly earlier"]
```

## 5. Repository architecture

```mermaid
flowchart TD
    RUN["run.py"] --> P["src/pipeline.py"]
    P --> CFG["src/config.py"]
    P --> D["src/data.py"]
    P --> F["src/features.py"]
    P --> R["src/recency.py"]
    P --> S["src/sleep.py"]
    P --> M["src/model.py"]
    P --> V["src/validation.py"]
    S -.reads.-> SV["data_processed/sleep_v3.parquet"]
    SCR["scripts/build_sleep_features.py"] -.writes.-> SV
    TEST["scripts/test_pipeline.py"] --> P
```

## 6. Experiment workflow (how the research loop ran)

```mermaid
flowchart LR
    H["hypothesis"] --> EX["experiment script (archive/exp_scripts/)"]
    EX --> CV["forward-time CV"]
    CV --> GATE{"beats best<br/>on BOTH schemes?"}
    GATE -- no --> LOG1["log + discard"] --> H
    GATE -- yes --> SUBMIT["build submission"]
    SUBMIT --> LB{"beats LB anchor?"}
    LB -- no --> LOG2["refute (validator was optimistic)"] --> H
    LB -- yes --> ANCHOR["new verified anchor"] --> H
```
