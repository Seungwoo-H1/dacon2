# =============================
# Deep Learning for Tabular Data (Dacon2 / ETRI)
# Baseline: FT-Transformer (pytabkit)
# Goal: Beat LGBM V10 (cal OOF: 0.6038)
# =============================

import os
import sys
import json
import hashlib
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
PROC_DATA_DIR = BASE_DIR / "data_processed"
RESULTS_DIR = BASE_DIR / "results"

# Fallback: parent workspace dacon2 project
_FALLBACK_DATA = "/home/mwoo423/projects/dacon2/data_processed/features.parquet"

TARGET_COLS_DEFAULT = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]
SUBJECT_COL = "subject_id"
DATE_COL = "lifelog_date"


def load_data():
    """Load features.parquet."""
    parquet_path = PROC_DATA_DIR / "features.parquet"
    if parquet_path.exists():
        print(f"[DATA] Loading from {parquet_path}")
        df = pd.read_parquet(parquet_path)
    elif Path(_FALLBACK_DATA).exists():
        print(f"[DATA] Loading from fallback: {_FALLBACK_DATA}")
        df = pd.read_parquet(_FALLBACK_DATA)
    else:
        raise FileNotFoundError(
            f"No data found at {parquet_path} or {_FALLBACK_DATA}"
        )
    print(f"[DATA] Loaded: {df.shape}")
    return df


def extract_meta(df, target_cols=None):
    """Extract meta columns and feature info."""
    if target_cols is None:
        target_cols = TARGET_COLS_DEFAULT
    
    meta_cols = [c for c in [SUBJECT_COL, DATE_COL, "date", "split"] if c in df.columns]
    
    # Identify which target cols actually exist
    existing_targets = [c for c in target_cols if c in df.columns]
    
    # Feature cols: numeric, excluding meta and target cols
    exclude = set(meta_cols + existing_targets)
    # Also exclude categorical cols
    for c in df.columns:
        if not pd.api.types.is_numeric_dtype(df[c]):
            exclude.add(c)
    feature_cols = [c for c in df.columns if c not in exclude]
    
    meta_info = {
        "meta_cols": meta_cols,
        "target_cols": existing_targets,
        "all_numeric_cols": feature_cols,
        "subject_col": SUBJECT_COL if SUBJECT_COL in df.columns else None,
    }
    print(f"[META] Meta cols: {meta_cols}")
    print(f"[META] Target cols: {existing_targets}")
    print(f"[META] Feature cols: {len(feature_cols)}")
    return meta_info, df


def prepare_for_dl(df, meta_info, val_subjects=None):
    """
    Prepare data for FT-Transformer:
    - Handle missing values (median imputation)
    - Standardize numeric features
    - Create subject_id mapping
    - Return X, y dicts
    """
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    
    subject_col = meta_info["subject_col"]
    target_cols = meta_info["target_cols"]
    feature_cols = meta_info["all_numeric_cols"]
    
    # Subject mapping
    if subject_col:
        subjects = df[subject_col].unique()
        subject_to_idx = {s: i for i, s in enumerate(subjects)}
        X_subjects = df[subject_col].map(subject_to_idx).values
    
    # Feature matrix
    X = df[feature_cols].values.astype(np.float32)
    
    # Impute
    imputer = SimpleImputer(strategy="median")
    X = imputer.fit_transform(X)
    
    # Scale
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Target dict (one per subject if multi-target)
    y = {}
    if target_cols:
        for tc in target_cols:
            if tc in df.columns:
                y[tc] = df[tc].values.astype(np.float32)
    
    # Split if available
    split_col = "split"
    split = df[split_col].values if split_col in df.columns else None
    
    # If val_subjects provided, create a custom split
    if val_subjects is not None:
        if subject_col:
            custom_split = np.array(["train" if s not in val_subjects else "val" 
                                      for s in df[subject_col]])
            split = custom_split
    
    result = {
        "X": X,
        "y": y if y else None,
        "feature_cols": feature_cols,
        "feature_mean": imputer.statistics_,
        "feature_std": scaler.scale_ + 1e-8,
    }
    
    if subject_col:
        result["X_subjects"] = X_subjects
        result["subject_to_idx"] = subject_to_idx
    
    if split is not None:
        result["split"] = split
    
    print(f"[PREP] X shape: {X.shape}")
    print(f"[PREP] Targets: {list(y.keys()) if y else 'None'}")
    
    return result
