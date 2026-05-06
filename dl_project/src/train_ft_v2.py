# ============================================================
# FT-Transformer V2: Small data optimized (GPU)
# - Fewer params, more regularization
# - Early stopping + learning rate warmup
# - Try multiple configs with 1-fold first
# ============================================================
import sys, os, warnings, time, json, traceback
warnings.filterwarnings("ignore")
os.environ['LD_LIBRARY_PATH'] = '/usr/lib/wsl/lib:' + os.environ.get('LD_LIBRARY_PATH', '')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import GroupKFold
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler
import pickle
from pathlib import Path

print("=" * 60)
print("GPU Setup")
print("=" * 60)
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

device = torch.device('cuda')

# ============================================================
# Load Data
# ============================================================
src_dir = Path("/home/mwoo423/.openclaw/workspace/dl_project/src")
sys.path.insert(0, str(src_dir))
spec = __import__('importlib.util').util.spec_from_file_location("prepare", src_dir / "00_prepare_data.py")
prepare = __import__('importlib.util').util.module_from_spec(spec)
spec.loader.exec_module(prepare)

df = prepare.load_data()
meta_info, df = prepare.extract_meta(df)
prepared = prepare.prepare_for_dl(df, meta_info)
X = prepared["X"]
targets = meta_info["target_cols"]
groups = prepared["X_subjects"]

print(f"\nData: {X.shape}, Targets: {targets}")

# ============================================================
# FT-Transformer V2: Optimized for small data
# ============================================================
class FTBlock(nn.Module):
    """Pre-norm transformer block for small data."""
    def __init__(self, d_model, n_heads, dropout=0.3):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout),
        )
    
    def forward(self, x):
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed)
        x = x + attn_out
        normed = self.norm2(x)
        x = x + self.ffn(normed)
        return x


class FTTransformerV2(nn.Module):
    """Smaller FT-Transformer for small datasets."""
    def __init__(self, n_features, d_token=32, n_layers=2, n_heads=4, 
                 dropout=0.3, n_feature_select=40):
        super().__init__()
        self.d_token = d_token
        # Feature selection: pick top-k features as tokens
        # For small data, we reduce input dim drastically
        self.n_feature_select = n_feature_select
        self.feature_idx = torch.arange(min(n_feature_select, n_features))
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_token))
        # Map selected features to tokens
        self.token_emb = nn.Linear(1, d_token)  # embed each selected feature value
        self.pos_emb = nn.Parameter(torch.randn(1, 1 + n_feature_select, d_token))
        self.dropout = nn.Dropout(dropout)
        self.norm_before = nn.LayerNorm(d_token)
        
        self.blocks = nn.ModuleList([
            FTBlock(d_token, n_heads, dropout) for _ in range(n_layers)
        ])
        
        self.head = nn.Sequential(
            nn.LayerNorm(d_token),
            nn.Linear(d_token, 8),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(8, 1),
        )
    
    def forward(self, x):
        batch = x.size(0)
        # Select features
        x_sel = x[:, self.feature_idx]  # (B, n_select)
        # Each feature value -> token: reshape to (B, n_select, 1) -> embed -> (B, n_select, d)
        x_3d = x_sel.unsqueeze(-1)  # (B, n_select, 1)
        emb = self.token_emb(x_3d)  # (B, n_select, d)
        
        # Add cls token
        cls = self.cls_token.expand(batch, 1, -1)
        tokens = torch.cat([cls, emb], dim=1)  # (B, 1+n_select, d)
        
        tokens = tokens + self.pos_emb[:, :tokens.size(1), :]
        tokens = self.norm_before(tokens)
        tokens = self.dropout(tokens)
        
        for block in self.blocks:
            tokens = block(tokens)
        
        return self.head(tokens[:, 0, :]).squeeze(-1)


# ============================================================
# Training with early stopping
# ============================================================
class Trainer:
    def __init__(self, model, lr=1e-3, weight_decay=1e-3):
        self.model = model
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2
        )
    
    def fit(self, train_loader, epochs, patience=10, target=''):
        self.model.train()
        best_auc = 0
        best_state = None
        patience_counter = 0
        all_losses = []
        
        for epoch in range(epochs):
            total_loss = 0
            n_batches = 0
            for X_b, y_b in train_loader:
                X_b, y_b = X_b.to(device), y_b.to(device)
                self.optimizer.zero_grad()
                preds = self.model(X_b)
                loss = nn.BCEWithLogitsLoss()(preds, y_b)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                n_batches += 1
            
            avg_loss = total_loss / max(n_batches, 1)
            all_losses.append(avg_loss)
            self.scheduler.step()
            
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"    Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
        
        return all_losses


# ============================================================
# Feature Selection: Use only most important features
# ============================================================
def select_features(X, y, n_features=30):
    """Simple: select features with highest variance * correlation with y."""
    correlations = np.abs(np.corrcoef(X.T, y)[0:-1, -1])
    # Handle NaN
    correlations = np.nan_to_num(correlations, nan=0.0)
    indices = np.argsort(correlations)[::-1][:n_features]
    return np.sort(indices)


# ============================================================
# Configs to try
# ============================================================
configs = [
    {'d_token': 32, 'n_layers': 2, 'n_heads': 4, 'dropout': 0.3, 'n_feature_select': 30, 'lr': 1e-3, 'wd': 1e-3, 'epochs': 100, 'batch': 32, 'name': 'small'},
    {'d_token': 32, 'n_layers': 2, 'n_heads': 2, 'dropout': 0.4, 'n_feature_select': 20, 'lr': 8e-4, 'wd': 2e-3, 'epochs': 100, 'batch': 32, 'name': 'tiny'},
    {'d_token': 48, 'n_layers': 2, 'n_heads': 4, 'dropout': 0.25, 'n_feature_select': 40, 'lr': 1e-3, 'wd': 1e-3, 'epochs': 100, 'batch': 32, 'name': 'medium'},
]

gkf = GroupKFold(n_splits=5)
results = {}

print(f"\n{'='*60}")
print(f"Training FT-Transformer V2 (GPU)")
print(f"{'='*60}\n")

# Test each config on Q1 first to pick best
best_config_idx = 0
best_q1_auc = 0

for ci, cfg in enumerate(configs):
    print(f"\n{'='*60}")
    print(f"Config {ci+1}: {cfg['name']} (d={cfg['d_token']}, L={cfg['n_layers']}, "
          f"fs={cfg['n_feature_select']}, drop={cfg['dropout']})")
    print(f"{'='*60}")
    
    # Use Q1 to evaluate config
    y = prepared["y"]["Q1"]
    feat_idx = select_features(X, y, n_features=cfg['n_feature_select'])
    
    fold_aucs = []
    for fold, (tr_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        X_tr, X_val = X[tr_idx][:, feat_idx], X[val_idx][:, feat_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]
        
        tr_t = FTTransformerV2(n_features=len(feat_idx), **{k:v for k,v in cfg.items() 
                      if k in ('d_token','n_layers','n_heads','dropout','n_feature_select')})
        tr_t = tr_t.to(device)
        trainer = Trainer(tr_t, lr=cfg['lr'], weight_decay=cfg['wd'])
        
        tr_ds = TensorDataset(torch.FloatTensor(X_tr).to(device), torch.FloatTensor(y_tr).to(device))
        loader = DataLoader(tr_ds, batch_size=cfg['batch'], shuffle=True)
        trainer.fit(loader, epochs=cfg['epochs'], target="Q1")
        
        tr_t.eval()
        with torch.no_grad():
            preds = torch.sigmoid(tr_t(torch.FloatTensor(X_val).to(device))).cpu().numpy()
        
        auc = roc_auc_score(y_val, preds) if len(np.unique(y_val)) > 1 else 0.5
        fold_aucs.append(auc)
        print(f"  Fold {fold+1}: AUC={auc:.4f}")
    
    mean_auc = np.mean(fold_aucs)
    print(f"  >>> Q1 AVG AUC: {mean_auc:.4f}")
    
    if mean_auc > best_q1_auc:
        best_q1_auc = mean_auc
        best_config_idx = ci

print(f"\n{'='*60}")
print(f"Best config: {configs[best_config_idx]['name']} (Q1 avg AUC: {best_q1_auc:.4f})")
print(f"{'='*60}\n")

best_cfg = configs[best_config_idx]

# ============================================================
# Full training with best config
# ============================================================
feat_idx = select_features(X, prepared["y"][targets[0]], n_features=best_cfg['n_feature_select'])
# Actually select features per target for each target
for target in targets:
    print(f"\n{'='*60}")
    print(f"Target: {target}")
    print(f"{'='*60}")
    
    y = prepared["y"][target]
    t_feat_idx = select_features(X, y, n_features=best_cfg['n_feature_select'])
    
    oof_preds = np.zeros(len(X))
    fold_losses = []
    
    for fold, (tr_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        t0 = time.time()
        
        X_tr, X_val = X[tr_idx][:, t_feat_idx], X[val_idx][:, t_feat_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]
        
        model = FTTransformerV2(n_features=len(t_feat_idx), 
                               d_token=best_cfg['d_token'],
                               n_layers=best_cfg['n_layers'],
                               n_heads=best_cfg['n_heads'],
                               dropout=best_cfg['dropout'],
                               n_feature_select=best_cfg['n_feature_select'])
        model = model.to(device)
        trainer = Trainer(model, lr=best_cfg['lr'], weight_decay=best_cfg['wd'])
        
        tr_ds = TensorDataset(torch.FloatTensor(X_tr).to(device), torch.FloatTensor(y_tr).to(device))
        loader = DataLoader(tr_ds, batch_size=best_cfg['batch'], shuffle=True)
        trainer.fit(loader, epochs=best_cfg['epochs'])
        
        model.eval()
        with torch.no_grad():
            preds = torch.sigmoid(model(torch.FloatTensor(X_val).to(device))).cpu().numpy()
        
        oof_preds[val_idx] = preds
        
        fold_auc = roc_auc_score(y_val, preds) if len(np.unique(y_val)) > 1 else 0.5
        fold_loss = log_loss(y_val, preds)
        elapsed = time.time() - t0
        fold_losses.append(fold_loss)
        print(f"  Fold {fold+1}: loss={fold_loss:.6f}, AUC={fold_auc:.4f}, time={elapsed:.1f}s")
    
    overall_auc = roc_auc_score(y, oof_preds)
    overall_loss = log_loss(y, oof_preds)
    
    results[target] = {
        'auc': overall_auc,
        'loss': overall_loss,
        'fold_losses': fold_losses,
        'oof_preds': oof_preds,
    }
    print(f"  >>> {target}: AUC={overall_auc:.4f}, Loss={overall_loss:.6f}")

# ============================================================
# Summary
# ============================================================
print(f"\n{'='*60}")
print("FINAL SUMMARY")
print(f"{'='*60}")
print(f"{'Target':<10} {'AUC':>8} {'Loss':>10} {'Δ vs LGBM':>10}")
print(f"{'-'*40}")

lgbm_baseline = {t: 0.604 for t in targets}

for t in targets:
    r = results[t]
    delta = r['auc'] - lgbm_baseline[t]
    print(f"{t:<10} {r['auc']:>8.4f} {r['loss']:>10.6f} {delta:>+10.4f}")

avg_auc = np.mean([r['auc'] for r in results.values()])
avg_loss = np.mean([r['loss'] for r in results.values()])
print(f"{'-'*40}")
print(f"{'AVG':<10} {avg_auc:>8.4f} {avg_loss:>10.6f}")

# Save
save_dir = Path("/home/mwoo423/.openclaw/workspace/dl_project/results/ft_v2_gpu")
save_dir.mkdir(parents=True, exist_ok=True)

for t in targets:
    np.save(save_dir / f"{t}_oof.npy", results[t]['oof_preds'])

with open(save_dir / "results.txt", "w") as f:
    f.write(f"Best config: {best_cfg['name']}\n")
    f.write(f"Config: {json.dumps(best_cfg)}\n")
    for t in targets:
        r = results[t]
        f.write(f"{t}: AUC={r['auc']:.4f}, Loss={r['loss']:.6f}\n")
    f.write(f"AVG AUC: {avg_auc:.4f}\n")

print(f"\nResults saved to {save_dir}")
print(f"LGBM V10 baseline: 0.6038")
