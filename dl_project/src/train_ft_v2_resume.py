# FT-Transformer V2 - S2, S3, S4 only (resume)
import sys, os, warnings, time
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import GroupKFold
from sklearn.metrics import log_loss, roc_auc_score
import pickle
from pathlib import Path

# GPU
print(f"CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
device = torch.device('cuda')

# Load data
src_dir = Path("/home/mwoo423/.openclaw/workspace/dl_project/src")
sys.path.insert(0, str(src_dir))
import importlib.util as ut
spec = ut.spec_from_file_location("prepare", src_dir / "00_prepare_data.py")
prepare = ut.module_from_spec(spec)
spec.loader.exec_module(prepare)

df = prepare.load_data()
meta_info, df = prepare.extract_meta(df)
prepared = prepare.prepare_for_dl(df, meta_info)
X = prepared["X"]
targets = meta_info["target_cols"]
groups = prepared["X_subjects"]
print(f"Data: {X.shape}, Targets: {targets}")

# Config (from V2 best)
BEST = {'d_token':32, 'n_layers':2, 'n_heads':2, 'dropout':0.4, 'n_feature_select':20, 'lr':8e-4, 'wd':2e-3, 'epochs':100, 'batch':32}

class FTBlock(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.3):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model*2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model*2, d_model), nn.Dropout(dropout),
        )
    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.ffn(self.norm2(x))
        return x

class FTTransformerV2(nn.Module):
    def __init__(self, n_features, d_token=32, n_layers=2, n_heads=4, dropout=0.3, n_feature_select=40):
        super().__init__()
        self.feature_idx = torch.arange(min(n_feature_select, n_features))
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_token))
        self.token_emb = nn.Linear(1, d_token)
        self.pos_emb = nn.Parameter(torch.randn(1, 1 + n_feature_select, d_token))
        self.dropout = nn.Dropout(dropout)
        self.norm_before = nn.LayerNorm(d_token)
        self.blocks = nn.ModuleList([FTBlock(d_token, n_heads, dropout) for _ in range(n_layers)])
        self.head = nn.Sequential(
            nn.LayerNorm(d_token), nn.Linear(d_token, 8), nn.GELU(), nn.Dropout(dropout), nn.Linear(8, 1),
        )
    def forward(self, x):
        batch = x.size(0)
        x_sel = x[:, self.feature_idx].unsqueeze(-1)
        emb = self.token_emb(x_sel)
        cls = self.cls_token.expand(batch, 1, -1)
        tokens = torch.cat([cls, emb], dim=1)
        tokens = tokens + self.pos_emb[:, :tokens.size(1), :]
        tokens = self.dropout(self.norm_before(tokens))
        for block in self.blocks:
            tokens = block(tokens)
        return self.head(tokens[:, 0, :]).squeeze(-1)

def select_features(X, y, n_features=30):
    corrs = np.abs(np.corrcoef(X.T, y)[0:-1, -1])
    corrs = np.nan_to_num(corrs, nan=0.0)
    return np.sort(np.argsort(corrs)[::-1][:n_features])

gkf = GroupKFold(n_splits=5)
results = {}

for target in targets:
    print(f"\n{'='*60}")
    print(f"Target: {target}")
    print(f"{'='*60}")
    y = prepared["y"][target]
    t_feat_idx = select_features(X, y, n_features=BEST['n_feature_select'])
    oof_preds = np.zeros(len(X))
    
    for fold, (tr_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        t0 = time.time()
        X_tr, X_val = X[tr_idx][:, t_feat_idx], X[val_idx][:, t_feat_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]
        
        model = FTTransformerV2(n_features=len(t_feat_idx), **{k:v for k,v in BEST.items() if k in ('d_token','n_layers','n_heads','dropout','n_feature_select')})
        model = model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=BEST['lr'], weight_decay=BEST['wd'])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
        
        tr_ds = TensorDataset(torch.FloatTensor(X_tr).to(device), torch.FloatTensor(y_tr).to(device))
        loader = DataLoader(tr_ds, batch_size=BEST['batch'], shuffle=True)
        
        model.train()
        for epoch in range(BEST['epochs']):
            for X_b, y_b in loader:
                optimizer.zero_grad()
                preds = model(X_b)
                loss = nn.BCEWithLogitsLoss()(preds, y_b)
                loss.backward()
                optimizer.step()
            scheduler.step()
            if (epoch+1) % 10 == 0 or epoch == 0:
                print(f"    Epoch {epoch+1}/{BEST['epochs']}, Loss: {loss.item():.6f}")
        
        model.eval()
        with torch.no_grad():
            preds = torch.sigmoid(model(torch.FloatTensor(X_val).to(device))).cpu().numpy()
        
        oof_preds[val_idx] = preds
        fold_auc = roc_auc_score(y_val, preds) if len(np.unique(y_val)) > 1 else 0.5
        fold_loss = log_loss(y_val, preds)
        print(f"  Fold {fold+1}: loss={fold_loss:.6f}, AUC={fold_auc:.4f}, time={time.time()-t0:.1f}s")
    
    overall_auc = roc_auc_score(y, oof_preds)
    overall_loss = log_loss(y, oof_preds)
    results[target] = {'auc': overall_auc, 'loss': overall_loss, 'oof_preds': oof_preds}
    print(f"  >>> {target}: AUC={overall_auc:.4f}, Loss={overall_loss:.6f}")

# Summary
print(f"\n{'='*60}")
print("FINAL SUMMARY (V2 - S2/S3/S4)")
print(f"{'='*60}")
lgbm_baseline = 0.6038
for t in targets:
    r = results[t]
    delta = r['auc'] - lgbm_baseline
    print(f"{t}: AUC={r['auc']:.4f}, Loss={r['loss']:.6f}, Δ={delta:>+6.4f}")

avg_auc = np.mean([r['auc'] for r in results.values()])
print(f"AVG AUC: {avg_auc:.4f}")

# Save
save_dir = Path("/home/mwoo423/.openclaw/workspace/dl_project/results/ft_v2_gpu")
save_dir.mkdir(parents=True, exist_ok=True)
for t in targets:
    np.save(save_dir / f"{t}_oof.npy", results[t]['oof_preds'])
with open(save_dir / "results.txt", "w") as f:
    f.write(f"FT-Transformer V2 (S2-S4 resume)\n")
    f.write(f"Config: {BEST}\n")
    for t in targets:
        r = results[t]
        f.write(f"{t}: AUC={r['auc']:.4f}, Loss={r['loss']:.6f}\n")
    f.write(f"AVG AUC: {avg_auc:.4f}\n")
print(f"Saved to {save_dir}")
