# ============================================================
# Full Training: FT-Transformer on Dacon2 (GPU)
# 5-fold GroupKFold × 7 targets
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
from pathlib import Path

print("=" * 60)
print("GPU Setup")
print("=" * 60)
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
print(f"CUDA: {torch.version.cuda}")
print(f"cuDNN: {torch.backends.cudnn.version()}")

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
print(f"Subjects: {len(np.unique(groups))}")

# Move to GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# ============================================================
# FT-Transformer Model (Manual Implementation)
# ============================================================
class FTTransformerBlock(nn.Module):
    """Single FT-Transformer block: attention + FFN with pre-norm."""
    def __init__(self, d_model, n_heads, dropout=0.1):
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
        # Pre-norm attention
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed)
        x = x + attn_out
        
        # Pre-norm FFN
        normed = self.norm2(x)
        ffn_out = self.ffn(normed)
        x = x + ffn_out
        return x


class FTTransformer(nn.Module):
    """FT-Transformer for binary classification."""
    def __init__(self, n_features, d_token=64, n_layers=4, n_heads=4, 
                 dropout=0.1, mlp_hidden_mult=2):
        super().__init__()
        self.d_token = d_token
        # Token embedding (maps each feature to a token)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_token))
        self.token_emb = nn.Linear(n_features, d_token)
        self.pos_emb = nn.Parameter(torch.randn(1, n_features + 1, d_token))  # cls + data tokens
        self.dropout = nn.Dropout(dropout)
        
        self.blocks = nn.ModuleList([
            FTTransformerBlock(d_token, n_heads, dropout) 
            for _ in range(n_layers)
        ])
        
        self.head = nn.Sequential(
            nn.LayerNorm(d_token),
            nn.Linear(d_token, d_token * mlp_hidden_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_token * mlp_hidden_mult, 1),
        )
    
    def forward(self, x):
        batch = x.size(0)
        # Embed each feature as a token: (B, n_feat, d_token)
        # x is (B, n_features), reshape to (B, n_features, 1) then linear
        x_2d = x.view(-1, x.size(-1))  # (B * n_feat, n_features)
        emb = self.token_emb(x_2d)  # (B * n_feat, d_token)
        emb = emb.view(batch, -1, self.d_token)  # (B, n_feat, d_token)
        
        # Add cls token: (1, 1, d_token) -> (B, 1, d_token)
        cls = self.cls_token.expand(batch, 1, -1)
        tokens = torch.cat([cls, emb], dim=1)  # (B, 1+n_feat, d_token)
        
        # Positional embedding
        tokens = tokens + self.pos_emb[:, :tokens.size(1), :]
        tokens = self.dropout(tokens)
        
        # Transformer blocks
        for block in self.blocks:
            tokens = block(tokens)
        
        # Use cls token output
        cls_out = tokens[:, 0, :]
        return self.head(cls_out).squeeze(-1)


# ============================================================
# Training Loop
# ============================================================
class FTTrainer:
    def __init__(self, n_features, config):
        self.model = FTTransformer(
            n_features=n_features,
            d_token=config['d_token'],
            n_layers=config['n_layers'],
            n_heads=config['n_heads'],
            dropout=config['dropout'],
            mlp_hidden_mult=config.get('mlp_hidden_mult', 2),
        ).to(device)
        self.config = config
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config['lr'],
            weight_decay=config['weight_decay'],
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config['max_epochs']
        )
    
    def train_epoch(self, loader, epochs):
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for X_batch, y_batch in loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                
                self.optimizer.zero_grad()
                preds = self.model(X_batch)
                loss = nn.BCEWithLogitsLoss()(preds, y_batch)
                loss.backward()
                self.optimizer.step()
            
            self.scheduler.step()
            if (epoch + 1) % 5 == 0:
                print(f"    Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")
    
    def predict(self, X_tensor):
        self.model.eval()
        X_tensor = X_tensor.to(device)
        with torch.no_grad():
            preds = torch.sigmoid(self.model(X_tensor)).cpu().numpy()
        return preds
    
    def save(self, path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
        }, path)


# ============================================================
# Train All Targets
# ============================================================
config = {
    'd_token': 64,
    'n_layers': 4,
    'n_heads': 4,
    'dropout': 0.1,
    'mlp_hidden_mult': 2,
    'lr': 1e-3,
    'weight_decay': 1e-4,
    'batch_size': 64,
    'max_epochs': 50,
    'n_splits': 5,
    'seed': 42,
}

results = {}
gkf = GroupKFold(n_splits=config['n_splits'])

print(f"\n{'='*60}")
print(f"Training FT-Transformer")
print(f"Config: d_token={config['d_token']}, layers={config['n_layers']}, heads={config['n_heads']}")
print(f"{'='*60}\n")

for target in targets:
    print(f"\n{'='*60}")
    print(f"Target: {target}")
    print(f"{'='*60}")
    
    y = prepared["y"][target]
    
    # OOF predictions
    oof_preds = np.zeros(len(X))
    fold_losses = []
    
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        t0 = time.time()
        
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Create GPU tensors
        X_train_t = torch.FloatTensor(X_train).to(device)
        y_train_t = torch.FloatTensor(y_train).to(device)
        X_val_t = torch.FloatTensor(X_val).to(device)
        
        # DataLoader
        train_ds = TensorDataset(X_train_t, y_train_t)
        train_loader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
        
        # Train
        trainer = FTTrainer(X.shape[1], config)
        trainer.train_epoch(train_loader, config['max_epochs'])
        
        # Predict
        val_preds = trainer.predict(X_val_t)
        oof_preds[val_idx] = val_preds
        
        fold_loss = log_loss(y_val, val_preds)
        fold_auc = roc_auc_score(y_val, val_preds) if len(np.unique(y_val)) > 1 else 0.5
        elapsed = time.time() - t0
        
        print(f"  Fold {fold+1}: loss={fold_loss:.6f}, AUC={fold_auc:.4f}, time={elapsed:.1f}s")
        fold_losses.append(fold_loss)
    
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

lgbm_baseline = {
    'Q1': 0.604, 'Q2': 0.604, 'Q3': 0.604,
    'S1': 0.604, 'S2': 0.604, 'S3': 0.604, 'S4': 0.604,
}

for target in targets:
    r = results[target]
    delta = r['auc'] - lgbm_baseline[target]
    print(f"{target:<10} {r['auc']:>8.4f} {r['loss']:>10.6f} {delta:>+10.4f}")

avg_auc = np.mean([r['auc'] for r in results.values()])
avg_loss = np.mean([r['loss'] for r in results.values()])
print(f"{'-'*40}")
print(f"{'AVG':<10} {avg_auc:>8.4f} {avg_loss:>10.6f}")

# Save results
save_dir = Path("/home/mwoo423/.openclaw/workspace/dl_project/results/ft_transformer_gpu")
save_dir.mkdir(parents=True, exist_ok=True)

with open(save_dir / "results.json", "w") as f:
    json.dump({k: {kk: (float(vv) if isinstance(vv, (float, np.floating)) else vv) 
                   for kk, vv in v.items()} 
               for k, v in results.items()}, f, indent=2)

for target in targets:
    np.save(save_dir / f"{target}_oof.npy", results[target]['oof_preds'])

print(f"\nResults saved to {save_dir}")
print(f"\nLGBM V10 baseline: avg cal OOF loss ≈ 0.6038")
print(f"Need to beat this with FT-Transformer!")
