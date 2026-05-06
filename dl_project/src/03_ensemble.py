# =============================
# Ensemble: Blend LGBM + FT-Transformer
# =============================

import json
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression

BASE_DIR = Path(__file__).parent.parent


def load_preds(model_type, target, fold="all"):
    """Load OOF predictions from a model type."""
    save_dir = BASE_DIR / "results" / model_type / target
    if fold == "all":
        preds_path = save_dir / "oof_preds.npy"
    else:
        preds_path = save_dir / f"oof_preds_fold{fold}.npy"
    if preds_path.exists():
        return np.load(preds_path)
    return None


def blend_ensemble(targets, weights=None):
    """
    Blend predictions from LGBM baseline and FT-Transformer.
    
    Uses logistic regression as a meta-learner on OOF predictions.
    """
    results = {}
    
    for target in targets:
        lgbm_preds = load_preds("lgbm_baseline", target)
        ft_preds = load_preds("ft_transformer", target)
        
        if lgbm_preds is None or ft_preds is None:
            print(f"[BLEND] Skipping {target}: missing preds")
            continue
        
        # Stack features
        X_blend = np.column_stack([lgbm_preds, ft_preds])
        # Need actual labels — load from config or data
        # For now, use placeholder
        
        if weights:
            w = weights.get(target, [0.5, 0.5])
            blended = w[0] * lgbm_preds + w[1] * ft_preds
        else:
            # Auto-weight by individual performance
            blended = (lgbm_preds + ft_preds) / 2
        
        results[target] = blended
        print(f"[BLEND] {target}: LGBM={lgbm_preds.mean():.4f}, FT={ft_preds.mean():.4f}, Blend={blended.mean():.4f}")
    
    return results


def save_submission(results, output_path=None):
    """Save submission file."""
    if output_path is None:
        output_path = BASE_DIR / "submission" / "blend_submission.csv"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create submission DataFrame
    if len(results) == 1:
        target = list(results.keys())[0]
        df = pd.DataFrame({
            "subject_id": [...],  # Need subject IDs
            target: results[target],
        })
    else:
        # Multi-target submission
        df_dict = {"subject_id": [...]}
        for target, preds in results.items():
            df_dict[target] = preds
        df = pd.DataFrame(df_dict)
    
    df.to_csv(output_path, index=False)
    print(f"[SUBMIT] Saved to {output_path}")


if __name__ == "__main__":
    import pandas as pd
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", nargs="+", default=["target1", "target2"])
    parser.add_argument("--weights", type=str, default=None, help="JSON string")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()
    
    weights = json.loads(args.weights) if args.weights else None
    results = blend_ensemble(args.targets, weights)
    save_submission(results, args.output)
