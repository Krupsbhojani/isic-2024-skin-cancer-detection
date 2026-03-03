"""
gbdt_stacking.py — LightGBM + CatBoost meta-learner for ISIC 2024

Stacks image model OOF predictions with tabular metadata features.
This is the pattern used by ALL top-10 ISIC 2024 solutions.

Usage:
    python src/gbdt_stacking.py --oof_path results/oof_predictions.csv
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
import lightgbm as lgb
from catboost import CatBoostClassifier

from feature_engineering import get_gbdt_feature_columns


def compute_pauc(y_true: np.ndarray, y_score: np.ndarray, min_tpr: float = 0.80) -> float:
    """Compute partial AUC at TPR >= min_tpr, normalized to [0, 0.2]."""
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_true, y_score)
    fpr_at_min  = np.interp(min_tpr, tpr, fpr)
    mask = fpr <= fpr_at_min
    if mask.sum() < 2:
        return 0.0
    pauc = np.trapz(tpr[mask], fpr[mask])
    pauc = pauc / fpr_at_min if fpr_at_min > 0 else 0.0
    return pauc * 0.2


def train_lgbm_folds(X: np.ndarray, y: np.ndarray, folds: np.ndarray,
                     params: dict, n_folds: int = 5) -> tuple:
    """Train LightGBM across K folds. Returns (oof_preds, fitted_models)."""
    oof_preds = np.zeros(len(y))
    models    = []

    for fold in range(n_folds):
        train_idx = np.where(folds != fold)[0]
        val_idx   = np.where(folds == fold)[0]

        model = lgb.LGBMClassifier(**params)
        model.fit(
            X[train_idx], y[train_idx],
            eval_set=[(X[val_idx], y[val_idx])],
            callbacks=[lgb.early_stopping(100, verbose=False),
                       lgb.log_evaluation(period=1000)]
        )
        oof_preds[val_idx] = model.predict_proba(X[val_idx])[:, 1]
        models.append(model)

        fold_auc  = roc_auc_score(y[val_idx], oof_preds[val_idx])
        fold_pauc = compute_pauc(y[val_idx], oof_preds[val_idx])
        print(f'  LGBM Fold {fold+1}: AUC={fold_auc:.4f} | pAUC={fold_pauc:.4f}')

    return oof_preds, models


def train_catboost_folds(X: np.ndarray, y: np.ndarray, folds: np.ndarray,
                         params: dict, n_folds: int = 5) -> tuple:
    """Train CatBoost across K folds. Returns (oof_preds, fitted_models)."""
    oof_preds = np.zeros(len(y))
    models    = []

    for fold in range(n_folds):
        train_idx = np.where(folds != fold)[0]
        val_idx   = np.where(folds == fold)[0]

        model = CatBoostClassifier(**params)
        model.fit(
            X[train_idx], y[train_idx],
            eval_set=(X[val_idx], y[val_idx]),
            verbose=500
        )
        oof_preds[val_idx] = model.predict_proba(X[val_idx])[:, 1]
        models.append(model)

        fold_auc  = roc_auc_score(y[val_idx], oof_preds[val_idx])
        fold_pauc = compute_pauc(y[val_idx], oof_preds[val_idx])
        print(f'  CatBoost Fold {fold+1}: AUC={fold_auc:.4f} | pAUC={fold_pauc:.4f}')

    return oof_preds, models


def main():
    parser = argparse.ArgumentParser(description='GBDT Stacking for ISIC 2024')
    parser.add_argument('--oof_path',   type=str, default='results/oof_predictions.csv')
    parser.add_argument('--output_dir', type=str, default='results')
    parser.add_argument('--n_folds',    type=int, default=5)
    parser.add_argument('--seed',       type=int, default=42)
    args = parser.parse_args()

    df      = pd.read_csv(args.oof_path)
    feat_cols = get_gbdt_feature_columns(df)
    print(f'GBDT features: {len(feat_cols)}')

    X      = df[feat_cols].fillna(df[feat_cols].median()).values
    y      = df['target'].values
    folds  = df['fold'].values

    lgbm_params = {
        'objective':     'binary',
        'metric':        'auc',
        'n_estimators':  3000,
        'learning_rate': 0.02,
        'max_depth':     6,
        'num_leaves':    31,
        'subsample':     0.8,
        'colsample_bytree': 0.8,
        'reg_alpha':     0.1,
        'reg_lambda':    1.0,
        'random_state':  args.seed,
        'n_jobs':        -1,
        'verbose':       -1,
    }
    catboost_params = {
        'iterations':    2000,
        'learning_rate': 0.02,
        'depth':         6,
        'eval_metric':   'AUC',
        'random_seed':   args.seed,
        'task_type':     'CPU',
        'silent':        True,
    }

    print('\n=== Training LightGBM ===')
    lgbm_oof, lgbm_models = train_lgbm_folds(X, y, folds, lgbm_params, args.n_folds)

    print('\n=== Training CatBoost ===')
    cb_oof, cb_models = train_catboost_folds(X, y, folds, catboost_params, args.n_folds)

    # Blend predictions (simple average)
    blended_oof = 0.5 * lgbm_oof + 0.5 * cb_oof

    print('\n' + '='*50)
    print('GBDT STACKING RESULTS')
    print('='*50)
    for name, preds in [('LGBM', lgbm_oof), ('CatBoost', cb_oof), ('Blended', blended_oof)]:
        auc  = roc_auc_score(y, preds)
        pauc = compute_pauc(y, preds)
        print(f'  {name:10s}: AUC={auc:.4f} | pAUC={pauc:.4f}')
    print('='*50)

    df['gbdt_oof_pred'] = blended_oof
    df.to_csv(Path(args.output_dir) / 'oof_predictions_stacked.csv', index=False)
    print(f'\nSaved stacked predictions to {args.output_dir}/oof_predictions_stacked.csv')


if __name__ == '__main__':
    main()
