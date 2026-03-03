"""
feature_engineering.py — Patient-relative feature engineering for ISIC 2024

The "ugly duckling" principle: a malignant lesion stands out from a patient's
baseline. This module encodes that signal as patient-relative features.

Inspired by top solutions' use of patient-level GBDT features.
"""

import numpy as np
import pandas as pd


# Features to compute patient-relative transformations for
FEATURES_FOR_PATIENT_STATS = [
    'clin_size_long_diam_mm',
    'tbp_lv_areaMM2',
    'tbp_lv_norm_color',
    'tbp_lv_color_std_mean',
    'tbp_lv_symm_2axis',
    'tbp_lv_deltaLBnorm',
    'tbp_lv_H',
    'tbp_lv_Hext',
    'tbp_lv_L',
    'tbp_lv_Lext',
    'tbp_lv_nevi_confidence',
]


def engineer_patient_relative_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute patient-relative features capturing the ugly duckling signal.
    
    For each numerical feature in FEATURES_FOR_PATIENT_STATS, computes:
      - {feat}_vs_patient_mean  : z-score relative to patient mean (deviation)
      - {feat}_ratio_to_patient : ratio to patient mean
      - {feat}_vs_patient_min   : difference from patient minimum
      - {feat}_vs_patient_max   : difference from patient maximum
    
    Additional:
      - patient_lesion_count         : total number of lesions this patient has
      - lesion_size_rank_in_patient  : percentile rank of this lesion's size
    
    Args:
        df: DataFrame with columns 'patient_id' and feature columns
    
    Returns:
        DataFrame with original columns plus all patient-relative features
    """
    df = df.copy()
    available = [f for f in FEATURES_FOR_PATIENT_STATS if f in df.columns]
    
    if not available:
        print('Warning: No matching features found for patient-relative engineering.')
        return df

    for feat in available:
        # Compute patient-level statistics
        patient_stats = (
            df.groupby('patient_id')[feat]
            .agg(['mean', 'std', 'min', 'max'])
            .rename(columns={
                'mean': f'patient_{feat}_mean',
                'std':  f'patient_{feat}_std',
                'min':  f'patient_{feat}_min',
                'max':  f'patient_{feat}_max',
            })
        )
        df = df.merge(patient_stats, on='patient_id', how='left')

        # z-score deviation from patient mean (primary ugly duckling signal)
        df[f'{feat}_vs_patient_mean'] = (
            df[feat] - df[f'patient_{feat}_mean']
        ) / (df[f'patient_{feat}_std'] + 1e-6)

        # Ratio to patient mean
        df[f'{feat}_ratio_to_patient'] = df[feat] / (df[f'patient_{feat}_mean'] + 1e-6)

        # Deviation from patient extremes
        df[f'{feat}_vs_patient_min'] = df[feat] - df[f'patient_{feat}_min']
        df[f'{feat}_vs_patient_max'] = df[f'patient_{feat}_max'] - df[feat]

    # Number of lesions this patient has
    df['patient_lesion_count'] = df.groupby('patient_id')['patient_id'].transform('count')

    # Percentile rank of lesion size within patient
    if 'clin_size_long_diam_mm' in df.columns:
        df['lesion_size_rank_in_patient'] = (
            df.groupby('patient_id')['clin_size_long_diam_mm']
            .rank(pct=True)
        )

    n_new = sum(1 for c in df.columns if 'vs_patient' in c or 'ratio_to_patient' in c)
    print(f'Engineered {n_new} patient-relative features from {len(available)} base features.')
    return df


def impute_missing_values(df: pd.DataFrame, strategy: str = 'median') -> pd.DataFrame:
    """
    Impute missing values in numerical columns.
    
    Args:
        df:       Input DataFrame
        strategy: 'median' (default) or 'mean'
    
    Returns:
        DataFrame with imputed values
    """
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    for col in numeric_cols:
        if df[col].isnull().any():
            fill_val = df[col].median() if strategy == 'median' else df[col].mean()
            df[col].fillna(fill_val, inplace=True)
    
    return df


def get_gbdt_feature_columns(df: pd.DataFrame) -> list:
    """
    Return the list of columns to use as GBDT features.
    Includes OOF image predictions, raw metadata, and patient-relative features.
    """
    exclude = {'isic_id', 'patient_id', 'target', 'fold', 'image_path',
               'attribution', 'copyright_license', 'lesion_id', 'iddx_full',
               'iddx_1', 'iddx_2', 'iddx_3', 'iddx_4', 'iddx_5', 'mel_mitotic_index',
               'mel_thick_mm', 'tbp_tile_type', 'tbp_lv_dnn_lesion_confidence'}
    return [c for c in df.columns if c not in exclude
            and df[c].dtype in [np.float64, np.float32, np.int64, np.int32]]
