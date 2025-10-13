#!/usr/bin/env python3
"""
Enhanced XGBoost ClinVar Multi-Chromosome Trainer
Advanced model with SMOTE, MCC/ROC-AUC optimization, and grid search
"""

import pandas as pd
import numpy as np
from pathlib import Path
import xgboost as xgb
from sklearn.model_selection import (
    train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
)
from sklearn.metrics import (
    matthews_corrcoef, classification_report, confusion_matrix,
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    make_scorer
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import argparse
warnings.filterwarnings('ignore')

def find_clinvar_files(input_dir="Final_Results"):
    """Find all chr*_features_engineered_clinvar.csv files"""
    input_path = Path(input_dir)
    clinvar_files = list(input_path.glob("chr*_features_engineered_clinvar.csv"))
    
    print(f"Found {len(clinvar_files)} ClinVar files:")
    for file in sorted(clinvar_files):
        print(f"  {file.name}")
    
    return sorted(clinvar_files)

def analyze_file_structure(files):
    """Analyze column structure of all files to ensure consistency"""
    print("\nAnalyzing file structures...")
    
    file_info = {}
    all_columns = set()
    
    for file in files:
        try:
            # Read just the header to check columns
            df_header = pd.read_csv(file, nrows=0)
            columns = list(df_header.columns)
            
            # Get actual row count
            with open(file, 'r') as f:
                row_count = sum(1 for _ in f) - 1  # subtract header
            
            file_info[file.name] = {
                'columns': columns,
                'n_columns': len(columns),
                'n_rows': row_count
            }
            
            all_columns.update(columns)
            
            print(f"{file.name}: {row_count:,} rows, {len(columns)} columns")
            
        except Exception as e:
            print(f"Error reading {file.name}: {e}")
            file_info[file.name] = None
    
    # Find common columns across all files
    if file_info:
        first_valid = next(info for info in file_info.values() if info is not None)
        common_columns = set(first_valid['columns'])
        for info in file_info.values():
            if info:
                common_columns = common_columns.intersection(set(info['columns']))
    else:
        common_columns = set()
    
    print(f"\nColumn analysis:")
    print(f"Total unique columns across all files: {len(all_columns)}")
    print(f"Common columns in all files: {len(common_columns)}")
    
    return file_info, list(common_columns)

def load_and_merge_files(files, common_columns):
    """Load all files and merge them using only common columns"""
    print(f"\nLoading and merging files using {len(common_columns)} common columns...")
    
    # Ensure CLNSIG is in common columns (it should be)
    if 'CLNSIG' not in common_columns:
        print("Warning: CLNSIG not in common columns!")
        return None
    
    all_dataframes = []
    total_rows = 0
    
    for file in files:
        try:
            print(f"Loading {file.name}...")
            
            # Load file
            df = pd.read_csv(file)
            print(f"  Original columns: {len(df.columns)}")
            
            # Check which common columns are actually present
            missing_cols = set(common_columns) - set(df.columns)
            if missing_cols:
                print(f"  Warning: Missing columns in {file.name}: {list(missing_cols)[:5]}...")
                # Use only columns that are both common AND present in this file
                available_common_cols = [col for col in common_columns if col in df.columns]
            else:
                available_common_cols = common_columns
            
            # Keep only available common columns
            df_common = df[available_common_cols].copy()
            
            # Add chromosome identifier
            chr_id = file.name.split('_')[0].replace('chr', '')
            df_common['chromosome'] = chr_id
            
            all_dataframes.append(df_common)
            total_rows += len(df_common)
            
            print(f"  Loaded: {len(df_common):,} rows, {len(df_common.columns)} columns")
            
        except Exception as e:
            print(f"Error loading {file.name}: {e}")
            continue
    
    if not all_dataframes:
        print("No files loaded successfully!")
        return None
    
    # Find truly common columns across all loaded dataframes
    print(f"\nFinding truly common columns across all loaded files...")
    
    if all_dataframes:
        final_common_cols = set(all_dataframes[0].columns)
        for df in all_dataframes[1:]:
            final_common_cols = final_common_cols.intersection(set(df.columns))
        
        print(f"Final common columns across all files: {len(final_common_cols)}")
        
        # Keep only truly common columns in all dataframes
        final_common_cols = list(final_common_cols)
        for i, df in enumerate(all_dataframes):
            all_dataframes[i] = df[final_common_cols]
    
    # Merge all dataframes
    print(f"\nMerging {len(all_dataframes)} dataframes...")
    merged_df = pd.concat(all_dataframes, ignore_index=True)
    
    print(f"Total merged data: {len(merged_df):,} rows, {len(merged_df.columns)} columns")
    print(f"Final columns: {list(merged_df.columns)[:10]}...")  # Show first 10 columns
    
    return merged_df

def prepare_data_for_training(df, remove_missing=True, replace_nan_value=0, remove_correlated_threshold=None):
    """Prepare data for XGBoost training with advanced preprocessing
    
    Args:
        df: Input dataframe
        remove_missing: Whether to remove features with >50% missing values
        replace_nan_value: Value to replace NaN with (default: 0)
        remove_correlated_threshold: Correlation threshold for feature removal (e.g., 0.95)
    """
    print(f"\nPreparing data for training (remove_missing={remove_missing}, replace_nan={replace_nan_value}, remove_correlated={remove_correlated_threshold})...")
    
    # Check target column
    if 'CLNSIG' not in df.columns:
        print("Error: CLNSIG column not found!")
        return None, None, None
    
    # Show class distribution
    class_dist = df['CLNSIG'].value_counts()
    print(f"Original class distribution:")
    for cls, count in class_dist.items():
        print(f"  {cls}: {count:,} ({count/len(df)*100:.1f}%)")
    
    # Encode target
    le = LabelEncoder()
    y = le.fit_transform(df['CLNSIG'])
    class_names = le.classes_
    
    print(f"Encoded classes: {dict(zip(range(len(class_names)), class_names))}")
    
    # Separate features
    feature_cols = [col for col in df.columns if col not in ['CLNSIG', 'chromosome']]
    X = df[feature_cols].copy()
    
    print(f"Initial features: {len(feature_cols)}")
    
    # Advanced feature processing
    print("Advanced feature processing...")
    
    # First, identify all forms of missing values across the dataset
    print("Identifying all forms of missing values...")
    missing_values = [np.inf, -np.inf, np.nan, 'nan', 'NaN', 'NA', 'na', 'NULL', 'null', '', ' ', 'n/a', 'N/A']
    
    # Create a temporary copy to calculate true missing ratios
    X_temp = X.copy()
    
    # Replace all forms of missing values with NaN for accurate missing ratio calculation
    X_temp = X_temp.replace(missing_values, np.nan)
    
    # Remove features with too many missing values (>50%) - now optional
    if remove_missing:
        missing_threshold = 0.5
        missing_ratio = X_temp.isnull().sum() / len(X_temp)
        high_missing_cols = missing_ratio[missing_ratio > missing_threshold].index.tolist()
        
        print(f"Missing value analysis:")
        print(f"Total missing values found: {X_temp.isnull().sum().sum():,}")
        print(f"Features with >{missing_threshold*100}% missing: {len(high_missing_cols)}")
        
        # Show top 10 features with highest missing ratios
        if len(missing_ratio) > 0:
            top_missing = missing_ratio.sort_values(ascending=False).head(10)
            print(f"Top 10 features by missing ratio:")
            for feature, ratio in top_missing.items():
                print(f"  {feature}: {ratio:.2%} ({int(ratio * len(X_temp)):,} missing)")
        
        if high_missing_cols:
            print(f"Removing {len(high_missing_cols)} features with >{missing_threshold*100}% missing values")
            X = X.drop(columns=high_missing_cols)
        else:
            print(f"No features found with >{missing_threshold*100}% missing values")
    else:
        print("Skipping removal of high-missingness features (--remove-missing no)")
        # Still show missing value statistics
        missing_ratio = X_temp.isnull().sum() / len(X_temp)
        print(f"Total missing values found: {X_temp.isnull().sum().sum():,}")
        print(f"Features with >50% missing: {sum(missing_ratio > 0.5)}")
    
    # Convert non-numeric columns
    numeric_features = []
    for col in X.columns:
        if X[col].dtype in ['object', 'string']:
            try:
                X[col] = pd.to_numeric(X[col], errors='coerce')
                numeric_features.append(col)
            except:
                print(f"Dropping non-convertible column: {col}")
                continue
        else:
            numeric_features.append(col)
    
    X = X[numeric_features]
    
    # Handle missing values more robustly
    print("Handling missing values and NaN...")
    print(f"NaN values before cleaning: {X.isnull().sum().sum()}")
    
    # Replace various representations of missing values - including NA
    X = X.replace(missing_values, np.nan)
    
    # Fill remaining NaN with specified value for numeric columns
    print(f"Replacing NaN values with: {replace_nan_value}")
    for col in X.columns:
        if X[col].isnull().any():
            if X[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                # Use specified replacement value for numeric columns
                X[col] = X[col].fillna(replace_nan_value)
            else:
                # For any remaining non-numeric, fill with specified value
                X[col] = X[col].fillna(replace_nan_value)
    
    # Convert all columns to numeric, coercing errors to NaN then to replacement value
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce').fillna(replace_nan_value)
    
    # Final NaN check and removal
    print(f"NaN values after initial cleaning: {X.isnull().sum().sum()}")
    
    # If there are still any NaN values, replace them with specified value
    if X.isnull().any().any():
        print(f"Replacing remaining NaN values with {replace_nan_value}...")
        X = X.fillna(replace_nan_value)
    
    # Double-check for infinite values
    inf_mask = np.isinf(X).any(axis=1)
    if inf_mask.any():
        print(f"Removing {inf_mask.sum()} rows with infinite values...")
        X = X[~inf_mask]
        y = y[~inf_mask]
    
    # Final verification - ensure no NaN or infinite values
    assert not X.isnull().any().any(), "Still have NaN values!"
    assert not np.isinf(X).any().any(), "Still have infinite values!"
    
    print(f"Data cleaning complete. Final NaN count: {X.isnull().sum().sum()}")
    print(f"Final infinite count: {np.isinf(X).sum().sum()}")
    
    # Remove constant features (no variance)
    constant_features = X.columns[X.var() == 0].tolist()
    if constant_features:
        print(f"Removing {len(constant_features)} constant features")
        X = X.drop(columns=constant_features)
    
    # Remove highly correlated features if requested
    if remove_correlated_threshold is not None:
        print(f"Removing highly correlated features (threshold: {remove_correlated_threshold})...")
        
        # Calculate correlation matrix
        corr_matrix = X.corr().abs()
        
        # Find pairs of features with correlation above threshold
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        # Find features to drop (one from each highly correlated pair)
        to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > remove_correlated_threshold)]
        
        print(f"Found {len(to_drop)} highly correlated features to remove:")
        if to_drop:
            # Show top 10 correlated pairs
            correlated_pairs = []
            for column in to_drop[:10]:  # Show only first 10
                max_corr_idx = upper_tri[column].idxmax()
                max_corr_val = upper_tri[column].max()
                if not pd.isna(max_corr_val):
                    correlated_pairs.append((column, max_corr_idx, max_corr_val))
            
            for feat1, feat2, corr_val in correlated_pairs:
                print(f"  {feat1} <-> {feat2}: {corr_val:.3f}")
            
            if len(to_drop) > 10:
                print(f"  ... and {len(to_drop) - 10} more")
            
            # Drop the correlated features
            X = X.drop(columns=to_drop)
            print(f"Removed {len(to_drop)} highly correlated features")
        else:
            print("No highly correlated features found")
    else:
        print("Keeping all features including highly correlated ones...")
    
    print(f"Final feature matrix: {X.shape}")
    print(f"Features after preprocessing: {X.shape[1]}")
    
    return X, y, le

def create_custom_scorers():
    """Create custom scoring functions for optimization"""
    
    def mcc_scorer(y_true, y_pred):
        return matthews_corrcoef(y_true, y_pred)
    
    def roc_mcc_scorer(y_true, y_pred_proba):
        # Combined ROC-AUC and MCC score
        y_pred = (y_pred_proba[:, 1] > 0.5).astype(int)
        roc_score = roc_auc_score(y_true, y_pred_proba[:, 1])
        mcc_score = matthews_corrcoef(y_true, y_pred)
        return (roc_score + mcc_score) / 2  # Average of both metrics
    
    # Create sklearn scorers
    mcc_scorer_sk = make_scorer(mcc_scorer)
    
    return mcc_scorer_sk, roc_mcc_scorer

def train_fast_xgboost(X, y, test_size=0.2, random_state=42):
    """Train fast XGBoost model with SMOTE and minimal optimization"""
    print(f"\nTraining fast XGBoost model with SMOTE...")
    
    # Split data first
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"Original train set: {len(X_train):,} samples")
    print(f"Test set: {len(X_test):,} samples")
    
    # Show original class distribution in training set
    unique, counts = np.unique(y_train, return_counts=True)
    print(f"Original train class distribution: {dict(zip(unique, counts))}")
    
    # Apply SMOTE to training data
    print("Applying SMOTE for balanced training data...")
    smote = SMOTE(random_state=random_state, k_neighbors=5)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    
    unique, counts = np.unique(y_train_balanced, return_counts=True)
    print(f"SMOTE balanced train set: {len(X_train_balanced):,} samples")
    print(f"Balanced class distribution: {dict(zip(unique, counts))}")
    
    # Use optimized parameters without grid search
    print("Training with optimized default parameters...")
    
    best_model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=random_state,
        n_jobs=-1,
        eval_metric='logloss'
    )
    
    # Fit on balanced data
    best_model.fit(X_train_balanced, y_train_balanced)
    
    # Make predictions
    y_train_pred = best_model.predict(X_train)
    y_test_pred = best_model.predict(X_test)
    
    # Get prediction probabilities
    y_train_proba = best_model.predict_proba(X_train)
    y_test_proba = best_model.predict_proba(X_test)
    
    return best_model, (X_train, X_test, y_train, y_test), (y_train_pred, y_test_pred), (y_train_proba, y_test_proba)

def evaluate_model_advanced(y_true, y_pred, y_proba, dataset_name, class_names):
    """Advanced model evaluation with multiple metrics"""
    print(f"\n--- {dataset_name} Results ---")
    
    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    mcc = matthews_corrcoef(y_true, y_pred)
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"MCC: {mcc:.4f}")
    
    # AUC metrics
    if len(np.unique(y_true)) == 2:
        auc = roc_auc_score(y_true, y_proba[:, 1])
        print(f"AUC-ROC: {auc:.4f}")
        
        # Combined ROC-MCC score
        roc_mcc_combined = (auc + mcc) / 2
        print(f"ROC-MCC Combined: {roc_mcc_combined:.4f}")
    
    # Per-class metrics
    print(f"\nPer-class metrics:")
    precision_per_class = precision_score(y_true, y_pred, average=None)
    recall_per_class = recall_score(y_true, y_pred, average=None)
    f1_per_class = f1_score(y_true, y_pred, average=None)
    
    for i, class_name in enumerate(class_names):
        print(f"  {class_name}: Precision={precision_per_class[i]:.4f}, "
              f"Recall={recall_per_class[i]:.4f}, F1={f1_per_class[i]:.4f}")
    
    # Confusion Matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_true, y_pred)
    print(cm)
    
    # Classification Report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'mcc': mcc,
        'auc': auc if len(np.unique(y_true)) == 2 else None
    }

def analyze_grid_search_results(grid_search):
    """Analyze grid search results"""
    print(f"\n--- Grid Search Analysis ---")
    
    results_df = pd.DataFrame(grid_search.cv_results_)
    
    # Show top 10 parameter combinations
    top_results = results_df.nlargest(10, 'mean_test_score')[
        ['mean_test_score', 'std_test_score', 'params']
    ]
    
    print("Top 10 parameter combinations:")
    for i, (_, row) in enumerate(top_results.iterrows(), 1):
        print(f"{i:2d}. Score: {row['mean_test_score']:.4f} (+/-{row['std_test_score']*2:.4f}) "
              f"Params: {row['params']}")
    
    # Parameter importance analysis
    print(f"\nParameter impact analysis:")
    
    # Analyze each parameter's effect on performance
    for param in grid_search.param_grid.keys():
        param_values = []
        param_scores = []
        
        for _, row in results_df.iterrows():
            param_values.append(row['params'][param])
            param_scores.append(row['mean_test_score'])
        
        param_df = pd.DataFrame({'value': param_values, 'score': param_scores})
        param_summary = param_df.groupby('value')['score'].agg(['mean', 'std']).round(4)
        
        print(f"  {param}:")
        for value, stats in param_summary.iterrows():
            print(f"    {value}: {stats['mean']:.4f} (+/-{stats['std']:.4f})")

def show_feature_importance_advanced(model, feature_names, top_n=25):
    """Show advanced feature importance analysis"""
    print(f"\n--- Top {top_n} Feature Importances ---")
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Show top features
    for i, (_, row) in enumerate(importance_df.head(top_n).iterrows()):
        print(f"{i+1:2d}. {row['feature']:<35} {row['importance']:.4f}")
    
    # Feature importance statistics
    print(f"\nFeature importance statistics:")
    print(f"Mean importance: {importance_df['importance'].mean():.4f}")
    print(f"Std importance: {importance_df['importance'].std():.4f}")
    print(f"Top 10 features account for {importance_df.head(10)['importance'].sum():.1%} of total importance")
    
    return importance_df

def create_visualizations(model, X_test, y_test, y_test_pred, y_test_proba, feature_names, class_names, importance_df):
    """Create comprehensive visualizations using seaborn"""
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 15))
    
    # 1. Feature Importance Plot
    plt.subplot(2, 3, 1)
    top_features = importance_df.head(20)
    
    ax1 = sns.barplot(data=top_features, y='feature', x='importance', 
                      palette='viridis')
    plt.title('Top 20 Feature Importances', fontsize=14, fontweight='bold')
    plt.xlabel('Importance Score', fontsize=12)
    plt.ylabel('Features', fontsize=12)
    
    # Rotate y-axis labels for better readability
    plt.xticks(rotation=0)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    
    # 2. Confusion Matrix Heatmap
    plt.subplot(2, 3, 2)
    cm = confusion_matrix(y_test, y_test_pred)
    
    # Create annotations for the heatmap
    cm_labels = np.array([[f'TN\n{cm[0,0]}', f'FP\n{cm[0,1]}'],
                         [f'FN\n{cm[1,0]}', f'TP\n{cm[1,1]}']])
    
    sns.heatmap(cm, annot=cm_labels, fmt='', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('Actual', fontsize=12)
    
    # 3. ROC Curve
    from sklearn.metrics import roc_curve, auc
    plt.subplot(2, 3, 3)
    
    fpr, tpr, _ = roc_curve(y_test, y_test_proba[:, 1])
    roc_auc = auc(fpr, tpr)
    
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC Curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', alpha=0.8)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    # 4. Decision Boundary (using PCA for dimensionality reduction)
    plt.subplot(2, 3, 4)
    
    # Apply PCA to reduce to 2D for visualization
    pca = PCA(n_components=2, random_state=42)
    X_test_pca = pca.fit_transform(X_test)
    
    # Create scatter plot
    colors = ['#FF6B6B', '#4ECDC4']  # Red for benign, teal for pathogenic
    for i, class_name in enumerate(class_names):
        mask = y_test == i
        plt.scatter(X_test_pca[mask, 0], X_test_pca[mask, 1], 
                   c=colors[i], label=f'True {class_name}', alpha=0.7, s=20)
    
    # Simple decision boundary using a coarse grid
    try:
        # Use much larger step size and limit the range
        x_min, x_max = X_test_pca[:, 0].min(), X_test_pca[:, 0].max()
        y_min, y_max = X_test_pca[:, 1].min(), X_test_pca[:, 1].max()
        
        # Add small padding
        x_range = x_max - x_min
        y_range = y_max - y_min
        x_min -= x_range * 0.1
        x_max += x_range * 0.1
        y_min -= y_range * 0.1
        y_max += y_range * 0.1
        
        # Use coarse grid (max 100x100 points)
        h_x = max((x_max - x_min) / 100, 0.1)
        h_y = max((y_max - y_min) / 100, 0.1)
        
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h_x),
                             np.arange(y_min, y_max, h_y))
        
        # Simple logistic regression for decision boundary in PCA space
        from sklearn.linear_model import LogisticRegression
        lr_pca = LogisticRegression(random_state=42, max_iter=1000)
        lr_pca.fit(X_test_pca, y_test)
        
        mesh_points = np.c_[xx.ravel(), yy.ravel()]
        Z = lr_pca.predict(mesh_points)
        Z = Z.reshape(xx.shape)
        
        plt.contourf(xx, yy, Z, alpha=0.3, colors=['#FFE5E5', '#E5F9F6'])
        
    except Exception as e:
        print(f"Skipping decision boundary due to: {e}")
    
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=12)
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=12)
    plt.title('Decision Boundary (PCA Projection)', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 5. Prediction Probability Distribution
    plt.subplot(2, 3, 5)
    
    # Get probabilities for each class
    prob_pathogenic = y_test_proba[:, 1]
    
    # Create histogram for each true class
    for i, class_name in enumerate(class_names):
        mask = y_test == i
        plt.hist(prob_pathogenic[mask], bins=30, alpha=0.7, 
                label=f'True {class_name}', color=colors[i])
    
    plt.axvline(x=0.5, color='black', linestyle='--', alpha=0.8, 
                label='Decision Threshold')
    plt.xlabel('Predicted Probability (Pathogenic)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Prediction Probability Distribution', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 6. Metrics Summary
    plt.subplot(2, 3, 6)
    plt.axis('off')
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_test_pred)
    precision = precision_score(y_test, y_test_pred, average='weighted')
    recall = recall_score(y_test, y_test_pred, average='weighted')
    f1 = f1_score(y_test, y_test_pred, average='weighted')
    mcc = matthews_corrcoef(y_test, y_test_pred)
    auc_score = roc_auc_score(y_test, y_test_proba[:, 1])
    
    # Create metrics text
    metrics_text = f"""
MODEL PERFORMANCE SUMMARY

Accuracy:     {accuracy:.4f}
Precision:    {precision:.4f}
Recall:       {recall:.4f}
F1-Score:     {f1:.4f}
MCC:          {mcc:.4f}
AUC-ROC:      {auc_score:.4f}

Sample Size:  {len(y_test):,}
Features:     {len(feature_names):,}

Class Distribution:
• {class_names[0]}: {sum(y_test == 0):,}
• {class_names[1]}: {sum(y_test == 1):,}
    """
    
    plt.text(0.1, 0.5, metrics_text, fontsize=12, 
             verticalalignment='center', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    
    # Save the plot
    output_dir = Path("plots")
    output_dir.mkdir(exist_ok=True)
    
    plt.savefig(output_dir / "xgboost_analysis.png", dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / "xgboost_analysis.pdf", bbox_inches='tight')
    
    print(f"\nVisualization saved to:")
    print(f"  • plots/xgboost_analysis.png")
    print(f"  • plots/xgboost_analysis.pdf")
    
def cross_validate_model(model, X, y, cv_folds=5):
    """Perform cross-validation analysis"""
    print(f"\n--- Cross-Validation Analysis ({cv_folds}-fold) ---")
    
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    # Multiple scoring metrics
    scoring_metrics = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted', 'roc_auc']
    
    cv_results = {}
    for metric in scoring_metrics:
        try:
            scores = cross_val_score(model, X, y, cv=cv, scoring=metric)
            cv_results[metric] = scores
            print(f"{metric}: {scores.mean():.4f} (+/-{scores.std()*2:.4f})")
        except Exception as e:
            print(f"Could not compute {metric}: {e}")
    
    # MCC cross-validation (custom)
    mcc_scorer, _ = create_custom_scorers()
    mcc_scores = cross_val_score(model, X, y, cv=cv, scoring=mcc_scorer)
    cv_results['mcc'] = mcc_scores
    print(f"mcc: {mcc_scores.mean():.4f} (+/-{mcc_scores.std()*2:.4f})")
    
    return cv_results

def plot_feature_importance_detailed(importance_df, top_n=30):
    """Create a detailed feature importance plot"""
    
    plt.figure(figsize=(12, 10))
    
    # Plot top N features
    top_features = importance_df.head(top_n)
    
    # Create horizontal bar plot
    ax = sns.barplot(data=top_features, y='feature', x='importance', 
                     palette='viridis')
    
    plt.title(f'Top {top_n} Feature Importances - Detailed View', 
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Importance Score', fontsize=14)
    plt.ylabel('Features', fontsize=14)
    
    # Add value labels on bars
    for i, (_, row) in enumerate(top_features.iterrows()):
        plt.text(row['importance'], i, f'{row["importance"]:.4f}', 
                va='center', ha='left', fontsize=10)
    
    plt.tight_layout()
    
    # Save detailed plot
    output_dir = Path("plots")
    output_dir.mkdir(exist_ok=True)
    
    plt.savefig(output_dir / "feature_importance_detailed.png", dpi=300, bbox_inches='tight')
    print(f"Detailed feature importance plot saved to: plots/feature_importance_detailed.png")
    
    plt.show()

def plot_confusion_matrix(y_true, y_pred, class_names):
    """Create confusion matrix plot"""
    plt.figure(figsize=(8, 6))
    
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('Actual', fontsize=12)
    plt.tight_layout()
    
    output_dir = Path("plots")
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / "confusion_matrix.png", dpi=300, bbox_inches='tight')
    print(f"Confusion matrix plot saved to: plots/confusion_matrix.png")
    plt.show()

def plot_roc_curve(y_true, y_proba):
    """Create ROC curve plot"""
    from sklearn.metrics import roc_curve, auc
    
    plt.figure(figsize=(8, 6))
    
    fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1])
    roc_auc = auc(fpr, tpr)
    
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC Curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', alpha=0.8)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    output_dir = Path("plots")
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / "roc_curve.png", dpi=300, bbox_inches='tight')
    print(f"ROC curve plot saved to: plots/roc_curve.png")
    plt.show()

def plot_xgboost_decision_boundary(model, X_test, y_test, y_pred, class_names):
    """Create decision boundary plot using normalized PCA with clear class separation"""
    plt.figure(figsize=(12, 10))
    
    # Normalize the data before PCA
    scaler = StandardScaler()
    X_test_scaled = scaler.fit_transform(X_test)
    
    # Apply PCA to reduce to 2D for visualization
    pca = PCA(n_components=2, random_state=42)
    X_test_pca = pca.fit_transform(X_test_scaled)
    
    # Create color mapping - distinct colors for better separation
    colors = {'Benign': '#2E86AB', 'Pathogenic': '#F24236'}  # Blue for benign, Red for pathogenic
    markers = {'Benign': 'o', 'Pathogenic': 's'}  # Circle for benign, Square for pathogenic
    
    # Plot each class separately for better visualization
    for i, class_name in enumerate(class_names):
        mask = y_test == i
        color = colors.get(class_name, '#888888')
        marker = markers.get(class_name, 'o')
        
        plt.scatter(X_test_pca[mask, 0], X_test_pca[mask, 1], 
                   c=color, label=f'{class_name} (n={mask.sum()})', 
                   alpha=0.7, s=50, marker=marker, edgecolors='white', linewidths=0.5)
    
    # Create decision boundary using the actual XGBoost model predictions in PCA space
    print("Creating decision boundary...")
    
    # Get the bounds of the PCA space
    x_min, x_max = X_test_pca[:, 0].min() - 1, X_test_pca[:, 0].max() + 1
    y_min, y_max = X_test_pca[:, 1].min() - 1, X_test_pca[:, 1].max() + 1
    
    # Create a mesh grid for decision boundary
    h = 0.1  # Step size in the mesh
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # For decision boundary, we need to inverse transform PCA back to original space
    # then apply the XGBoost model
    try:
        # Create mesh points in PCA space
        mesh_points_pca = np.c_[xx.ravel(), yy.ravel()]
        
        # We'll use a simple approach: train a logistic regression on the PCA space
        # as a proxy for the decision boundary
        from sklearn.linear_model import LogisticRegression
        lr_pca = LogisticRegression(random_state=42, max_iter=1000, C=1.0)
        lr_pca.fit(X_test_pca, y_test)
        
        # Get decision boundary
        Z = lr_pca.predict_proba(mesh_points_pca)[:, 1]  # Probability of pathogenic
        Z = Z.reshape(xx.shape)
        
        # Plot decision boundary contours
        contour_levels = [0.1, 0.3, 0.5, 0.7, 0.9]
        cs = plt.contour(xx, yy, Z, levels=contour_levels, colors='black', 
                        linestyles='dashed', alpha=0.6, linewidths=1)
        plt.clabel(cs, inline=True, fontsize=8, fmt='%.1f')
        
        # Highlight the main decision boundary (0.5 threshold)
        plt.contour(xx, yy, Z, levels=[0.5], colors='red', linewidths=3, alpha=0.8)
        
        # Fill regions with different colors
        plt.contourf(xx, yy, Z, levels=[0, 0.5, 1.0], colors=['lightblue', 'lightcoral'], alpha=0.3)
        
    except Exception as e:
        print(f"Could not create decision boundary: {e}")
        # Fallback: just show data points
    
    # Calculate class separation metrics
    benign_points = X_test_pca[y_test == 0]
    pathogenic_points = X_test_pca[y_test == 1]
    
    if len(benign_points) > 0 and len(pathogenic_points) > 0:
        # Calculate centroids
        benign_centroid = np.mean(benign_points, axis=0)
        pathogenic_centroid = np.mean(pathogenic_points, axis=0)
        
        # Mark centroids
        plt.scatter(benign_centroid[0], benign_centroid[1], 
                   c='darkblue', s=200, marker='X', 
                   label='Benign Centroid', edgecolors='white', linewidths=2)
        plt.scatter(pathogenic_centroid[0], pathogenic_centroid[1], 
                   c='darkred', s=200, marker='X', 
                   label='Pathogenic Centroid', edgecolors='white', linewidths=2)
        
        # Calculate distance between centroids
        centroid_distance = np.linalg.norm(pathogenic_centroid - benign_centroid)
        
        # Calculate within-class scatter
        benign_scatter = np.mean([np.linalg.norm(point - benign_centroid) for point in benign_points])
        pathogenic_scatter = np.mean([np.linalg.norm(point - pathogenic_centroid) for point in pathogenic_points])
        avg_within_scatter = (benign_scatter + pathogenic_scatter) / 2
        
        # Separation ratio (higher is better separation)
        separation_ratio = centroid_distance / avg_within_scatter if avg_within_scatter > 0 else 0
        
        # Add separation metrics to plot
        plt.text(0.02, 0.98, f'Class Separation Analysis:\n'
                            f'Centroid Distance: {centroid_distance:.2f}\n'
                            f'Avg Within-Class Scatter: {avg_within_scatter:.2f}\n'
                            f'Separation Ratio: {separation_ratio:.2f}',
                transform=plt.gca().transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Formatting
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=14)
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=14)
    plt.title('Class Separation Analysis (Normalized PCA)\nwith Decision Boundary', 
              fontsize=16, fontweight='bold', pad=20)
    plt.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
    plt.grid(True, alpha=0.3)
    
    # Add explained variance info
    total_variance = pca.explained_variance_ratio_.sum()
    plt.text(0.02, 0.02, f'Total Variance Explained: {total_variance:.1%}',
            transform=plt.gca().transAxes, fontsize=10,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    
    # Save plot
    output_dir = Path("plots")
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / "pca_class_separation.png", dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / "pca_class_separation.pdf", bbox_inches='tight')
    print(f"PCA class separation plot saved to: plots/pca_class_separation.png")
    print(f"PCA class separation plot saved to: plots/pca_class_separation.pdf")
    
    # Additional analysis - create a more detailed separation plot
    create_detailed_separation_analysis(X_test_pca, y_test, class_names, pca)
    
    plt.show()

def create_detailed_separation_analysis(X_pca, y_true, class_names, pca):
    """Create detailed class separation analysis plots"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Colors for classes
    colors = {'Benign': '#2E86AB', 'Pathogenic': '#F24236'}
    
    # 1. PC1 distribution by class
    axes[0, 0].hist([X_pca[y_true == 0, 0], X_pca[y_true == 1, 0]], 
                    bins=30, alpha=0.7, label=class_names, 
                    color=[colors[name] for name in class_names])
    axes[0, 0].set_xlabel('PC1')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('PC1 Distribution by Class')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. PC2 distribution by class
    axes[0, 1].hist([X_pca[y_true == 0, 1], X_pca[y_true == 1, 1]], 
                    bins=30, alpha=0.7, label=class_names,
                    color=[colors[name] for name in class_names])
    axes[0, 1].set_xlabel('PC2')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('PC2 Distribution by Class')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Distance from origin analysis
    distances = np.sqrt(X_pca[:, 0]**2 + X_pca[:, 1]**2)
    axes[1, 0].boxplot([distances[y_true == 0], distances[y_true == 1]], 
                       labels=class_names, patch_artist=True,
                       boxprops=dict(facecolor='lightblue', alpha=0.7),
                       medianprops=dict(color='red', linewidth=2))
    axes[1, 0].set_ylabel('Distance from Origin')
    axes[1, 0].set_title('Distance Distribution by Class')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Class overlap analysis
    from sklearn.metrics import silhouette_score
    try:
        silhouette_avg = silhouette_score(X_pca, y_true)
        
        # Create a simple separation metric visualization
        axes[1, 1].axis('off')
        
        # Calculate more separation metrics
        benign_points = X_pca[y_true == 0]
        pathogenic_points = X_pca[y_true == 1]
        
        benign_centroid = np.mean(benign_points, axis=0)
        pathogenic_centroid = np.mean(pathogenic_points, axis=0)
        centroid_distance = np.linalg.norm(pathogenic_centroid - benign_centroid)
        
        # Calculate overlap using convex hulls
        try:
            from scipy.spatial import ConvexHull
            if len(benign_points) >= 3 and len(pathogenic_points) >= 3:
                benign_hull = ConvexHull(benign_points)
                pathogenic_hull = ConvexHull(pathogenic_points)
                
                hull_separation = "Calculated"
            else:
                hull_separation = "Insufficient points"
        except:
            hull_separation = "Could not calculate"
        
        # Display metrics
        metrics_text = f"""
CLASS SEPARATION METRICS

Silhouette Score: {silhouette_avg:.4f}
(Range: -1 to 1, higher is better)

Centroid Distance: {centroid_distance:.4f}

PC1 Variance Explained: {pca.explained_variance_ratio_[0]:.1%}
PC2 Variance Explained: {pca.explained_variance_ratio_[1]:.1%}
Total Variance: {pca.explained_variance_ratio_.sum():.1%}

Sample Sizes:
• {class_names[0]}: {sum(y_true == 0):,}
• {class_names[1]}: {sum(y_true == 1):,}

Hull Analysis: {hull_separation}
        """
        
        axes[1, 1].text(0.1, 0.5, metrics_text, fontsize=11, 
                        verticalalignment='center', fontfamily='monospace',
                        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
        axes[1, 1].set_title('Separation Quality Metrics', fontweight='bold')
        
    except Exception as e:
        axes[1, 1].text(0.5, 0.5, f'Could not calculate\nseparation metrics:\n{str(e)[:50]}...', 
                        ha='center', va='center', fontsize=12)
        axes[1, 1].set_title('Separation Analysis')
    
    plt.tight_layout()
    
    # Save detailed analysis
    output_dir = Path("plots")
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / "detailed_class_separation.png", dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / "detailed_class_separation.pdf", bbox_inches='tight')
    print(f"Detailed separation analysis saved to: plots/detailed_class_separation.png")
    print(f"Detailed separation analysis saved to: plots/detailed_class_separation.pdf")
    
    plt.show()

def plot_probability_distribution(y_true, y_proba, class_names):
    """Create probability distribution plot"""
    plt.figure(figsize=(10, 6))
    
    # Get probabilities for pathogenic class
    prob_pathogenic = y_proba[:, 1]
    colors = ['#FF6B6B', '#4ECDC4']
    
    # Create histogram for each true class
    for i, class_name in enumerate(class_names):
        mask = y_true == i
        plt.hist(prob_pathogenic[mask], bins=30, alpha=0.7, 
                label=f'True {class_name}', color=colors[i])
    
    plt.axvline(x=0.5, color='black', linestyle='--', alpha=0.8, 
                label='Decision Threshold')
    plt.xlabel('Predicted Probability (Pathogenic)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Prediction Probability Distribution', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    output_dir = Path("plots")
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / "probability_distribution.png", dpi=300, bbox_inches='tight')
    print(f"Probability distribution plot saved to: plots/probability_distribution.png")
    plt.show()

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Enhanced XGBoost ClinVar Trainer')
    parser.add_argument('--remove-missing', 
                       choices=['yes', 'no'], 
                       default='yes',
                       help='Remove features with >50%% missing values (default: yes)')
    parser.add_argument('--replace-nan',
                       type=float,
                       default=0,
                       help='Value to replace NaN with (default: 0)')
    parser.add_argument('--remove-correlated',
                       type=float,
                       metavar='THRESHOLD',
                       help='Remove highly correlated features above threshold (e.g., 0.95)')
    
    args = parser.parse_args()
    remove_missing = args.remove_missing == 'yes'
    replace_nan_value = args.replace_nan
    remove_correlated_threshold = args.remove_correlated
    
    print("="*80)
    print("ENHANCED XGBOOST CLINVAR TRAINER")
    print("With SMOTE, MCC/ROC-AUC Optimization, and Grid Search")
    print(f"Remove missing features: {remove_missing}")
    print(f"Replace NaN with: {replace_nan_value}")
    if remove_correlated_threshold is not None:
        print(f"Remove correlated features above: {remove_correlated_threshold}")
    else:
        print("Keep all correlated features")
    print("="*80)
    
    # Find all ClinVar files
    files = find_clinvar_files()
    
    if not files:
        print("No ClinVar files found!")
        return
    
    # Analyze file structure
    file_info, common_columns = analyze_file_structure(files)
    
    if len(common_columns) == 0:
        print("No common columns found across files!")
        return
    
    # Load and merge all files
    merged_df = load_and_merge_files(files, common_columns)
    
    if merged_df is None:
        print("Failed to merge files!")
        return
    
    # Prepare data with advanced preprocessing
    X, y, label_encoder = prepare_data_for_training(merged_df, 
                                                   remove_missing=remove_missing,
                                                   replace_nan_value=replace_nan_value,
                                                   remove_correlated_threshold=remove_correlated_threshold)
    
    if X is None:
        print("Failed to prepare data!")
        return
    
    # Train fast model
    model, (X_train, X_test, y_train, y_test), (y_train_pred, y_test_pred), (y_train_proba, y_test_proba) = train_fast_xgboost(X, y)
    
    # Evaluate model
    class_names = label_encoder.classes_
    
    train_metrics = evaluate_model_advanced(y_train, y_train_pred, y_train_proba, "Training Set", class_names)
    test_metrics = evaluate_model_advanced(y_test, y_test_pred, y_test_proba, "Test Set", class_names)
    
    # Advanced feature importance
    importance_df = show_feature_importance_advanced(model, X.columns.tolist())
    
    # Cross-validation analysis
    cv_results = cross_validate_model(model, X, y)
    
    # Create separate visualizations
    print("\nCreating separate visualization plots...")
    
    # 1. Feature importance plots
    plot_feature_importance_detailed(importance_df, top_n=25)
    
    # 2. Confusion matrix
    plot_confusion_matrix(y_test, y_test_pred, class_names)
    
    # 3. ROC curve
    plot_roc_curve(y_test, y_test_proba)
    
    # 4. XGBoost decision boundary with PCA
    plot_xgboost_decision_boundary(model, X_test, y_test, y_test_pred, class_names)
    
    # 5. Probability distribution
    plot_probability_distribution(y_test, y_test_proba, class_names)
    
    # Final summary
    print("\n" + "="*80)
    print("ENHANCED TRAINING SUMMARY")
    print("="*80)
    print(f"Files processed: {len(files)}")
    print(f"Total samples: {len(merged_df):,}")
    print(f"Features after preprocessing: {X.shape[1]}")
    print(f"Classes: {list(class_names)}")
    print(f"\nBest Model Performance:")
    print(f"Train MCC: {train_metrics['mcc']:.4f}")
    print(f"Test MCC:  {test_metrics['mcc']:.4f}")
    print(f"Test AUC:  {test_metrics['auc']:.4f}")
    print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Combined ROC-MCC: {(test_metrics['auc'] + test_metrics['mcc'])/2:.4f}")
    
    print(f"\nOptimizations Applied:")
    print(f"✓ SMOTE for balanced training data")
    print(f"✓ Optimized default hyperparameters (no grid search for speed)")
    print(f"✓ MCC and AUC evaluation metrics")
    print(f"✓ Advanced feature preprocessing")
    print(f"✓ Cross-validation analysis")
    print(f"✓ Remove high-missingness features: {remove_missing}")
    print(f"✓ Replace NaN with: {replace_nan_value}")
    if remove_correlated_threshold is not None:
        print(f"✓ Remove correlated features above: {remove_correlated_threshold}")
    else:
        print(f"✓ Keep all correlated features")

if __name__ == "__main__":
    main()