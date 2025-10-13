#!/usr/bin/env python3
"""
ClinVar LazyClassifier Analysis with SMOTE and Cross-Validation
Comprehensive model comparison using LazyPredict with advanced evaluation metrics
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    matthews_corrcoef, accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, classification_report, confusion_matrix
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from lazypredict.Supervised import LazyClassifier
import warnings
warnings.filterwarnings('ignore')

def find_clinvar_files(input_dir="Final_Results"):
    """Find all chr*_features_engineered_clinvar.csv files"""
    input_path = Path(input_dir)
    clinvar_files = list(input_path.glob("chr*_features_engineered_clinvar.csv"))
    
    print(f"Found {len(clinvar_files)} ClinVar files:")
    for file in sorted(clinvar_files):
        print(f"  {file.name}")
    
    return sorted(clinvar_files)

def load_and_merge_files(files, max_files=None):
    """Load and merge ClinVar files"""
    print(f"\nLoading and merging files...")
    
    if max_files:
        files = files[:max_files]
        print(f"Limited to first {max_files} files for faster processing")
    
    all_dataframes = []
    
    for file in files:
        try:
            print(f"Loading {file.name}...")
            df = pd.read_csv(file)
            
            # Add chromosome identifier
            chr_id = file.name.split('_')[0].replace('chr', '')
            df['chromosome'] = chr_id
            
            all_dataframes.append(df)
            print(f"  Loaded: {len(df):,} rows, {len(df.columns)} columns")
            
        except Exception as e:
            print(f"Error loading {file.name}: {e}")
            continue
    
    if not all_dataframes:
        print("No files loaded successfully!")
        return None
    
    # Find common columns across all dataframes
    common_columns = set(all_dataframes[0].columns)
    for df in all_dataframes[1:]:
        common_columns = common_columns.intersection(set(df.columns))
    
    print(f"Common columns across all files: {len(common_columns)}")
    
    # Keep only common columns
    common_columns = list(common_columns)
    for i, df in enumerate(all_dataframes):
        all_dataframes[i] = df[common_columns]
    
    # Merge all dataframes
    merged_df = pd.concat(all_dataframes, ignore_index=True)
    print(f"Total merged data: {len(merged_df):,} rows, {len(merged_df.columns)} columns")
    
    return merged_df

def prepare_data_for_modeling(df, sample_size=None):
    """Prepare data for machine learning models"""
    print("\nPreparing data for modeling...")
    
    # Check target column
    if 'CLNSIG' not in df.columns:
        print("Error: CLNSIG column not found!")
        return None, None, None
    
    # Sample data if needed
    if sample_size and len(df) > sample_size:
        print(f"Sampling {sample_size:,} rows from {len(df):,} total rows")
        df = df.sample(n=sample_size, random_state=42)
    
    # Show class distribution
    class_dist = df['CLNSIG'].value_counts()
    print(f"Class distribution:")
    for cls, count in class_dist.items():
        print(f"  {cls}: {count:,} ({count/len(df)*100:.1f}%)")
    
    # Encode target
    le = LabelEncoder()
    y = le.fit_transform(df['CLNSIG'])
    class_names = le.classes_
    
    # Separate features
    feature_cols = [col for col in df.columns if col not in ['CLNSIG', 'chromosome']]
    X = df[feature_cols].copy()
    
    print(f"Initial features: {len(feature_cols)}")
    
    # Convert to numeric and handle missing values
    print("Processing features...")
    numeric_features = []
    
    for col in X.columns:
        if X[col].dtype in ['object', 'string']:
            try:
                X[col] = pd.to_numeric(X[col], errors='coerce')
                numeric_features.append(col)
            except:
                continue
        else:
            numeric_features.append(col)
    
    X = X[numeric_features]
    
    # Handle missing values and infinities
    X = X.replace([np.inf, -np.inf, np.nan], 0)
    
    # Remove constant features
    constant_features = X.columns[X.var() == 0].tolist()
    if constant_features:
        print(f"Removing {len(constant_features)} constant features")
        X = X.drop(columns=constant_features)
    
    # Remove features with too many zero values
    zero_threshold = 0.95
    zero_ratio = (X == 0).sum() / len(X)
    high_zero_cols = zero_ratio[zero_ratio > zero_threshold].index.tolist()
    
    if high_zero_cols:
        print(f"Removing {len(high_zero_cols)} features with >{zero_threshold*100}% zero values")
        X = X.drop(columns=high_zero_cols)
    
    print(f"Final feature matrix: {X.shape}")
    
    return X, y, class_names

def run_lazy_classifier_with_smote(X, y, test_size=0.2, random_state=42):
    """Run LazyClassifier with SMOTE preprocessing"""
    print(f"\nRunning LazyClassifier with SMOTE...")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"Train set: {len(X_train):,} samples")
    print(f"Test set: {len(X_test):,} samples")
    
    # Show original class distribution
    unique, counts = np.unique(y_train, return_counts=True)
    print(f"Original train class distribution: {dict(zip(unique, counts))}")
    
    # Apply SMOTE to training data
    print("Applying SMOTE to training data...")
    smote = SMOTE(random_state=random_state, k_neighbors=5)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    
    unique, counts = np.unique(y_train_balanced, return_counts=True)
    print(f"SMOTE balanced train set: {len(X_train_balanced):,} samples")
    print(f"Balanced class distribution: {dict(zip(unique, counts))}")
    
    # Initialize LazyClassifier
    print("Running LazyClassifier...")
    clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
    
    # Fit LazyClassifier on balanced data
    models, predictions = clf.fit(X_train_balanced, X_test, y_train_balanced, y_test)
    
    print(f"LazyClassifier completed. Found {len(models)} models.")
    
    return models, predictions, (X_train, X_test, y_train, y_test), (X_train_balanced, y_train_balanced)

def calculate_comprehensive_metrics(models, X_train, X_test, y_train, y_test, X_train_balanced, y_train_balanced):
    """Calculate comprehensive metrics including MCC for all models"""
    print("\nCalculating comprehensive metrics...")
    
    results = []
    
    # Get model instances from LazyClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
    
    # Define model instances (common ones that LazyClassifier uses)
    model_instances = {
        'RandomForestClassifier': RandomForestClassifier(random_state=42),
        'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000),
        'SVC': SVC(random_state=42, probability=True),
        'KNeighborsClassifier': KNeighborsClassifier(),
        'DecisionTreeClassifier': DecisionTreeClassifier(random_state=42),
        'GaussianNB': GaussianNB(),
        'GradientBoostingClassifier': GradientBoostingClassifier(random_state=42),
        'AdaBoostClassifier': AdaBoostClassifier(random_state=42),
        'LinearDiscriminantAnalysis': LinearDiscriminantAnalysis(),
        'QuadraticDiscriminantAnalysis': QuadraticDiscriminantAnalysis()
    }
    
    for model_name in models.index:
        try:
            # Get the base model name (remove any suffixes)
            base_name = model_name.replace('Classifier', 'Classifier')
            
            if base_name in model_instances:
                model = model_instances[base_name]
            else:
                # Try to find a matching model
                matching_models = [k for k in model_instances.keys() if k in model_name or model_name in k]
                if matching_models:
                    model = model_instances[matching_models[0]]
                else:
                    print(f"Skipping {model_name} - model instance not found")
                    continue
            
            # Fit on balanced training data
            model.fit(X_train_balanced, y_train_balanced)
            
            # Make predictions
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            
            # Calculate metrics
            train_accuracy = accuracy_score(y_train, y_pred_train)
            test_accuracy = accuracy_score(y_test, y_pred_test)
            
            train_precision = precision_score(y_train, y_pred_train, average='weighted')
            test_precision = precision_score(y_test, y_pred_test, average='weighted')
            
            train_recall = recall_score(y_train, y_pred_train, average='weighted')
            test_recall = recall_score(y_test, y_pred_test, average='weighted')
            
            train_f1 = f1_score(y_train, y_pred_train, average='weighted')
            test_f1 = f1_score(y_test, y_pred_test, average='weighted')
            
            train_mcc = matthews_corrcoef(y_train, y_pred_train)
            test_mcc = matthews_corrcoef(y_test, y_pred_test)
            
            # AUC (for binary classification)
            if len(np.unique(y_test)) == 2:
                try:
                    y_pred_proba = model.predict_proba(X_test)[:, 1]
                    test_auc = roc_auc_score(y_test, y_pred_proba)
                except:
                    test_auc = 0.5  # Default for models without predict_proba
            else:
                test_auc = None
            
            results.append({
                'Model': model_name,
                'Train_Accuracy': train_accuracy,
                'Test_Accuracy': test_accuracy,
                'Train_Precision': train_precision,
                'Test_Precision': test_precision,
                'Train_Recall': train_recall,
                'Test_Recall': test_recall,
                'Train_F1': train_f1,
                'Test_F1': test_f1,
                'Train_MCC': train_mcc,
                'Test_MCC': test_mcc,
                'Test_AUC': test_auc if test_auc is not None else 0.5,
                'Accuracy_Diff': train_accuracy - test_accuracy,
                'MCC_Diff': train_mcc - test_mcc
            })
            
        except Exception as e:
            print(f"Error processing {model_name}: {e}")
            continue
    
    results_df = pd.DataFrame(results)
    
    # Sort by Test_MCC descending
    results_df = results_df.sort_values('Test_MCC', ascending=False)
    
    return results_df

def perform_cross_validation_analysis(X, y, class_names, top_models=5):
    """Perform cross-validation analysis on top models with SMOTE"""
    print(f"\nPerforming cross-validation analysis on top {top_models} models...")
    
    # Define models to test
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier
    
    models_cv = {
        'RandomForest': RandomForestClassifier(random_state=42, n_jobs=-1),
        'GradientBoosting': GradientBoostingClassifier(random_state=42),
        'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000),
        'SVC': SVC(random_state=42),
        'KNeighbors': KNeighborsClassifier()
    }
    
    cv_results = []
    cv_folds = 5
    
    # Create cross-validation strategy
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    for model_name, model in models_cv.items():
        print(f"Cross-validating {model_name}...")
        
        try:
            # Create pipeline with SMOTE and model
            pipeline = ImbPipeline([
                ('smote', SMOTE(random_state=42)),
                ('classifier', model)
            ])
            
            # Perform cross-validation for multiple metrics
            cv_accuracy = cross_val_score(pipeline, X, y, cv=cv, scoring='accuracy')
            cv_precision = cross_val_score(pipeline, X, y, cv=cv, scoring='precision_weighted')
            cv_recall = cross_val_score(pipeline, X, y, cv=cv, scoring='recall_weighted')
            cv_f1 = cross_val_score(pipeline, X, y, cv=cv, scoring='f1_weighted')
            
            # Custom MCC cross-validation
            from sklearn.metrics import make_scorer
            mcc_scorer = make_scorer(matthews_corrcoef)
            cv_mcc = cross_val_score(pipeline, X, y, cv=cv, scoring=mcc_scorer)
            
            cv_results.append({
                'Model': model_name,
                'CV_Accuracy_Mean': cv_accuracy.mean(),
                'CV_Accuracy_Std': cv_accuracy.std(),
                'CV_Precision_Mean': cv_precision.mean(),
                'CV_Precision_Std': cv_precision.std(),
                'CV_Recall_Mean': cv_recall.mean(),
                'CV_Recall_Std': cv_recall.std(),
                'CV_F1_Mean': cv_f1.mean(),
                'CV_F1_Std': cv_f1.std(),
                'CV_MCC_Mean': cv_mcc.mean(),
                'CV_MCC_Std': cv_mcc.std()
            })
            
        except Exception as e:
            print(f"Error in cross-validation for {model_name}: {e}")
            continue
    
    cv_results_df = pd.DataFrame(cv_results)
    cv_results_df = cv_results_df.sort_values('CV_MCC_Mean', ascending=False)
    
    return cv_results_df

def create_results_visualizations(results_df, cv_results_df, class_names):
    """Create comprehensive visualization of results"""
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    
    # 1. Test MCC Comparison
    ax1 = axes[0, 0]
    top_10 = results_df.head(10)
    bars = ax1.barh(range(len(top_10)), top_10['Test_MCC'])
    ax1.set_yticks(range(len(top_10)))
    ax1.set_yticklabels(top_10['Model'], fontsize=8)
    ax1.set_xlabel('Test MCC Score')
    ax1.set_title('Top 10 Models - Test MCC Performance')
    ax1.grid(True, alpha=0.3)
    
    # Add score labels
    for i, (bar, score) in enumerate(zip(bars, top_10['Test_MCC'])):
        ax1.text(score + 0.005, bar.get_y() + bar.get_height()/2, 
                f'{score:.3f}', va='center', fontsize=8)
    
    # 2. Train vs Test MCC
    ax2 = axes[0, 1]
    ax2.scatter(results_df['Train_MCC'], results_df['Test_MCC'], alpha=0.7)
    ax2.plot([0, 1], [0, 1], 'r--', alpha=0.8)
    ax2.set_xlabel('Train MCC')
    ax2.set_ylabel('Test MCC')
    ax2.set_title('Train vs Test MCC (Overfitting Check)')
    ax2.grid(True, alpha=0.3)
    
    # Add diagonal line and ideal region
    ax2.text(0.05, 0.95, 'Overfitting', transform=ax2.transAxes, 
             bbox=dict(boxstyle='round', facecolor='red', alpha=0.3))
    ax2.text(0.7, 0.05, 'Good Performance', transform=ax2.transAxes,
             bbox=dict(boxstyle='round', facecolor='green', alpha=0.3))
    
    # 3. Multiple Metrics Comparison
    ax3 = axes[0, 2]
    top_5 = results_df.head(5)
    
    x = np.arange(len(top_5))
    width = 0.15
    
    metrics = ['Test_Accuracy', 'Test_Precision', 'Test_Recall', 'Test_F1', 'Test_MCC']
    colors = ['skyblue', 'lightgreen', 'lightcoral', 'lightyellow', 'lightpink']
    
    for i, (metric, color) in enumerate(zip(metrics, colors)):
        ax3.bar(x + i*width, top_5[metric], width, label=metric.replace('Test_', ''), 
               color=color, alpha=0.8)
    
    ax3.set_xlabel('Models')
    ax3.set_ylabel('Score')
    ax3.set_title('Top 5 Models - Multiple Metrics')
    ax3.set_xticks(x + width * 2)
    ax3.set_xticklabels([name[:10] + '...' if len(name) > 10 else name 
                        for name in top_5['Model']], rotation=45, ha='right')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Cross-Validation Results
    ax4 = axes[1, 0]
    if not cv_results_df.empty:
        bars = ax4.barh(range(len(cv_results_df)), cv_results_df['CV_MCC_Mean'])
        ax4.set_yticks(range(len(cv_results_df)))
        ax4.set_yticklabels(cv_results_df['Model'])
        ax4.set_xlabel('CV MCC Mean')
        ax4.set_title('Cross-Validation MCC Results')
        ax4.grid(True, alpha=0.3)
        
        # Add error bars
        ax4.errorbar(cv_results_df['CV_MCC_Mean'], range(len(cv_results_df)),
                    xerr=cv_results_df['CV_MCC_Std'], fmt='none', color='red', alpha=0.7)
    else:
        ax4.text(0.5, 0.5, 'No CV Results', ha='center', va='center')
        ax4.set_title('Cross-Validation Results')
    
    # 5. Model Performance Distribution
    ax5 = axes[1, 1]
    ax5.hist(results_df['Test_MCC'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    ax5.axvline(results_df['Test_MCC'].mean(), color='red', linestyle='--', 
                label=f'Mean: {results_df["Test_MCC"].mean():.3f}')
    ax5.axvline(results_df['Test_MCC'].median(), color='green', linestyle='--',
                label=f'Median: {results_df["Test_MCC"].median():.3f}')
    ax5.set_xlabel('Test MCC Score')
    ax5.set_ylabel('Number of Models')
    ax5.set_title('MCC Score Distribution Across All Models')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Summary Statistics
    ax6 = axes[1, 2]
    ax6.axis('off')
    
    # Create summary text
    best_model = results_df.iloc[0]
    worst_model = results_df.iloc[-1]
    
    summary_text = f"""
MODEL COMPARISON SUMMARY

Best Performing Model:
• {best_model['Model'][:25]}
• Test MCC: {best_model['Test_MCC']:.4f}
• Test Accuracy: {best_model['Test_Accuracy']:.4f}
• Test AUC: {best_model['Test_AUC']:.4f}

Worst Performing Model:
• {worst_model['Model'][:25]}
• Test MCC: {worst_model['Test_MCC']:.4f}

Overall Statistics:
• Models Tested: {len(results_df)}
• Best MCC: {results_df['Test_MCC'].max():.4f}
• Worst MCC: {results_df['Test_MCC'].min():.4f}
• Mean MCC: {results_df['Test_MCC'].mean():.4f}
• Std MCC: {results_df['Test_MCC'].std():.4f}

Class Distribution:
• {class_names[0]}: {len([c for c in class_names if c == class_names[0]])}
• {class_names[1]}: {len([c for c in class_names if c == class_names[1]])}
    """
    
    ax6.text(0.1, 0.5, summary_text, fontsize=10, 
             verticalalignment='center', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    
    # Save plots
    output_dir = Path("plots")
    output_dir.mkdir(exist_ok=True)
    
    plt.savefig(output_dir / "lazy_classifier_analysis.png", dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / "lazy_classifier_analysis.pdf", bbox_inches='tight')
    
    print(f"\nVisualization saved to:")
    print(f"  • plots/lazy_classifier_analysis.png")
    print(f"  • plots/lazy_classifier_analysis.pdf")
    
    plt.show()

def print_detailed_results(results_df, cv_results_df):
    """Print detailed results in a nice format"""
    
    print("\n" + "="*100)
    print("DETAILED MODEL COMPARISON RESULTS")
    print("="*100)
    
    print("\nTop 10 Models by Test MCC:")
    print("-" * 100)
    top_10 = results_df.head(10)
    
    print(f"{'Rank':<4} {'Model':<25} {'Test_MCC':<9} {'Test_Acc':<9} {'Test_AUC':<9} {'Overfit':<8}")
    print("-" * 100)
    
    for i, (_, row) in enumerate(top_10.iterrows()):
        overfit_indicator = "⚠️" if row['MCC_Diff'] > 0.1 else "✅"
        print(f"{i+1:<4} {row['Model'][:24]:<25} {row['Test_MCC']:<9.4f} "
              f"{row['Test_Accuracy']:<9.4f} {row['Test_AUC']:<9.4f} {overfit_indicator:<8}")
    
    if not cv_results_df.empty:
        print(f"\nCross-Validation Results (5-fold):")
        print("-" * 80)
        print(f"{'Model':<20} {'CV_MCC':<15} {'CV_Accuracy':<15} {'CV_F1':<15}")
        print("-" * 80)
        
        for _, row in cv_results_df.iterrows():
            print(f"{row['Model']:<20} {row['CV_MCC_Mean']:.3f}±{row['CV_MCC_Std']:.3f}"
                  f"{'':3} {row['CV_Accuracy_Mean']:.3f}±{row['CV_Accuracy_Std']:.3f}"
                  f"{'':3} {row['CV_F1_Mean']:.3f}±{row['CV_F1_Std']:.3f}")
    
    print(f"\nModel Performance Summary:")
    print(f"• Best Test MCC: {results_df['Test_MCC'].max():.4f}")
    print(f"• Median Test MCC: {results_df['Test_MCC'].median():.4f}")
    print(f"• Models with MCC > 0.5: {sum(results_df['Test_MCC'] > 0.5)}/{len(results_df)}")
    print(f"• Models with MCC > 0.7: {sum(results_df['Test_MCC'] > 0.7)}/{len(results_df)}")

def main():
    print("="*80)
    print("CLINVAR LAZY CLASSIFIER ANALYSIS WITH SMOTE AND CROSS-VALIDATION")
    print("Comprehensive model comparison using LazyPredict")
    print("="*80)
    
    # Find ClinVar files
    files = find_clinvar_files()
    
    if not files:
        print("No ClinVar files found!")
        return
    
    # Load data (limit files for reasonable processing time)
    max_files = 3  # Adjust as needed
    merged_df = load_and_merge_files(files, max_files=max_files)
    
    if merged_df is None:
        print("Failed to load data!")
        return
    
    # Prepare data
    sample_size = 20000  # Limit for reasonable LazyClassifier runtime
    X, y, class_names = prepare_data_for_modeling(merged_df, sample_size=sample_size)
    
    if X is None:
        print("Failed to prepare data!")
        return
    
    # Run LazyClassifier with SMOTE
    models, predictions, train_test_split_data, balanced_data = run_lazy_classifier_with_smote(X, y)
    X_train, X_test, y_train, y_test = train_test_split_data
    X_train_balanced, y_train_balanced = balanced_data
    
    # Calculate comprehensive metrics
    results_df = calculate_comprehensive_metrics(
        models, X_train, X_test, y_train, y_test, X_train_balanced, y_train_balanced
    )
    
    # Perform cross-validation analysis
    cv_results_df = perform_cross_validation_analysis(X, y, class_names)
    
    # Create visualizations
    create_results_visualizations(results_df, cv_results_df, class_names)
    
    # Print detailed results
    print_detailed_results(results_df, cv_results_df)
    
    # Save results to CSV
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    
    results_df.to_csv(output_dir / "lazy_classifier_results.csv", index=False)
    if not cv_results_df.empty:
        cv_results_df.to_csv(output_dir / "cross_validation_results.csv", index=False)
    
    print(f"\nResults saved to:")
    print(f"  • results/lazy_classifier_results.csv")
    if not cv_results_df.empty:
        print(f"  • results/cross_validation_results.csv")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"Files processed: {max_files}")
    print(f"Samples analyzed: {len(y):,}")
    print(f"Features used: {X.shape[1]}")
    print(f"Models tested: {len(results_df)}")
    print(f"Best model: {results_df.iloc[0]['Model']}")
    print(f"Best MCC score: {results_df.iloc[0]['Test_MCC']:.4f}")

if __name__ == "__main__":
    main()