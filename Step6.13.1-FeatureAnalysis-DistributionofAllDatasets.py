import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import glob
import os
import warnings
warnings.filterwarnings('ignore')

def load_and_merge_datasets():
    """Load and merge all datasets with proper Dataset 5 handling"""
    datasets = {}
    
    # Dataset 1
    print("Loading Dataset 1...")
    try:
        testdataset1 = pd.read_csv("Clinvar_Dataset/Fold_1/X_test_processed.csv")
        traindataset1 = pd.read_csv("Clinvar_Dataset/Fold_1/X_train_processed.csv")
        predictiontraindataset1 = pd.read_csv("Clinvar_Dataset/Fold_1/Y_train.csv")
        predictiontestdataset1 = pd.read_csv("Clinvar_Dataset/Fold_1/Y_test.csv")

        merged_df1X = pd.concat([traindataset1, testdataset1], ignore_index=True)
        merged_df1Y = pd.concat([predictiontraindataset1, predictiontestdataset1], ignore_index=True)
        datasets['Dataset1_X'] = merged_df1X
        datasets['Dataset1_Y'] = merged_df1Y
        print(f"  Dataset 1 loaded: {merged_df1X.shape[0]} samples, {merged_df1X.shape[1]} features")
    except Exception as e:
        print(f"  Error loading Dataset 1: {e}")
    
    # Dataset 2
    print("Loading Dataset 2...")
    try:
        testdataset2 = pd.read_csv("Clinvar_Dataset2/Fold_1/X_test_processed.csv")
        traindataset2 = pd.read_csv("Clinvar_Dataset2/Fold_1/X_train_processed.csv")
        predictiontraindataset2 = pd.read_csv("Clinvar_Dataset2/Fold_1/Y_train.csv")
        predictiontestdataset2 = pd.read_csv("Clinvar_Dataset2/Fold_1/Y_test.csv")

        merged_df2X = pd.concat([traindataset2, testdataset2], ignore_index=True)
        merged_df2Y = pd.concat([predictiontraindataset2, predictiontestdataset2], ignore_index=True)
        datasets['Dataset2_X'] = merged_df2X
        datasets['Dataset2_Y'] = merged_df2Y
        print(f"  Dataset 2 loaded: {merged_df2X.shape[0]} samples, {merged_df2X.shape[1]} features")
    except Exception as e:
        print(f"  Error loading Dataset 2: {e}")
    
    # Dataset 3
    print("Loading Dataset 3...")
    try:
        testdataset3 = pd.read_csv("Clinvar_Dataset3/Fold_1/X_test_processed.csv")
        traindataset3 = pd.read_csv("Clinvar_Dataset3/Fold_1/X_train_processed.csv")
        predictiontraindataset3 = pd.read_csv("Clinvar_Dataset3/Fold_1/Y_train.csv")
        predictiontestdataset3 = pd.read_csv("Clinvar_Dataset3/Fold_1/Y_test.csv")

        merged_df3X = pd.concat([traindataset3, testdataset3], ignore_index=True)
        merged_df3Y = pd.concat([predictiontraindataset3, predictiontestdataset3], ignore_index=True)
        datasets['Dataset3_X'] = merged_df3X
        datasets['Dataset3_Y'] = merged_df3Y
        print(f"  Dataset 3 loaded: {merged_df3X.shape[0]} samples, {merged_df3X.shape[1]} features")
    except Exception as e:
        print(f"  Error loading Dataset 3: {e}")
    
    # Dataset 4
    print("Loading Dataset 4...")
    try:
        testdataset4 = pd.read_csv("Clinvar_Dataset4/Fold_1/X_test_processed.csv")
        traindataset4 = pd.read_csv("Clinvar_Dataset4/Fold_1/X_train_processed.csv")
        predictiontraindataset4 = pd.read_csv("Clinvar_Dataset4/Fold_1/Y_train.csv")
        predictiontestdataset4 = pd.read_csv("Clinvar_Dataset4/Fold_1/Y_test.csv")

        merged_df4X = pd.concat([traindataset4, testdataset4], ignore_index=True)
        merged_df4Y = pd.concat([predictiontraindataset4, predictiontestdataset4], ignore_index=True)
        datasets['Dataset4_X'] = merged_df4X
        datasets['Dataset4_Y'] = merged_df4Y
        print(f"  Dataset 4 loaded: {merged_df4X.shape[0]} samples, {merged_df4X.shape[1]} features")
    except Exception as e:
        print(f"  Error loading Dataset 4: {e}")
    
    # Dataset 5 - Handle different CLNSIG column names
    print("Loading Dataset 5 (chromosome files)...")
    try:
        features_files = glob.glob("Final_Results/chr*_features_engineered_clinvar.csv")
        predictions_files = glob.glob("Final_Results/chr*_predictions.csv")
        
        print(f"  Found {len(features_files)} feature files")
        print(f"  Found {len(predictions_files)} prediction files")
        
        all_X_data = []
        all_Y_data = []
        
        # Process each chromosome
        for pred_file in predictions_files:
            chr_name = pred_file.split('/')[-1].split('_')[0]
            
            # Find matching feature file
            feature_file = None
            for feat_file in features_files:
                if chr_name in feat_file:
                    feature_file = feat_file
                    break
            
            if feature_file is None:
                print(f"  Warning: No matching feature file for {chr_name}")
                continue
            
            # Load both files
            X_chr = pd.read_csv(feature_file)
            Y_chr = pd.read_csv(pred_file)
            
            print(f"  Processing {chr_name}: X={X_chr.shape}, Y={Y_chr.shape}")
            
            # Find CLNSIG column (handle different names)
            clnsig_col = None
            for col in Y_chr.columns:
                if 'CLNSIG' in col and 'original' not in col.lower():
                    clnsig_col = col
                    break
            
            if clnsig_col is None:
                print(f"    Warning: No CLNSIG column found for {chr_name}")
                continue
            
            print(f"    Using column: {clnsig_col}")
            
            # Filter for Benign and Pathogenic only
            benign_mask = Y_chr[clnsig_col] == 'Benign'
            pathogenic_mask = Y_chr[clnsig_col] == 'Pathogenic'
            keep_mask = benign_mask | pathogenic_mask
            
            if keep_mask.sum() > 0:
                # Apply filter to Y data
                Y_filtered = Y_chr[keep_mask].reset_index(drop=True)
                
                # Take matching number of X samples (assuming row correspondence)
                n_samples = len(Y_filtered)
                if len(X_chr) >= n_samples:
                    X_filtered = X_chr.iloc[:n_samples].reset_index(drop=True)
                else:
                    # If X has fewer samples, take all X and match Y
                    X_filtered = X_chr.reset_index(drop=True)
                    Y_filtered = Y_filtered.iloc[:len(X_chr)].reset_index(drop=True)
                
                # Final length check
                if len(X_filtered) == len(Y_filtered):
                    all_X_data.append(X_filtered)
                    all_Y_data.append(Y_filtered)
                    print(f"    Kept {len(X_filtered)} samples (Benign: {benign_mask.sum()}, Pathogenic: {pathogenic_mask.sum()})")
                else:
                    print(f"    Warning: Length mismatch for {chr_name}: X={len(X_filtered)}, Y={len(Y_filtered)}")
            else:
                print(f"    Warning: No Benign/Pathogenic samples found for {chr_name}")
        
        # Combine all chromosomes
        if all_X_data and all_Y_data:
            dataset5X = pd.concat(all_X_data, ignore_index=True)
            dataset5Y = pd.concat(all_Y_data, ignore_index=True)
            
            # Final verification
            assert len(dataset5X) == len(dataset5Y), f"Final length mismatch: X={len(dataset5X)}, Y={len(dataset5Y)}"
            
            datasets['Dataset5_X'] = dataset5X
            datasets['Dataset5_Y'] = dataset5Y
            
            print(f"  Dataset 5 final: {len(dataset5X)} samples, {dataset5X.shape[1]} features")
            
            # Check final Y distribution
            clnsig_col_final = None
            for col in dataset5Y.columns:
                if 'CLNSIG' in col and 'original' not in col.lower():
                    clnsig_col_final = col
                    break
            
            if clnsig_col_final:
                print(f"  Y distribution: {dataset5Y[clnsig_col_final].value_counts().to_dict()}")
        else:
            print("  Warning: No valid data for Dataset 5")
            datasets['Dataset5_X'] = pd.DataFrame()
            datasets['Dataset5_Y'] = pd.DataFrame()
            
    except Exception as e:
        print(f"  Error loading Dataset 5: {e}")
        datasets['Dataset5_X'] = pd.DataFrame()
        datasets['Dataset5_Y'] = pd.DataFrame()
    
    # Summary
    valid_datasets = [k for k, v in datasets.items() if '_X' in k and not v.empty]
    print(f"\nSuccessfully loaded {len(valid_datasets)} datasets")
    
    return datasets

def get_numeric_columns(df):
    """Get only numeric columns from dataframe"""
    return df.select_dtypes(include=[np.number]).columns.tolist()

def perform_pca_analysis(datasets):
    """Perform PCA analysis on each dataset"""
    pca_results = {}
    
    for dataset_name in ['Dataset1', 'Dataset2', 'Dataset3', 'Dataset4', 'Dataset5']:
        x_key = f'{dataset_name}_X'
        y_key = f'{dataset_name}_Y'
        
        if x_key in datasets and y_key in datasets and not datasets[x_key].empty and not datasets[y_key].empty:
            print(f"\nProcessing {dataset_name}...")
            
            X_data = datasets[x_key]
            y_data = datasets[y_key]
            
            print(f"  Data shapes - X: {X_data.shape}, Y: {y_data.shape}")
            
            # Get numeric columns only
            numeric_cols = get_numeric_columns(X_data)
            
            if len(numeric_cols) == 0:
                print(f"  Warning: No numeric columns in {dataset_name}")
                continue
            
            X_numeric = X_data[numeric_cols].copy()
            
            # Clean data
            X_numeric = X_numeric.replace([np.inf, -np.inf], np.nan)
            
            # Remove constant columns
            valid_cols = []
            for col in numeric_cols:
                if not X_numeric[col].isnull().all() and X_numeric[col].nunique() > 1:
                    valid_cols.append(col)
            
            if len(valid_cols) == 0:
                print(f"  Error: No valid columns in {dataset_name}")
                continue
            
            X_clean = X_numeric[valid_cols].copy()
            
            # Impute missing values
            imputer = SimpleImputer(strategy='median')
            X_imputed = pd.DataFrame(imputer.fit_transform(X_clean), columns=valid_cols)
            X_imputed = X_imputed.fillna(0).replace([np.inf, -np.inf], 0)
            
            # Remove zero variance columns
            X_final = X_imputed.loc[:, X_imputed.var() != 0]
            
            if X_final.shape[1] == 0:
                print(f"  Error: No features with variance in {dataset_name}")
                continue
            
            # Get labels - use CLNSIG column
            if 'CLNSIG' in y_data.columns:
                labels = y_data['CLNSIG'].values
            elif 'ML_Class' in y_data.columns:
                labels = y_data['ML_Class'].values
            else:
                labels = y_data.iloc[:, -1].values
            
            # Ensure matching lengths
            min_length = min(len(X_final), len(labels))
            X_final = X_final.iloc[:min_length]
            labels = labels[:min_length]
            
            print(f"  Final data: {X_final.shape[0]} samples, {X_final.shape[1]} features")
            
            # Standardize and perform PCA
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_final)
            
            n_components = min(2, X_scaled.shape[1], X_scaled.shape[0] - 1)
            pca = PCA(n_components=n_components, random_state=42)
            X_pca = pca.fit_transform(X_scaled)
            
            pca_results[dataset_name] = {
                'pca_data': X_pca,
                'labels': labels,
                'pca_model': pca,
                'n_samples': len(X_final),
                'n_features': X_final.shape[1]
            }
            
            print(f"  ✓ PCA completed: PC1={pca.explained_variance_ratio_[0]*100:.1f}% variance")
            
    return pca_results

def create_pca_plots(pca_results):
    """Create PCA visualization plots"""
    n_datasets = len(pca_results)
    
    if n_datasets == 0:
        print("No datasets for plotting")
        return
    
    # Calculate layout
    if n_datasets <= 2:
        rows, cols = 1, n_datasets
    elif n_datasets <= 4:
        rows, cols = 2, 2
    else:
        rows, cols = 2, 3
    
    fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 5*rows))
    
    if n_datasets == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes if hasattr(axes, '__len__') else [axes]
    else:
        axes = axes.flatten()
    
    # Colors for classes
    colors = {'Benign': '#2E8B57', 'Pathogenic': '#DC143C', 'Unknown': '#708090'}
    
    for i, (dataset_name, results) in enumerate(pca_results.items()):
        ax = axes[i]
        
        pca_data = results['pca_data']
        labels = results['labels']
        pca_model = results['pca_model']
        
        # Map labels to standard format (case sensitive exact match)
        mapped_labels = []
        for label in labels:
            label_str = str(label)
            if label_str == 'Benign':
                mapped_labels.append('Benign')
            elif label_str == 'Pathogenic':
                mapped_labels.append('Pathogenic')
            else:
                mapped_labels.append('Unknown')
        
        # Plot each class
        for class_name in ['Benign', 'Pathogenic', 'Unknown']:
            mask = [label == class_name for label in mapped_labels]
            if any(mask):
                indices = [j for j, m in enumerate(mask) if m]
                if len(indices) > 0:
                    ax.scatter(pca_data[indices, 0], pca_data[indices, 1], 
                             c=colors[class_name], 
                             label=f'{class_name} (n={len(indices)})', 
                             alpha=0.7, s=30)
        
        # Labels and title
        if pca_data.shape[1] >= 2:
            pc1_var = pca_model.explained_variance_ratio_[0] * 100
            pc2_var = pca_model.explained_variance_ratio_[1] * 100
            ax.set_xlabel(f'PC1 ({pc1_var:.1f}% variance)')
            ax.set_ylabel(f'PC2 ({pc2_var:.1f}% variance)')
        
        ax.set_title(f'{dataset_name}\n({results["n_samples"]:,} samples)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Hide empty subplots
    for i in range(len(pca_results), len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle('PCA Analysis: Benign vs Pathogenic Classifications', fontsize=16)
    plt.tight_layout()
    plt.savefig('pca_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main function"""
    print("Starting PCA Analysis")
    print("="*50)
    
    # Load datasets
    datasets = load_and_merge_datasets()
    
    # Perform PCA analysis
    pca_results = perform_pca_analysis(datasets)
    
    if len(pca_results) == 0:
        print("No valid datasets found for analysis")
        return
    
    # Create plots
    create_pca_plots(pca_results)
    
    # Print summary
    print("\nAnalysis Summary:")
    print("="*50)
    for name, results in pca_results.items():
        print(f"{name}: {results['n_samples']} samples, {results['n_features']} features")
    
    print("\nAnalysis completed!")
    return pca_results

if __name__ == "__main__":
    results = main()