#!/usr/bin/env python3
"""
Complete correlation analysis with all prediction tools from feature analysis
Normalizes all scores between 0-1 and creates publication-ready heatmaps
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import matthews_corrcoef

def create_comprehensive_correlation_analysis(fold_number=1, 
                                            data_dir="Clinvar_Dataset",
                                            ml_method_paths=[]):
    """
    Complete correlation analysis with all prediction tools and normalized scores
    """
    
    print("Loading base data...")
    fold_dir = Path(data_dir) / f"Fold_{fold_number}"
    X_train = pd.read_csv(fold_dir / "X_train.csv")
    X_test = pd.read_csv(fold_dir / "X_test.csv")
    X_combined = pd.concat([X_train, X_test], axis=0, ignore_index=True)
    
    y_train = pd.read_csv(fold_dir / "Y_train.csv")
    y_test = pd.read_csv(fold_dir / "Y_test.csv")
    y_combined = pd.concat([y_train, y_test], axis=0, ignore_index=True)
     
    # COMPREHENSIVE PREDICTION TOOL SCORES (from feature analysis)
    all_score_tools = [
        # CRITICAL SCORES
        # 'CADD_phred', 'CADD_raw', 'CADD_raw_rankscore',
        # 'SIFT4G_score', 'dbnsfp42c_SIFT4G_score',
        # 'SIFT_score', 'dbnsfp42c_SIFT_score', 
        # 'SIFT_converted_rankscore', 'dbnsfp42c_SIFT_converted_rankscore',
        # 'SIFT4G_converted_rankscore', 'dbnsfp42c_SIFT4G_converted_rankscore',
        # 'AlphaMissense', 'am_pathogenicity',
        # 'REVEL_score', 'REVEL_rankscore',
        
        'combined_score_merged',
            'esm1v_t33_650M_UR90S_5_merged',
            'esm1v_t33_650M_UR90S_4_merged',
            'esm1v_t33_650M_UR90S_1_merged',
            'esm1v_t33_650M_UR90S_2_merged',
            'esm1v_t33_650M_UR90S_3_merged',

        # # IMPORTANT SCORES  
        # 'BayesDel_addAF_score', 'dbnsfp42c_BayesDel_addAF_score',
        # 'BayesDel_noAF_score', 'dbnsfp42c_BayesDel_noAF_score',
        # 'BayesDel_addAF_rankscore', 'dbnsfp42c_BayesDel_addAF_rankscore',
        # 'BayesDel_noAF_rankscore', 'dbnsfp42c_BayesDel_noAF_rankscore',
        # 'fathmm-MKL_coding_score', 'dbnsfp42c_fathmm-MKL_coding_score',
        # 'fathmm-MKL_coding_rankscore', 'dbnsfp42c_fathmm-MKL_coding_rankscore',
        # 'DANN_score', 'dbnsfp42c_DANN_score',
        # 'DANN_rankscore', 'dbnsfp42c_DANN_rankscore',
        # 'VEST4_score', 'VEST4_rankscore',
        # 'GenoCanyon_score', 'GenoCanyon_rankscore',
        # 'MutationTaster_score', 'dbnsfp42c_MutationTaster_score',
        # 'MutationTaster_converted_rankscore', 'dbnsfp42c_MutationTaster_converted_rankscore',
        # 'MetaRNN_score', 'dbnsfp42c_MetaRNN_score',
        # 'MetaRNN_rankscore', 'dbnsfp42c_MetaRNN_rankscore',
        # 'LIST-S2_score', 'dbnsfp42c_LIST-S2_score',
        # 'LIST-S2_rankscore', 'dbnsfp42c_LIST-S2_rankscore',
        # 'MVP_score', 'dbnsfp42c_MVP_score',
        # 'MVP_rankscore', 'dbnsfp42c_MVP_rankscore',
        # 'integrated_fitCons_score', 'dbnsfp42c_integrated_fitCons_score',
        # 'integrated_fitCons_rankscore', 'dbnsfp42c_integrated_fitCons_rankscore',
        # 'integrated_confidence_value', 'dbnsfp42c_integrated_confidence_value',
        # 'DEOGEN2_score', 'dbnsfp42c_DEOGEN2_score',
        # 'DEOGEN2_rankscore', 'dbnsfp42c_DEOGEN2_rankscore',
        # 'PROVEAN_score', 'dbnsfp42c_PROVEAN_score',
        # 'PROVEAN_converted_rankscore', 'dbnsfp42c_PROVEAN_converted_rankscore',
        # 'M-CAP_score', 'dbnsfp42c_M-CAP_score',
        # 'M-CAP_rankscore', 'dbnsfp42c_M-CAP_rankscore',
        # 'fathmm-XF_coding_score', 'dbnsfp42c_fathmm-XF_coding_score',
        # 'fathmm-XF_coding_rankscore', 'dbnsfp42c_fathmm-XF_coding_rankscore',
        # 'ClinPred_score', 'dbnsfp42c_ClinPred_score',
        # 'ClinPred_rankscore', 'dbnsfp42c_ClinPred_rankscore',
        # 'MetaLR_score', 'dbnsfp42c_MetaLR_score',
        # 'MetaLR_rankscore', 'dbnsfp42c_MetaLR_rankscore',
        # 'MetaSVM_score', 'dbnsfp42c_MetaSVM_score',
        # 'MetaSVM_rankscore', 'dbnsfp42c_MetaSVM_rankscore',
        # 'Eigen-raw_coding', 'dbnsfp42c_Eigen-raw_coding',
        # 'Eigen-raw_coding_rankscore', 'dbnsfp42c_Eigen-raw_coding_rankscore',
        # 'Eigen-PC-raw_coding', 'dbnsfp42c_Eigen-PC-raw_coding',
        # 'Eigen-PC-raw_coding_rankscore', 'dbnsfp42c_Eigen-PC-raw_coding_rankscore',
        # 'FATHMM_score', 'dbnsfp42c_FATHMM_score',
        # 'FATHMM_converted_rankscore', 'dbnsfp42c_FATHMM_converted_rankscore',
        # 'PrimateAI_score', 'dbnsfp42c_PrimateAI_score',
        # 'PrimateAI_rankscore', 'dbnsfp42c_PrimateAI_rankscore',
        # 'MutPred_score', 'dbnsfp42c_MutPred_score',
        # 'MutPred_rankscore', 'dbnsfp42c_MutPred_rankscore',
        # 'LRT_score', 'dbnsfp42c_LRT_score',
        # 'LRT_converted_rankscore', 'dbnsfp42c_LRT_converted_rankscore',
        # 'Polyphen2_HDIV_score', 'Polyphen2_HDIV_rankscore',
        # 'Polyphen2_HVAR_score', 'Polyphen2_HVAR_rankscore',
        # 'MutationAssessor_score', 'dbnsfp42c_MutationAssessor_score',
        # 'MutationAssessor_rankscore', 'dbnsfp42c_MutationAssessor_rankscore',
        # 'MPC_score', 'dbnsfp42c_MPC_score',
        # 'MPC_rankscore', 'dbnsfp42c_MPC_rankscore',
        # 'SiPhy_29way_logOdds', 'dbnsfp42c_SiPhy_29way_logOdds',
        # 'SiPhy_29way_logOdds_rankscore', 'dbnsfp42c_SiPhy_29way_logOdds_rankscore'
    ]
    
    # COMPREHENSIVE PREDICTION METHODS
    all_pred_tools = [
        'SIFT4G_pred', 'dbnsfp42c_SIFT4G_pred','am_class',
        'SIFT_pred', 'dbnsfp42c_SIFT_pred',
         
        # 'BayesDel_addAF_pred', 'dbnsfp42c_BayesDel_addAF_pred',
        # 'BayesDel_noAF_pred', 'dbnsfp42c_BayesDel_noAF_pred',
        # 'fathmm-MKL_coding_pred', 'dbnsfp42c_fathmm-MKL_coding_pred',
        # 'MutationTaster_pred', 'dbnsfp42c_MutationTaster_pred',
        # 'MetaRNN_pred', 'dbnsfp42c_MetaRNN_pred',
        # 'LIST-S2_pred', 'dbnsfp42c_LIST-S2_pred',
        # 'DEOGEN2_pred', 'dbnsfp42c_DEOGEN2_pred',
        # 'PROVEAN_pred', 'dbnsfp42c_PROVEAN_pred',
        # 'M-CAP_pred', 'dbnsfp42c_M-CAP_pred',
        # 'fathmm-XF_coding_pred', 'dbnsfp42c_fathmm-XF_coding_pred',
        # 'ClinPred_pred', 'dbnsfp42c_ClinPred_pred',
        # 'MetaLR_pred', 'dbnsfp42c_MetaLR_pred',
        # 'MetaSVM_pred', 'dbnsfp42c_MetaSVM_pred',
        # 'FATHMM_pred', 'dbnsfp42c_FATHMM_pred',
        # 'PrimateAI_pred', 'dbnsfp42c_PrimateAI_pred',
        # 'LRT_pred', 'dbnsfp42c_LRT_pred',
        # 'Polyphen2_HDIV_pred', 'Polyphen2_HVAR_pred',
        # 'MutationAssessor_pred', 'dbnsfp42c_MutationAssessor_pred',
        # 'Aloft_pred', 'dbnsfp42c_Aloft_pred'
    ]
    
    # Filter available columns
    available_scores = [col for col in all_score_tools if col in X_combined.columns]
    available_preds = [col for col in all_pred_tools if col in X_combined.columns]
    
    print(f"Found {len(available_scores)} score tools and {len(available_preds)} prediction tools")
    
    # Prediction mapping
    pred_map = {
        'D': 1, 'P': 1, 'H': 1, 'M': 1, 'A': 1,  # Pathogenic
        'T': 0, 'B': 0, 'N': 0, 'L': 0,          # Benign
        'U': -1, '.': -1, '': -1, 'nan': -1       # Unknown
    }
    
    def map_pred(val):
        if pd.isna(val):
            return -1
        return pred_map.get(str(val).strip().upper(), -1)
    
    # Apply prediction mapping
    for col in available_preds:
        X_combined[f'{col}_mapped'] = X_combined[col].apply(map_pred)
    
    # ClinVar mapping
    clinvar_map = {
        'Benign': 0, 'Likely_benign': 0, 'Benign/Likely_benign': 0,
        'Pathogenic': 1, 'Likely_pathogenic': 1, 'Pathogenic/Likely_pathogenic': 1
    }
    
    def map_clinvar(val):
        if pd.isna(val):
            return -1
        val_str = str(val).strip()
        for key in clinvar_map:
            if key in val_str:
                return clinvar_map[key]
        return -1
    
    # Find and map ClinVar
    clinvar_col = None
    for col in y_combined.columns:
        if any(term in col.lower() for term in ['label', 'class', 'target']):
            clinvar_col = col
            break
    
    clinvar_mapped = None
    if clinvar_col:
        clinvar_mapped = y_combined[clinvar_col].apply(map_clinvar)
    
    # Load ML methods
    print("Loading ML methods...")
    ml_data = {}
    
    for path in ml_method_paths:
        fold_path = Path(path) / f"Fold_{fold_number}"
        
        if fold_path.exists():
            try:
                train_file = fold_path / "train_predictions.csv"
                test_file = fold_path / "test_predictions.csv"
                
                if train_file.exists() and test_file.exists():
                    ml_train = pd.read_csv(train_file)
                    ml_test = pd.read_csv(test_file)
                    
                    # Extract train data
                    train_data = pd.DataFrame({
                        'y_pred': ml_train['y_train_pred'],
                        'y_true': ml_train['y_train_true'], 
                        'y_proba': ml_train['y_train_proba']
                    })
                    
                    # Extract test data
                    test_data = pd.DataFrame({
                        'y_pred': ml_test['y_test_pred'],
                        'y_true': ml_test['y_test_true'],
                        'y_proba': ml_test['y_test_proba']
                    })
                    
                    # Combine vertically: train first, then test
                    ml_combined = pd.concat([train_data, test_data], ignore_index=True)
                    
                    # Create method name
                    parts = str(path).split('/')
                    dataset = [p for p in parts if 'Clinvar_Dataset' in p][0]
                    algorithm = parts[-1]
                    method_name = f"{dataset}_{algorithm}"
                    
                    # Store the data
                    ml_data[method_name] = {
                        'predictions': pd.to_numeric(ml_combined['y_pred'], errors='coerce').fillna(0),
                        'scores': pd.to_numeric(ml_combined['y_proba'], errors='coerce').fillna(0)
                    }
                    print(f"Loaded: {method_name}")
                    
            except Exception as e:
                print(f"Error loading {path}: {e}")
    
    print(f"Total ML methods loaded: {len(ml_data)}")
    
    # NORMALIZE ALL SCORES BETWEEN 0-1
    print("Normalizing scores between 0-1...")
    
    # Create score data and normalize
    score_data = {}
    
    # Add ML method scores (these are already probabilities, don't normalize)
    for name, data in ml_data.items():
        # Keep original probability values - they're already meaningful 0-1 probabilities
        score_data[name] = data['scores'].values
    
    # Add and normalize prediction tool scores INDIVIDUALLY
    for col in available_scores:
        raw_scores = pd.to_numeric(X_combined[col], errors='coerce').fillna(0)
        if raw_scores.std() > 0:  # Only normalize if there's variation
            # Create individual scaler for each column
            individual_scaler = MinMaxScaler()
            normalized = individual_scaler.fit_transform(raw_scores.values.reshape(-1, 1)).flatten()
            score_data[col] = normalized
        else:
            score_data[col] = raw_scores  # Keep constant values as is
    
    # Add normalized ClinVar
    if clinvar_mapped is not None:
        # ClinVar is already 0/1, so just handle -1 values
        clinvar_clean = clinvar_mapped.copy()
        clinvar_clean[clinvar_clean == -1] = 0  # Convert unknown to 0 for correlation
        score_data['ClinVar'] = clinvar_clean
    
    score_df = pd.DataFrame(score_data)
    score_corr = score_df.corr()
    
    # Create prediction correlation matrix
    print("Creating prediction correlations...")
    pred_data = {}
    
    # Add ML method predictions
    for name, data in ml_data.items():
        pred_data[name] = data['predictions']
    
    # Add prediction tool predictions (mapped)
    for col in available_preds:
        pred_data[col] = X_combined[f'{col}_mapped'].fillna(-1)
    
    # Add ClinVar
    if clinvar_mapped is not None:
        pred_data['ClinVar'] = clinvar_mapped.fillna(-1)
    
    pred_df = pd.DataFrame(pred_data)
    pred_corr = pred_df.corr()
    
    # Plot settings
    plt.style.use('default')
    sns.set_context("paper", font_scale=1.0)
    
    # Create score heatmap (20x17)
    print("Creating normalized score heatmap...")
    fig1, ax1 = plt.subplots(figsize=(20, 17))
    
    mask1 = np.triu(np.ones_like(score_corr, dtype=bool), k=1)
    
    sns.heatmap(score_corr,
                annot=True,
                cmap='RdBu_r',
                center=0,
                square=True,
                fmt='.1f',
                cbar_kws={'label': 'Correlation Coefficient'},
                mask=mask1,
                ax=ax1,
                annot_kws={'size': 8})
    
    ax1.set_title('Normalized Score Correlations (0-1) - All Methods', 
                  fontsize=18, fontweight='bold', pad=20)
    ax1.set_xlabel('Methods', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Methods', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()
    
    score_path = Path(data_dir) / f"normalized_score_correlations_fold_{fold_number}.png"
    plt.savefig(score_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    print(f"Normalized score heatmap saved: {score_path}")
    
    # Create prediction heatmap (20x17)
    print("Creating prediction heatmap...")
    fig2, ax2 = plt.subplots(figsize=(20, 17))
    
    mask2 = np.triu(np.ones_like(pred_corr, dtype=bool), k=1)
    
    sns.heatmap(pred_corr,
                annot=True,
                cmap='RdBu_r',
                center=0,
                square=True,
                fmt='.1f',
                cbar_kws={'label': 'Correlation Coefficient'},
                mask=mask2,
                ax=ax2,
                annot_kws={'size': 8})
    
    ax2.set_title('Prediction Correlations - All Methods', 
                  fontsize=18, fontweight='bold', pad=20)
    ax2.set_xlabel('Methods', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Methods', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()
    
    pred_path = Path(data_dir) / f"prediction_correlations_fold_{fold_number}.png"
    plt.savefig(pred_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    print(f"Prediction heatmap saved: {pred_path}")
    
    # Calculate and plot MCC comparison
    print("Calculating MCC with ClinVar...")
    mcc_results = calculate_mcc_with_clinvar(score_data, pred_data, clinvar_mapped)
    plot_mcc_comparison(mcc_results, data_dir, fold_number)
    
    # Print summary
    print(f"\n{'='*80}")
    print("ANALYSIS SUMMARY")
    print(f"{'='*80}")
    print(f"Total prediction tools analyzed:")
    print(f"  - Score tools: {len(available_scores)}")
    print(f"  - Prediction tools: {len(available_preds)}")
    print(f"  - ML methods: {len(ml_data)}")
    print(f"  - All scores normalized to 0-1 range")
    print(f"  - Figure size: 20x17 inches")
    print(f"  - Resolution: 300 DPI")
    print(f"  - MCC calculated for {len(mcc_results)} methods")
    
    return score_corr, pred_corr, mcc_results

def calculate_mcc_with_clinvar(score_data, pred_data, clinvar_mapped):
    """
    Calculate Matthews Correlation Coefficient for all methods vs ClinVar
    """
    mcc_results = {}
    
    if clinvar_mapped is None:
        print("No ClinVar data available for MCC calculation")
        return {}
    
    # Clean ClinVar data (remove unknown values)
    clinvar_clean = clinvar_mapped.copy()
    valid_mask = clinvar_clean != -1
    clinvar_clean = clinvar_clean[valid_mask]
    
    print(f"Calculating MCC for {len(clinvar_clean)} valid ClinVar entries...")
    
    # Calculate MCC for score-based methods (using 0.5 threshold)
    for method, scores in score_data.items():
        if method == 'ClinVar':
            continue
            
        try:
            scores_clean = scores[valid_mask]
            # Convert scores to binary predictions using 0.5 threshold
            pred_binary = (scores_clean >= 0.5).astype(int)
            mcc = matthews_corrcoef(clinvar_clean, pred_binary)
            mcc_results[f"{method}_score"] = mcc
        except Exception as e:
            print(f"Error calculating MCC for {method} scores: {e}")
            mcc_results[f"{method}_score"] = 0
    
    # Calculate MCC for prediction-based methods
    for method, preds in pred_data.items():
        if method == 'ClinVar':
            continue
            
        try:
            preds_clean = preds[valid_mask]
            # Only use known predictions (not -1)
            pred_mask = preds_clean != -1
            if pred_mask.sum() > 0:
                mcc = matthews_corrcoef(clinvar_clean[pred_mask], preds_clean[pred_mask])
                mcc_results[f"{method}_pred"] = mcc
            else:
                mcc_results[f"{method}_pred"] = 0
        except Exception as e:
            print(f"Error calculating MCC for {method} predictions: {e}")
            mcc_results[f"{method}_pred"] = 0
    
    return mcc_results

def plot_mcc_comparison(mcc_results, data_dir, fold_number):
    """
    Create bar plot of MCC values compared to ClinVar
    """
    if not mcc_results:
        print("No MCC results to plot")
        return
    
    # Sort by MCC value (descending)
    sorted_mcc = dict(sorted(mcc_results.items(), key=lambda x: x[1], reverse=True))
    
    methods = list(sorted_mcc.keys())
    mcc_values = list(sorted_mcc.values())
    
    # Create color scheme
    colors = ['darkblue' if 'MachineLearning' in method or 'FLAML' in method or 'XGBoost' in method or 'RandomForest' in method 
              else 'darkred' if any(tool in method for tool in ['AlphaMissense', 'CADD', 'REVEL', 'SIFT']) 
              else 'darkgreen' for method in methods]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(20, 12))
    
    bars = ax.bar(range(len(methods)), mcc_values, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
    
    # Customize plot
    ax.set_xlabel('Prediction Methods', fontsize=14, fontweight='bold')
    ax.set_ylabel('Matthews Correlation Coefficient (MCC)', fontsize=14, fontweight='bold')
    ax.set_title('Matthews Correlation Coefficient vs ClinVar\n(All Prediction Methods)', 
                 fontsize=18, fontweight='bold', pad=20)
    
    # Set x-axis labels
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=45, ha='right', fontsize=10)
    
    # Add horizontal line at MCC = 0
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars, mcc_values)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01 if height >= 0 else height - 0.03,
                f'{value:.3f}', ha='center', va='bottom' if height >= 0 else 'top', 
                fontsize=8, fontweight='bold')
    
    # Set y-axis limits
    y_min = min(mcc_values) - 0.1
    y_max = max(mcc_values) + 0.1
    ax.set_ylim(y_min, y_max)
    
    # Add grid
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='darkblue', alpha=0.7, label='ML Methods'),
        Patch(facecolor='darkred', alpha=0.7, label='Key Prediction Tools'),
        Patch(facecolor='darkgreen', alpha=0.7, label='Other Tools')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    
    # Save plot
    mcc_path = Path(data_dir) / f"mcc_vs_clinvar_fold_{fold_number}.png"
    plt.savefig(mcc_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    print(f"MCC comparison plot saved: {mcc_path}")
    
    # Print top 10 results
    print(f"\nTop 10 Methods by MCC with ClinVar:")
    print("=" * 50)
    for i, (method, mcc) in enumerate(list(sorted_mcc.items())[:10], 1):
        print(f"{i:2d}. {method:<40} MCC: {mcc:6.3f}")

# Usage
if __name__ == "__main__":
    ml_paths = [
        "Clinvar_Dataset/MachineLearning/FLAML_AutoML",
        "Clinvar_Dataset/MachineLearning/XGBoost",
        "Clinvar_Dataset/MachineLearning/RandomForest",
        "Clinvar_Dataset2/MachineLearning/FLAML_AutoML", 
        "Clinvar_Dataset2/MachineLearning/XGBoost",
        "Clinvar_Dataset2/MachineLearning/RandomForest",
        "Clinvar_Dataset3/MachineLearning/FLAML_AutoML",
        "Clinvar_Dataset3/MachineLearning/XGBoost",
        "Clinvar_Dataset3/MachineLearning/RandomForest"
        "Clinvar_Dataset4/MachineLearning/FLAML_AutoML",
        "Clinvar_Dataset4/MachineLearning/XGBoost",
        "Clinvar_Dataset4/MachineLearning/RandomForest"

        "Clinvar_Dataset4/MachineLearning/FLAML_AutoML",
        "Clinvar_Dataset4/MachineLearning/XGBoost",
        "Clinvar_Dataset4/MachineLearning/RandomForest",
        "Clinvar_Dataset/DeepLearning/predictions/simpleBuildFnnModel/"


    ]
    
    score_corr, pred_corr, mcc_results = create_comprehensive_correlation_analysis(
        fold_number=1,
        data_dir="Clinvar_Dataset",
        ml_method_paths=ml_paths
    )