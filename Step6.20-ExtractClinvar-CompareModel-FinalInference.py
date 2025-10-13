#!/usr/bin/env python3
"""
Comprehensive Chromosome Analysis - POLARS OPTIMIZED WITH COMPLETE VISUALIZATIONS
Compare predictions across submissions + annotations for ALL chromosomes
"""

import polars as pl
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

def comprehensive_chromosome_analysis(
    main_predictions_pattern="Final_Dataset/Predictions/chr{}_predictions.txt",
    annotations_pattern="Annovar_Merge/chr{}_concatenated_annotations.csv", 
    submission_dirs=[
        "/data/ascher01/uqmmune1/submission2",
        "/data/ascher01/uqmmune1/submission3"
    ],
    output_dir="all_chromosomes_results",
    chromosomes=['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', 
                '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', 
                '21', '22', 'X', 'Y'],
    chunk_size=100000
):
    """Comprehensive analysis with COMPLETE visualizations"""
    
    print("=" * 80)
    print("COMPREHENSIVE CHROMOSOME ANALYSIS - POLARS OPTIMIZED")
    print(f"Processing chunk size: {chunk_size:,} rows")
    print("=" * 80)
    
    Path(output_dir).mkdir(exist_ok=True)
    
    accumulated_correlations = {}
    chromosome_summaries = []
    total_variants_processed = 0
    all_data_for_distributions = []  # Store data for distribution plots
    
    # Process each chromosome
    for chrom in chromosomes:
        print(f"\n{'='*60}")
        print(f"PROCESSING CHROMOSOME {chrom}")
        print(f"{'='*60}")
        
        try:
            chrom_result = process_chromosome_polars(
                chrom, main_predictions_pattern, annotations_pattern, 
                submission_dirs, chunk_size
            )
            
            if chrom_result is not None:
                accumulate_correlations(accumulated_correlations, chrom_result['correlations'])
                chromosome_summaries.append(chrom_result['summary'])
                total_variants_processed += chrom_result['n_variants']
                
                # Store sample data for distribution plots
                if 'sample_data' in chrom_result:
                    all_data_for_distributions.append(chrom_result['sample_data'])
                
                print(f"✓ Chromosome {chrom} processed: {chrom_result['n_variants']:,} variants")
            else:
                print(f"✗ Chromosome {chrom} failed to process")
                
        except Exception as e:
            print(f"✗ Error processing chromosome {chrom}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Average results
    print(f"\n{'='*80}")
    print("AVERAGING RESULTS AND CREATING VISUALIZATIONS")
    print(f"{'='*80}")
    
    if len(accumulated_correlations) > 0:
        averaged_results = finalize_averaged_correlations(
            accumulated_correlations, len(chromosome_summaries), output_dir
        )
        
        # Create ALL comprehensive plots
        create_all_comprehensive_plots(
            averaged_results, chromosome_summaries, 
            all_data_for_distributions, output_dir
        )
        
        print_comprehensive_summary(averaged_results, chromosome_summaries, total_variants_processed)
        
        return averaged_results, chromosome_summaries
    else:
        print("No chromosomes were successfully processed!")
        return None, None

def process_chromosome_polars(chrom, main_predictions_pattern, annotations_pattern, 
                              submission_dirs, chunk_size):
    """Process a single chromosome using Polars"""
    
    main_predictions = main_predictions_pattern.format(chrom)
    annotations_file = annotations_pattern.format(chrom)
    
    if not Path(main_predictions).exists() or not Path(annotations_file).exists():
        print(f"Files not found for chromosome {chrom}")
        return None
    
    print(f"Loading files with Polars...")
    
    try:
        # Load annotations
        annotations = pl.read_csv(
            annotations_file,
            infer_schema_length=10000,
            low_memory=False
        )
        print(f"  Annotations loaded: {len(annotations):,} rows")
        
        # Load predictions
        main_pred = load_prediction_file_polars(main_predictions, f"Main_Model_chr{chrom}")
        if main_pred is None:
            return None
        
        submission_preds = {}
        for submission_dir in submission_dirs:
            submission_name = Path(submission_dir).name
            chr_file = Path(submission_dir) / f"dbNSFP4_nsSNV.chr{chrom}.template"
            
            if chr_file.exists():
                sub_pred = load_prediction_file_polars(chr_file, f"Model_{submission_name}_chr{chrom}")
                if sub_pred is not None:
                    submission_preds[f"Model_{submission_name}_chr{chrom}"] = sub_pred
        
        # Combine all data
        min_length = min(len(annotations), len(main_pred))
        for pred_df in submission_preds.values():
            min_length = min(min_length, len(pred_df))
        
        print(f"  Processing {min_length:,} variants")
        
        annotations = annotations.head(min_length)
        main_pred = main_pred.head(min_length)
        submission_preds = {k: v.head(min_length) for k, v in submission_preds.items()}
        
        all_predictions = {f"Main_Model_chr{chrom}": main_pred}
        all_predictions.update(submission_preds)
        
        # Calculate correlations
        correlations = calculate_correlations_polars(annotations, all_predictions, chrom)
        
        # Sample data for distribution plots (take first 10000 rows)
        sample_size = min(10000, min_length)
        sample_data = extract_sample_data(
            annotations.head(sample_size), 
            {k: v.head(sample_size) for k, v in all_predictions.items()},
            chrom
        )
        
        if correlations:
            summary = create_chromosome_summary(correlations, chrom, min_length)
            return {
                'chromosome': chrom,
                'correlations': correlations,
                'summary': summary,
                'n_variants': min_length,
                'sample_data': sample_data
            }
        
        return None
        
    except Exception as e:
        print(f"Error processing chromosome {chrom}: {e}")
        import traceback
        traceback.print_exc()
        return None

def load_prediction_file_polars(file_path, model_name):
    """Load prediction file with Polars"""
    
    try:
        with open(file_path, 'r') as f:
            first_line = f.readline().strip()
        
        has_header = first_line.startswith('#')
        
        if has_header:
            column_names = first_line[1:].strip().split('\t')
            df = pl.read_csv(
                file_path,
                separator='\t',
                skip_rows=1,
                has_header=False,
                new_columns=column_names,
                infer_schema_length=0,
                truncate_ragged_lines=True
            )
        else:
            df = pl.read_csv(
                file_path,
                separator='\t',
                infer_schema_length=0,
                truncate_ragged_lines=True
            )
        
        if 'pred' not in df.columns or 'score' not in df.columns:
            print(f"  Missing required columns in {model_name}")
            return None
        
        pred_map = {
            'Pathogenic': 1, 'Benign': 0, 'D': 1, 'T': 0, 'P': 1, 'B': 0, 'U': -1,
            'likely_pathogenic': 1, 'likely_benign': 0, 'unknown': -1,
            'LIKELY_PATHOGENIC': 1, 'LIKELY_BENIGN': 0, 'UNKNOWN': -1
        }
        
        df = df.with_columns([
            pl.col('score').cast(pl.Float64, strict=False).fill_null(0).alias(f'{model_name}_Score'),
            pl.col('pred').map_elements(
                lambda x: pred_map.get(str(x).strip(), pred_map.get(str(x).strip().upper(), 
                          pred_map.get(str(x).strip().lower(), -1))),
                return_dtype=pl.Int64
            ).alias(f'{model_name}_Pred'),
            pl.col('pred').alias(f'{model_name}_Pred_Label')  # Keep original labels
        ])
        
        return df.select([f'{model_name}_Score', f'{model_name}_Pred', f'{model_name}_Pred_Label'])
        
    except Exception as e:
        print(f"  Error loading {file_path}: {e}")
        return None

def extract_sample_data(annotations, predictions_dict, chrom):
    """Extract sample data for distribution plots"""
    
    sample_data = {
        'chromosome': chrom,
        'scores': {},
        'predictions': {}
    }
    
    # Convert to pandas for easier manipulation
    for model_name, pred_df in predictions_dict.items():
        pred_pd = pred_df.to_pandas()
        score_col = f'{model_name}_Score'
        pred_col = f'{model_name}_Pred'
        pred_label_col = f'{model_name}_Pred_Label'
        
        if score_col in pred_pd.columns:
            sample_data['scores'][model_name] = pred_pd[score_col].values
        if pred_col in pred_pd.columns and pred_label_col in pred_pd.columns:
            sample_data['predictions'][model_name] = {
                'values': pred_pd[pred_col].values,
                'labels': pred_pd[pred_label_col].values
            }
    
    return sample_data

def calculate_correlations_polars(annotations, predictions_dict, chrom):
    """Calculate correlations using Polars"""
    
    key_tools = [
        'AlphaMissense', 'am_pathogenicity', 'am_class',
        'CADD_phred', 'CADD_raw', 'CADD_raw_rankscore',
        'REVEL_score', 'REVEL_rankscore', 
        'SIFT4G_score', 'SIFT4G_pred',
        'SIFT_score', 'SIFT_pred',
        'BayesDel_addAF_score', 'BayesDel_addAF_pred',
        'MetaRNN_score', 'MetaRNN_pred',
        'PrimateAI_score', 'PrimateAI_pred',
        'combined_score_merged',
        'esm1v_t33_650M_UR90S_5_merged', 'esm1v_t33_650M_UR90S_4_merged'
    ]
    
    available_tools = [col for col in key_tools if col in annotations.columns]
    
    combined = annotations.select(available_tools)
    for model_name, pred_df in predictions_dict.items():
        combined = combined.hstack(pred_df.select([col for col in pred_df.columns if not col.endswith('_Label')]))
    
    combined_pd = combined.to_pandas()
    
    # Score correlations
    score_columns = [col for col in combined_pd.columns if '_Score' in col]
    score_columns.extend([tool for tool in available_tools 
                         if 'score' in tool.lower() or tool in ['AlphaMissense', 'am_pathogenicity', 'combined_score_merged']
                         or 'esm1v' in tool.lower()])
    
    score_data = {}
    for col in score_columns:
        if col in combined_pd.columns:
            raw_scores = combined_pd[col].astype(float, errors='ignore').fillna(0)
            
            if any(model in col for model in predictions_dict.keys()):
                score_data[col] = raw_scores.values
            else:
                if raw_scores.std() > 0:
                    scaler = MinMaxScaler()
                    normalized = scaler.fit_transform(raw_scores.values.reshape(-1, 1)).flatten()
                    score_data[col] = normalized
                else:
                    score_data[col] = raw_scores.values
    
    # Prediction correlations
    pred_columns = [col for col in combined_pd.columns if '_Pred' in col]
    pred_columns.extend([tool for tool in available_tools 
                        if 'pred' in tool.lower() or 'class' in tool.lower()])
    
    pred_map = {
        'D': 1, 'P': 1, 'T': 0, 'B': 0, 'U': -1, 
        'Pathogenic': 1, 'Benign': 0,
        'likely_pathogenic': 1, 'likely_benign': 0, 'unknown': -1,
        'H': 1, 'M': 1, 'A': 1, 'N': 0, 'L': 0,
        '.': -1, '': -1, 'nan': -1
    }
    
    pred_data = {}
    for col in pred_columns:
        if col in combined_pd.columns:
            if any(model in col for model in predictions_dict.keys()):
                pred_data[col] = combined_pd[col].values
            else:
                mapped = combined_pd[col].apply(
                    lambda x: pred_map.get(str(x).strip(), 
                             pred_map.get(str(x).strip().upper(), 
                             pred_map.get(str(x).strip().lower(), -1)))
                )
                pred_data[col] = mapped.values
    
    score_df = pd.DataFrame(score_data)
    pred_df = pd.DataFrame(pred_data)
    
    score_corr = score_df.corr() if len(score_data) > 1 else pd.DataFrame()
    pred_corr = pred_df.corr() if len(pred_data) > 1 else pd.DataFrame()
    
    # Convert to dictionary format
    result = {
        'score_corr': {},
        'pred_corr': {}
    }
    
    for i, row in enumerate(score_corr.index):
        for j, col in enumerate(score_corr.columns):
            key = f"score_{row}_{col}"
            result['score_corr'][key] = score_corr.iloc[i, j]
    
    for i, row in enumerate(pred_corr.index):
        for j, col in enumerate(pred_corr.columns):
            key = f"pred_{row}_{col}"
            result['pred_corr'][key] = pred_corr.iloc[i, j]
    
    return result

def create_chromosome_summary(correlations, chrom, total_variants):
    """Create summary for a single chromosome"""
    
    summary = {
        'chromosome': chrom,
        'n_variants': total_variants,
        'score_correlations': {},
        'pred_correlations': {}
    }
    
    main_model_key = f"Main_Model_chr{chrom}"
    
    for key, corr_val in correlations['score_corr'].items():
        if pd.notna(corr_val):
            parts = key.replace('score_', '').split('_', 1)
            if len(parts) >= 2:
                row_name, col_name = parts[0], parts[1]
                if main_model_key in row_name and 'Model_' not in row_name:
                    summary['score_correlations'][col_name] = corr_val
                elif main_model_key in col_name and 'Model_' not in col_name:
                    summary['score_correlations'][row_name] = corr_val
    
    return summary

def accumulate_correlations(accumulated_correlations, chrom_correlations):
    """Accumulate correlations across chromosomes"""
    
    for key, corr_val in chrom_correlations['score_corr'].items():
        if pd.notna(corr_val):
            if key not in accumulated_correlations:
                accumulated_correlations[key] = {'sum': 0, 'count': 0}
            accumulated_correlations[key]['sum'] += corr_val
            accumulated_correlations[key]['count'] += 1
    
    for key, corr_val in chrom_correlations['pred_corr'].items():
        if pd.notna(corr_val):
            if key not in accumulated_correlations:
                accumulated_correlations[key] = {'sum': 0, 'count': 0}
            accumulated_correlations[key]['sum'] += corr_val
            accumulated_correlations[key]['count'] += 1

def finalize_averaged_correlations(accumulated_correlations, n_chromosomes, output_dir):
    """Convert accumulated correlations to averages"""
    
    print(f"Finalizing averaged correlations from {n_chromosomes} chromosomes...")
    
    avg_score_corr_dict = {}
    avg_pred_corr_dict = {}
    
    for key, data in accumulated_correlations.items():
        avg_corr = data['sum'] / data['count']
        
        if key.startswith('score_'):
            avg_score_corr_dict[key] = avg_corr
        elif key.startswith('pred_'):
            avg_pred_corr_dict[key] = avg_corr
    
    avg_score_corr = create_correlation_matrix_from_dict(avg_score_corr_dict, 'score_')
    avg_pred_corr = create_correlation_matrix_from_dict(avg_pred_corr_dict, 'pred_')
    
    if len(avg_score_corr) > 0:
        avg_score_corr.to_csv(Path(output_dir) / "averaged_score_correlations.csv")
        print(f"Saved averaged score correlations: {avg_score_corr.shape}")
    
    if len(avg_pred_corr) > 0:
        avg_pred_corr.to_csv(Path(output_dir) / "averaged_prediction_correlations.csv")
        print(f"Saved averaged prediction correlations: {avg_pred_corr.shape}")
    
    return {
        'avg_score_corr': avg_score_corr,
        'avg_pred_corr': avg_pred_corr,
        'n_chromosomes': n_chromosomes
    }

def create_correlation_matrix_from_dict(corr_dict, prefix):
    """Create correlation matrix from dictionary"""
    
    if not corr_dict:
        return pd.DataFrame()
    
    rows = set()
    cols = set()
    
    for key in corr_dict.keys():
        key_clean = key.replace(prefix, '')
        parts = key_clean.split('_', 1)
        if len(parts) >= 2:
            row_name, col_name = parts[0], parts[1]
            rows.add(row_name)
            cols.add(col_name)
    
    if not rows or not cols:
        return pd.DataFrame()
    
    rows = sorted(list(rows))
    cols = sorted(list(cols))
    
    corr_matrix = pd.DataFrame(index=rows, columns=cols, dtype=float)
    
    for key, corr_val in corr_dict.items():
        key_clean = key.replace(prefix, '')
        parts = key_clean.split('_', 1)
        if len(parts) >= 2:
            row_name, col_name = parts[0], parts[1]
            if row_name in rows and col_name in cols:
                corr_matrix.loc[row_name, col_name] = corr_val
                if row_name != col_name and col_name in rows and row_name in cols:
                    corr_matrix.loc[col_name, row_name] = corr_val
    
    for idx in corr_matrix.index:
        if idx in corr_matrix.columns:
            corr_matrix.loc[idx, idx] = 1.0
    
    return corr_matrix

def create_all_comprehensive_plots(averaged_results, chromosome_summaries, 
                                   all_sample_data, output_dir):
    """Create ALL comprehensive plots including distributions"""
    
    plt.style.use('default')
    sns.set_context("paper", font_scale=0.8)
    
    print("\nCreating comprehensive visualizations...")
    
    # 1. SCORE Correlation Heatmap
    create_score_correlation_heatmap(averaged_results, output_dir)
    
    # 2. PREDICTION Correlation Heatmap
    create_prediction_correlation_heatmap(averaged_results, output_dir)
    
    # 3. Score Distribution Plots
    create_score_distributions(all_sample_data, output_dir)
    
    # 4. Prediction Category Distributions (with colors)
    create_prediction_distributions(all_sample_data, output_dir)
    
    # 5. Scatter plots: Scores colored by Predictions
    create_score_vs_prediction_scatter(all_sample_data, output_dir)
    
    # 6. Chromosome variation plots
    create_chromosome_variation_plots(chromosome_summaries, output_dir)
    
    # 7. Chromosome trends
    create_chromosome_trends_plot(chromosome_summaries, output_dir)

def create_score_correlation_heatmap(averaged_results, output_dir):
    """Create score correlation heatmap"""
    
    if len(averaged_results['avg_score_corr']) == 0:
        return
    
    fig, ax = plt.subplots(figsize=(20, 16))
    mask = np.triu(np.ones_like(averaged_results['avg_score_corr'], dtype=bool), k=1)
    
    sns.heatmap(averaged_results['avg_score_corr'],
                annot=True, cmap='RdBu_r', center=0, square=True, fmt='.2f',
                cbar_kws={'label': 'Correlation Coefficient'}, 
                mask=mask, ax=ax, annot_kws={'size': 7})
    
    ax.set_title(f'SCORE Correlations Across {averaged_results["n_chromosomes"]} Chromosomes', 
                fontsize=16, fontweight='bold', pad=20)
    
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    
    path = Path(output_dir) / "1_score_correlations_heatmap.png"
    plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    plt.close()
    print(f"✓ Score correlation heatmap saved: {path}")

def create_prediction_correlation_heatmap(averaged_results, output_dir):
    """Create PREDICTION correlation heatmap"""
    
    if len(averaged_results['avg_pred_corr']) == 0:
        return
    
    fig, ax = plt.subplots(figsize=(20, 16))
    mask = np.triu(np.ones_like(averaged_results['avg_pred_corr'], dtype=bool), k=1)
    
    sns.heatmap(averaged_results['avg_pred_corr'],
                annot=True, cmap='RdBu_r', center=0, square=True, fmt='.2f',
                cbar_kws={'label': 'Correlation Coefficient'}, 
                mask=mask, ax=ax, annot_kws={'size': 7})
    
    ax.set_title(f'PREDICTION Correlations Across {averaged_results["n_chromosomes"]} Chromosomes', 
                fontsize=16, fontweight='bold', pad=20)
    
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    
    path = Path(output_dir) / "2_prediction_correlations_heatmap.png"
    plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    plt.close()
    print(f"✓ Prediction correlation heatmap saved: {path}")

def create_score_distributions(all_sample_data, output_dir):
    """Create individual score distribution histograms"""
    
    if not all_sample_data:
        return
    
    # Combine all scores from all chromosomes
    all_scores = {}
    for sample in all_sample_data:
        for model_name, scores in sample['scores'].items():
            if model_name not in all_scores:
                all_scores[model_name] = []
            all_scores[model_name].extend(scores)
    
    if not all_scores:
        return
    
    # Create subplots for each model
    n_models = len(all_scores)
    ncols = 3
    nrows = (n_models + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 4*nrows))
    axes = axes.flatten() if n_models > 1 else [axes]
    
    for idx, (model_name, scores) in enumerate(all_scores.items()):
        ax = axes[idx]
        scores_clean = [s for s in scores if not np.isnan(s)]
        
        ax.hist(scores_clean, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
        ax.axvline(np.mean(scores_clean), color='red', linestyle='--', 
                  linewidth=2, label=f'Mean: {np.mean(scores_clean):.3f}')
        ax.axvline(np.median(scores_clean), color='green', linestyle='--', 
                  linewidth=2, label=f'Median: {np.median(scores_clean):.3f}')
        
        ax.set_xlabel('Score', fontsize=10)
        ax.set_ylabel('Frequency', fontsize=10)
        ax.set_title(model_name.replace('_Score', ''), fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(n_models, len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle('Score Distributions Across All Chromosomes', 
                fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    path = Path(output_dir) / "3_score_distributions.png"
    plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    plt.close()
    print(f"✓ Score distributions saved: {path}")

def create_prediction_distributions(all_sample_data, output_dir):
    """Create prediction category distributions with color coding"""
    
    if not all_sample_data:
        return
    
    # Combine all predictions
    all_predictions = {}
    for sample in all_sample_data:
        for model_name, pred_data in sample['predictions'].items():
            if model_name not in all_predictions:
                all_predictions[model_name] = {'values': [], 'labels': []}
            all_predictions[model_name]['values'].extend(pred_data['values'])
            all_predictions[model_name]['labels'].extend(pred_data['labels'])
    
    if not all_predictions:
        return
    
    # Create subplots
    n_models = len(all_predictions)
    ncols = 3
    nrows = (n_models + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 5*nrows))
    axes = axes.flatten() if n_models > 1 else [axes]
    
    # Color scheme: Pathogenic=Red, Benign=Green, Unknown=Gray
    colors = {1: '#d62728', 0: '#2ca02c', -1: '#7f7f7f'}
    labels_map = {1: 'Pathogenic', 0: 'Benign', -1: 'Unknown'}
    
    for idx, (model_name, pred_data) in enumerate(all_predictions.items()):
        ax = axes[idx]
        
        # Count predictions
        pred_counts = {}
        for val in pred_data['values']:
            pred_counts[val] = pred_counts.get(val, 0) + 1
        
        # Create bar plot
        categories = sorted(pred_counts.keys())
        counts = [pred_counts[cat] for cat in categories]
        bar_colors = [colors.get(cat, '#7f7f7f') for cat in categories]
        cat_labels = [labels_map.get(cat, f'Value_{cat}') for cat in categories]
        
        bars = ax.bar(range(len(categories)), counts, color=bar_colors, 
                     alpha=0.7, edgecolor='black', linewidth=1.5)
        
        ax.set_xticks(range(len(categories)))
        ax.set_xticklabels(cat_labels, fontsize=10)
        ax.set_ylabel('Count', fontsize=10)
        ax.set_title(model_name.replace('_Pred', ''), fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add percentage labels on bars
        total = sum(counts)
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            percentage = (count / total) * 100
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{count:,}\n({percentage:.1f}%)',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Hide unused subplots
    for idx in range(n_models, len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle('Prediction Category Distributions (Red=Pathogenic, Green=Benign, Gray=Unknown)', 
                fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    path = Path(output_dir) / "4_prediction_distributions.png"
    plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    plt.close()
    print(f"✓ Prediction distributions saved: {path}")

def create_score_vs_prediction_scatter(all_sample_data, output_dir):
    """Create scatter plots of scores colored by prediction category"""
    
    if not all_sample_data or len(all_sample_data) == 0:
        return
    
    # Get first sample with data
    sample = all_sample_data[0]
    if 'scores' not in sample or 'predictions' not in sample:
        return
    
    # Find models present in both scores and predictions
    score_models = list(sample['scores'].keys())
    pred_models = list(sample['predictions'].keys())
    common_models = [m for m in score_models if m in pred_models]
    
    if len(common_models) < 2:
        return
    
    # Take first two models for scatter plot
    model1 = common_models[0]
    model2 = common_models[1] if len(common_models) > 1 else common_models[0]
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Color scheme
    colors_map = {1: '#d62728', 0: '#2ca02c', -1: '#7f7f7f'}
    labels_map = {1: 'Pathogenic', 0: 'Benign', -1: 'Unknown'}
    
    # Plot 1: Model1 scores colored by Model1 predictions
    ax1 = axes[0]
    scores1 = sample['scores'][model1]
    preds1 = sample['predictions'][model1]['values']
    
    for pred_val, label in labels_map.items():
        mask = np.array(preds1) == pred_val
        ax1.scatter(np.arange(len(scores1))[mask], np.array(scores1)[mask],
                   c=colors_map[pred_val], label=label, alpha=0.6, s=10)
    
    ax1.set_xlabel('Variant Index', fontsize=11)
    ax1.set_ylabel('Score', fontsize=11)
    ax1.set_title(f'{model1.replace("_Score", "")} Scores\nColored by Prediction', 
                 fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Model1 vs Model2 scores
    ax2 = axes[1]
    scores2 = sample['scores'][model2]
    min_len = min(len(scores1), len(scores2), len(preds1))
    
    for pred_val, label in labels_map.items():
        mask = np.array(preds1[:min_len]) == pred_val
        ax2.scatter(np.array(scores1[:min_len])[mask], np.array(scores2[:min_len])[mask],
                   c=colors_map[pred_val], label=label, alpha=0.6, s=10)
    
    ax2.set_xlabel(model1.replace('_Score', ''), fontsize=11)
    ax2.set_ylabel(model2.replace('_Score', ''), fontsize=11)
    ax2.set_title(f'Score Comparison\nColored by {model1.replace("_Score", "")} Prediction', 
                 fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle('Scores Colored by Prediction Categories', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    path = Path(output_dir) / "5_scores_vs_predictions_scatter.png"
    plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    plt.close()
    print(f"✓ Score vs prediction scatter plots saved: {path}")

def create_chromosome_variation_plots(chromosome_summaries, output_dir):
    """Create chromosome variation plots"""
    
    chrom_data = []
    for summary in chromosome_summaries:
        for model, corr_val in summary['score_correlations'].items():
            chrom_data.append({
                'Chromosome': summary['chromosome'],
                'Model': model.replace('_Score', ''),
                'Correlation': corr_val,
                'Variants': summary['n_variants']
            })
    
    if not chrom_data:
        return
    
    chrom_df = pd.DataFrame(chrom_data)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
    
    # Box plot
    if len(chrom_df['Model'].unique()) > 0:
        sns.boxplot(data=chrom_df, x='Model', y='Correlation', ax=ax1, palette='Set2')
        sns.stripplot(data=chrom_df, x='Model', y='Correlation', 
                     color='red', alpha=0.6, size=4, ax=ax1)
        
        ax1.set_title('Correlation Variation Across Chromosomes', 
                     fontsize=14, fontweight='bold')
        ax1.set_ylabel('Correlation Coefficient', fontsize=12)
        ax1.set_xlabel('Model/Tool', fontsize=12)
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    # Variants bar plot
    variant_data = [(s['chromosome'], s['n_variants']) for s in chromosome_summaries]
    if variant_data:
        chroms, variants = zip(*variant_data)
        ax2.bar(range(len(chroms)), variants, alpha=0.7, color='steelblue', edgecolor='black')
        ax2.set_xlabel('Chromosome', fontsize=12)
        ax2.set_ylabel('Number of Variants', fontsize=12)
        ax2.set_title('Variants Processed per Chromosome', fontsize=14, fontweight='bold')
        ax2.set_xticks(range(len(chroms)))
        ax2.set_xticklabels(chroms, rotation=45)
        ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    path = Path(output_dir) / "6_chromosome_variation.png"
    plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    plt.close()
    print(f"✓ Chromosome variation plots saved: {path}")

def create_chromosome_trends_plot(chromosome_summaries, output_dir):
    """Create line plots showing trends across chromosomes"""
    
    chrom_data = []
    for summary in chromosome_summaries:
        for model, corr_val in summary['score_correlations'].items():
            chrom_data.append({
                'Chromosome': summary['chromosome'],
                'Model': model.replace('_Score', ''),
                'Correlation': corr_val
            })
    
    if not chrom_data:
        return
    
    chrom_df = pd.DataFrame(chrom_data)
    
    def sort_key(chrom):
        if chrom.isdigit():
            return (0, int(chrom))
        elif chrom == 'X':
            return (1, 0)
        elif chrom == 'Y':
            return (1, 1)
        else:
            return (2, 0)
    
    chrom_order = sorted(chrom_df['Chromosome'].unique(), key=sort_key)
    
    fig, ax = plt.subplots(figsize=(18, 8))
    
    models = chrom_df['Model'].unique()
    colors = plt.cm.tab20(np.linspace(0, 1, len(models)))
    
    for model, color in zip(models, colors):
        model_data = chrom_df[chrom_df['Model'] == model]
        model_data = model_data.set_index('Chromosome').reindex(chrom_order).reset_index()
        
        ax.plot(range(len(chrom_order)), model_data['Correlation'], 
               marker='o', label=model[:30], linewidth=2, markersize=6, 
               color=color, alpha=0.7)
    
    ax.set_xlabel('Chromosome', fontsize=12, fontweight='bold')
    ax.set_ylabel('Correlation Coefficient', fontsize=12, fontweight='bold')
    ax.set_title('Correlation Trends Across Chromosomes', fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(range(len(chrom_order)))
    ax.set_xticklabels(chrom_order, rotation=45)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    
    plt.tight_layout()
    
    path = Path(output_dir) / "7_chromosome_trends.png"
    plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    plt.close()
    print(f"✓ Chromosome trends saved: {path}")

def print_comprehensive_summary(averaged_results, chromosome_summaries, total_variants):
    """Print comprehensive summary"""
    
    print(f"\n{'='*80}")
    print("COMPREHENSIVE ANALYSIS SUMMARY")
    print(f"{'='*80}")
    
    print(f"Chromosomes processed: {averaged_results['n_chromosomes']}")
    print(f"Total variants processed: {total_variants:,}")
    if len(chromosome_summaries) > 0:
        avg_variants = total_variants / len(chromosome_summaries)
        print(f"Average variants per chromosome: {avg_variants:,.0f}")
    
    print(f"\nPer-chromosome breakdown:")
    print("-" * 40)
    for summary in chromosome_summaries:
        print(f"Chromosome {summary['chromosome']:>2}: {summary['n_variants']:>8,} variants")
    
    if len(averaged_results['avg_score_corr']) > 0:
        print(f"\nTop 10 Average Score Correlations:")
        print("-" * 50)
        
        main_cols = [col for col in averaged_results['avg_score_corr'].columns 
                    if 'Main_Model' in str(col)]
        if len(main_cols) > 0:
            main_col = main_cols[0]
            correlations = averaged_results['avg_score_corr'][main_col].drop(
                main_cols, errors='ignore'
            )
            top_corr = correlations.sort_values(ascending=False).head(10)
            
            for i, (tool, corr) in enumerate(top_corr.items(), 1):
                print(f"{i:2d}. {str(tool):<35} r = {corr:6.3f}")

if __name__ == "__main__":
    averaged_results, all_summaries = comprehensive_chromosome_analysis(
        main_predictions_pattern="Final_Dataset/Predictions/chr{}_predictions.txt",
        annotations_pattern="Annovar_Merge/chr{}_concatenated_annotations.csv",
        submission_dirs=[
            "/data/ascher01/uqmmune1/submission2",
            "/data/ascher01/uqmmune1/submission3"
        ],
        output_dir="all_chromosomes_results",
        chromosomes=['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', 
                    '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', 
                    '21', '22', 'X', 'Y'],
        chunk_size=100000
    )