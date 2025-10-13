import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import matthews_corrcoef, silhouette_score
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# PyCaret imports
from pycaret.classification import *
from pycaret.clustering import *

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

def prepare_data_for_pycaret(df, sample_size=None):
    """Prepare data for PyCaret analysis"""
    print("\nPreparing data for PyCaret...")
    
    # Check target column
    if 'CLNSIG' not in df.columns:
        print("Error: CLNSIG column not found!")
        return None, None
    
    # Sample data if needed
    if sample_size and len(df) > sample_size:
        print(f"Sampling {sample_size:,} rows from {len(df):,} total rows")
        df = df.sample(n=sample_size, random_state=42)
    
    # Show class distribution
    class_dist = df['CLNSIG'].value_counts()
    print(f"Class distribution:")
    for cls, count in class_dist.items():
        print(f"  {cls}: {count:,} ({count/len(df)*100:.1f}%)")
    
    # Rename target column for PyCaret
    df = df.rename(columns={'CLNSIG': 'target'})
    
    # Separate features
    feature_cols = [col for col in df.columns if col not in ['target', 'chromosome']]
    
    print(f"Initial features: {len(feature_cols)}")
    
    # Convert to numeric and handle missing values
    print("Processing features...")
    numeric_features = []
    
    for col in feature_cols:
        if df[col].dtype in ['object', 'string']:
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                numeric_features.append(col)
            except:
                continue
        else:
            numeric_features.append(col)
    
    # Keep only numeric features plus target
    final_cols = numeric_features + ['target']
    df_clean = df[final_cols].copy()
    
    # Handle missing values and infinities
    df_clean = df_clean.replace([np.inf, -np.inf, np.nan], 0)
    
    # Remove constant features
    feature_variance = df_clean[numeric_features].var()
    constant_features = feature_variance[feature_variance == 0].index.tolist()
    
    if constant_features:
        print(f"Removing {len(constant_features)} constant features")
        df_clean = df_clean.drop(columns=constant_features)
        numeric_features = [f for f in numeric_features if f not in constant_features]
    
    # Remove features with too many zero values
    zero_threshold = 0.95
    zero_ratio = (df_clean[numeric_features] == 0).sum() / len(df_clean)
    high_zero_cols = zero_ratio[zero_ratio > zero_threshold].index.tolist()
    
    if high_zero_cols:
        print(f"Removing {len(high_zero_cols)} features with >{zero_threshold*100}% zero values")
        df_clean = df_clean.drop(columns=high_zero_cols)
    
    print(f"Final dataset shape: {df_clean.shape}")
    
    # Get class names for later use
    class_names = df_clean['target'].unique()
    
    return df_clean, class_names

def run_pycaret_classification(df, class_names):
    """Run comprehensive classification analysis using PyCaret"""
    print("\n" + "="*80)
    print("PYCARET CLASSIFICATION ANALYSIS")
    print("="*80)
    
    # Setup PyCaret classification environment
    print("Setting up PyCaret classification environment...")
    
    try:
        # Try the most basic setup first (oldest PyCaret versions)
        clf = setup(
            data=df, 
            target_col='target',
            session_id=42,
            silent=True
        )
        print("Using oldest PyCaret API (target_col parameter)")
    except (TypeError, NameError):
        try:
            # Try PyCaret 2.x API
            clf = setup(
                data=df, 
                target='target',
                session_id=42,
                train_size=0.8,
                silent=True,
                fix_imbalance=True,
                normalize=True,
                transformation=True,
                remove_multicollinearity=True,
                multicollinearity_threshold=0.95,
                feature_selection=True,
                feature_selection_threshold=0.8,
                fold=5
            )
            print("Using PyCaret 2.x API")
        except (TypeError, NameError):
            try:
                # Try PyCaret 3.x API
                clf = setup(
                    df, 
                    target='target',
                    session_id=42,
                    train_size=0.8,
                    preprocess=True,
                    imputation_type='simple',
                    fix_imbalance=True,
                    fix_imbalance_method='smote',
                    normalize=True,
                    transformation=True,
                    remove_multicollinearity=True,
                    multicollinearity_threshold=0.95,
                    feature_selection=True,
                    feature_selection_threshold=0.8,
                    fold=5,
                    verbose=False
                )
                print("Using PyCaret 3.x API")
            except Exception as e:
                print(f"All PyCaret setup attempts failed: {e}")
                # Try absolute minimal setup
                try:
                    clf = setup(data=df, target_col='target')
                    print("Using minimal PyCaret setup")
                except:
                    try:
                        clf = setup(df, target='target')
                        print("Using minimal PyCaret setup (v2)")
                    except Exception as final_e:
                        print(f"Final setup attempt failed: {final_e}")
                        return [], pd.DataFrame(), pd.DataFrame()
    
    print("PyCaret setup complete!")
    
    # Compare all models in one line
    print("\nRunning comprehensive model comparison...")
    print("This will train and evaluate multiple classification algorithms...")
    
    try:
        models = compare_models(
            fold=3,  # Reduce folds for compatibility
            round=4,
            sort='MCC',
            n_select=-1,
            verbose=False
        )
    except TypeError:
        try:
            # Try without some parameters
            models = compare_models(
                fold=3,
                sort='MCC',
                verbose=False
            )
        except:
            try:
                # Most basic compare_models
                models = compare_models()
            except Exception as e:
                print(f"compare_models failed: {e}")
                return [], pd.DataFrame(), pd.DataFrame()
    
    if isinstance(models, list):
        print(f"Model comparison complete! Trained {len(models)} models.")
    else:
        # Single model returned, convert to list
        models = [models]
        print(f"Model comparison complete! Trained 1 model.")
    
    # Get the comparison results
    try:
        comparison_results = pull()
    except Exception as e:
        print(f"Error pulling comparison results: {e}")
        comparison_results = pd.DataFrame()
    
    # Calculate additional metrics for each model
    print("\nCalculating additional metrics...")
    
    detailed_results = []
    
    for i, model in enumerate(models):
        try:
            model_name = str(type(model).__name__)
            
            # Get predictions
            try:
                train_pred = predict_model(model, verbose=False)
                
                # Find the correct column names
                target_col = 'target'
                pred_col = None
                
                possible_pred_cols = ['prediction_label', 'Label', 'prediction', 'predicted']
                for col in possible_pred_cols:
                    if col in train_pred.columns:
                        pred_col = col
                        break
                
                if pred_col and target_col in train_pred.columns:
                    y_true = train_pred[target_col]
                    y_pred = train_pred[pred_col]
                    train_mcc = matthews_corrcoef(y_true, y_pred)
                else:
                    print(f"Skipping {model_name} - prediction columns not found")
                    print(f"Available columns: {list(train_pred.columns)}")
                    train_mcc = 0.0
                
            except Exception as pred_error:
                print(f"Error predicting for {model_name}: {pred_error}")
                train_mcc = 0.0
            
            # Get metrics from comparison results if available
            try:
                if not comparison_results.empty and i < len(comparison_results):
                    model_row = comparison_results.iloc[i]
                    test_mcc = model_row.get('MCC', 0.0)
                    test_accuracy = model_row.get('Accuracy', 0.0)
                    test_auc = model_row.get('AUC', 0.0)
                    test_recall = model_row.get('Recall', 0.0)
                    test_precision = model_row.get('Prec.', model_row.get('Precision', 0.0))
                    test_f1 = model_row.get('F1', 0.0)
                else:
                    test_mcc = test_accuracy = test_auc = test_recall = test_precision = test_f1 = 0.0
            except Exception as metric_error:
                print(f"Error getting metrics for {model_name}: {metric_error}")
                test_mcc = test_accuracy = test_auc = test_recall = test_precision = test_f1 = 0.0
            
            detailed_results.append({
                'Model': model_name,
                'Train_MCC': train_mcc,
                'Test_MCC': test_mcc,
                'Test_Accuracy': test_accuracy,
                'Test_AUC': test_auc,
                'Test_Precision': test_precision,
                'Test_Recall': test_recall,
                'Test_F1': test_f1,
                'MCC_Difference': train_mcc - test_mcc,
                'Cross_Validation': True
            })
            
        except Exception as e:
            print(f"Error processing model {i}: {e}")
            continue
    
    detailed_results_df = pd.DataFrame(detailed_results)
    if not detailed_results_df.empty:
        detailed_results_df = detailed_results_df.sort_values('Test_MCC', ascending=False)
    
    return models, comparison_results, detailed_results_df

def run_pycaret_clustering(df):
    """Run comprehensive clustering analysis using PyCaret"""
    print("\n" + "="*80)
    print("PYCARET CLUSTERING ANALYSIS")
    print("="*80)
    
    # Prepare data for clustering (remove target)
    df_clustering = df.drop('target', axis=1)
    
    # Setup PyCaret clustering environment
    print("Setting up PyCaret clustering environment...")
    
    # Reset any existing PyCaret session
    try:
        from pycaret.clustering import finalize_model
        # Try to finalize any existing model to clean up
    except:
        pass
    
    setup_success = False
    
    try:
        # Try the most basic setup first (oldest PyCaret versions)
        from pycaret.clustering import setup as cluster_setup
        clu = cluster_setup(
            data=df_clustering,
            session_id=42,
            silent=True
        )
        print("Using oldest PyCaret clustering API")
        setup_success = True
    except Exception as e1:
        print(f"Oldest API failed: {e1}")
        try:
            # Try PyCaret 2.x API
            clu = cluster_setup(
                data=df_clustering,
                session_id=42,
                silent=True,
                normalize=True,
                transformation=True,
                remove_multicollinearity=True,
                multicollinearity_threshold=0.95
            )
            print("Using PyCaret 2.x clustering API")
            setup_success = True
        except Exception as e2:
            print(f"PyCaret 2.x API failed: {e2}")
            try:
                # Try PyCaret 3.x API
                clu = cluster_setup(
                    df_clustering,
                    session_id=42,
                    preprocess=True,
                    normalize=True,
                    transformation=True,
                    remove_multicollinearity=True,
                    multicollinearity_threshold=0.95,
                    verbose=False
                )
                print("Using PyCaret 3.x clustering API")
                setup_success = True
            except Exception as e3:
                print(f"PyCaret 3.x API failed: {e3}")
                try:
                    # Try without any parameters except data
                    clu = cluster_setup(df_clustering)
                    print("Using minimal clustering setup")
                    setup_success = True
                except Exception as e4:
                    print(f"All clustering setup attempts failed: {e4}")
                    print("Skipping clustering analysis...")
                    return [], pd.DataFrame(), {}, pd.DataFrame()
    
    if not setup_success:
        print("Failed to setup PyCaret clustering environment")
        return [], pd.DataFrame(), {}, pd.DataFrame()
    
    print("PyCaret clustering setup complete!")
    
    # Compare clustering models
    print("\nRunning comprehensive clustering comparison...")
    print("This will evaluate multiple clustering algorithms...")
    
    try:
        from pycaret.clustering import compare_models as cluster_compare_models
        from pycaret.clustering import pull as cluster_pull
        
        clustering_models = cluster_compare_models(verbose=False, round=4)
        clustering_results = cluster_pull()
        print("Clustering comparison successful!")
    except Exception as e:
        print(f"Error with clustering compare_models: {e}")
        clustering_models = []
        clustering_results = pd.DataFrame()
        print("Continuing with individual clustering models...")
    
    print(f"Clustering comparison complete!")
    
    # Create specific cluster models for detailed analysis
    print("\nCreating detailed clustering models...")
    
    cluster_models = {}
    cluster_detailed_results = []
    
    # Import clustering functions
    try:
        from pycaret.clustering import create_model as cluster_create_model
        from pycaret.clustering import assign_model as cluster_assign_model
    except ImportError as e:
        print(f"Error importing clustering functions: {e}")
        return clustering_models, clustering_results, cluster_models, pd.DataFrame(cluster_detailed_results)
    
    # K-Means clustering
    try:
        print("Creating K-Means model...")
        kmeans = cluster_create_model('kmeans', num_clusters=2)
        kmeans_results = cluster_assign_model(kmeans)
        
        # Calculate silhouette score
        cluster_col = None
        possible_cluster_cols = ['Cluster', 'cluster', 'labels', 'Labels']
        for col in possible_cluster_cols:
            if col in kmeans_results.columns:
                cluster_col = col
                break
        
        if cluster_col:
            cluster_labels = kmeans_results[cluster_col]
            if len(np.unique(cluster_labels)) > 1:
                silhouette_kmeans = silhouette_score(df_clustering, cluster_labels)
            else:
                silhouette_kmeans = 0.0
        else:
            print(f"Available columns in K-means results: {list(kmeans_results.columns)}")
            silhouette_kmeans = 0.0
        
        cluster_models['KMeans'] = kmeans
        cluster_detailed_results.append({
            'Model': 'KMeans',
            'Silhouette_Score': silhouette_kmeans,
            'Num_Clusters': 2
        })
        print(f"K-Means silhouette score: {silhouette_kmeans:.4f}")
        
    except Exception as e:
        print(f"Error with K-Means: {e}")
    
    # Hierarchical clustering
    try:
        print("Creating Hierarchical clustering model...")
        hclust = cluster_create_model('hclust', num_clusters=2)
        hclust_results = cluster_assign_model(hclust)
        
        cluster_col = None
        possible_cluster_cols = ['Cluster', 'cluster', 'labels', 'Labels']
        for col in possible_cluster_cols:
            if col in hclust_results.columns:
                cluster_col = col
                break
        
        if cluster_col:
            cluster_labels = hclust_results[cluster_col]
            if len(np.unique(cluster_labels)) > 1:
                silhouette_hclust = silhouette_score(df_clustering, cluster_labels)
            else:
                silhouette_hclust = 0.0
        else:
            print(f"Available columns in hierarchical results: {list(hclust_results.columns)}")
            silhouette_hclust = 0.0
        
        cluster_models['Hierarchical'] = hclust
        cluster_detailed_results.append({
            'Model': 'Hierarchical',
            'Silhouette_Score': silhouette_hclust,
            'Num_Clusters': 2
        })
        print(f"Hierarchical silhouette score: {silhouette_hclust:.4f}")
        
    except Exception as e:
        print(f"Error with Hierarchical clustering: {e}")
    
    # DBSCAN
    try:
        print("Creating DBSCAN model...")
        dbscan = cluster_create_model('dbscan')
        dbscan_results = cluster_assign_model(dbscan)
        
        cluster_col = None
        possible_cluster_cols = ['Cluster', 'cluster', 'labels', 'Labels']
        for col in possible_cluster_cols:
            if col in dbscan_results.columns:
                cluster_col = col
                break
        
        if cluster_col:
            cluster_labels = dbscan_results[cluster_col]
            n_clusters_dbscan = len(np.unique(cluster_labels))
            if n_clusters_dbscan > 1:
                silhouette_dbscan = silhouette_score(df_clustering, cluster_labels)
            else:
                silhouette_dbscan = 0.0
        else:
            print(f"Available columns in DBSCAN results: {list(dbscan_results.columns)}")
            n_clusters_dbscan = 1
            silhouette_dbscan = 0.0
        
        cluster_models['DBSCAN'] = dbscan
        cluster_detailed_results.append({
            'Model': 'DBSCAN',
            'Silhouette_Score': silhouette_dbscan,
            'Num_Clusters': n_clusters_dbscan
        })
        print(f"DBSCAN silhouette score: {silhouette_dbscan:.4f} ({n_clusters_dbscan} clusters)")
        
    except Exception as e:
        print(f"Error with DBSCAN: {e}")
    
    # Additional clustering algorithms
    try:
        print("Creating Gaussian Mixture Model...")
        gmm = cluster_create_model('gcluster', num_clusters=2)
        gmm_results = cluster_assign_model(gmm)
        
        cluster_col = None
        possible_cluster_cols = ['Cluster', 'cluster', 'labels', 'Labels']
        for col in possible_cluster_cols:
            if col in gmm_results.columns:
                cluster_col = col
                break
        
        if cluster_col:
            cluster_labels = gmm_results[cluster_col]
            if len(np.unique(cluster_labels)) > 1:
                silhouette_gmm = silhouette_score(df_clustering, cluster_labels)
            else:
                silhouette_gmm = 0.0
        else:
            silhouette_gmm = 0.0
        
        cluster_models['GaussianMixture'] = gmm
        cluster_detailed_results.append({
            'Model': 'GaussianMixture',
            'Silhouette_Score': silhouette_gmm,
            'Num_Clusters': 2
        })
        print(f"Gaussian Mixture silhouette score: {silhouette_gmm:.4f}")
        
    except Exception as e:
        print(f"Error with Gaussian Mixture: {e}")
    
    cluster_detailed_df = pd.DataFrame(cluster_detailed_results)
    if not cluster_detailed_df.empty:
        cluster_detailed_df = cluster_detailed_df.sort_values('Silhouette_Score', ascending=False)
        print(f"\nBest clustering model: {cluster_detailed_df.iloc[0]['Model']} with silhouette score: {cluster_detailed_df.iloc[0]['Silhouette_Score']:.4f}")
    
    return clustering_models, clustering_results, cluster_models, cluster_detailed_df

def create_comprehensive_visualizations(classification_results, clustering_results, class_names):
    """Create comprehensive visualizations for both classification and clustering"""
    
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Check if we have valid results
    has_classification = not classification_results.empty and 'Test_MCC' in classification_results.columns
    has_clustering = not clustering_results.empty and 'Silhouette_Score' in clustering_results.columns
    
    if not has_classification and not has_clustering:
        print("No valid results to visualize. Creating basic data summary...")
        
        # Create a simple summary plot
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        # Show class distribution only
        class_counts = [sum(class_names == cls) for cls in np.unique(class_names)]
        class_labels = np.unique(class_names)
        colors_class = plt.cm.Set1(np.linspace(0, 1, len(class_labels)))
        
        bars = ax.bar(class_labels, class_counts, color=colors_class)
        ax.set_ylabel('Count')
        ax.set_title('Class Distribution - Analysis Failed')
        ax.grid(True, alpha=0.3)
        
        # Add count labels
        for bar, count in zip(bars, class_counts):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(class_counts)*0.01,
                    f'{count:,}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        # Save plot
        output_dir = Path("plots")
        output_dir.mkdir(exist_ok=True)
        
        plt.savefig(output_dir / "pycaret_basic_summary.png", dpi=300, bbox_inches='tight')
        print(f"\nBasic summary saved to: plots/pycaret_basic_summary.png")
        plt.show()
        return
    
    # Create figure with subplots
    fig, axes = plt.subplots(3, 3, figsize=(20, 18))
    
    # 1. Classification Model Comparison (MCC)
    ax1 = axes[0, 0]
    if has_classification:
        top_10_class = classification_results.head(10)
        bars = ax1.barh(range(len(top_10_class)), top_10_class['Test_MCC'])
        ax1.set_yticks(range(len(top_10_class)))
        ax1.set_yticklabels(top_10_class['Model'], fontsize=8)
        ax1.set_xlabel('Test MCC Score')
        ax1.set_title('Top 10 Classification Models - MCC Performance')
        ax1.grid(True, alpha=0.3)
        
        # Add score labels
        for i, (bar, score) in enumerate(zip(bars, top_10_class['Test_MCC'])):
            ax1.text(score + 0.005, bar.get_y() + bar.get_height()/2, 
                    f'{score:.3f}', va='center', fontsize=8)
    else:
        ax1.text(0.5, 0.5, 'Classification Failed', ha='center', va='center', transform=ax1.transAxes)
        ax1.set_title('Classification Results - Failed')
    
    # 2. Classification Train vs Test MCC
    ax2 = axes[0, 1]
    if has_classification and 'Train_MCC' in classification_results.columns:
        ax2.scatter(classification_results['Train_MCC'], classification_results['Test_MCC'], alpha=0.7)
        ax2.plot([0, 1], [0, 1], 'r--', alpha=0.8)
        ax2.set_xlabel('Train MCC')
        ax2.set_ylabel('Test MCC')
        ax2.set_title('Classification: Train vs Test MCC')
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'Train/Test MCC\nNot Available', ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Train vs Test MCC - N/A')
    
    # 3. Classification Multiple Metrics
    ax3 = axes[0, 2]
    if has_classification:
        top_5_class = classification_results.head(5)
        
        x = np.arange(len(top_5_class))
        width = 0.15
        
        # Check which metrics are available
        available_metrics = []
        metric_names = []
        colors = []
        base_metrics = ['Test_Accuracy', 'Test_Precision', 'Test_Recall', 'Test_F1', 'Test_MCC']
        base_colors = ['skyblue', 'lightgreen', 'lightcoral', 'lightyellow', 'lightpink']
        
        for metric, color in zip(base_metrics, base_colors):
            if metric in top_5_class.columns:
                available_metrics.append(metric)
                metric_names.append(metric.replace('Test_', ''))
                colors.append(color)
        
        if available_metrics:
            for i, (metric, color) in enumerate(zip(available_metrics, colors)):
                ax3.bar(x + i*width, top_5_class[metric], width, 
                       label=metric.replace('Test_', ''), color=color, alpha=0.8)
            
            ax3.set_xlabel('Models')
            ax3.set_ylabel('Score')
            ax3.set_title('Top 5 Classification Models - Multiple Metrics')
            ax3.set_xticks(x + width * len(available_metrics)//2)
            ax3.set_xticklabels([name[:8] + '...' if len(name) > 8 else name 
                                for name in top_5_class['Model']], rotation=45, ha='right')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, 'No Metrics Available', ha='center', va='center', transform=ax3.transAxes)
    else:
        ax3.text(0.5, 0.5, 'Classification Failed', ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Multiple Metrics - N/A')
    
    # 4. Clustering Results
    ax4 = axes[1, 0]
    if has_clustering:
        bars = ax4.barh(range(len(clustering_results)), clustering_results['Silhouette_Score'])
        ax4.set_yticks(range(len(clustering_results)))
        ax4.set_yticklabels(clustering_results['Model'], fontsize=8)
        ax4.set_xlabel('Silhouette Score')
        ax4.set_title('Clustering Models - Silhouette Score')
        ax4.grid(True, alpha=0.3)
        
        # Add score labels
        for i, (bar, score) in enumerate(zip(bars, clustering_results['Silhouette_Score'])):
            ax4.text(score + 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{score:.3f}', va='center', fontsize=8)
    else:
        ax4.text(0.5, 0.5, 'No Clustering Results', ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Clustering Results - N/A')
    
    # 5. Classification Performance Distribution
    ax5 = axes[1, 1]
    if has_classification:
        ax5.hist(classification_results['Test_MCC'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax5.axvline(classification_results['Test_MCC'].mean(), color='red', linestyle='--', 
                    label=f'Mean: {classification_results["Test_MCC"].mean():.3f}')
        ax5.set_xlabel('Test MCC Score')
        ax5.set_ylabel('Number of Models')
        ax5.set_title('Classification MCC Distribution')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
    else:
        ax5.text(0.5, 0.5, 'No MCC Data', ha='center', va='center', transform=ax5.transAxes)
        ax5.set_title('MCC Distribution - N/A')
    
    # 6. Model Performance vs Complexity
    ax6 = axes[1, 2]
    if has_classification and 'Test_AUC' in classification_results.columns:
        ax6.scatter(classification_results['Test_AUC'], classification_results['Test_MCC'], alpha=0.7)
        ax6.set_xlabel('Test AUC (Model Complexity Proxy)')
        ax6.set_ylabel('Test MCC')
        ax6.set_title('Classification: Performance vs Complexity')
        ax6.grid(True, alpha=0.3)
    else:
        ax6.text(0.5, 0.5, 'AUC vs MCC\nNot Available', ha='center', va='center', transform=ax6.transAxes)
        ax6.set_title('Performance vs Complexity - N/A')
    
    # 7. Class Distribution
    ax7 = axes[2, 0]
    class_counts = [sum(class_names == cls) for cls in np.unique(class_names)]
    class_labels = np.unique(class_names)
    colors_class = plt.cm.Set1(np.linspace(0, 1, len(class_labels)))
    
    bars = ax7.bar(class_labels, class_counts, color=colors_class)
    ax7.set_ylabel('Count')
    ax7.set_title('Original Class Distribution')
    ax7.grid(True, alpha=0.3)
    
    # Add count labels
    for bar, count in zip(bars, class_counts):
        ax7.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(class_counts)*0.01,
                f'{count:,}', ha='center', va='bottom')
    
    # 8. Top Models Comparison
    ax8 = axes[2, 1]
    if has_classification and len(classification_results) >= 3:
        top_3_models = classification_results.head(3)
        
        # Check available metrics
        available_metrics = []
        for metric in ['Test_Accuracy', 'Test_Precision', 'Test_Recall', 'Test_F1', 'Test_MCC']:
            if metric in top_3_models.columns:
                available_metrics.append(metric)
        
        if available_metrics:
            x_pos = np.arange(len(available_metrics))
            width = 0.25
            
            for i, (_, model_row) in enumerate(top_3_models.iterrows()):
                values = [model_row[metric] for metric in available_metrics]
                ax8.bar(x_pos + i*width, values, width, label=model_row['Model'][:10], alpha=0.8)
            
            ax8.set_xlabel('Metrics')
            ax8.set_ylabel('Score')
            ax8.set_title('Top 3 Models - Detailed Comparison')
            ax8.set_xticks(x_pos + width)
            ax8.set_xticklabels([m.replace('Test_', '') for m in available_metrics], rotation=45)
            ax8.legend()
            ax8.grid(True, alpha=0.3)
        else:
            ax8.text(0.5, 0.5, 'No Metrics for\nComparison', ha='center', va='center', transform=ax8.transAxes)
    else:
        ax8.text(0.5, 0.5, 'Insufficient Models\nfor Comparison', ha='center', va='center', transform=ax8.transAxes)
        ax8.set_title('Top Models Comparison - N/A')
    
    # 9. Summary Statistics
    ax9 = axes[2, 2]
    ax9.axis('off')
    
    # Create summary text
    if has_classification:
        best_class_model = classification_results.iloc[0]
        classification_summary = f"""
CLASSIFICATION:
Best Model: {best_class_model['Model'][:20]}
• Test MCC: {best_class_model.get('Test_MCC', 'N/A'):.4f if best_class_model.get('Test_MCC') else 'N/A'}
• Test Accuracy: {best_class_model.get('Test_Accuracy', 'N/A'):.4f if best_class_model.get('Test_Accuracy') else 'N/A'}
• Test AUC: {best_class_model.get('Test_AUC', 'N/A'):.4f if best_class_model.get('Test_AUC') else 'N/A'}

Models Tested: {len(classification_results)}
Models with MCC > 0.5: {sum(classification_results['Test_MCC'] > 0.5) if 'Test_MCC' in classification_results.columns else 'N/A'}
Mean MCC: {classification_results['Test_MCC'].mean():.3f if 'Test_MCC' in classification_results.columns else 'N/A'}"""
    else:
        classification_summary = """
CLASSIFICATION:
Status: FAILED
No models were trained successfully"""

    if has_clustering:
        best_cluster_model = clustering_results.iloc[0]
        clustering_summary = f"""
CLUSTERING:
Best Model: {best_cluster_model['Model']}
Best Silhouette: {best_cluster_model['Silhouette_Score']:.3f}"""
    else:
        clustering_summary = """
CLUSTERING:
Status: FAILED or NO RESULTS"""

    summary_text = f"""
COMPREHENSIVE ANALYSIS SUMMARY
{classification_summary}
{clustering_summary}

DATASET:
Classes: {len(np.unique(class_names))}
Features: Auto-selected by PyCaret
SMOTE: Applied if successful
Cross-Validation: 3-fold
    """
    
    ax9.text(0.1, 0.5, summary_text, fontsize=9, 
             verticalalignment='center', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    
    # Save plots
    output_dir = Path("plots")
    output_dir.mkdir(exist_ok=True)
    
    plt.savefig(output_dir / "pycaret_comprehensive_analysis.png", dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / "pycaret_comprehensive_analysis.pdf", bbox_inches='tight')
    
    print(f"\nComprehensive visualization saved to:")
    print(f"  • plots/pycaret_comprehensive_analysis.png")
    print(f"  • plots/pycaret_comprehensive_analysis.pdf")
    
    plt.show()

def print_comprehensive_results(classification_results, clustering_results):
    """Print comprehensive results in a nice format"""
    
    print("\n" + "="*100)
    print("PYCARET COMPREHENSIVE ANALYSIS RESULTS")
    print("="*100)
    
    if not classification_results.empty:
        print("\nCLASSIFICATION RESULTS:")
        print("-" * 100)
        print(f"{'Rank':<4} {'Model':<25} {'Test_MCC':<9} {'Test_Acc':<9} {'Test_AUC':<9} {'CV_Used':<8}")
        print("-" * 100)
        
        for i, (_, row) in enumerate(classification_results.head(10).iterrows()):
            cv_indicator = "✅" if row.get('Cross_Validation', False) else "❌"
            test_mcc = f"{row.get('Test_MCC', 0.0):.4f}" if row.get('Test_MCC') is not None else "N/A"
            test_acc = f"{row.get('Test_Accuracy', 0.0):.4f}" if row.get('Test_Accuracy') is not None else "N/A"
            test_auc = f"{row.get('Test_AUC', 0.0):.4f}" if row.get('Test_AUC') is not None else "N/A"
            
            print(f"{i+1:<4} {row['Model'][:24]:<25} {test_mcc:<9} "
                  f"{test_acc:<9} {test_auc:<9} {cv_indicator:<8}")
    else:
        print("\nCLASSIFICATION RESULTS: FAILED")
        print("No classification models were trained successfully.")
    
    if not clustering_results.empty:
        print(f"\nCLUSTERING RESULTS:")
        print("-" * 80)
        print(f"{'Rank':<4} {'Model':<20} {'Silhouette':<12} {'Clusters':<10}")
        print("-" * 80)
        
        for i, (_, row) in enumerate(clustering_results.iterrows()):
            print(f"{i+1:<4} {row['Model']:<20} {row['Silhouette_Score']:<12.4f} {row['Num_Clusters']:<10}")
    else:
        print(f"\nCLUSTERING RESULTS: FAILED OR NO RESULTS")
    
    print(f"\nSUMMARY STATISTICS:")
    if not classification_results.empty:
        print(f"• Total Classification Models: {len(classification_results)}")
        if 'Test_MCC' in classification_results.columns:
            print(f"• Best Classification MCC: {classification_results['Test_MCC'].max():.4f}")
            print(f"• Mean Classification MCC: {classification_results['Test_MCC'].mean():.4f}")
            print(f"• Models with MCC > 0.5: {sum(classification_results['Test_MCC'] > 0.5)}/{len(classification_results)}")
        else:
            print("• MCC metrics not available")
    else:
        print("• Classification analysis failed")
    
    if not clustering_results.empty:
        print(f"• Best Clustering Silhouette: {clustering_results['Silhouette_Score'].max():.4f}")
        print(f"• Mean Clustering Silhouette: {clustering_results['Silhouette_Score'].mean():.4f}")
    else:
        print("• Clustering analysis failed or no results")

def main():
    print("="*80)
    print("PYCARET COMPREHENSIVE ANALYSIS - CLASSIFICATION AND CLUSTERING")
    print("Efficient one-line model comparison with automatic SMOTE and feature engineering")
    print("="*80)
    
    # Find ClinVar files
    files = find_clinvar_files()
    
    if not files:
        print("No ClinVar files found!")
        return
    
    # Load data
    max_files = 2  # Limit for reasonable PyCaret runtime
    merged_df = load_and_merge_files(files, max_files=max_files)
    
    if merged_df is None:
        print("Failed to load data!")
        return
    
    # Prepare data
    sample_size = 15000  # Reasonable size for PyCaret
    df_clean, class_names = prepare_data_for_pycaret(merged_df, sample_size=sample_size)
    
    if df_clean is None:
        print("Failed to prepare data!")
        return
    
    # Run classification analysis
    classification_models, comparison_results, detailed_classification_results = run_pycaret_classification(df_clean, class_names)
    
    # Run clustering analysis
    clustering_models, clustering_comparison, cluster_models, detailed_clustering_results = run_pycaret_clustering(df_clean)
    
    # Create comprehensive visualizations
    create_comprehensive_visualizations(detailed_classification_results, detailed_clustering_results, class_names)
    
    # Print results
    print_comprehensive_results(detailed_classification_results, detailed_clustering_results)
    
    # Save results
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    
    if not detailed_classification_results.empty:
        detailed_classification_results.to_csv(output_dir / "pycaret_classification_results.csv", index=False)
        print(f"  • results/pycaret_classification_results.csv")
    
    if not comparison_results.empty:
        comparison_results.to_csv(output_dir / "pycaret_comparison_results.csv")
        print(f"  • results/pycaret_comparison_results.csv")
    
    if not detailed_clustering_results.empty:
        detailed_clustering_results.to_csv(output_dir / "pycaret_clustering_results.csv", index=False)
        print(f"  • results/pycaret_clustering_results.csv")
        
    if not clustering_comparison.empty:
        clustering_comparison.to_csv(output_dir / "pycaret_clustering_comparison.csv")
    
    print(f"\nResults saved to results/ directory")
    
    print("\n" + "="*80)
    print("PYCARET ANALYSIS COMPLETE")
    print("="*80)
    print(f"Files processed: {max_files}")
    print(f"Samples analyzed: {len(df_clean):,}")
    
    if not detailed_classification_results.empty:
        print(f"Classification models: {len(classification_models)}")
        print(f"Best classification model: {detailed_classification_results.iloc[0]['Model']}")
        if 'Test_MCC' in detailed_classification_results.columns:
            print(f"Best MCC score: {detailed_classification_results.iloc[0]['Test_MCC']:.4f}")
    else:
        print("Classification analysis failed")
    
    if not detailed_clustering_results.empty:
        print(f"Clustering models: {len(cluster_models)}")
        print(f"Best clustering model: {detailed_clustering_results.iloc[0]['Model']}")
        print(f"Best silhouette score: {detailed_clustering_results.iloc[0]['Silhouette_Score']:.4f}")
    else:
        print("Clustering analysis failed")

if __name__ == "__main__":
    main()