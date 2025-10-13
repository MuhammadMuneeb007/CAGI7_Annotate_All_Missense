#!/usr/bin/env python3
"""
Fold-based ML Pipeline for ClinVar Pathogenicity Prediction
Reads data from Clinvar_Dataset2/Fold_* structure
Trains XGBoost, Random Forest, and AutoML models on each fold
Evaluates with MCC, confusion matrices, and saves results
"""

import pandas as pd
import numpy as np
import glob
from pathlib import Path
import joblib
import warnings
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (matthews_corrcoef, confusion_matrix, classification_report, 
                           roc_curve, auc, roc_auc_score, precision_recall_curve, 
                           accuracy_score, precision_score, recall_score, f1_score)
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

# AutoML libraries (install if needed)
try:
    from flaml import AutoML
    FLAML_AVAILABLE = True
except ImportError:
    print("FLAML not available. Install with: pip install flaml")
    FLAML_AVAILABLE = False

warnings.filterwarnings('ignore')

class ClinVarFoldMLPipeline:
    def __init__(self, data_dir="Clinvar_Dataset2", results_dir="Clinvar_Dataset2/MachineLearning"):
        self.data_dir = Path(data_dir)
        self.results_dir = Path(results_dir)
        
        # Create results directories - for XGBoost, RandomForest, and FLAML_AutoML
        self.results_dir.mkdir(exist_ok=True, parents=True)
        (self.results_dir / "XGBoost").mkdir(exist_ok=True)
        (self.results_dir / "RandomForest").mkdir(exist_ok=True)
        (self.results_dir / "FLAML_AutoML").mkdir(exist_ok=True)
        (self.results_dir / "plots").mkdir(exist_ok=True)
        (self.results_dir / "summary").mkdir(exist_ok=True)
        
        self.fold_results = {}
        self.all_results = []
        
        # Set up plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
    def find_folds(self):
        """Find all available fold directories"""
        fold_dirs = [d for d in self.data_dir.iterdir() 
                    if d.is_dir() and d.name.startswith('Fold_')]
        fold_dirs.sort(key=lambda x: int(x.name.split('_')[1]))
        
        print(f"Found {len(fold_dirs)} folds: {[d.name for d in fold_dirs]}")
        return fold_dirs
    
    def load_fold_data(self, fold_dir):
        """Load data for a specific fold"""
        fold_name = fold_dir.name
        print(f"\nLoading data for {fold_name}...")
        
        # Define file paths
        files = {
            'X_train': fold_dir / "X_train_processed.csv",
            'X_test': fold_dir / "X_test_processed.csv", 
            'y_train': fold_dir / "Y_train.csv",
            'y_test': fold_dir / "Y_test.csv"
        }
        
        # Check if all files exist
        missing_files = [name for name, path in files.items() if not path.exists()]
        if missing_files:
            print(f"  Missing files in {fold_name}: {missing_files}")
            return None
        
        try:
            # Load data
            X_train = pd.read_csv(files['X_train'])  # Limiting rows for testing purposes
            X_test = pd.read_csv(files['X_test'])
            y_train = pd.read_csv(files['y_train'])
            y_test = pd.read_csv(files['y_test'])

            # Extract target values and convert to binary
            if 'label' in y_train.columns:
                y_train_raw = y_train['label'].values
                y_test_raw = y_test['label'].values
            elif 'target' in y_train.columns:
                y_train_raw = y_train['target'].values
                y_test_raw = y_test['target'].values
            elif 'ML_Class' in y_train.columns:
                y_train_raw = y_train['ML_Class'].values
                y_test_raw = y_test['ML_Class'].values
            elif 'class' in y_train.columns:
                y_train_raw = y_train['class'].values
                y_test_raw = y_test['class'].values
            else:
                # Take the first column if structure is unclear
                y_train_raw = y_train.iloc[:, 0].values
                y_test_raw = y_test.iloc[:, 0].values
            
            # Convert string labels to binary (1 for Pathogenic, 0 for Benign)
            def convert_to_binary(labels):
                binary_labels = []
                for label in labels:
                    if pd.isna(label):
                        print(f"    Warning: Missing label found, treating as benign (0)")
                        binary_labels.append(0)  # Default to benign for missing
                    elif str(label).lower() in ['pathogenic', 'pathogenic/likely_pathogenic', 'likely_pathogenic', 
                                               'disease_causing', 'deleterious', 'damaging', 'high', 
                                               'positive', 'pos', 'p', 'd', 'h', '1', 1]:
                        binary_labels.append(1)  # Pathogenic
                    elif str(label).lower() in ['benign', 'benign/likely_benign', 'likely_benign', 
                                               'tolerated', 'neutral', 'polymorphism', 'low', 
                                               'negative', 'neg', 'b', 't', 'n', 'l', '0', 0]:
                        binary_labels.append(0)  # Benign
                    else:
                        # Try to convert to int/float directly
                        try:
                            num_val = float(label)
                            if num_val >= 0.5:
                                binary_labels.append(1)  # Pathogenic
                            else:
                                binary_labels.append(0)  # Benign
                        except:
                            print(f"    Warning: Unknown label '{label}', treating as benign (0)")
                            binary_labels.append(0)  # Default to benign
                return np.array(binary_labels, dtype=int)
            
            y_train = convert_to_binary(y_train_raw)
            y_test = convert_to_binary(y_test_raw)
            
            # Ensure same features in train and test
            common_features = list(set(X_train.columns) & set(X_test.columns))
            X_train = X_train[common_features]
            X_test = X_test[common_features]
            
            # Handle missing values
            X_train = X_train.fillna(0)
            X_test = X_test.fillna(0)
            
            # Calculate statistics
            train_pathogenic = np.sum(y_train)
            train_benign = len(y_train) - train_pathogenic
            test_pathogenic = np.sum(y_test)
            test_benign = len(y_test) - test_pathogenic
            
            print(f"  Loaded {fold_name}:")
            print(f"    X_train: {X_train.shape}")
            print(f"    X_test: {X_test.shape}")
            print(f"    y_train: {len(y_train)} (Pathogenic: {train_pathogenic}, Benign: {train_benign})")
            print(f"    y_test: {len(y_test)} (Pathogenic: {test_pathogenic}, Benign: {test_benign})")
            print(f"    Features: {len(common_features)}")
            
            # Check for class imbalance
            if train_pathogenic == 0 or train_benign == 0:
                print(f"    Warning: {fold_name} has only one class in training set!")
                return None
            
            return {
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test,
                'feature_names': common_features,
                'fold_name': fold_name
            }
            
        except Exception as e:
            print(f"  Error loading {fold_name}: {e}")
            import traceback
            print(f"  Traceback: {traceback.format_exc()}")
            return None
    
    def setup_models(self, y_train):
        """Initialize models with appropriate class weights"""
        # Calculate class weights
        classes = np.unique(y_train)
        class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
        class_weight_dict = dict(zip(classes, class_weights))
        scale_pos_weight = class_weights[1] / class_weights[0] if len(class_weights) > 1 else 1.0
        
        print(f"    Class distribution - Benign (0): {np.sum(y_train == 0)}, Pathogenic (1): {np.sum(y_train == 1)}")
        print(f"    Class weights: {class_weight_dict}")
        
        models = {}
        
        # XGBoost with enhanced parameters
        models['XGBoost'] = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            n_jobs=-1,
            eval_metric='logloss',
            min_child_weight=1,
            gamma=0,
            reg_alpha=0,
            reg_lambda=1
        )
        
        # Random Forest with enhanced parameters
        models['RandomForest'] = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            class_weight='balanced',
            random_state=42,
            n_jobs=-1,
            bootstrap=True,
            oob_score=True
        )
        
        # Enhanced FLAML AutoML if available
        if FLAML_AVAILABLE:
            models['FLAML_AutoML'] = AutoML(
                task='classification',
                metric='roc_auc',
                time_budget=600,  # 1 minute per fold for better exploration
                estimator_list=['xgboost', 'rf', 'lgbm', 'extra_tree'],
                eval_method='cv',  # Cross-validation for better model selection
                split_ratio=0.8,
                n_splits=3,
                log_file_name='flaml_log.txt',
                seed=42,
                verbose=1,
                early_stop=True,
                retrain_full=True,  # Retrain on full dataset after finding best config
                sample_weight_flag=True,  # Use sample weights for imbalanced data
                mem_thres=4 * 1024**3,  # 4GB memory limit
                pred_time_limit=1e-3,  # Fast prediction time
            )
        
        return models
    
    def calculate_all_metrics(self, y_true, y_pred, y_proba=None):
        """Calculate comprehensive metrics"""
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, average='binary', zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, average='binary', zero_division=0)
        metrics['f1'] = f1_score(y_true, y_pred, average='binary', zero_division=0)
        metrics['mcc'] = matthews_corrcoef(y_true, y_pred)
        
        # Confusion matrix elements
        cm = confusion_matrix(y_true, y_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            metrics['tn'] = int(tn)
            metrics['fp'] = int(fp)
            metrics['fn'] = int(fn)
            metrics['tp'] = int(tp)
            
            # Additional metrics
            metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
            metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        # AUC if probabilities available
        if y_proba is not None:
            try:
                metrics['auc_roc'] = roc_auc_score(y_true, y_proba)
            except:
                metrics['auc_roc'] = 0.0
        
        return metrics
    
    def train_and_evaluate_fold(self, fold_data):
        """Train all models on a single fold and evaluate"""
        fold_name = fold_data['fold_name']
        print(f"\nTraining models for {fold_name}...")
        
        X_train = fold_data['X_train']
        X_test = fold_data['X_test']
        y_train = fold_data['y_train']
        y_test = fold_data['y_test']
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Setup models
        models = self.setup_models(y_train)
        
        fold_results = {
            'fold_name': fold_name,
            'feature_names': fold_data['feature_names'],  # Add feature names to fold results
            'data_info': {
                'train_samples': len(y_train),
                'test_samples': len(y_test),
                'features': len(fold_data['feature_names']),
                'train_positive_rate': np.mean(y_train),
                'test_positive_rate': np.mean(y_test)
            },
            'models': {}
        }
        
        # Train each model
        for model_name, model in models.items():
            print(f"  Training {model_name}...")
            
            try:
                # Train model
                if model_name == 'FLAML_AutoML':
                    model.fit(X_train_scaled, y_train)
                else:
                    model.fit(X_train_scaled, y_train)
                
                # Predictions
                y_train_pred = model.predict(X_train_scaled)
                y_test_pred = model.predict(X_test_scaled)
                
                # Probabilities
                if hasattr(model, 'predict_proba'):
                    y_train_proba = model.predict_proba(X_train_scaled)[:, 1]
                    y_test_proba = model.predict_proba(X_test_scaled)[:, 1]
                else:
                    y_train_proba = None
                    y_test_proba = None
                
                # Calculate metrics
                train_metrics = self.calculate_all_metrics(y_train, y_train_pred, y_train_proba)
                test_metrics = self.calculate_all_metrics(y_test, y_test_pred, y_test_proba)
                
                # Store results
                model_results = {
                    'model': model,
                    'scaler': scaler,
                    'train_metrics': train_metrics,
                    'test_metrics': test_metrics,
                    'predictions': {
                        'y_train_pred': y_train_pred,
                        'y_test_pred': y_test_pred,
                        'y_train_proba': y_train_proba,
                        'y_test_proba': y_test_proba
                    }
                }
                
                fold_results['models'][model_name] = model_results
                
                print(f"    {model_name} - Train MCC: {train_metrics['mcc']:.4f}, Test MCC: {test_metrics['mcc']:.4f}")
                
            except Exception as e:
                print(f"    Error training {model_name}: {e}")
                continue
        
        return fold_results
    
    def save_fold_results(self, fold_results):
        """Save results for a single fold"""
        fold_name = fold_results['fold_name']
        
        # Save each model's results
        for model_name, model_results in fold_results['models'].items():
            model_dir = self.results_dir / model_name
            fold_dir = model_dir / fold_name
            fold_dir.mkdir(exist_ok=True, parents=True)
            
            # Save model and scaler
            joblib.dump(model_results['model'], fold_dir / 'model.joblib')
            joblib.dump(model_results['scaler'], fold_dir / 'scaler.joblib')
            
            # Save metrics
            metrics_data = {
                'fold_name': fold_name,
                'train_metrics': model_results['train_metrics'],
                'test_metrics': model_results['test_metrics'],
                'data_info': fold_results['data_info']
            }
            
            with open(fold_dir / 'metrics.json', 'w') as f:
                json.dump(metrics_data, f, indent=2)
            
            # Fix predictions DataFrame creation - ensure all arrays have same length
            predictions_dict = {
                'y_train_pred': model_results['predictions']['y_train_pred'],
                'y_test_pred': model_results['predictions']['y_test_pred']
            }
            
            # Only add y_true values if they exist and have correct length
            if 'y_train' in fold_results and len(fold_results['y_train']) == len(predictions_dict['y_train_pred']):
                predictions_dict['y_train_true'] = fold_results['y_train']
            
            if 'y_test' in fold_results and len(fold_results['y_test']) == len(predictions_dict['y_test_pred']):
                predictions_dict['y_test_true'] = fold_results['y_test']
            
            # Add probabilities if available and same length
            if (model_results['predictions']['y_train_proba'] is not None and 
                len(model_results['predictions']['y_train_proba']) == len(predictions_dict['y_train_pred'])):
                predictions_dict['y_train_proba'] = model_results['predictions']['y_train_proba']
                
            if (model_results['predictions']['y_test_proba'] is not None and 
                len(model_results['predictions']['y_test_proba']) == len(predictions_dict['y_test_pred'])):
                predictions_dict['y_test_proba'] = model_results['predictions']['y_test_proba']
            
            # Create DataFrame with consistent lengths
            max_train_len = len(predictions_dict['y_train_pred'])
            max_test_len = len(predictions_dict['y_test_pred'])
            
            # Create separate DataFrames for train and test to avoid length mismatch
            train_df_dict = {}
            test_df_dict = {}
            
            for key, values in predictions_dict.items():
                if 'train' in key:
                    train_df_dict[key] = values
                elif 'test' in key:
                    test_df_dict[key] = values
            
            # Save train predictions
            if train_df_dict:
                train_predictions_df = pd.DataFrame(train_df_dict)
                train_predictions_df.to_csv(fold_dir / 'train_predictions.csv', index=False)
            
            # Save test predictions  
            if test_df_dict:
                test_predictions_df = pd.DataFrame(test_df_dict)
                test_predictions_df.to_csv(fold_dir / 'test_predictions.csv', index=False)

    def plot_confusion_matrix(self, y_true, y_pred, title, save_path):
        """Plot and save confusion matrix"""
        try:
            print(f"    Generating confusion matrix: {save_path}")
            plt.figure(figsize=(8, 6))
            cm = confusion_matrix(y_true, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=['Benign (0)', 'Pathogenic (1)'],
                       yticklabels=['Benign (0)', 'Pathogenic (1)'])
            plt.title(title)
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"    Saved confusion matrix: {save_path}")
        except Exception as e:
            print(f"    Error generating confusion matrix {save_path}: {e}")
    
    def plot_roc_curve(self, y_true, y_proba, title, save_path):
        """Plot and save ROC curve"""
        try:
            print(f"    Generating ROC curve: {save_path}")
            plt.figure(figsize=(8, 6))
            fpr, tpr, _ = roc_curve(y_true, y_proba)
            auc_score = auc(fpr, tpr)
            
            plt.plot(fpr, tpr, color='darkorange', lw=2, 
                    label=f'ROC curve (AUC = {auc_score:.3f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
                    label='Random classifier')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(title)
            plt.legend(loc="lower right")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"    Saved ROC curve: {save_path}")
        except Exception as e:
            print(f"    Error generating ROC curve {save_path}: {e}")

    def plot_feature_importance(self, model, feature_names, model_name, title, save_path, top_n=20):
        """Plot and save feature importance"""
        try:
            print(f"    Generating feature importance: {save_path}")
            
            # Get feature importance based on model type
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
            elif hasattr(model, 'coef_'):
                importances = np.abs(model.coef_[0]) if len(model.coef_.shape) > 1 else np.abs(model.coef_)
            else:
                print(f"    {model_name} does not support feature importance")
                return
            
            # Create feature importance DataFrame
            feature_importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            # Take top N features
            top_features = feature_importance_df.head(top_n)
            
            # Create plot
            plt.figure(figsize=(12, 8))
            sns.barplot(data=top_features, x='importance', y='feature', palette='viridis')
            plt.title(f'{title}\nTop {min(top_n, len(top_features))} Most Important Features')
            plt.xlabel('Feature Importance')
            plt.ylabel('Features')
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            # Save feature importance as CSV
            csv_path = save_path.with_suffix('.csv')
            feature_importance_df.to_csv(csv_path, index=False)
            
            print(f"    Saved feature importance plot: {save_path}")
            print(f"    Saved feature importance CSV: {csv_path}")
            
        except Exception as e:
            print(f"    Error generating feature importance {save_path}: {e}")
            import traceback
            print(f"    Traceback: {traceback.format_exc()}")

    def generate_fold_plots(self, fold_results):
        """Generate plots for a single fold"""
        fold_name = fold_results['fold_name']
        print(f"  Generating plots for {fold_name}...")
        
        plots_dir = self.results_dir / "plots" / fold_name
        plots_dir.mkdir(exist_ok=True, parents=True)
        print(f"  Plots directory: {plots_dir}")
        
        # Get actual y values
        y_train = fold_results.get('y_train', [])
        y_test = fold_results.get('y_test', [])
        
        print(f"  Available data - y_train: {len(y_train)}, y_test: {len(y_test)}")
        print(f"  Models available: {list(fold_results['models'].keys())}")
        
        # Generate confusion matrices for each model
        for model_name, model_results in fold_results['models'].items():
            print(f"    Processing {model_name}...")
            model_plots_dir = plots_dir / model_name
            model_plots_dir.mkdir(exist_ok=True, parents=True)
            print(f"    Model plots directory: {model_plots_dir}")
            
            try:
                # Training confusion matrix
                if len(y_train) > 0:
                    y_train_pred = model_results['predictions']['y_train_pred']
                    print(f"    Train predictions shape: {len(y_train_pred)}")
                    train_cm_path = model_plots_dir / 'confusion_matrix_train.png'
                    self.plot_confusion_matrix(
                        y_train, y_train_pred,
                        f'{model_name} - Training Set Confusion Matrix\n{fold_name}',
                        train_cm_path
                    )
                
                # Test confusion matrix
                if len(y_test) > 0:
                    y_test_pred = model_results['predictions']['y_test_pred']
                    print(f"    Test predictions shape: {len(y_test_pred)}")
                    test_cm_path = model_plots_dir / 'confusion_matrix_test.png'
                    self.plot_confusion_matrix(
                        y_test, y_test_pred,
                        f'{model_name} - Test Set Confusion Matrix\n{fold_name}',
                        test_cm_path
                    )
                
                # ROC curves if probabilities are available
                if (model_results['predictions']['y_test_proba'] is not None and 
                    len(y_test) > 0):
                    y_test_proba = model_results['predictions']['y_test_proba']
                    print(f"    Test probabilities shape: {len(y_test_proba)}")
                    roc_path = model_plots_dir / 'roc_curve.png'
                    self.plot_roc_curve(
                        y_test, y_test_proba,
                        f'{model_name} - ROC Curve\n{fold_name}',
                        roc_path
                    )
                else:
                    print(f"    No probabilities available for {model_name}")
                
                # Feature importance plot
                model = model_results['model']
                if hasattr(model, 'feature_importances_') or hasattr(model, 'coef_'):
                    # Get feature names from fold_results or create generic ones
                    if 'feature_names' in fold_results:
                        feature_names = fold_results['feature_names']
                    else:
                        # Try to get from the first fold's data structure
                        feature_names = [f'feature_{i}' for i in range(len(model.feature_importances_))] if hasattr(model, 'feature_importances_') else []
                    
                    if feature_names:
                        importance_path = model_plots_dir / 'feature_importance.png'
                        self.plot_feature_importance(
                            model, feature_names, model_name,
                            f'{model_name} - Feature Importance\n{fold_name}',
                            importance_path
                        )
                    else:
                        print(f"    No feature names available for {model_name}")
                else:
                    print(f"    {model_name} does not support feature importance")
                    
            except Exception as e:
                print(f"    Error processing {model_name} plots: {e}")
                import traceback
                print(f"    Traceback: {traceback.format_exc()}")
        
        print(f"  Completed plots for {fold_name}")

    def compile_summary_results(self):
        """Compile results across all folds"""
        print("\nCompiling summary results...")
        
        summary_results = {
            'overall_stats': {
                'total_folds': len(self.all_results),
                'models_evaluated': [],
                'avg_metrics_by_model': {}
            },
            'fold_details': self.all_results
        }
        
        if not self.all_results:
            print("No results to summarize")
            return summary_results
        
        # Get model names from first fold
        model_names = list(self.all_results[0]['models'].keys())
        summary_results['overall_stats']['models_evaluated'] = model_names
        
        # Calculate average metrics across folds for each model
        for model_name in model_names:
            model_fold_results = []
            
            for fold_result in self.all_results:
                if model_name in fold_result['models']:
                    model_fold_results.append(fold_result['models'][model_name])
            
            if model_fold_results:
                # Calculate averages for test metrics
                test_metrics_keys = model_fold_results[0]['test_metrics'].keys()
                avg_test_metrics = {}
                std_test_metrics = {}
                
                for metric in test_metrics_keys:
                    values = [r['test_metrics'][metric] for r in model_fold_results 
                             if metric in r['test_metrics']]
                    if values:
                        avg_test_metrics[metric] = np.mean(values)
                        std_test_metrics[metric] = np.std(values)
                
                summary_results['overall_stats']['avg_metrics_by_model'][model_name] = {
                    'test_metrics_mean': avg_test_metrics,
                    'test_metrics_std': std_test_metrics,
                    'num_folds': len(model_fold_results)
                }
        
        return summary_results
    
    def save_summary_results(self, summary_results):
        """Save comprehensive summary results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create a clean, serializable version of the results
        clean_summary = {
            'overall_stats': summary_results['overall_stats'].copy(),
            'fold_details': []
        }
        
        # Clean fold details - remove non-serializable objects
        for fold_result in summary_results['fold_details']:
            clean_fold = {
                'fold_name': fold_result['fold_name'],
                'data_info': fold_result['data_info'].copy(),
                'models': {}
            }
            
            # Extract only serializable model results
            for model_name, model_results in fold_result['models'].items():
                clean_fold['models'][model_name] = {
                    'train_metrics': model_results['train_metrics'].copy(),
                    'test_metrics': model_results['test_metrics'].copy()
                    # Exclude 'model', 'scaler', and 'predictions' as they're not JSON serializable
                }
            
            clean_summary['fold_details'].append(clean_fold)
        
        # Save JSON summary
        summary_path = self.results_dir / "summary" / f"summary_results_{timestamp}.json"
        with open(summary_path, 'w') as f:
            # Convert numpy types for JSON serialization
            def convert_types(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif hasattr(obj, '__dict__'):
                    # Skip complex objects that can't be serialized
                    return str(type(obj).__name__)
                return obj
            
            json.dump(clean_summary, f, indent=2, default=convert_types)
        
        # Create readable report
        report_path = self.results_dir / "summary" / f"summary_report_{timestamp}.txt"
        with open(report_path, 'w') as f:
            f.write("CLINVAR FOLD-BASED ML PIPELINE RESULTS\n")
            f.write("=" * 60 + "\n")
            f.write(f"Generated: {datetime.now()}\n")
            f.write(f"Total Folds: {summary_results['overall_stats']['total_folds']}\n")
            f.write(f"Models: {', '.join(summary_results['overall_stats']['models_evaluated'])}\n")
            f.write("Class Labels: 0 = Benign, 1 = Pathogenic\n\n")
            
            f.write("AVERAGE TEST METRICS ACROSS FOLDS:\n")
            f.write("-" * 40 + "\n")
            
            for model_name, model_stats in summary_results['overall_stats']['avg_metrics_by_model'].items():
                f.write(f"\n{model_name} ({model_stats['num_folds']} folds):\n")
                
                # Group metrics for better readability
                main_metrics = ['mcc', 'auc_roc', 'accuracy', 'f1', 'precision', 'recall']
                other_metrics = [m for m in model_stats['test_metrics_mean'].keys() if m not in main_metrics]
                
                f.write("  Main Performance Metrics:\n")
                for metric in main_metrics:
                    if metric in model_stats['test_metrics_mean']:
                        mean_val = model_stats['test_metrics_mean'][metric]
                        std_val = model_stats['test_metrics_std'].get(metric, 0)
                        f.write(f"    {metric:15}: {mean_val:.4f} (±{std_val:.4f})\n")
                
                if other_metrics:
                    f.write("  Additional Metrics:\n")
                    for metric in other_metrics:
                        mean_val = model_stats['test_metrics_mean'][metric]
                        std_val = model_stats['test_metrics_std'].get(metric, 0)
                        f.write(f"    {metric:15}: {mean_val:.4f} (±{std_val:.4f})\n")
            
            f.write("\nFOLD-BY-FOLD DETAILS:\n")
            f.write("-" * 25 + "\n")
            
            for fold_result in summary_results['fold_details']:
                f.write(f"\n{fold_result['fold_name']}:\n")
                f.write(f"  Train samples: {fold_result['data_info']['train_samples']}\n")
                f.write(f"  Test samples:  {fold_result['data_info']['test_samples']}\n")
                f.write(f"  Features:      {fold_result['data_info']['features']}\n")
                f.write(f"  Train Pathogenic Rate: {fold_result['data_info']['train_positive_rate']:.3f}\n")
                f.write(f"  Test Pathogenic Rate:  {fold_result['data_info']['test_positive_rate']:.3f}\n")
                
                for model_name, model_result in fold_result['models'].items():
                    test_mcc = model_result['test_metrics']['mcc']
                    test_auc = model_result['test_metrics'].get('auc_roc', 'N/A')
                    test_acc = model_result['test_metrics'].get('accuracy', 'N/A')
                    f.write(f"    {model_name:15}: MCC={test_mcc:.4f}, AUC={test_auc:.4f}, ACC={test_acc:.4f}\n")
        
        print(f"Summary saved: {summary_path}")
        print(f"Report saved: {report_path}")
        
        return summary_path, report_path
    
    def plot_summary_results(self, summary_results):
        """Create summary plots across all folds"""
        print("Creating summary plots...")
        
        if not summary_results['fold_details']:
            print("No data for summary plots")
            return
        
        # Prepare data for plotting
        model_names = summary_results['overall_stats']['models_evaluated']
        fold_names = [fold['fold_name'] for fold in summary_results['fold_details']]
        
        # Create MCC comparison plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: MCC by fold and model
        mcc_data = []
        for fold_result in summary_results['fold_details']:
            for model_name in model_names:
                if model_name in fold_result['models']:
                    mcc = fold_result['models'][model_name]['test_metrics']['mcc']
                    mcc_data.append({
                        'Fold': fold_result['fold_name'],
                        'Model': model_name,
                        'MCC': mcc
                    })
        
        if mcc_data:
            mcc_df = pd.DataFrame(mcc_data)
            
            # Box plot of MCC by model
            sns.boxplot(data=mcc_df, x='Model', y='MCC', ax=ax1)
            ax1.set_title('Test MCC Distribution by Model')
            ax1.tick_params(axis='x', rotation=45)
            ax1.grid(True, alpha=0.3)
            
            # Line plot of MCC by fold
            for model_name in model_names:
                model_data = mcc_df[mcc_df['Model'] == model_name]
                if not model_data.empty:
                    ax2.plot(model_data['Fold'], model_data['MCC'], 
                            marker='o', label=model_name)
            
            ax2.set_title('Test MCC by Fold')
            ax2.set_xlabel('Fold')
            ax2.set_ylabel('MCC')
            ax2.legend()
            ax2.tick_params(axis='x', rotation=45)
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plots_path = self.results_dir / "summary" / "summary_plots.png"
        plt.savefig(plots_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Summary plots saved: {plots_path}")
    
    def plot_summary_feature_importance(self, summary_results):
        """Create summary feature importance plots across all folds"""
        print("Creating summary feature importance plots...")
        
        if not summary_results['fold_details']:
            print("No data for summary feature importance plots")
            return
        
        model_names = summary_results['overall_stats']['models_evaluated']
        
        # Aggregate feature importance across folds for each model
        for model_name in model_names:
            print(f"  Processing {model_name} feature importance...")
            
            # Collect feature importance from all folds
            all_importances = {}
            fold_count = 0
            
            for fold_result in summary_results['fold_details']:
                if model_name in fold_result['models']:
                    # Try to get feature names and importances
                    feature_names = fold_result.get('feature_names', [])
                    model = fold_result['models'][model_name]['model']
                    
                    if (feature_names and 
                        (hasattr(model, 'feature_importances_') or hasattr(model, 'coef_'))):
                        
                        if hasattr(model, 'feature_importances_'):
                            importances = model.feature_importances_
                        else:
                            importances = np.abs(model.coef_[0]) if len(model.coef_.shape) > 1 else np.abs(model.coef_)
                        
                        # Accumulate importances
                        for feature, importance in zip(feature_names, importances):
                            if feature not in all_importances:
                                all_importances[feature] = []
                            all_importances[feature].append(importance)
                        
                        fold_count += 1
            
            # Create summary plot if we have data
            if all_importances and fold_count > 0:
                # Calculate mean and std for each feature
                feature_stats = {}
                for feature, importance_list in all_importances.items():
                    feature_stats[feature] = {
                        'mean': np.mean(importance_list),
                        'std': np.std(importance_list),
                        'count': len(importance_list)
                    }
                
                # Convert to DataFrame and sort
                importance_df = pd.DataFrame([
                    {
                        'feature': feature,
                        'mean_importance': stats['mean'],
                        'std_importance': stats['std'],
                        'fold_count': stats['count']
                    }
                    for feature, stats in feature_stats.items()
                ]).sort_values('mean_importance', ascending=False)
                
                # Plot top 20 features
                top_features = importance_df.head(20)
                
                plt.figure(figsize=(12, 8))
                plt.errorbar(top_features['mean_importance'], 
                           range(len(top_features)), 
                           xerr=top_features['std_importance'],
                           fmt='o', capsize=5, capthick=2, elinewidth=2)
                
                plt.yticks(range(len(top_features)), top_features['feature'])
                plt.xlabel('Mean Feature Importance')
                plt.ylabel('Features')
                plt.title(f'{model_name} - Mean Feature Importance Across Folds\n(Top 20 Features)')
                plt.gca().invert_yaxis()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                
                # Save plot
                importance_plot_path = self.results_dir / "summary" / f"feature_importance_{model_name}.png"
                plt.savefig(importance_plot_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                # Save CSV
                importance_csv_path = self.results_dir / "summary" / f"feature_importance_{model_name}.csv"
                importance_df.to_csv(importance_csv_path, index=False)
                
                print(f"  Saved {model_name} feature importance: {importance_plot_path}")
                print(f"  Saved {model_name} feature importance CSV: {importance_csv_path}")
            else:
                print(f"  No feature importance data available for {model_name}")

    def plot_averaged_confusion_matrices(self, summary_results):
        """Create averaged confusion matrices for all models across all folds"""
        print("Creating averaged confusion matrices for all models...")
        
        if not summary_results['fold_details']:
            print("No data for averaged confusion matrices")
            return
        
        model_names = summary_results['overall_stats']['models_evaluated']
        
        # Collect confusion matrix data for each model
        model_cm_data = {}
        
        for model_name in model_names:
            all_y_true = []
            all_y_pred = []
            
            for fold_result in summary_results['fold_details']:
                if model_name in fold_result['models']:
                    # Get test data
                    y_test = fold_result.get('y_test', [])
                    y_test_pred = fold_result['models'][model_name]['predictions']['y_test_pred']
                    
                    if len(y_test) > 0 and len(y_test_pred) > 0:
                        all_y_true.extend(y_test)
                        all_y_pred.extend(y_test_pred)
            
            if all_y_true and all_y_pred:
                model_cm_data[model_name] = {
                    'y_true': all_y_true,
                    'y_pred': all_y_pred
                }
        
        if not model_cm_data:
            print("No confusion matrix data available")
            return
        
        # Create subplots for all models
        n_models = len(model_cm_data)
        fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))
        if n_models == 1:
            axes = [axes]
        
        for idx, (model_name, data) in enumerate(model_cm_data.items()):
            # Calculate confusion matrix
            cm = confusion_matrix(data['y_true'], data['y_pred'])
            
            # Calculate percentages
            cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
            
            # Create heatmap
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                       xticklabels=['Benign (0)', 'Pathogenic (1)'],
                       yticklabels=['Benign (0)', 'Pathogenic (1)'])
            
            # Add percentage annotations
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    axes[idx].text(j + 0.5, i + 0.7, f'({cm_percent[i,j]:.1f}%)', 
                                 ha='center', va='center', fontsize=10, color='red')
            
            axes[idx].set_title(f'{model_name}\nAveraged Across All Folds')
            axes[idx].set_xlabel('Predicted')
            axes[idx].set_ylabel('Actual')
        
        plt.tight_layout()
        cm_plot_path = self.results_dir / "summary" / "averaged_confusion_matrices_all_models.png"
        plt.savefig(cm_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved averaged confusion matrices: {cm_plot_path}")
        
        # Also save detailed metrics
        metrics_summary = {}
        for model_name, data in model_cm_data.items():
            cm = confusion_matrix(data['y_true'], data['y_pred'])
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
                metrics_summary[model_name] = {
                    'total_samples': len(data['y_true']),
                    'true_negatives': int(tn),
                    'false_positives': int(fp), 
                    'false_negatives': int(fn),
                    'true_positives': int(tp),
                    'accuracy': (tp + tn) / (tp + tn + fp + fn),
                    'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
                    'recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
                    'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
                    'mcc': matthews_corrcoef(data['y_true'], data['y_pred'])
                }
        
        # Save metrics
        metrics_path = self.results_dir / "summary" / "averaged_confusion_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics_summary, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)
        
        print(f"Saved averaged confusion metrics: {metrics_path}")

    def plot_merged_feature_importance_all_models(self, summary_results):
        """Create merged feature importance plot for all models across all folds"""
        print("Creating merged feature importance for all models...")
        
        if not summary_results['fold_details']:
            print("No data for merged feature importance")
            return
        
        model_names = summary_results['overall_stats']['models_evaluated']
        
        # Collect feature importance data for all models
        all_model_importances = {}
        
        for model_name in model_names:
            print(f"  Processing {model_name} for merged plot...")
            
            # Collect feature importance from all folds
            model_importances = {}
            fold_count = 0
            
            for fold_result in summary_results['fold_details']:
                if model_name in fold_result['models']:
                    feature_names = fold_result.get('feature_names', [])
                    model = fold_result['models'][model_name]['model']
                    
                    if (feature_names and 
                        (hasattr(model, 'feature_importances_') or hasattr(model, 'coef_'))):
                        
                        if hasattr(model, 'feature_importances_'):
                            importances = model.feature_importances_
                        else:
                            importances = np.abs(model.coef_[0]) if len(model.coef_.shape) > 1 else np.abs(model.coef_)
                        
                        # Accumulate importances
                        for feature, importance in zip(feature_names, importances):
                            if feature not in model_importances:
                                model_importances[feature] = []
                            model_importances[feature].append(importance)
                        
                        fold_count += 1
            
            # Calculate mean importance for this model
            if model_importances and fold_count > 0:
                model_mean_importance = {}
                for feature, importance_list in model_importances.items():
                    model_mean_importance[feature] = np.mean(importance_list)
                
                all_model_importances[model_name] = model_mean_importance
        
        if not all_model_importances:
            print("No feature importance data available for any model")
            return
        
        # Find common features across all models
        all_features = set()
        for model_importances in all_model_importances.values():
            all_features.update(model_importances.keys())
        
        # Get top features based on maximum importance across any model
        feature_max_importance = {}
        for feature in all_features:
            max_imp = 0
            for model_importances in all_model_importances.values():
                if feature in model_importances:
                    max_imp = max(max_imp, model_importances[feature])
            feature_max_importance[feature] = max_imp
        
        # Sort features by maximum importance and take top N
        top_features = sorted(feature_max_importance.items(), 
                            key=lambda x: x[1], reverse=True)[:25]  # Top 25 features
        top_feature_names = [f[0] for f in top_features]
        
        # Create DataFrame for plotting
        plot_data = []
        for model_name, model_importances in all_model_importances.items():
            for feature in top_feature_names:
                importance = model_importances.get(feature, 0)
                plot_data.append({
                    'Feature': feature,
                    'Model': model_name,
                    'Importance': importance
                })
        
        if not plot_data:
            print("No plot data generated")
            return
        
        importance_df = pd.DataFrame(plot_data)
        
        # Create grouped bar plot
        plt.figure(figsize=(16, 10))
        
        # Create pivot table for easier plotting
        pivot_df = importance_df.pivot(index='Feature', columns='Model', values='Importance')
        pivot_df = pivot_df.fillna(0)  # Fill missing values with 0
        
        # Create the plot
        ax = pivot_df.plot(kind='barh', figsize=(16, 10), width=0.8)
        
        plt.title('Top 25 Feature Importance Comparison Across All Models\n(Averaged Across All Folds)', 
                 fontsize=16, fontweight='bold')
        plt.xlabel('Feature Importance', fontsize=14)
        plt.ylabel('Features', fontsize=14)
        plt.legend(title='Models', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        
        # Save plot
        merged_importance_path = self.results_dir / "summary" / "merged_feature_importance_all_models.png"
        plt.savefig(merged_importance_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create heatmap version
        plt.figure(figsize=(12, 16))
        sns.heatmap(pivot_df, annot=False, cmap='YlOrRd', cbar_kws={'label': 'Feature Importance'})
        plt.title('Feature Importance Heatmap - All Models\n(Averaged Across All Folds)', 
                 fontsize=16, fontweight='bold')
        plt.xlabel('Models', fontsize=14)
        plt.ylabel('Features', fontsize=14)
        plt.tight_layout()
        
        heatmap_path = self.results_dir / "summary" / "feature_importance_heatmap_all_models.png"
        plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save the data as CSV
        csv_path = self.results_dir / "summary" / "merged_feature_importance_all_models.csv"
        pivot_df.to_csv(csv_path)
        
        print(f"Saved merged feature importance bar plot: {merged_importance_path}")
        print(f"Saved feature importance heatmap: {heatmap_path}")
        print(f"Saved feature importance data: {csv_path}")

    def run_complete_pipeline(self):
        """Run the complete fold-based ML pipeline"""
        print("CLINVAR FOLD-BASED ML PIPELINE")
        print("=" * 60)
        print("Classification Task: 0 = Benign, 1 = Pathogenic")
        print("=" * 60)
        
        # Find all folds
        fold_dirs = self.find_folds()
        
        if not fold_dirs:
            print("No fold directories found!")
            return
        
        # Process each fold
        for fold_dir in fold_dirs:
            # Load fold data
            fold_data = self.load_fold_data(fold_dir)
            
            if fold_data is None:
                continue
            
            # Train and evaluate models
            fold_results = self.train_and_evaluate_fold(fold_data)
            
            # Store actual y values for potential plotting
            fold_results['y_train'] = fold_data['y_train']
            fold_results['y_test'] = fold_data['y_test']
            
            # Save fold results
            self.save_fold_results(fold_results)
            
            # Generate plots for this fold
            self.generate_fold_plots(fold_results)
            
            # Store for summary
            self.all_results.append(fold_results)
            
            print(f"Completed {fold_data['fold_name']}")
        
        # Compile and save summary results
        summary_results = self.compile_summary_results()
        summary_path, report_path = self.save_summary_results(summary_results)
        
        # Create summary plots
        self.plot_summary_results(summary_results)
        
        # Create summary feature importance plots (individual models)
        self.plot_summary_feature_importance(summary_results)
        
        # Create averaged confusion matrices for all models
        self.plot_averaged_confusion_matrices(summary_results)
        
        # Create merged feature importance plot for all models
        self.plot_merged_feature_importance_all_models(summary_results)
        
        # Print final summary
        print("\n" + "=" * 60)
        print("PIPELINE COMPLETE!")
        print("=" * 60)
        print("Classification: 0 = Benign, 1 = Pathogenic")
        print(f"Processed {len(self.all_results)} folds")
        print(f"Results saved in: {self.results_dir}")
        
        if summary_results['overall_stats']['avg_metrics_by_model']:
            print("\nBest models by average test MCC:")
            sorted_models = sorted(
                summary_results['overall_stats']['avg_metrics_by_model'].items(),
                key=lambda x: x[1]['test_metrics_mean'].get('mcc', 0),
                reverse=True
            )
            for i, (name, stats) in enumerate(sorted_models, 1):
                avg_mcc = stats['test_metrics_mean'].get('mcc', 0)
                std_mcc = stats['test_metrics_std'].get('mcc', 0)
                avg_auc = stats['test_metrics_mean'].get('auc_roc', 0)
                std_auc = stats['test_metrics_std'].get('auc_roc', 0)
                print(f"{i}. {name}:")
                print(f"   MCC: {avg_mcc:.4f} (±{std_mcc:.4f})")
                print(f"   AUC: {avg_auc:.4f} (±{std_auc:.4f})")
        
        return summary_results


def main():
    """Run the complete pipeline"""
    # Initialize pipeline
    pipeline = ClinVarFoldMLPipeline()
    
    # Run complete pipeline
    results = pipeline.run_complete_pipeline()
    
    return results


if __name__ == "__main__":
    results = main()