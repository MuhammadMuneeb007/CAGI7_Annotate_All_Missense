#!/usr/bin/env python3
"""
XGBoost-Only Fold-based ML Pipeline for ClinVar Pathogenicity Prediction
Reads data from Clinvar_Dataset2/Fold_* structure
Trains XGBoost models on each fold and saves models/features for inference
"""

import pandas as pd
import numpy as np
import glob
from pathlib import Path
import joblib
import pickle
import json
import warnings
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (matthews_corrcoef, confusion_matrix, classification_report, 
                           roc_curve, auc, roc_auc_score, precision_recall_curve, 
                           accuracy_score, precision_score, recall_score, f1_score)
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import StandardScaler, LabelEncoder
import xgboost as xgb
import sys

warnings.filterwarnings('ignore')

class XGBoostClinVarPipeline:
    def __init__(self, data_dir="Clinvar_Dataset2", results_dir="Clinvar_Dataset2/MachineLearning"):
        self.data_dir = Path(data_dir)
        self.results_dir = Path(results_dir)
        self.features_dir = Path("Clinvar_Dataset2/Features")  # New directory for saved models
        
        # Create results directories
        self.results_dir.mkdir(exist_ok=True, parents=True)
        self.features_dir.mkdir(exist_ok=True, parents=True)
        (self.results_dir / "XGBoost").mkdir(exist_ok=True)
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
            X_train = pd.read_csv(files['X_train'])
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
    
    def setup_xgboost_model(self, y_train):
        """Initialize XGBoost model with appropriate class weights"""
        # Calculate class weights
        classes = np.unique(y_train)
        class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
        class_weight_dict = dict(zip(classes, class_weights))
        scale_pos_weight = class_weights[1] / class_weights[0] if len(class_weights) > 1 else 1.0
        
        print(f"    Class distribution - Benign (0): {np.sum(y_train == 0)}, Pathogenic (1): {np.sum(y_train == 1)}")
        print(f"    Class weights: {class_weight_dict}")
        print(f"    Scale pos weight: {scale_pos_weight:.3f}")
        
        # XGBoost with enhanced parameters
        model = xgb.XGBClassifier(
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
        
        return model
    
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
        """Train XGBoost model on a single fold and evaluate"""
        fold_name = fold_data['fold_name']
        print(f"\nTraining XGBoost for {fold_name}...")
        
        X_train = fold_data['X_train']
        X_test = fold_data['X_test']
        y_train = fold_data['y_train']
        y_test = fold_data['y_test']
        feature_names = fold_data['feature_names']
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Setup XGBoost model
        model = self.setup_xgboost_model(y_train)
        
        fold_results = {
            'fold_name': fold_name,
            'feature_names': feature_names,
            'data_info': {
                'train_samples': len(y_train),
                'test_samples': len(y_test),
                'features': len(feature_names),
                'train_positive_rate': np.mean(y_train),
                'test_positive_rate': np.mean(y_test)
            }
        }
        
        try:
            # Train model
            print(f"  Training XGBoost...")
            model.fit(X_train_scaled, y_train)
            
            # Predictions
            y_train_pred = model.predict(X_train_scaled)
            y_test_pred = model.predict(X_test_scaled)
            
            # Probabilities
            y_train_proba = model.predict_proba(X_train_scaled)[:, 1]
            y_test_proba = model.predict_proba(X_test_scaled)[:, 1]
            
            # Calculate metrics
            train_metrics = self.calculate_all_metrics(y_train, y_train_pred, y_train_proba)
            test_metrics = self.calculate_all_metrics(y_test, y_test_pred, y_test_proba)
            
            # Store results
            fold_results['model'] = model
            fold_results['scaler'] = scaler
            fold_results['train_metrics'] = train_metrics
            fold_results['test_metrics'] = test_metrics
            fold_results['predictions'] = {
                'y_train_pred': y_train_pred,
                'y_test_pred': y_test_pred,
                'y_train_proba': y_train_proba,
                'y_test_proba': y_test_proba
            }
            fold_results['y_train'] = y_train
            fold_results['y_test'] = y_test
            
            print(f"  XGBoost - Train MCC: {train_metrics['mcc']:.4f}, Test MCC: {test_metrics['mcc']:.4f}")
            print(f"  XGBoost - Train AUC: {train_metrics['auc_roc']:.4f}, Test AUC: {test_metrics['auc_roc']:.4f}")
            
        except Exception as e:
            print(f"  Error training XGBoost: {e}")
            return None
        
        return fold_results
    
    def save_model_for_inference(self, fold_results, fold_idx=None):
        """Save model and features for inference use"""
        fold_name = fold_results['fold_name']
        
        # Create fold-specific directory in Features folder
        fold_features_dir = self.features_dir / fold_name
        fold_features_dir.mkdir(exist_ok=True, parents=True)
        
        print(f"  Saving model and features for {fold_name} to {fold_features_dir}")
        
        model = fold_results['model']
        scaler = fold_results['scaler']
        feature_names = fold_results['feature_names']
        
        # Create label encoder for consistency with inference script
        label_encoder = LabelEncoder()
        label_encoder.fit(['Benign', 'Pathogenic'])  # Consistent class names
        
        # Save the trained XGBoost model
        model_path = fold_features_dir / "xgboost_clinvar_model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"    ✓ Model saved to: {model_path}")
        
        # Save scaler
        scaler_path = fold_features_dir / "scaler.pkl"
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        print(f"    ✓ Scaler saved to: {scaler_path}")
        
        # Save feature names as JSON
        features_json_path = fold_features_dir / "feature_names.json"
        with open(features_json_path, 'w') as f:
            json.dump(feature_names, f, indent=2)
        print(f"    ✓ Feature names (JSON) saved to: {features_json_path}")
        
        # Save label encoder
        label_encoder_path = fold_features_dir / "label_encoder.pkl"
        with open(label_encoder_path, 'wb') as f:
            pickle.dump(label_encoder, f)
        print(f"    ✓ Label encoder saved to: {label_encoder_path}")
        
        # Save comprehensive metadata
        metadata = {
            'model_info': {
                'model_type': 'XGBoostClassifier',
                'n_estimators': model.n_estimators,
                'max_depth': model.max_depth,
                'learning_rate': model.learning_rate,
                'subsample': model.subsample,
                'colsample_bytree': model.colsample_bytree,
                'reg_alpha': model.reg_alpha,
                'reg_lambda': model.reg_lambda,
                'scale_pos_weight': model.scale_pos_weight,
                'random_state': model.random_state
            },
            'data_info': {
                'n_features': len(feature_names),
                'n_samples': fold_results['data_info']['train_samples'] + fold_results['data_info']['test_samples'],
                'n_classes': 2,
                'class_names': ['Benign', 'Pathogenic'],
                'feature_count': len(feature_names),
                'fold_name': fold_name
            },
            'preprocessing': {
                'scaling_method': 'StandardScaler',
                'missing_value_strategy': 'fill_with_zero',
                'feature_selection': None
            },
            'performance': {
                'train_metrics': fold_results['train_metrics'],
                'test_metrics': fold_results['test_metrics']
            },
            'training_info': {
                'timestamp': datetime.now().isoformat(),
                'script_version': 'XGBoost_Fold_Pipeline.py',
                'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
                'xgboost_version': xgb.__version__,
                'fold_index': fold_idx
            }
        }
        
        metadata_path = fold_features_dir / "model_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        print(f"    ✓ Model metadata saved to: {metadata_path}")
        
        # Save feature importance
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            importance_path = fold_features_dir / "feature_importance.csv"
            importance_df.to_csv(importance_path, index=False)
            print(f"    ✓ Feature importance saved to: {importance_path}")
    
    def save_fold_results(self, fold_results):
        """Save detailed results for a single fold"""
        fold_name = fold_results['fold_name']
        
        # Save in the traditional results directory
        fold_dir = self.results_dir / "XGBoost" / fold_name
        fold_dir.mkdir(exist_ok=True, parents=True)
        
        # Save model and scaler (for backwards compatibility)
        joblib.dump(fold_results['model'], fold_dir / 'model.joblib')
        joblib.dump(fold_results['scaler'], fold_dir / 'scaler.joblib')
        
        # Save metrics
        metrics_data = {
            'fold_name': fold_name,
            'train_metrics': fold_results['train_metrics'],
            'test_metrics': fold_results['test_metrics'],
            'data_info': fold_results['data_info']
        }
        
        with open(fold_dir / 'metrics.json', 'w') as f:
            json.dump(metrics_data, f, indent=2)
        
        # Save predictions
        train_predictions_df = pd.DataFrame({
            'y_train_true': fold_results['y_train'],
            'y_train_pred': fold_results['predictions']['y_train_pred'],
            'y_train_proba': fold_results['predictions']['y_train_proba']
        })
        train_predictions_df.to_csv(fold_dir / 'train_predictions.csv', index=False)
        
        test_predictions_df = pd.DataFrame({
            'y_test_true': fold_results['y_test'],
            'y_test_pred': fold_results['predictions']['y_test_pred'],
            'y_test_proba': fold_results['predictions']['y_test_proba']
        })
        test_predictions_df.to_csv(fold_dir / 'test_predictions.csv', index=False)

    def plot_confusion_matrix(self, y_true, y_pred, title, save_path):
        """Plot and save confusion matrix"""
        try:
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

    def plot_feature_importance(self, model, feature_names, title, save_path, top_n=20):
        """Plot and save feature importance"""
        try:
            if not hasattr(model, 'feature_importances_'):
                print(f"    Model does not support feature importance")
                return
            
            # Create feature importance DataFrame
            feature_importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': model.feature_importances_
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

    def generate_fold_plots(self, fold_results):
        """Generate plots for a single fold"""
        fold_name = fold_results['fold_name']
        print(f"  Generating plots for {fold_name}...")
        
        plots_dir = self.results_dir / "plots" / fold_name
        plots_dir.mkdir(exist_ok=True, parents=True)
        
        y_train = fold_results['y_train']
        y_test = fold_results['y_test']
        
        try:
            # Training confusion matrix
            y_train_pred = fold_results['predictions']['y_train_pred']
            train_cm_path = plots_dir / 'confusion_matrix_train.png'
            self.plot_confusion_matrix(
                y_train, y_train_pred,
                f'XGBoost - Training Set Confusion Matrix\n{fold_name}',
                train_cm_path
            )
            
            # Test confusion matrix
            y_test_pred = fold_results['predictions']['y_test_pred']
            test_cm_path = plots_dir / 'confusion_matrix_test.png'
            self.plot_confusion_matrix(
                y_test, y_test_pred,
                f'XGBoost - Test Set Confusion Matrix\n{fold_name}',
                test_cm_path
            )
            
            # ROC curve
            y_test_proba = fold_results['predictions']['y_test_proba']
            roc_path = plots_dir / 'roc_curve.png'
            self.plot_roc_curve(
                y_test, y_test_proba,
                f'XGBoost - ROC Curve\n{fold_name}',
                roc_path
            )
            
            # Feature importance plot
            model = fold_results['model']
            feature_names = fold_results['feature_names']
            importance_path = plots_dir / 'feature_importance.png'
            self.plot_feature_importance(
                model, feature_names,
                f'XGBoost - Feature Importance\n{fold_name}',
                importance_path
            )
                    
        except Exception as e:
            print(f"    Error processing plots: {e}")
        
        print(f"  Completed plots for {fold_name}")

    def compile_summary_results(self):
        """Compile results across all folds"""
        print("\nCompiling summary results...")
        
        summary_results = {
            'overall_stats': {
                'total_folds': len(self.all_results),
                'model': 'XGBoost',
                'avg_metrics': {}
            },
            'fold_details': self.all_results
        }
        
        if not self.all_results:
            print("No results to summarize")
            return summary_results
        
        # Calculate average metrics across folds
        test_metrics_keys = self.all_results[0]['test_metrics'].keys()
        avg_test_metrics = {}
        std_test_metrics = {}
        
        for metric in test_metrics_keys:
            values = [r['test_metrics'][metric] for r in self.all_results 
                     if metric in r['test_metrics']]
            if values:
                avg_test_metrics[metric] = np.mean(values)
                std_test_metrics[metric] = np.std(values)
        
        summary_results['overall_stats']['avg_metrics'] = {
            'test_metrics_mean': avg_test_metrics,
            'test_metrics_std': std_test_metrics,
            'num_folds': len(self.all_results)
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
                'train_metrics': fold_result['train_metrics'].copy(),
                'test_metrics': fold_result['test_metrics'].copy()
            }
            clean_summary['fold_details'].append(clean_fold)
        
        # Save JSON summary
        summary_path = self.results_dir / "summary" / f"xgboost_summary_{timestamp}.json"
        with open(summary_path, 'w') as f:
            def convert_types(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return obj
            
            json.dump(clean_summary, f, indent=2, default=convert_types)
        
        # Create readable report
        report_path = self.results_dir / "summary" / f"xgboost_report_{timestamp}.txt"
        with open(report_path, 'w') as f:
            f.write("XGBOOST CLINVAR FOLD-BASED PIPELINE RESULTS\n")
            f.write("=" * 60 + "\n")
            f.write(f"Generated: {datetime.now()}\n")
            f.write(f"Total Folds: {summary_results['overall_stats']['total_folds']}\n")
            f.write("Model: XGBoost\n")
            f.write("Class Labels: 0 = Benign, 1 = Pathogenic\n\n")
            
            f.write("AVERAGE TEST METRICS ACROSS FOLDS:\n")
            f.write("-" * 40 + "\n")
            
            avg_metrics = summary_results['overall_stats']['avg_metrics']
            main_metrics = ['mcc', 'auc_roc', 'accuracy', 'f1', 'precision', 'recall']
            
            f.write("Main Performance Metrics:\n")
            for metric in main_metrics:
                if metric in avg_metrics['test_metrics_mean']:
                    mean_val = avg_metrics['test_metrics_mean'][metric]
                    std_val = avg_metrics['test_metrics_std'].get(metric, 0)
                    f.write(f"  {metric:15}: {mean_val:.4f} (±{std_val:.4f})\n")
            
            f.write("\nFOLD-BY-FOLD DETAILS:\n")
            f.write("-" * 25 + "\n")
            
            for fold_result in summary_results['fold_details']:
                f.write(f"\n{fold_result['fold_name']}:\n")
                f.write(f"  Train samples: {fold_result['data_info']['train_samples']}\n")
                f.write(f"  Test samples:  {fold_result['data_info']['test_samples']}\n")
                f.write(f"  Features:      {fold_result['data_info']['features']}\n")
                
                test_mcc = fold_result['test_metrics']['mcc']
                test_auc = fold_result['test_metrics'].get('auc_roc', 'N/A')
                test_acc = fold_result['test_metrics'].get('accuracy', 'N/A')
                f.write(f"  XGBoost: MCC={test_mcc:.4f}, AUC={test_auc:.4f}, ACC={test_acc:.4f}\n")
        
        print(f"Summary saved: {summary_path}")
        print(f"Report saved: {report_path}")
        
        return summary_path, report_path

    def run_complete_pipeline(self):
        """Run the complete XGBoost fold-based pipeline"""
        print("XGBOOST CLINVAR FOLD-BASED PIPELINE")
        print("=" * 60)
        print("Classification Task: 0 = Benign, 1 = Pathogenic")
        print("Model: XGBoost only")
        print("=" * 60)
        
        # Find all folds
        fold_dirs = self.find_folds()
        
        if not fold_dirs:
            print("No fold directories found!")
            return
        
        # Process each fold
        for fold_idx, fold_dir in enumerate(fold_dirs):
            # Load fold data
            fold_data = self.load_fold_data(fold_dir)
            
            if fold_data is None:
                continue
            
            # Train and evaluate XGBoost
            fold_results = self.train_and_evaluate_fold(fold_data)
            
            if fold_results is None:
                continue
            
            # Save model and features for inference
            self.save_model_for_inference(fold_results, fold_idx)
            
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
        
        # Print final summary
        print("\n" + "=" * 60)
        print("PIPELINE COMPLETE!")
        print("=" * 60)
        print("Classification: 0 = Benign, 1 = Pathogenic")
        print(f"Processed {len(self.all_results)} folds")
        print(f"Results saved in: {self.results_dir}")
        print(f"Models and features saved in: {self.features_dir}")
        
        if summary_results['overall_stats']['avg_metrics']:
            avg_metrics = summary_results['overall_stats']['avg_metrics']
            avg_mcc = avg_metrics['test_metrics_mean'].get('mcc', 0)
            std_mcc = avg_metrics['test_metrics_std'].get('mcc', 0)
            avg_auc = avg_metrics['test_metrics_mean'].get('auc_roc', 0)
            std_auc = avg_metrics['test_metrics_std'].get('auc_roc', 0)
            print(f"\nOverall XGBoost Performance:")
            print(f"  MCC: {avg_mcc:.4f} (±{std_mcc:.4f})")
            print(f"  AUC: {avg_auc:.4f} (±{std_auc:.4f})")
        
        print(f"\nSaved models for inference:")
        for fold_result in self.all_results:
            fold_name = fold_result['fold_name']
            print(f"  {fold_name}: Clinvar_Dataset2/Features/{fold_name}/")
        
        return summary_results


def main():
    """Run the complete XGBoost pipeline"""
    # Initialize pipeline
    pipeline = XGBoostClinVarPipeline()
    
    # Run complete pipeline
    results = pipeline.run_complete_pipeline()
    
    return results


if __name__ == "__main__":
    results = main()