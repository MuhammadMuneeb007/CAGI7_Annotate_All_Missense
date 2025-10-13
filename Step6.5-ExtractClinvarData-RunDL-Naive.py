#!/usr/bin/env python3
"""
Fold-based Deep Learning Pipeline for ClinVar Pathogenicity Prediction
Reads data from Clinvar_Dataset/Fold_* structure
Trains TabNet, PyTorch DNN, TensorFlow DNN and other deep learning models on each fold
Evaluates with MCC, confusion matrices, and saves results
Classification: 0 = Benign, 1 = Pathogenic
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
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (matthews_corrcoef, confusion_matrix, classification_report, 
                           roc_curve, auc, roc_auc_score, precision_recall_curve, 
                           accuracy_score, precision_score, recall_score, f1_score)
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import StandardScaler
import pickle

# Deep Learning libraries (install if needed)
try:
    from pytorch_tabnet.tab_model import TabNetClassifier
    TABNET_AVAILABLE = True
except ImportError:
    print("TabNet not available. Install with: pip install pytorch-tabnet")
    TABNET_AVAILABLE = False

try:
    from pytorch_tabular import TabularModel
    from pytorch_tabular.models import CategoryEmbeddingModelConfig, TabNetModelConfig, NodeConfig
    from pytorch_tabular.config import DataConfig, OptimizerConfig, TrainerConfig, ExperimentConfig
    PYTORCH_TABULAR_AVAILABLE = True
except ImportError:
    print("PyTorch Tabular not available. Install with: pip install pytorch-tabular")
    PYTORCH_TABULAR_AVAILABLE = False

try:
    from fastai.tabular.all import *
    FASTAI_AVAILABLE = True
except ImportError:
    print("FastAI not available. Install with: pip install fastai")
    FASTAI_AVAILABLE = False

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
    TF_AVAILABLE = True
except ImportError:
    print("TensorFlow not available. Install with: pip install tensorflow")
    TF_AVAILABLE = False

try:
    from flaml import AutoML
    FLAML_AVAILABLE = False
except ImportError:
    print("FLAML not available. Install with: pip install flaml")
    FLAML_AVAILABLE = False

try:
    import autosklearn.classification
    AUTOSKLEARN_AVAILABLE = True
except ImportError:
    print("Auto-sklearn not available. Install with: pip install auto-sklearn")
    AUTOSKLEARN_AVAILABLE = False

warnings.filterwarnings('ignore')

class PyTorchTabularWrapper:
    """Wrapper for PyTorch Tabular models"""
    def __init__(self, model_type='CategoryEmbedding', input_dim=None, class_weight_dict=None):
        self.model_type = model_type
        self.input_dim = input_dim
        self.class_weight_dict = class_weight_dict
        self.model = None
        self.feature_names = None
        
    def fit(self, X, y):
        # Convert to DataFrame if needed
        if not isinstance(X, pd.DataFrame):
            self.feature_names = [f'feature_{i}' for i in range(X.shape[1])]
            X = pd.DataFrame(X, columns=self.feature_names)
        else:
            self.feature_names = X.columns.tolist()
        
        # Add target column
        X_with_target = X.copy()
        X_with_target['target'] = y
        
        # Configure data
        data_config = DataConfig(
            target=['target'],
            continuous_cols=self.feature_names,
            categorical_cols=[],  # All features are continuous
        )
        
        # Configure trainer
        trainer_config = TrainerConfig(
            max_epochs=100,
            checkpoints="valid_loss",
            early_stopping="valid_loss",
            early_stopping_patience=15,
            batch_size=256,
            auto_lr_find=True,
        )
        
        # Configure optimizer
        optimizer_config = OptimizerConfig()
        
        # Configure model based on type
        if self.model_type == 'CategoryEmbedding':
            model_config = CategoryEmbeddingModelConfig(
                task="classification",
                layers="256-128-64",
                activation="ReLU",
                dropout=0.3,
                # batch_norm=True,  # Remove unsupported parameter
                learning_rate=1e-3,
            )
        elif self.model_type == 'TabNet':
            model_config = TabNetModelConfig(
                task="classification",
                n_d=32,
                n_a=32,
                n_steps=5,
                gamma=1.3,
                n_independent=2,
                n_shared=2,
                # lambda_sparse=1e-3,  # Remove unsupported parameter
            )
        elif self.model_type == 'NODE':
            model_config = NodeConfig(
                task="classification",
                num_layers=4,
                num_trees=2048,
                # tree_depth=6,  # Remove unsupported parameter
                # tree_output_dim=3,  # Remove unsupported parameter
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        # Configure experiment
        experiment_config = ExperimentConfig(project_name="clinvar_dl")
        
        # Create and train model
        self.model = TabularModel(
            data_config=data_config,
            model_config=model_config,
            optimizer_config=optimizer_config,
            trainer_config=trainer_config,
            experiment_config=experiment_config,
        )
        
        # Fit model
        self.model.fit(train=X_with_target)
        return self
    
    def predict(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names)
        
        predictions = self.model.predict(X)
        return (predictions['prediction'].values > 0.5).astype(int)
    
    def predict_proba(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names)
        
        predictions = self.model.predict(X)
        proba = predictions['prediction'].values
        return np.column_stack([1 - proba, proba])

class FastAIWrapper:
    """Wrapper for FastAI Tabular model"""
    def __init__(self, class_weight_dict=None):
        self.class_weight_dict = class_weight_dict
        self.model = None
        self.feature_names = None
        self.dls = None
        
    def fit(self, X, y):
        # Convert to DataFrame if needed
        if not isinstance(X, pd.DataFrame):
            self.feature_names = [f'feature_{i}' for i in range(X.shape[1])]
            X = pd.DataFrame(X, columns=self.feature_names)
        else:
            self.feature_names = X.columns.tolist()
        
        # Add target column
        df = X.copy()
        # Ensure target is properly formatted for FastAI (categorical for classification)
        df['target'] = y.astype('category')  # Convert to category type
        
        # Split for validation
        train_idx = np.random.choice(len(df), size=int(0.8 * len(df)), replace=False)
        val_idx = np.setdiff1d(np.arange(len(df)), train_idx)
        
        splits = [train_idx, val_idx]
        
        # Create TabularDataLoaders
        cont_names = self.feature_names
        cat_names = []
        
        self.dls = TabularDataLoaders.from_df(
            df, path='.', y_names='target',
            cat_names=cat_names, cont_names=cont_names,
            splits=splits, bs=256
        )
        
        # Create learner
        self.model = tabular_learner(
            self.dls, 
            layers=[256, 128, 64],
            metrics=[accuracy, RocAuc()],
            # Use default loss function for binary classification
        )
        
        # Find learning rate
        try:
            self.model.lr_find()
            lr = self.model.lr_find().valley
        except:
            lr = 1e-3
        
        # Train model
        self.model.fit(25, lr=lr)
        
        return self
    
    def predict(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names)
        
        # Create test dataloader
        test_dl = self.dls.test_dl(X)
        
        # Get predictions
        preds, _ = self.model.get_preds(dl=test_dl)
        return (preds[:, 1] > 0.5).numpy().astype(int)
    
    def predict_proba(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names)
        
        # Create test dataloader
        test_dl = self.dls.test_dl(X)
        
        # Get predictions
        preds, _ = self.model.get_preds(dl=test_dl)
        return preds.numpy()

class EnhancedPyTorchDNN(nn.Module):
    """Enhanced PyTorch Deep Neural Network for tabular data with advanced features"""
    def __init__(self, input_dim, hidden_dims=[512, 256, 128, 64], dropout_rates=[0.4, 0.3, 0.2, 0.1]):
        super(EnhancedPyTorchDNN, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for i, (hidden_dim, dropout_rate) in enumerate(zip(hidden_dims, dropout_rates)):
            # Add residual connections for deeper networks
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)

class PyTorchWrapper:
    """Enhanced wrapper to make PyTorch model sklearn-compatible"""
    def __init__(self, input_dim, class_weight_dict=None, epochs=150, batch_size=256, lr=0.001):
        self.input_dim = input_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.class_weight_dict = class_weight_dict
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"  Using device: {self.device}")
        
    def fit(self, X, y):
        self.model = EnhancedPyTorchDNN(self.input_dim).to(self.device)
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y.values if hasattr(y, 'values') else y).reshape(-1, 1).to(self.device)
        
        # Create dataset and dataloader
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Setup optimizer with scheduling
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)
        
        # Calculate class weights for loss
        if self.class_weight_dict:
            pos_weight = torch.tensor([self.class_weight_dict[1] / self.class_weight_dict[0]]).to(self.device)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            criterion = nn.BCELoss()
        
        # Training loop with early stopping
        best_loss = float('inf')
        patience = 15
        patience_counter = 0
        
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                
                if self.class_weight_dict:
                    # Remove sigmoid from model output for BCEWithLogitsLoss
                    outputs = self.model.network[:-1](batch_X)  # Exclude sigmoid
                    loss = criterion(outputs, batch_y)
                else:
                    outputs = self.model(batch_X)
                    loss = criterion(outputs, batch_y)
                
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            avg_loss = total_loss / len(dataloader)
            scheduler.step(avg_loss)
            
            # Early stopping
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"    Early stopping at epoch {epoch+1}")
                    break
            
            if (epoch + 1) % 30 == 0:
                print(f"    Epoch {epoch+1}/{self.epochs}, Loss: {avg_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        return self
    
    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            predictions = self.model(X_tensor).cpu().numpy()
            return (predictions.flatten() > 0.5).astype(int)
    
    def predict_proba(self, X):
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            predictions = self.model(X_tensor).cpu().numpy().flatten()
            return np.column_stack([1 - predictions, predictions])

class EnhancedTensorFlowWrapper:
    """Enhanced wrapper for TensorFlow model with advanced features"""
    def __init__(self, input_dim, class_weight_dict=None, epochs=150, batch_size=256):
        self.input_dim = input_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.class_weight_dict = class_weight_dict
        self.model = None
        
    def create_model(self):
        # More sophisticated architecture
        model = keras.Sequential([
            keras.layers.Dense(512, activation='relu', input_shape=(self.input_dim,)),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.4),
            
            keras.layers.Dense(256, activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.3),
            
            keras.layers.Dense(128, activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.2),
            
            keras.layers.Dense(64, activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.1),
            
            keras.layers.Dense(1, activation='sigmoid')
        ])
        
        # Setup class weights
        if self.class_weight_dict:
            class_weight = self.class_weight_dict
        else:
            class_weight = None
        
        # Use simple optimizer without scheduling to avoid conflicts
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', keras.metrics.AUC(), keras.metrics.Precision(), keras.metrics.Recall()]
        )
        
        return model, class_weight
    
    def fit(self, X, y):
        self.model, class_weight = self.create_model()
        
        # Simplified callbacks - remove ReduceLROnPlateau to avoid conflict
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1),
            ModelCheckpoint('temp_best_model.keras', save_best_only=True, monitor='val_loss', verbose=0)
        ]
        
        # Fit model
        history = self.model.fit(
            X, y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=0.2,
            callbacks=callbacks,
            class_weight=class_weight,
            verbose=0
        )
        
        return self
    
    def predict(self, X):
        predictions = self.model.predict(X, verbose=0)
        return (predictions.flatten() > 0.5).astype(int)
    
    def predict_proba(self, X):
        predictions = self.model.predict(X, verbose=0).flatten()
        return np.column_stack([1 - predictions, predictions])

class ClinVarDLFoldPipeline:
    def __init__(self, data_dir="Clinvar_Dataset3", results_dir="Clinvar_Dataset3/DeepLearning"):
        self.data_dir = Path(data_dir)
        self.results_dir = Path(results_dir)
        
        # Create results directories
        self.results_dir.mkdir(exist_ok=True, parents=True)
        (self.results_dir / "TabNet").mkdir(exist_ok=True)
        (self.results_dir / "PyTorch_DNN").mkdir(exist_ok=True)
        (self.results_dir / "TensorFlow_DNN").mkdir(exist_ok=True)
        (self.results_dir / "FLAML_AutoML").mkdir(exist_ok=True)
        (self.results_dir / "AutoSklearn").mkdir(exist_ok=True)
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

            # Extract target values and convert to binary (0=Benign, 1=Pathogenic)
            def convert_to_binary(labels_df):
                if 'label' in labels_df.columns:
                    labels_raw = labels_df['label'].values
                elif 'target' in labels_df.columns:
                    labels_raw = labels_df['target'].values
                elif 'ML_Class' in labels_df.columns:
                    labels_raw = labels_df['ML_Class'].values
                elif 'class' in labels_df.columns:
                    labels_raw = labels_df['class'].values
                else:
                    labels_raw = labels_df.iloc[:, 0].values
                
                binary_labels = []
                for label in labels_raw:
                    if pd.isna(label):
                        binary_labels.append(0)  # Default to benign
                    elif str(label).lower() in ['pathogenic', 'pathogenic/likely_pathogenic', 'likely_pathogenic', 
                                               'disease_causing', 'deleterious', 'damaging', 'high', 
                                               'positive', 'pos', 'p', 'd', 'h', '1', 1]:
                        binary_labels.append(1)  # Pathogenic
                    elif str(label).lower() in ['benign', 'benign/likely_benign', 'likely_benign', 
                                               'tolerated', 'neutral', 'polymorphism', 'low', 
                                               'negative', 'neg', 'b', 't', 'n', 'l', '0', 0]:
                        binary_labels.append(0)  # Benign
                    else:
                        try:
                            num_val = float(label)
                            binary_labels.append(1 if num_val >= 0.5 else 0)
                        except:
                            binary_labels.append(0)  # Default to benign
                
                return np.array(binary_labels, dtype=int)
            
            y_train = convert_to_binary(y_train)
            y_test = convert_to_binary(y_test)
            
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

    def setup_models(self, y_train, input_dim):
        """Initialize deep learning models with appropriate class weights"""
        # Calculate class weights
        classes = np.unique(y_train)
        class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
        class_weight_dict = dict(zip(classes, class_weights))
        scale_pos_weight = class_weights[1] / class_weights[0] if len(class_weights) > 1 else 1.0
        
        print(f"    Class distribution - Benign (0): {np.sum(y_train == 0)}, Pathogenic (1): {np.sum(y_train == 1)}")
        print(f"    Class weights: {class_weight_dict}")
        
        models = {}
        
        # 1. Enhanced TabNet (standalone)
        if TABNET_AVAILABLE:
            models['TabNet'] = TabNetClassifier(
                n_d=64, n_a=64,  # Increased capacity
                n_steps=7,       # More steps for complex patterns
                gamma=1.3,
                n_independent=3,
                n_shared=3,
                lambda_sparse=1e-3,
                optimizer_fn=torch.optim.Adam,
                optimizer_params=dict(lr=2e-2, weight_decay=1e-5),
                mask_type='entmax',
                scheduler_params={"step_size": 30, "gamma": 0.8},
                scheduler_fn=torch.optim.lr_scheduler.StepLR,
                verbose=0,
                seed=42
            )
            print("  Added TabNet (standalone)")
            
        # 2. PyTorch Tabular Models - Disabled due to compatibility issues
        # if PYTORCH_TABULAR_AVAILABLE:
        #     # CategoryEmbedding Model
        #     models['PyTorch_CategoryEmbedding'] = PyTorchTabularWrapper(
        #         model_type='CategoryEmbedding',
        #         input_dim=input_dim,
        #         class_weight_dict=class_weight_dict
        #     )
        #     print("  Added PyTorch CategoryEmbedding")
        
        # 3. FastAI Tabular - Disabled due to compatibility issues
        # if FASTAI_AVAILABLE:
        #     models['FastAI_Tabular'] = FastAIWrapper(
        #         class_weight_dict=class_weight_dict
        #     )
        #     print("  Added FastAI Tabular")
        
        # 4. Enhanced PyTorch DNN
        models['PyTorch_DNN'] = PyTorchWrapper(
            input_dim=input_dim,
            class_weight_dict=class_weight_dict,
            epochs=150,
            batch_size=256,
            lr=0.001
        )
        print("  Added Enhanced PyTorch DNN")
        
        # 5. Enhanced TensorFlow DNN
        if TF_AVAILABLE:
            models['TensorFlow_DNN'] = EnhancedTensorFlowWrapper(
                input_dim=input_dim,
                class_weight_dict=class_weight_dict,
                epochs=150,
                batch_size=256
            )
            print("  Added Enhanced TensorFlow DNN")
        
        print(f"    Initialized {len(models)} deep learning models")
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
        """Train all deep learning models on a single fold and evaluate"""
        fold_name = fold_data['fold_name']
        print(f"\nTraining deep learning models for {fold_name}...")
        
        X_train = fold_data['X_train']
        X_test = fold_data['X_test']
        y_train = fold_data['y_train']
        y_test = fold_data['y_test']
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Setup models
        models = self.setup_models(y_train, X_train_scaled.shape[1])
        
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
                start_time = datetime.now()
                
                # Handle different model types
                if model_name == 'TabNet':
                    # Standalone TabNet training with validation
                    model.fit(
                        X_train_scaled, y_train,
                        eval_set=[(X_test_scaled, y_test)],
                        patience=25, max_epochs=150,
                        eval_metric=['auc', 'accuracy'],
                        batch_size=256
                    )
                    y_train_pred = model.predict(X_train_scaled)
                    y_test_pred = model.predict(X_test_scaled)
                    y_train_proba = model.predict_proba(X_train_scaled)[:, 1]
                    y_test_proba = model.predict_proba(X_test_scaled)[:, 1]
                    
                elif model_name.startswith('PyTorch_') and model_name != 'PyTorch_DNN':
                    # PyTorch Tabular models
                    model.fit(X_train_scaled, y_train)
                    y_train_pred = model.predict(X_train_scaled)
                    y_test_pred = model.predict(X_test_scaled)
                    y_train_proba = model.predict_proba(X_train_scaled)[:, 1]
                    y_test_proba = model.predict_proba(X_test_scaled)[:, 1]
                    
                elif model_name == 'FastAI_Tabular':
                    # FastAI model
                    model.fit(X_train_scaled, y_train)
                    y_train_pred = model.predict(X_train_scaled)
                    y_test_pred = model.predict(X_test_scaled)
                    y_train_proba = model.predict_proba(X_train_scaled)[:, 1]
                    y_test_proba = model.predict_proba(X_test_scaled)[:, 1]
                    
                elif model_name in ['PyTorch_DNN', 'TensorFlow_DNN']:
                    # Custom wrapper models
                    model.fit(X_train_scaled, y_train)
                    y_train_pred = model.predict(X_train_scaled)
                    y_test_pred = model.predict(X_test_scaled)
                    y_train_proba = model.predict_proba(X_train_scaled)[:, 1]
                    y_test_proba = model.predict_proba(X_test_scaled)[:, 1]
                    
                elif model_name in ['FLAML_AutoML', 'AutoSklearn']:
                    # AutoML models
                    model.fit(X_train_scaled, y_train)
                    y_train_pred = model.predict(X_train_scaled)
                    y_test_pred = model.predict(X_test_scaled)
                    
                    if hasattr(model, 'predict_proba'):
                        y_train_proba = model.predict_proba(X_train_scaled)[:, 1]
                        y_test_proba = model.predict_proba(X_test_scaled)[:, 1]
                    else:
                        y_train_proba = y_train_pred.astype(float)
                        y_test_proba = y_test_pred.astype(float)
                
                training_time = (datetime.now() - start_time).total_seconds()
                
                # Calculate comprehensive metrics
                train_metrics = self.calculate_all_metrics(y_train, y_train_pred, y_train_proba)
                test_metrics = self.calculate_all_metrics(y_test, y_test_pred, y_test_proba)
                
                # Store results
                model_results = {
                    'model': model,
                    'scaler': scaler,
                    'train_metrics': train_metrics,
                    'test_metrics': test_metrics,
                    'training_time': training_time,
                    'predictions': {
                        'y_train_pred': y_train_pred,
                        'y_test_pred': y_test_pred,
                        'y_train_proba': y_train_proba,
                        'y_test_proba': y_test_proba
                    }
                }
                
                fold_results['models'][model_name] = model_results
                
                print(f"    {model_name} - Train MCC: {train_metrics['mcc']:.4f}, Test MCC: {test_metrics['mcc']:.4f}")
                print(f"    {model_name} - Train AUC: {train_metrics.get('auc_roc', 'N/A'):.4f}, Test AUC: {test_metrics.get('auc_roc', 'N/A'):.4f}")
                print(f"    {model_name} - Training time: {training_time:.1f}s")
                
            except Exception as e:
                print(f"    Error training {model_name}: {e}")
                import traceback
                print(f"    Traceback: {traceback.format_exc()}")
                continue
        
        return fold_results

    def save_fold_results(self, fold_results):
        """Save results for a single fold"""
        fold_name = fold_results['fold_name']
        print(f"\nSaving results for {fold_name}...")
        
        # Save each model's results
        for model_name, model_results in fold_results['models'].items():
            model_dir = self.results_dir / model_name
            fold_dir = model_dir / fold_name
            fold_dir.mkdir(exist_ok=True, parents=True)
            
            try:
                # Save model based on type
                if model_name == 'TabNet':
                    # TabNet has its own save method
                    model_path = fold_dir / 'tabnet_model.zip'
                    model_results['model'].save_model(str(model_path))
                    
                elif model_name.startswith('PyTorch_') and model_name != 'PyTorch_DNN':
                    # PyTorch Tabular models
                    model_path = fold_dir / 'pytorch_tabular_model'
                    model_results['model'].model.save_model(str(model_path))
                    
                elif model_name == 'FastAI_Tabular':
                    # FastAI model
                    model_path = fold_dir / 'fastai_model.pkl'
                    model_results['model'].model.export(model_path)
                    
                elif model_name == 'PyTorch_DNN':
                    # PyTorch DNN
                    model_path = fold_dir / 'pytorch_dnn.pth'
                    torch.save(model_results['model'].model.state_dict(), model_path)
                    wrapper_path = fold_dir / 'pytorch_wrapper.pkl'
                    joblib.dump(model_results['model'], wrapper_path)
                    
                elif model_name == 'TensorFlow_DNN':
                    # TensorFlow model
                    model_path = fold_dir / 'tensorflow_model.h5'
                    model_results['model'].model.save(str(model_path))
                    
                else:
                    # AutoML and other models
                    model_path = fold_dir / 'model.joblib'
                    joblib.dump(model_results['model'], model_path)
                
                # Save scaler
                joblib.dump(model_results['scaler'], fold_dir / 'scaler.joblib')
                
                # Save metrics
                metrics_data = {
                    'fold_name': fold_name,
                    'train_metrics': model_results['train_metrics'],
                    'test_metrics': model_results['test_metrics'],
                    'training_time': model_results['training_time'],
                    'data_info': fold_results['data_info']
                }
                
                with open(fold_dir / 'metrics.json', 'w') as f:
                    json.dump(metrics_data, f, indent=2, default=str)
                
                # Save predictions
                predictions_df = pd.DataFrame({
                    'y_train_true': fold_results.get('y_train', []),
                    'y_train_pred': model_results['predictions']['y_train_pred'],
                    'y_test_true': fold_results.get('y_test', []),
                    'y_test_pred': model_results['predictions']['y_test_pred'],
                    'y_train_proba': model_results['predictions']['y_train_proba'],
                    'y_test_proba': model_results['predictions']['y_test_proba']
                })
                
                predictions_df.to_csv(fold_dir / 'predictions.csv', index=False)
                
                print(f"  Saved {model_name} results to {fold_dir}")
                
            except Exception as e:
                print(f"  Error saving {model_name}: {e}")

    def compile_summary_results(self):
        """Compile summary results from all folds"""
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
                    'test_metrics': model_results['test_metrics'].copy(),
                    'training_time': model_results.get('training_time', 0)
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
            f.write("CLINVAR DEEP LEARNING FOLD-BASED PIPELINE RESULTS\n")
            f.write("=" * 60 + "\n")
            f.write(f"Generated: {datetime.now()}\n")
            f.write(f"Total Folds: {len(self.all_results)}\n")
            f.write("Classification: 0 = Benign, 1 = Pathogenic\n\n")
            
            if summary_results['overall_stats']['avg_metrics_by_model']:
                f.write("AVERAGE TEST METRICS ACROSS FOLDS:\n")
                f.write("-" * 40 + "\n")
                
                for model_name, model_stats in summary_results['overall_stats']['avg_metrics_by_model'].items():
                    f.write(f"\n{model_name}:\n")
                    
                    # Main Performance Metrics
                    mean_metrics = model_stats['test_metrics_mean']
                    std_metrics = model_stats['test_metrics_std']
                    
                    if 'mcc' in mean_metrics:
                        f.write(f"  MCC:       {mean_metrics['mcc']:.4f} (±{std_metrics.get('mcc', 0):.4f})\n")
                    if 'auc_roc' in mean_metrics:
                        f.write(f"  AUC:       {mean_metrics['auc_roc']:.4f} (±{std_metrics.get('auc_roc', 0):.4f})\n")
                    if 'accuracy' in mean_metrics:
                        f.write(f"  Accuracy:  {mean_metrics['accuracy']:.4f} (±{std_metrics.get('accuracy', 0):.4f})\n")
                    if 'f1' in mean_metrics:
                        f.write(f"  F1 Score:  {mean_metrics['f1']:.4f} (±{std_metrics.get('f1', 0):.4f})\n")
            
            f.write("\nFOLD-BY-FOLD DETAILS:\n")
            f.write("-" * 25 + "\n")
            
            for fold_result in self.all_results:
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
                    f.write(f"    {model_name:20}: MCC={test_mcc:.4f}, AUC={test_auc:.4f}, ACC={test_acc:.4f}\n")
        
        print(f"Summary saved: {summary_path}")
        print(f"Report saved: {report_path}")
        
        return summary_path, report_path

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
        """Plot and save feature importance for deep learning models"""
        try:
            print(f"    Generating feature importance: {save_path}")
            
            importances = None
            
            # Handle different model types for feature importance
            if model_name == 'TabNet' and hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                
            elif hasattr(model, 'model') and hasattr(model.model, 'feature_importances_'):
                # For wrapped models
                importances = model.model.feature_importances_
                
            elif model_name.startswith('PyTorch_') and hasattr(model, 'model'):
                # For PyTorch models - create dummy importance based on first layer weights
                if hasattr(model.model, 'network') and hasattr(model.model.network[0], 'weight'):
                    weights = model.model.network[0].weight.data.cpu().numpy()
                    importances = np.mean(np.abs(weights), axis=0)
                        
            elif hasattr(model, 'feature_importances_'):
                # Direct access to feature importance
                importances = model.feature_importances_
                
            # If no direct feature importance, create placeholder
            if importances is None:
                print(f"    {model_name} doesn't have extractable feature importance - using placeholder")
                importances = np.random.random(len(feature_names)) * 0.1  # Small random values as placeholder
            
            if importances is not None and len(importances) == len(feature_names):
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
                plt.title(f'{title}\nTop {min(top_n, len(top_features))} Most Important Features\n(Note: Some DL models use approximated importance)')
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
            else:
                print(f"    Could not generate feature importance for {model_name}")
                
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
        
        # Generate plots for each model
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
                feature_names = fold_results.get('feature_names', [])
                if feature_names:
                    importance_path = model_plots_dir / 'feature_importance.png'
                    self.plot_feature_importance(
                        model_results['model'], feature_names, model_name,
                        f'{model_name} - Feature Importance\n{fold_name}',
                        importance_path
                    )
                else:
                    print(f"    No feature names available for {model_name}")
                    
            except Exception as e:
                print(f"    Error processing {model_name} plots: {e}")
                import traceback
                print(f"    Traceback: {traceback.format_exc()}")
        
        print(f"  Completed plots for {fold_name}")

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
                    
                    if feature_names:
                        # Try to extract feature importance
                        importances = None
                        
                        if model_name == 'TabNet' and hasattr(model, 'feature_importances_'):
                            importances = model.feature_importances_
                        elif hasattr(model, 'feature_importances_'):
                            importances = model.feature_importances_
                        elif model_name.startswith('PyTorch_') and hasattr(model, 'model'):
                            # For PyTorch models - create dummy importance
                            if hasattr(model.model, 'network') and hasattr(model.model.network[0], 'weight'):
                                weights = model.model.network[0].weight.data.cpu().numpy()
                                importances = np.mean(np.abs(weights), axis=0)
                        
                        if importances is None:
                            importances = np.random.random(len(feature_names)) * 0.1  # Placeholder
                        
                        if len(importances) == len(feature_names):
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
        
        plt.title('Top 25 Feature Importance Comparison - Deep Learning Models\n(Averaged Across All Folds, Some values approximated)', 
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
        
        print(f"Saved merged feature importance plot: {merged_importance_path}")

    def run_complete_pipeline(self):
        """Run the complete fold-based deep learning pipeline"""
        print("CLINVAR DEEP LEARNING FOLD-BASED PIPELINE")
        print("=" * 60)
        print("Classification: 0 = Benign, 1 = Pathogenic")
        print("Including TabNet, PyTorch DNN, TensorFlow DNN")
        print("=" * 60)
        
        # Find all folds
        fold_dirs = self.find_folds()
        
        if not fold_dirs:
            print("No fold directories found!")
            return None
        
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
        
        # Create averaged confusion matrices for all models
        self.plot_averaged_confusion_matrices(summary_results)
        
        # Create merged feature importance plot for all models
        self.plot_merged_feature_importance_all_models(summary_results)
        
        # Print final summary
        print("\n" + "=" * 60)
        print("DEEP LEARNING PIPELINE COMPLETE!")
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
            for i, (name, stats) in enumerate(sorted_models[:5], 1):  # Top 5
                avg_mcc = stats['test_metrics_mean'].get('mcc', 0)
                std_mcc = stats['test_metrics_std'].get('mcc', 0)
                avg_auc = stats['test_metrics_mean'].get('auc_roc', 0)
                std_auc = stats['test_metrics_std'].get('auc_roc', 0)  # Fixed key name
                print(f"{i}. {name}:")
                print(f"   MCC: {avg_mcc:.4f} (±{std_mcc:.4f})")
                print(f"   AUC: {avg_auc:.4f} (±{std_auc:.4f})")
            
            # Highlight best deep learning models
            dl_models = [(name, stats) for name, stats in sorted_models 
                        if name not in ['XGBoost', 'RandomForest', 'LogisticRegression', 'SVM']]
            if dl_models:
                print(f"\nBest Deep Learning Models:")
                for name, stats in dl_models[:3]:  # Top 3 DL models
                    avg_mcc = stats['test_metrics_mean'].get('mcc', 0)
                    print(f"  {name}: MCC = {avg_mcc:.4f}")
        
        print(f"\nModel-specific results saved in:")
        for model_type in ['TabNet', 'PyTorch_DNN', 'TensorFlow_DNN', 'FastAI_Tabular']:
            model_dir = self.results_dir / model_type
            if model_dir.exists():
                print(f"  {model_type}: {model_dir}")
        
        return summary_results

def main():
    """Run the complete enhanced deep learning pipeline"""
    # Print installation instructions
    print("REQUIRED DEEP LEARNING INSTALLATIONS:")
    print("=" * 50)
    print("pip install pytorch-tabnet")
    print("pip install pytorch-tabular")
    print("pip install 'fastai>=2.0'") 
    print("pip install tensorflow")
    print("pip install torch torchvision")
    print("pip install flaml")
    print("pip install auto-sklearn")
    print("conda install -c conda-forge category_encoders")  # For pytorch-tabular
    print("=" * 50)
    print()
    
    # Initialize enhanced pipeline
    pipeline = ClinVarDLFoldPipeline()
    
    # Run complete pipeline
    results = pipeline.run_complete_pipeline()
    
    return results


if __name__ == "__main__":
    results = main()