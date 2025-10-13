#!/usr/bin/env python3
"""
ClinVar 5-Fold Cross-Validation Splitter
Creates stratified 5-fold splits from merged ClinVar data
Usage: python create_cv_folds.py [input_dir] [output_dir]
"""

import pandas as pd
import numpy as np
from pathlib import Path
import os
import sys
import argparse
from datetime import datetime
from sklearn.model_selection import StratifiedKFold
import warnings

warnings.filterwarnings('ignore')

class ClinVarFoldCreator:
    def __init__(self, input_dir="MachineLearningInput", output_dir="Clinvar_Dataset", n_folds=5):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.n_folds = n_folds
        self.random_state = 42
        
        # Create output directory
        self.output_dir.mkdir(exist_ok=True)
        
        # Statistics tracking
        self.stats = {
            'total_variants': 0,
            'total_features': 0,
            'class_distribution': {},
            'fold_stats': {}
        }

    def load_data(self):
        """Load the merged feature and prediction files"""
        print("Loading merged ClinVar data...")
        
        features_file = self.input_dir / "all_chromosomes_features.csv"
        predictions_file = self.input_dir / "all_chromosomes_predictions.csv"
        
        if not features_file.exists():
            print(f"ERROR: Features file not found: {features_file}")
            print("Please run the merger script first to create merged files")
            return None, None
        
        if not predictions_file.exists():
            print(f"ERROR: Predictions file not found: {predictions_file}")
            print("Please run the merger script first to create merged files")
            return None, None
        
        # Load files
        print(f"  Loading features: {features_file.name}")
        X = pd.read_csv(features_file, low_memory=False)
        
        print(f"  Loading predictions: {predictions_file.name}")
        y_df = pd.read_csv(predictions_file)
        
        print(f"  Initial shapes - Features: {X.shape}, Predictions: {y_df.shape}")
        
        # Verify same number of rows
        if len(X) != len(y_df):
            print(f"ERROR: Mismatch in row counts - Features: {len(X)}, Predictions: {len(y_df)}")
            return None, None
        
        # Show initial class distribution
        print(f"\n  Initial class distribution:")
        initial_class_counts = y_df['ML_Class'].value_counts()
        for class_name, count in initial_class_counts.items():
            print(f"    {class_name}: {count:,} ({count/len(y_df)*100:.1f}%)")
        
        # CRITICAL: Filter to only keep Benign and Pathogenic rows
        binary_mask = y_df['ML_Class'].isin(['Pathogenic', 'Benign'])
        
        print(f"\n  Filtering to binary classes only...")
        print(f"    Rows before filtering: {len(y_df):,}")
        print(f"    Rows with Benign/Pathogenic: {binary_mask.sum():,}")
        print(f"    Rows to discard (Unknown/VUS/etc): {(~binary_mask).sum():,} ({(~binary_mask).mean()*100:.1f}%)")
        
        # Apply the filter to both dataframes
        X_binary = X[binary_mask].copy()
        y_df_binary = y_df[binary_mask].copy()
        
        # Reset indices
        X_binary = X_binary.reset_index(drop=True)
        y_df_binary = y_df_binary.reset_index(drop=True)
        
        print(f"    Final shapes - Features: {X_binary.shape}, Predictions: {y_df_binary.shape}")
        
        # Create binary target variable (now only from Benign/Pathogenic)
        y_binary = (y_df_binary['ML_Class'] == 'Pathogenic').astype(int)
        
        # Show final clean class distribution
        final_class_counts = y_df_binary['ML_Class'].value_counts()
        print(f"\n  Final clean class distribution:")
        for class_name, count in final_class_counts.items():
            print(f"    {class_name}: {count:,} ({count/len(y_df_binary)*100:.1f}%)")
        
        print(f"  Final binary target distribution:")
        print(f"    Benign (0): {(y_binary == 0).sum():,} ({(y_binary == 0).mean()*100:.1f}%)")
        print(f"    Pathogenic (1): {(y_binary == 1).sum():,} ({(y_binary == 1).mean()*100:.1f}%)")
        
        # Store statistics
        self.stats['total_variants'] = len(X_binary)
        self.stats['total_features'] = X_binary.shape[1]
        self.stats['class_distribution'] = {
            'Benign': (y_binary == 0).sum(),
            'Pathogenic': (y_binary == 1).sum()
        }
        
        return X_binary, y_binary, y_df_binary

    def create_fold_directories(self):
        """Create directory structure for folds"""
        print(f"\nCreating fold directories...")
        
        fold_dirs = []
        for fold in range(1, self.n_folds + 1):
            fold_dir = self.output_dir / f"Fold_{fold}"
            fold_dir.mkdir(exist_ok=True)
            fold_dirs.append(fold_dir)
            print(f"  Created: {fold_dir}")
        
        return fold_dirs

    def create_stratified_folds(self, X, y, y_df):
        """Create stratified 5-fold splits"""
        print(f"\nCreating {self.n_folds}-fold stratified cross-validation splits...")
        
        # Create stratified k-fold splitter
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        
        # Create fold directories
        fold_dirs = self.create_fold_directories()
        
        # Generate folds
        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            fold_num = fold_idx + 1
            fold_dir = fold_dirs[fold_idx]
            
            print(f"\n  Processing Fold {fold_num}...")
            
            # Split the data
            X_train = X.iloc[train_idx].copy()
            X_test = X.iloc[test_idx].copy()
            y_train = y.iloc[train_idx].copy()
            y_test = y.iloc[test_idx].copy()
            y_df_train = y_df.iloc[train_idx].copy()
            y_df_test = y_df.iloc[test_idx].copy()
            
            # Print fold statistics
            print(f"    Train set: {len(X_train):,} samples")
            print(f"      Benign: {(y_train == 0).sum():,} ({(y_train == 0).mean()*100:.1f}%)")
            print(f"      Pathogenic: {(y_train == 1).sum():,} ({(y_train == 1).mean()*100:.1f}%)")
            
            print(f"    Test set: {len(X_test):,} samples")
            print(f"      Benign: {(y_test == 0).sum():,} ({(y_test == 0).mean()*100:.1f}%)")
            print(f"      Pathogenic: {(y_test == 1).sum():,} ({(y_test == 1).mean()*100:.1f}%)")
            
            # Save files
            X_train_file = fold_dir / "X_train.csv"
            X_test_file = fold_dir / "X_test.csv"
            Y_train_file = fold_dir / "Y_train.csv"
            Y_test_file = fold_dir / "Y_test.csv"
            
            print(f"    Saving fold files...")
            X_train.to_csv(X_train_file, index=False)
            X_test.to_csv(X_test_file, index=False)
            y_df_train.to_csv(Y_train_file, index=False)
            y_df_test.to_csv(Y_test_file, index=False)
            
            # Calculate file sizes
            x_train_size = X_train_file.stat().st_size / 1024 / 1024
            x_test_size = X_test_file.stat().st_size / 1024 / 1024
            y_train_size = Y_train_file.stat().st_size / 1024 / 1024
            y_test_size = Y_test_file.stat().st_size / 1024 / 1024
            
            print(f"      X_train.csv: {x_train_size:.1f} MB")
            print(f"      X_test.csv: {x_test_size:.1f} MB")
            print(f"      Y_train.csv: {y_train_size:.1f} MB")
            print(f"      Y_test.csv: {y_test_size:.1f} MB")
            
            # Store fold statistics
            self.stats['fold_stats'][fold_num] = {
                'train_samples': len(X_train),
                'test_samples': len(X_test),
                'train_benign': (y_train == 0).sum(),
                'train_pathogenic': (y_train == 1).sum(),
                'test_benign': (y_test == 0).sum(),
                'test_pathogenic': (y_test == 1).sum(),
                'file_sizes': {
                    'X_train_mb': x_train_size,
                    'X_test_mb': x_test_size,
                    'Y_train_mb': y_train_size,
                    'Y_test_mb': y_test_size
                }
            }

    def create_fold_summary(self):
        """Create summary of fold creation"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_file = self.output_dir / f"fold_creation_summary_{timestamp}.txt"
        
        with open(summary_file, 'w') as f:
            f.write("CLINVAR 5-FOLD CROSS-VALIDATION SUMMARY\n")
            f.write("=" * 60 + "\n")
            f.write(f"Generated: {datetime.now()}\n")
            f.write(f"Input directory: {self.input_dir.absolute()}\n")
            f.write(f"Output directory: {self.output_dir.absolute()}\n")
            f.write(f"Number of folds: {self.n_folds}\n")
            f.write(f"Random state: {self.random_state}\n\n")
            
            f.write("DATASET STATISTICS:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Total variants: {self.stats['total_variants']:,}\n")
            f.write(f"Total features: {self.stats['total_features']:,}\n")
            f.write(f"Benign variants: {self.stats['class_distribution']['Benign']:,}\n")
            f.write(f"Pathogenic variants: {self.stats['class_distribution']['Pathogenic']:,}\n\n")
            
            f.write("FOLD-BY-FOLD BREAKDOWN:\n")
            f.write("-" * 35 + "\n")
            for fold_num, stats in self.stats['fold_stats'].items():
                f.write(f"FOLD {fold_num}:\n")
                f.write(f"  Training set: {stats['train_samples']:,} samples\n")
                f.write(f"    Benign: {stats['train_benign']:,} ({stats['train_benign']/stats['train_samples']*100:.1f}%)\n")
                f.write(f"    Pathogenic: {stats['train_pathogenic']:,} ({stats['train_pathogenic']/stats['train_samples']*100:.1f}%)\n")
                f.write(f"  Test set: {stats['test_samples']:,} samples\n")
                f.write(f"    Benign: {stats['test_benign']:,} ({stats['test_benign']/stats['test_samples']*100:.1f}%)\n")
                f.write(f"    Pathogenic: {stats['test_pathogenic']:,} ({stats['test_pathogenic']/stats['test_samples']*100:.1f}%)\n")
                f.write(f"  File sizes:\n")
                f.write(f"    X_train.csv: {stats['file_sizes']['X_train_mb']:.1f} MB\n")
                f.write(f"    X_test.csv: {stats['file_sizes']['X_test_mb']:.1f} MB\n")
                f.write(f"    Y_train.csv: {stats['file_sizes']['Y_train_mb']:.1f} MB\n")
                f.write(f"    Y_test.csv: {stats['file_sizes']['Y_test_mb']:.1f} MB\n")
                f.write("\n")
            
            f.write("USAGE EXAMPLE:\n")
            f.write("-" * 20 + "\n")
            f.write("import pandas as pd\n")
            f.write("from sklearn.ensemble import RandomForestClassifier\n")
            f.write("from sklearn.metrics import classification_report\n\n")
            f.write("# Load a specific fold\n")
            f.write("fold = 1\n")
            f.write("X_train = pd.read_csv(f'Clinvar_Dataset/Fold_{fold}/X_train.csv')\n")
            f.write("X_test = pd.read_csv(f'Clinvar_Dataset/Fold_{fold}/X_test.csv')\n")
            f.write("y_train_df = pd.read_csv(f'Clinvar_Dataset/Fold_{fold}/Y_train.csv')\n")
            f.write("y_test_df = pd.read_csv(f'Clinvar_Dataset/Fold_{fold}/Y_test.csv')\n\n")
            f.write("# Convert to binary targets\n")
            f.write("y_train = (y_train_df['ML_Class'] == 'Pathogenic').astype(int)\n")
            f.write("y_test = (y_test_df['ML_Class'] == 'Pathogenic').astype(int)\n\n")
            f.write("# Train and evaluate model\n")
            f.write("model = RandomForestClassifier(random_state=42)\n")
            f.write("model.fit(X_train, y_train)\n")
            f.write("y_pred = model.predict(X_test)\n")
            f.write("print(classification_report(y_test, y_pred))\n")
        
        print(f"\nSummary saved: {summary_file.name}")

    def run_fold_creation(self):
        """Run the complete fold creation process"""
        print("CLINVAR 5-FOLD CROSS-VALIDATION CREATOR")
        print("=" * 60)
        print(f"Input: {self.input_dir.absolute()}")
        print(f"Output: {self.output_dir.absolute()}")
        print(f"Folds: {self.n_folds}")
        
        # Load data
        X, y, y_df = self.load_data()
        
        if X is None or y is None:
            print("ERROR: Could not load data")
            return False
        
        # Create folds
        self.create_stratified_folds(X, y, y_df)
        
        # Create summary
        self.create_fold_summary()
        
        # Final summary
        print(f"\n" + "=" * 60)
        print("FOLD CREATION COMPLETE!")
        print(f"Created {self.n_folds} stratified folds with {self.stats['total_variants']:,} variants each")
        print(f"Features per fold: {self.stats['total_features']:,}")
        print(f"Output directory: {self.output_dir.absolute()}")
        
        # Show directory structure
        print(f"\nDirectory structure created:")
        for fold in range(1, self.n_folds + 1):
            print(f"  Clinvar_Dataset/Fold_{fold}/")
            print(f"    ├── X_train.csv")
            print(f"    ├── X_test.csv") 
            print(f"    ├── Y_train.csv")
            print(f"    └── Y_test.csv")
        
        print(f"\nReady for {self.n_folds}-fold cross-validation!")
        
        return True


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Create 5-fold cross-validation splits from merged ClinVar data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python create_cv_folds.py                                    # Use default directories
  python create_cv_folds.py MachineLearningInput Clinvar_Dataset
  python create_cv_folds.py --folds 10                        # Create 10 folds instead
        """
    )
    
    parser.add_argument(
        'input_dir',
        nargs='?',
        default='MachineLearningInput',
        help='Input directory with merged files (default: MachineLearningInput)'
    )
    
    parser.add_argument(
        'output_dir',
        nargs='?',
        default='Clinvar_Dataset',
        help='Output directory for folds (default: Clinvar_Dataset)'
    )
    
    parser.add_argument(
        '--folds',
        type=int,
        default=5,
        help='Number of folds to create (default: 5)'
    )
    
    parser.add_argument(
        '--random-state',
        type=int,
        default=42,
        help='Random state for reproducible splits (default: 42)'
    )
    
    return parser.parse_args()


def main():
    """Main execution function"""
    args = parse_arguments()
    
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Number of folds: {args.folds}")
    print(f"Random state: {args.random_state}")
    
    # Create fold creator
    creator = ClinVarFoldCreator(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        n_folds=args.folds
    )
    creator.random_state = args.random_state
    
    try:
        success = creator.run_fold_creation()
        
        if success:
            print(f"\nSUCCESS: {args.folds}-fold cross-validation splits created!")
        else:
            print(f"\nFAILED: Could not create fold splits")
            
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()