#!/usr/bin/env python3
"""
ClinVar Data Merger Script
Loads all chromosome files, filters for ClinVar data, and creates two merged files:
- all_chromosomes_features.csv (all features for ML)
- all_chromosomes_predictions.csv (variant IDs + ML_Class)
Usage: python merge_clinvar.py [input_directory] [output_directory]
"""

import pandas as pd
import numpy as np
from pathlib import Path
import os
import glob
import sys
import argparse
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

class ClinVarMerger:
    def __init__(self, input_dir="Annovar_Merge", output_dir="MachineLearningInput"):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # ML Class mapping for CLNSIG
        self.ml_class_mapping = {
            # BENIGN CLASS
            'Benign': 'Benign',
            'Likely_benign': 'Benign',
            'Benign/Likely_benign': 'Benign',
            'Benign|_other': 'Benign',
            'Benign|_risk_factor': 'Benign',
            'Benign|_drug_response': 'Benign',
            'Benign|_confers_sensitivity': 'Benign',
            'Benign|_association|_confers_sensitivity': 'Benign',
            'Benign/Likely_benign|_other': 'Benign',
            'Benign/Likely_benign|_risk_factor': 'Benign',
            'Benign/Likely_benign|_drug_response': 'Benign',
            'Benign/Likely_benign|_drug_response|_other': 'Benign',
            'Benign/Likely_benign|_other|_risk_factor': 'Benign',
            'Likely_benign|_other': 'Benign',
            'Likely_benign|_drug_response|_other': 'Benign',
            
            # PATHOGENIC CLASS  
            'Pathogenic': 'Pathogenic',
            'Likely_pathogenic': 'Pathogenic',
            'Pathogenic/Likely_pathogenic': 'Pathogenic',
            'Pathogenic|_other': 'Pathogenic',
            'Pathogenic|_risk_factor': 'Pathogenic',
            'Pathogenic|_drug_response': 'Pathogenic',
            'Pathogenic|_Affects': 'Pathogenic',
            'Pathogenic|_association': 'Pathogenic',
            'Pathogenic|_confers_sensitivity': 'Pathogenic',
            'Pathogenic|_protective': 'Pathogenic',
            'Pathogenic/Likely_pathogenic|_other': 'Pathogenic',
            'Pathogenic/Likely_pathogenic|_risk_factor': 'Pathogenic',
            'Pathogenic/Likely_pathogenic|_drug_response': 'Pathogenic',
            'Likely_pathogenic|_risk_factor': 'Pathogenic',
            'Likely_pathogenic|_drug_response': 'Pathogenic',
            'Likely_pathogenic|_other': 'Pathogenic',
            'Likely_pathogenic|_Affects': 'Pathogenic',
        }
        
        # Essential variant identifier features
        self.identifier_features = [
            'Chr', 'Start', 'End', 'Ref', 'Alt'
        ]
        
        # Features to exclude (data leakage and problematic columns)
        self.exclude_features = [
            # Data leakage - ClinVar annotations (except CLNSIG for ML_Class)
            'CLNDN', 'CLNREVSTAT', 'CLNDISDB', 'CLNALLELEID',
           
        ]
        
        # Statistics tracking
        self.stats = {
            'files_processed': 0,
            'files_successful': 0,
            'total_variants_input': 0,
            'total_variants_output': 0,
            'class_distribution': {},
            'feature_count': 0
        }

    def find_chromosome_files(self):
        """Find all chromosome CSV files"""
        print(f"Looking for chromosome files in: {self.input_dir.absolute()}")
        
        patterns = [
            "chr*.csv",
            "*chr*.csv", 
            "chromosome*.csv",
            "*chromosome*.csv"
        ]
        
        all_files = []
        for pattern in patterns:
            files = list(self.input_dir.glob(pattern))
            all_files.extend(files)
        
        unique_files = list(set(all_files))
        unique_files.sort()
        
        if not unique_files:
            print(f"ERROR: No chromosome files found in {self.input_dir}")
            csv_files = list(self.input_dir.glob("*.csv"))
            if csv_files:
                print(f"Available CSV files: {[f.name for f in csv_files[:10]]}")
            return []
        
        print(f"Found {len(unique_files)} chromosome files:")
        for file in unique_files:
            print(f"  - {file.name}")
        
        return unique_files

    def extract_chromosome_id(self, filename):
        """Extract chromosome identifier from filename"""
        import re
        
        filename_lower = filename.lower().replace('.csv', '')
        
        patterns = [
            r'chr(\w+)_',
            r'chr(\w+)$', 
            r'chromosome_?(\w+)',
            r'(\d+|x|y|mt|m)_chr',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, filename_lower)
            if match:
                chr_id = match.group(1).upper()
                chr_id = re.split(r'[_\-\.]', chr_id)[0]
                
                if chr_id in ['M', 'MITO']:
                    chr_id = 'MT'
                
                return chr_id
        
        return filename_lower

    def process_chromosome_file(self, file_path):
        """Process a single chromosome file and return filtered data"""
        chr_id = self.extract_chromosome_id(file_path.name)
        print(f"  Processing: {file_path.name} (Chromosome {chr_id})")
        
        try:
            # Load file
            df = pd.read_csv(file_path, low_memory=False)
            print(f"    Loaded: {len(df):,} variants, {len(df.columns)} columns")
            
            original_count = len(df)
            self.stats['total_variants_input'] += original_count
            
            # Check for CLNSIG column
            if 'CLNSIG' not in df.columns:
                print(f"    WARNING: No CLNSIG column found, skipping file")
                return None
            
            # Filter for valid CLNSIG
            df_filtered = df[
                df['CLNSIG'].notna() & 
                (df['CLNSIG'] != '') & 
                (df['CLNSIG'] != '.') &
                (df['CLNSIG'] != 'nan')
            ].copy()
            
            print(f"    After CLNSIG filtering: {len(df_filtered):,} variants")
            
            if len(df_filtered) == 0:
                print(f"    WARNING: No variants with valid CLNSIG found")
                return None
            
            # Create ML_Class
            df_filtered['ML_Class'] = df_filtered['CLNSIG'].map(self.ml_class_mapping).fillna('Unknown')
            
            # Show class distribution
            class_counts = df_filtered['ML_Class'].value_counts()
            print(f"    ML_Class distribution:")
            for class_name, count in class_counts.items():
                print(f"      {class_name}: {count:,}")
            
            # Filter to Pathogenic/Benign only
            df_binary = df_filtered[df_filtered['ML_Class'].isin(['Pathogenic', 'Benign'])].copy()
            print(f"    Binary classification variants: {len(df_binary):,}")
            
            if len(df_binary) == 0:
                print(f"    WARNING: No Pathogenic/Benign variants found")
                return None
            
            # Add chromosome identifier column if not present
            if 'Chr' not in df_binary.columns:
                df_binary['Chr'] = chr_id
            
            self.stats['total_variants_output'] += len(df_binary)
            return df_binary
            
        except Exception as e:
            print(f"    ERROR: Failed to process {file_path.name}: {e}")
            return None

    def merge_all_chromosomes(self):
        """Merge all chromosome files into features and predictions"""
        print("CLINVAR DATA MERGER")
        print("=" * 50)
        
        chromosome_files = self.find_chromosome_files()
        
        if not chromosome_files:
            print("No chromosome files found. Exiting.")
            return False
        
        print(f"\nMerging {len(chromosome_files)} chromosome files...")
        
        all_dataframes = []
        
        # Process each chromosome file
        for file_path in chromosome_files:
            self.stats['files_processed'] += 1
            
            df_processed = self.process_chromosome_file(file_path)
            
            if df_processed is not None:
                all_dataframes.append(df_processed)
                self.stats['files_successful'] += 1
        
        if not all_dataframes:
            print("ERROR: No valid data found in any files")
            return False
        
        print(f"\nMerging data from {len(all_dataframes)} successful files...")
        
        # Concatenate all dataframes
        merged_df = pd.concat(all_dataframes, ignore_index=True, sort=False)
        print(f"Total merged variants: {len(merged_df):,}")
        
        # Handle duplicate columns by keeping first occurrence
        if merged_df.columns.duplicated().any():
            print("Removing duplicate column names...")
            merged_df = merged_df.loc[:, ~merged_df.columns.duplicated()]
            print(f"Columns after deduplication: {len(merged_df.columns)}")
        
        # Show final class distribution
        final_class_counts = merged_df['ML_Class'].value_counts()
        print(f"\nFinal ML_Class distribution:")
        for class_name, count in final_class_counts.items():
            print(f"  {class_name}: {count:,} ({count/len(merged_df)*100:.1f}%)")
            self.stats['class_distribution'][class_name] = count
        
        # Create predictions dataframe (variant identifiers + ML_Class)
        pred_cols = [col for col in self.identifier_features + ['ML_Class'] if col in merged_df.columns]
        predictions_df = merged_df[pred_cols].copy()
        
        # Create features dataframe (everything except identifiers, CLNSIG, ML_Class, and excluded features)
        exclude_for_features = self.identifier_features + ['CLNSIG', 'ML_Class'] + self.exclude_features
        feature_cols = [col for col in merged_df.columns if col not in exclude_for_features]
        features_df = merged_df[feature_cols].copy()
        
        self.stats['feature_count'] = len(feature_cols)
        
        print(f"\nDataset split:")
        print(f"  Features: {features_df.shape[1]} columns, {features_df.shape[0]} rows")
        print(f"  Predictions: {predictions_df.shape[1]} columns, {predictions_df.shape[0]} rows")
        
        # Save files
        features_file = self.output_dir / "all_chromosomes_features.csv"
        predictions_file = self.output_dir / "all_chromosomes_predictions.csv"
        
        print(f"\nSaving merged files...")
        features_df.to_csv(features_file, index=False)
        predictions_df.to_csv(predictions_file, index=False)
        
        features_size_mb = features_file.stat().st_size / 1024 / 1024
        predictions_size_mb = predictions_file.stat().st_size / 1024 / 1024
        
        print(f"  Features saved: {features_file.name} ({features_size_mb:.1f} MB)")
        print(f"  Predictions saved: {predictions_file.name} ({predictions_size_mb:.1f} MB)")
        
        # Create summary
        self.create_summary_report()
        
        return True

    def create_summary_report(self):
        """Create summary report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_file = self.output_dir / f"merge_summary_{timestamp}.txt"
        
        with open(summary_file, 'w') as f:
            f.write("CLINVAR DATA MERGER SUMMARY\n")
            f.write("=" * 50 + "\n")
            f.write(f"Generated: {datetime.now()}\n")
            f.write(f"Input directory: {self.input_dir.absolute()}\n")
            f.write(f"Output directory: {self.output_dir.absolute()}\n\n")
            
            f.write("PROCESSING STATISTICS:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Files processed: {self.stats['files_processed']}\n")
            f.write(f"Files successful: {self.stats['files_successful']}\n")
            f.write(f"Files failed: {self.stats['files_processed'] - self.stats['files_successful']}\n\n")
            
            f.write("VARIANT STATISTICS:\n")
            f.write("-" * 25 + "\n")
            f.write(f"Total input variants: {self.stats['total_variants_input']:,}\n")
            f.write(f"Total output variants: {self.stats['total_variants_output']:,}\n")
            
            if self.stats['total_variants_input'] > 0:
                retention_rate = self.stats['total_variants_output'] / self.stats['total_variants_input'] * 100
                f.write(f"Retention rate: {retention_rate:.1f}%\n\n")
            
            f.write("CLASS DISTRIBUTION:\n")
            f.write("-" * 25 + "\n")
            for class_name, count in self.stats['class_distribution'].items():
                f.write(f"  {class_name}: {count:,}\n")
            f.write(f"\nTotal features: {self.stats['feature_count']}\n")
            
            f.write("\nOUTPUT FILES:\n")
            f.write("-" * 15 + "\n")
            f.write("all_chromosomes_features.csv - Complete feature matrix for ML\n")
            f.write("all_chromosomes_predictions.csv - Variant IDs + ML_Class labels\n")
            
            f.write("\nREADY FOR MACHINE LEARNING:\n")
            f.write("-" * 30 + "\n")
            f.write("import pandas as pd\n")
            f.write("X = pd.read_csv('all_chromosomes_features.csv')\n")
            f.write("y_df = pd.read_csv('all_chromosomes_predictions.csv')\n")
            f.write("y = (y_df['ML_Class'] == 'Pathogenic').astype(int)\n")
        
        print(f"Summary report saved: {summary_file.name}")


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Merge chromosome files with ClinVar data into ML-ready format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python merge_clinvar.py                           # Use default directories
  python merge_clinvar.py Annovar_Merge Output     # Specify directories
        """
    )
    
    parser.add_argument(
        'input_dir',
        nargs='?',
        default='Annovar_Merge',
        help='Input directory containing chromosome CSV files (default: Annovar_Merge)'
    )
    
    parser.add_argument(
        'output_dir',
        nargs='?', 
        default='Clinvar_Dataset',
        help='Output directory for merged files (default: MachineLearningInput)'
    )
    
    return parser.parse_args()


def main():
    """Main execution function"""
    args = parse_arguments()
    
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    
    merger = ClinVarMerger(input_dir=args.input_dir, output_dir=args.output_dir)
    
    try:
        success = merger.merge_all_chromosomes()
        
        if success:
            print(f"\nSUCCESS: ClinVar data merged successfully!")
            print(f"Files created in: {args.output_dir}")
            print("  - all_chromosomes_features.csv")
            print("  - all_chromosomes_predictions.csv")
            print("\nReady for machine learning!")
        else:
            print(f"\nFAILED: Could not merge ClinVar data")
            
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()