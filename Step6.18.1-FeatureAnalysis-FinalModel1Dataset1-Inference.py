#!/usr/bin/env python3
"""
ClinVar XGBoost Inference Script
Reads chr*_features_engineered.csv files and makes predictions using saved XGBoost models
Outputs results in dbNSFP format
"""

import pandas as pd
import numpy as np
import pickle
import json
import argparse
from pathlib import Path
import gc
import warnings
from datetime import datetime
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

class ClinVarInference:
    def __init__(self, features_dir="Clinvar_Dataset2/Features", 
                 input_dir="Final_Results", 
                 output_dir="Clinvar_Dataset2/predictions"):
        self.features_dir = Path(features_dir)
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.label_encoder = None
        self.metadata = None
        
    def find_available_models(self):
        """Find all available trained models"""
        if not self.features_dir.exists():
            print(f"Features directory not found: {self.features_dir}")
            return []
        
        model_dirs = [d for d in self.features_dir.iterdir() 
                     if d.is_dir() and (d / "xgboost_clinvar_model.pkl").exists()]
        
        print(f"Found {len(model_dirs)} trained models:")
        for model_dir in sorted(model_dirs):
            print(f"  {model_dir.name}")
        
        return sorted(model_dirs)
    
    def load_best_model(self, model_dirs=None):
        """Load the best model based on test MCC score"""
        if model_dirs is None:
            model_dirs = self.find_available_models()
        
        if not model_dirs:
            print("No trained models found!")
            return False
        
        best_model_dir = None
        best_mcc = -1
        
        # Find model with best test MCC
        for model_dir in model_dirs:
            metadata_path = model_dir / "model_metadata.json"
            if metadata_path.exists():
                try:
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    
                    test_mcc = metadata['performance']['test_metrics']['mcc']
                    print(f"  {model_dir.name}: Test MCC = {test_mcc:.4f}")
                    
                    if test_mcc > best_mcc:
                        best_mcc = test_mcc
                        best_model_dir = model_dir
                        
                except Exception as e:
                    print(f"  Error reading metadata for {model_dir.name}: {e}")
                    continue
        
        if best_model_dir is None:
            print("No valid model found, using first available model")
            best_model_dir = model_dirs[0]
        
        print(f"\nUsing best model: {best_model_dir.name} (Test MCC: {best_mcc:.4f})")
        
        # Load model components
        return self.load_model_components(best_model_dir)
    
    def load_model_components(self, model_dir):
        """Load all model components from a directory"""
        try:
            print(f"Loading model components from {model_dir}...")
            
            # Load model
            model_path = model_dir / "xgboost_clinvar_model.pkl"
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            print(f"  ✓ Model loaded")
            
            # Load scaler
            scaler_path = model_dir / "scaler.pkl"
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            print(f"  ✓ Scaler loaded")
            
            # Load feature names
            features_path = model_dir / "feature_names.json"
            with open(features_path, 'r') as f:
                self.feature_names = json.load(f)
            print(f"  ✓ Feature names loaded ({len(self.feature_names)} features)")
            
            # Load label encoder
            label_encoder_path = model_dir / "label_encoder.pkl"
            with open(label_encoder_path, 'rb') as f:
                self.label_encoder = pickle.load(f)
            print(f"  ✓ Label encoder loaded")
            
            # Load metadata
            metadata_path = model_dir / "model_metadata.json"
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
            print(f"  ✓ Metadata loaded")
            
            return True
            
        except Exception as e:
            print(f"Error loading model components: {e}")
            return False
    
    def find_chromosome_files(self):
        """Find all chr*_features_engineered.csv files"""
        pattern = "chr*_features_engineered.csv"
        files = list(self.input_dir.glob(pattern))
        
        print(f"Found {len(files)} chromosome files:")
        for file in sorted(files):
            size_mb = file.stat().st_size / (1024 * 1024)
            print(f"  {file.name} ({size_mb:.1f} MB)")
        
        return sorted(files)
    
    def extract_chromosome_info(self, file_path):
        """Extract chromosome number from filename"""
        # Extract chromosome from filename like chr1_features_engineered.csv
        filename = file_path.name
        chr_part = filename.split('_')[0]  # Get 'chr1' part
        chr_num = chr_part.replace('chr', '')  # Remove 'chr' to get '1'
        return chr_num
    
    def get_essential_columns(self, df):
        """Extract essential columns for output format"""
        essential_cols = {}
        
        # Map common column names to standard format
        column_mappings = {
            'chr': ['chr', 'chromosome', 'CHROM', '#CHROM'],
            'pos': ['pos', 'position', 'POS', 'pos(1-based)', 'start', 'Start'],
            'ref': ['ref', 'reference', 'REF', 'ref_allele', 'Ref'],
            'alt': ['alt', 'alternate', 'ALT', 'alt_allele', 'Alt']
        }
        
        for standard_name, possible_names in column_mappings.items():
            found_col = None
            for col_name in possible_names:
                if col_name in df.columns:
                    found_col = col_name
                    break
            
            if found_col:
                essential_cols[standard_name] = df[found_col]
                print(f"    Found {standard_name} column: {found_col}")
            else:
                essential_cols[standard_name] = None
                print(f"    Warning: {standard_name} column not found")
        
        return essential_cols
    
    def smart_feature_mapping(self, df_columns):
        """Create smart mapping between model features and available columns"""
        if hasattr(self, '_feature_mapping'):
            return self._feature_mapping
            
        print(f"    Creating smart feature mapping...")
        print(f"    Model expects {len(self.feature_names)} features")
        print(f"    Data has {len(df_columns)} columns")
        
        feature_mapping = {}
        exact_matches = 0
        fuzzy_matches = 0
        missing_features = []
        
        # First pass: exact matches
        for feature in self.feature_names:
            if feature in df_columns:
                feature_mapping[feature] = feature
                exact_matches += 1
        
        # Second pass: fuzzy matching for missing features
        remaining_features = [f for f in self.feature_names if f not in feature_mapping]
        available_columns = [c for c in df_columns if c not in feature_mapping.values()]
        
        for feature in remaining_features:
            best_match = self.find_best_column_match(feature, available_columns)
            if best_match:
                feature_mapping[feature] = best_match
                available_columns.remove(best_match)
                fuzzy_matches += 1
            else:
                missing_features.append(feature)
        
        print(f"    Feature mapping results:")
        print(f"      Exact matches: {exact_matches}")
        print(f"      Fuzzy matches: {fuzzy_matches}")
        print(f"      Missing features: {len(missing_features)}")
        
        if missing_features and len(missing_features) <= 10:
            print(f"      Missing: {missing_features}")
        elif missing_features:
            print(f"      Missing: {missing_features[:5]} ... and {len(missing_features)-5} more")
        
        # Store mapping for reuse
        self._feature_mapping = feature_mapping
        self._missing_features = missing_features
        
        return feature_mapping
    
    def find_best_column_match(self, target_feature, available_columns):
        """Find the best matching column for a target feature"""
        # Normalize target feature name
        target_clean = self.normalize_column_name(target_feature)
        
        best_match = None
        best_score = 0
        
        for col in available_columns:
            col_clean = self.normalize_column_name(col)
            
            # Calculate similarity score
            score = self.calculate_similarity(target_clean, col_clean)
            
            if score > best_score and score > 0.7:  # Minimum similarity threshold
                best_score = score
                best_match = col
        
        return best_match
    
    def normalize_column_name(self, name):
        """Normalize column name for comparison"""
        import re
        
        # Convert to lowercase
        clean_name = str(name).lower()
        
        # Remove common prefixes/suffixes
        prefixes_to_remove = ['feat_', 'feature_', 'col_', 'var_', 'eng_', 'final_']
        suffixes_to_remove = ['_feat', '_feature', '_col', '_var', '_eng', '_final']
        
        for prefix in prefixes_to_remove:
            if clean_name.startswith(prefix):
                clean_name = clean_name[len(prefix):]
                break
        
        for suffix in suffixes_to_remove:
            if clean_name.endswith(suffix):
                clean_name = clean_name[:-len(suffix)]
                break
        
        # Replace separators with underscores
        clean_name = re.sub(r'[.\-\s]+', '_', clean_name)
        
        # Remove multiple underscores
        clean_name = re.sub(r'_+', '_', clean_name)
        
        # Remove leading/trailing underscores
        clean_name = clean_name.strip('_')
        
        return clean_name
    
    def calculate_similarity(self, str1, str2):
        """Calculate similarity between two strings"""
        # Exact match
        if str1 == str2:
            return 1.0
        
        # Substring match
        if str1 in str2 or str2 in str1:
            return 0.9
        
        # Jaccard similarity on character n-grams
        def get_ngrams(s, n=3):
            return set([s[i:i+n] for i in range(len(s)-n+1)])
        
        ngrams1 = get_ngrams(str1, 3)
        ngrams2 = get_ngrams(str2, 3)
        
        if not ngrams1 and not ngrams2:
            return 1.0 if str1 == str2 else 0.0
        
        if not ngrams1 or not ngrams2:
            return 0.0
        
        intersection = len(ngrams1.intersection(ngrams2))
        union = len(ngrams1.union(ngrams2))
        
        return intersection / union if union > 0 else 0.0
    
    def prepare_features_for_inference(self, df, chunk_idx=None):
        """Prepare chunk data for inference with smart feature mapping"""
        # Create feature mapping on first chunk
        if chunk_idx == 0 or not hasattr(self, '_feature_mapping'):
            feature_mapping = self.smart_feature_mapping(df.columns.tolist())
        else:
            feature_mapping = self._feature_mapping
        
        # Create feature matrix in correct order
        feature_data = {}
        
        for required_feature in self.feature_names:
            if required_feature in feature_mapping:
                # Use mapped column
                mapped_column = feature_mapping[required_feature]
                feature_data[required_feature] = df[mapped_column]
            else:
                # Feature is missing, use default value
                feature_data[required_feature] = 0
                
        # Create DataFrame with features in correct order
        X = pd.DataFrame(feature_data, columns=self.feature_names)
        
        # Handle missing values
        missing_values = [np.inf, -np.inf, np.nan, 'nan', 'NaN', 'NA', 'na', 'NULL', 'null', '', ' ', 'n/a', 'N/A']
        X = X.replace(missing_values, np.nan)
        
        # Convert to numeric and fill NaN
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
        
        # Final checks
        if X.isnull().any().any():
            X = X.fillna(0)
        
        if np.isinf(X).any().any():
            X = X.replace([np.inf, -np.inf], 0)
        
        return X
    
    def make_predictions(self, X):
        """Make predictions using loaded model"""
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Make predictions
        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)
        
        # Decode predictions to class names
        predicted_classes = self.label_encoder.inverse_transform(predictions)
        
        # Create results
        results = {
            'predictions': predicted_classes,
            'probabilities': probabilities,
            'pathogenic_prob': probabilities[:, 1],  # Probability of pathogenic class
            'confidence': np.max(probabilities, axis=1)
        }
        
        return results
    
    def format_output_chunk(self, essential_cols, results, chr_num, chunk_idx, total_samples):
        """Format predictions into dbNSFP format"""
        output_rows = []
        
        for i in range(len(results['predictions'])):
            # Get essential information
            chr_val = chr_num
            pos_val = essential_cols['pos'][i] if essential_cols['pos'] is not None else "*"
            ref_val = essential_cols['ref'][i] if essential_cols['ref'] is not None else "*"
            alt_val = essential_cols['alt'][i] if essential_cols['alt'] is not None else "*"
            
            # Handle missing values in essential columns
            if pd.isna(pos_val):
                pos_val = "*"
            if pd.isna(ref_val) or ref_val == "":
                ref_val = "*"
            if pd.isna(alt_val) or alt_val == "":
                alt_val = "*"
            
            # Get prediction information
            pathogenic_prob = results['pathogenic_prob'][i]
            confidence = results['confidence'][i]
            pred_class = results['predictions'][i]
            
            # Create output row
            row = [
                str(chr_val),                    # chr
                str(pos_val),                    # pos(1-based)
                str(ref_val),                    # ref
                str(alt_val),                    # alt
                f"{pathogenic_prob:.6f}",        # score (pathogenic probability)
                f"{confidence:.6f}",             # sd (confidence as proxy for standard deviation)
                str(pred_class),                 # pred (Benign/Pathogenic)
                f"chunk_{chunk_idx}"             # comments
            ]
            
            output_rows.append('\t'.join(row))
        
        return output_rows
    
    def process_chromosome_file(self, file_path, chunk_size=10000):
        """Process a single chromosome file in chunks"""
        chr_num = self.extract_chromosome_info(file_path)
        print(f"\nProcessing chromosome {chr_num}: {file_path.name}")
        
        # Get file size info
        try:
            with open(file_path, 'r') as f:
                total_lines = sum(1 for _ in f) - 1  # Subtract header
            size_mb = file_path.stat().st_size / (1024 * 1024)
            print(f"  File info: {total_lines:,} rows, {size_mb:.1f} MB")
        except Exception as e:
            print(f"  Error reading file info: {e}")
            return
        
        # Calculate number of chunks
        n_chunks = (total_lines + chunk_size - 1) // chunk_size
        print(f"  Processing in {n_chunks} chunks of {chunk_size:,} rows each")
        
        # Create output file
        output_file = self.output_dir / f"dbNSFP4_nsSNV.chr{chr_num}.txt"
        
        # Write header
        header = "#chr\tpos(1-based)\tref\talt\tscore\tsd\tpred\tcomments\n"
        with open(output_file, 'w') as f:
            f.write(header)
        
        total_predictions = 0
        prediction_summary = {'Benign': 0, 'Pathogenic': 0}
        
        try:
            # Process file in chunks
            chunk_reader = pd.read_csv(file_path, chunksize=chunk_size)
            
            for chunk_idx, chunk_df in enumerate(chunk_reader):
                if chunk_idx == 0 or chunk_idx % 10 == 0:  # Print progress every 10 chunks
                    print(f"    Processing chunk {chunk_idx + 1}/{n_chunks} ({len(chunk_df):,} rows)...")
                
                try:
                    # Extract essential columns
                    essential_cols = self.get_essential_columns(chunk_df)
                    
                    # Prepare features for inference
                    X = self.prepare_features_for_inference(chunk_df, chunk_idx)
                    
                    # Make predictions
                    results = self.make_predictions(X)
                    
                    # Format output
                    output_lines = self.format_output_chunk(
                        essential_cols, results, chr_num, chunk_idx, len(chunk_df)
                    )
                    
                    # Write to file
                    with open(output_file, 'a') as f:
                        for line in output_lines:
                            f.write(line + '\n')
                    
                    # Update statistics
                    total_predictions += len(results['predictions'])
                    unique, counts = np.unique(results['predictions'], return_counts=True)
                    for pred_class, count in zip(unique, counts):
                        prediction_summary[pred_class] += count
                    
                    # Clean up memory
                    del chunk_df, X, results, output_lines
                    gc.collect()
                    
                except Exception as e:
                    print(f"      Error processing chunk {chunk_idx}: {e}")
                    continue
        
        except Exception as e:
            print(f"  Error reading file: {e}")
            return
        
        # Print summary
        print(f"  ✓ Completed chromosome {chr_num}")
        print(f"    Total predictions: {total_predictions:,}")
        print(f"    Benign: {prediction_summary.get('Benign', 0):,} ({prediction_summary.get('Benign', 0)/total_predictions*100:.1f}%)")
        print(f"    Pathogenic: {prediction_summary.get('Pathogenic', 0):,} ({prediction_summary.get('Pathogenic', 0)/total_predictions*100:.1f}%)")
        print(f"    Output saved: {output_file}")
        
        return total_predictions, prediction_summary
    
    def validate_output_format(self, output_file, n_samples=5):
        """Validate output file format"""
        try:
            print(f"\n  Validating output format: {output_file.name}")
            
            with open(output_file, 'r') as f:
                lines = [f.readline().strip() for _ in range(n_samples + 1)]
            
            # Check header
            expected_header = "#chr\tpos(1-based)\tref\talt\tscore\tsd\tpred\tcomments"
            if lines[0] == expected_header:
                print("    ✓ Header format correct")
            else:
                print("    ⚠ Header format mismatch")
                print(f"    Expected: {expected_header}")
                print(f"    Got:      {lines[0]}")
            
            # Show sample data
            print("    Sample rows:")
            for i, line in enumerate(lines[1:], 1):
                if line:
                    parts = line.split('\t')
                    print(f"      Row {i}: chr={parts[0]}, pos={parts[1]}, ref={parts[2]}, alt={parts[3]}, score={parts[4]}, pred={parts[6]}")
            
        except Exception as e:
            print(f"    Error validating output: {e}")
    
    def run_inference(self, chromosomes=None, chunk_size=10000):
        """Run complete inference pipeline"""
        print("CLINVAR XGBOOST INFERENCE PIPELINE")
        print("=" * 60)
        print(f"Input directory: {self.input_dir}")
        print(f"Output directory: {self.output_dir}")
        print(f"Chunk size: {chunk_size:,}")
        print("=" * 60)
        
        # Load best model
        if not self.load_best_model():
            print("Failed to load model!")
            return
        
        print(f"\nModel Information:")
        if self.metadata:
            print(f"  Training fold: {self.metadata['data_info']['fold_name']}")
            print(f"  Training samples: {self.metadata['data_info']['n_samples']:,}")
            print(f"  Features: {self.metadata['data_info']['n_features']:,}")
            print(f"  Test MCC: {self.metadata['performance']['test_metrics']['mcc']:.4f}")
            print(f"  Test AUC: {self.metadata['performance']['test_metrics']['auc_roc']:.4f}")
        
        # Find chromosome files
        chromosome_files = self.find_chromosome_files()
        
        if not chromosome_files:
            print("No chromosome files found!")
            return
        
        # Filter by specific chromosomes if requested
        if chromosomes:
            filtered_files = []
            for file in chromosome_files:
                chr_num = self.extract_chromosome_info(file)
                if chr_num in [str(c) for c in chromosomes]:
                    filtered_files.append(file)
            chromosome_files = filtered_files
            print(f"Filtered to {len(chromosome_files)} files for chromosomes: {chromosomes}")
        
        # Process each chromosome
        total_processed = 0
        overall_summary = {'Benign': 0, 'Pathogenic': 0}
        successful_files = 0
        
        for file_idx, file_path in enumerate(chromosome_files):
            try:
                predictions_count, pred_summary = self.process_chromosome_file(file_path, chunk_size)
                
                if predictions_count and predictions_count > 0:
                    total_processed += predictions_count
                    successful_files += 1
                    
                    # Update overall summary
                    for class_name, count in pred_summary.items():
                        overall_summary[class_name] += count
                    
                    # Validate first file
                    if file_idx == 0:
                        chr_num = self.extract_chromosome_info(file_path)
                        output_file = self.output_dir / f"dbNSFP4_nsSNV.chr{chr_num}.txt"
                        self.validate_output_format(output_file)
                
            except Exception as e:
                print(f"Error processing {file_path.name}: {e}")
                continue
        
        # Final summary
        print(f"\n" + "=" * 60)
        print("INFERENCE COMPLETED")
        print("=" * 60)
        print(f"Files processed: {successful_files}/{len(chromosome_files)}")
        print(f"Total predictions: {total_processed:,}")
        print(f"Overall results:")
        if total_processed > 0:
            benign_count = overall_summary.get('Benign', 0)
            pathogenic_count = overall_summary.get('Pathogenic', 0)
            print(f"  Benign: {benign_count:,} ({benign_count/total_processed*100:.1f}%)")
            print(f"  Pathogenic: {pathogenic_count:,} ({pathogenic_count/total_processed*100:.1f}%)")
        
        print(f"\nOutput files saved in: {self.output_dir}")
        output_files = list(self.output_dir.glob("dbNSFP4_nsSNV.chr*.txt"))
        for output_file in sorted(output_files):
            size_mb = output_file.stat().st_size / (1024 * 1024)
            print(f"  {output_file.name} ({size_mb:.1f} MB)")
        
        if successful_files == len(chromosome_files):
            print(f"\n✓ All files processed successfully!")
        else:
            print(f"\n⚠ {len(chromosome_files) - successful_files} files failed")
        
        return total_processed, overall_summary


def main():
    parser = argparse.ArgumentParser(description='ClinVar XGBoost Inference')
    parser.add_argument('--features-dir', type=str, default='Clinvar_Dataset2/Features',
                       help='Directory containing saved models')
    parser.add_argument('--input-dir', type=str, default='Final_Results',
                       help='Directory containing chr*_features_engineered.csv files')
    parser.add_argument('--output-dir', type=str, default='Clinvar_Dataset2/predictions',
                       help='Output directory for predictions')
    parser.add_argument('--chunk-size', type=int, default=10000,
                       help='Chunk size for processing large files')
    parser.add_argument('--chromosomes', nargs='+', type=str,
                       help='Specific chromosomes to process (e.g., --chromosomes 1 2 X)')
    
    args = parser.parse_args()
    
    # Initialize inference pipeline
    inference = ClinVarInference(
        features_dir=args.features_dir,
        input_dir=args.input_dir,
        output_dir=args.output_dir
    )
    
    # Run inference
    total_processed, summary = inference.run_inference(
        chromosomes=args.chromosomes,
        chunk_size=args.chunk_size
    )
    
    return total_processed, summary


if __name__ == "__main__":
    main()