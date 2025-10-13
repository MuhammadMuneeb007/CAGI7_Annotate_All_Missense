#!/usr/bin/env python3
"""
Chunked Inference script for XGBoost ClinVar model
Processes chr*_features_engineered.csv files in chunks and saves predictions in dbNSFP format
"""

import pandas as pd
import numpy as np
import pickle
import json
import argparse
from pathlib import Path
import gc
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def load_model_and_features(model_dir="Final_Dataset/Features"):
    """Load the trained model and required features"""
    
    model_dir = Path(model_dir)
    
    print(f"Loading model from {model_dir}...")
    
    # Load the trained model
    model_path = model_dir / "xgboost_clinvar_model.pkl"
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    print(f"✓ Model loaded from: {model_path}")
    
    # Load feature names
    features_path = model_dir / "feature_names.json"
    with open(features_path, "r") as f:
        feature_names = json.load(f)
    print(f"✓ Feature names loaded: {len(feature_names)} features")
    
    # Load label encoder
    label_encoder_path = model_dir / "label_encoder.pkl"
    with open(label_encoder_path, "rb") as f:
        label_encoder = pickle.load(f)
    print(f"✓ Label encoder loaded")
    
    # Load metadata
    metadata_path = model_dir / "model_metadata.json"
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    print(f"✓ Model metadata loaded")
    
    return model, feature_names, label_encoder, metadata

def find_feature_files(input_dir="Final_Results"):
    """Find all chr*_features_engineered.csv files"""
    input_path = Path(input_dir)
    feature_files = list(input_path.glob("chr*_features_engineered.csv"))
    
    print(f"Found {len(feature_files)} feature files:")
    for file in sorted(feature_files):
        # Get file size in MB
        size_mb = file.stat().st_size / (1024 * 1024)
        print(f"  {file.name} ({size_mb:.1f} MB)")
    
    return sorted(feature_files)

def get_file_info(file_path):
    """Get basic information about the file"""
    try:
        # Read just the header to get column info
        df_header = pd.read_csv(file_path, nrows=0)
        columns = list(df_header.columns)
        
        # Get actual row count efficiently
        with open(file_path, 'r') as f:
            row_count = sum(1 for _ in f) - 1  # subtract header
        
        # Get file size
        size_mb = file_path.stat().st_size / (1024 * 1024)
        
        return {
            'columns': columns,
            'n_columns': len(columns),
            'n_rows': row_count,
            'size_mb': size_mb
        }
    except Exception as e:
        print(f"Error reading {file_path.name}: {e}")
        return None

def smart_feature_mapping(df_columns, feature_names, chunk_idx=0):
    """Create smart mapping between model features and available columns"""
    
    if chunk_idx == 0:  # Only print detailed info for first chunk
        print(f"  Creating smart feature mapping...")
        print(f"  Model expects {len(feature_names)} features")
        print(f"  Data has {len(df_columns)} columns")
    
    feature_mapping = {}
    exact_matches = 0
    fuzzy_matches = 0
    missing_features = []
    
    # First pass: exact matches
    for feature in feature_names:
        if feature in df_columns:
            feature_mapping[feature] = feature
            exact_matches += 1
    
    # Second pass: fuzzy matching for missing features
    remaining_features = [f for f in feature_names if f not in feature_mapping]
    available_columns = [c for c in df_columns if c not in feature_mapping.values()]
    
    for feature in remaining_features:
        best_match = find_best_column_match(feature, available_columns)
        if best_match:
            feature_mapping[feature] = best_match
            available_columns.remove(best_match)
            fuzzy_matches += 1
        else:
            missing_features.append(feature)
    
    if chunk_idx == 0:  # Only print for first chunk
        print(f"  Feature mapping results:")
        print(f"    Exact matches: {exact_matches}")
        print(f"    Fuzzy matches: {fuzzy_matches}")
        print(f"    Missing features: {len(missing_features)}")
        
        if missing_features and len(missing_features) <= 10:
            print(f"    Missing: {missing_features}")
        elif missing_features:
            print(f"    Missing: {missing_features[:5]} ... and {len(missing_features)-5} more")
    
    return feature_mapping, missing_features

def find_best_column_match(target_feature, available_columns):
    """Find the best matching column for a target feature"""
    # Normalize target feature name
    target_clean = normalize_column_name(target_feature)
    
    best_match = None
    best_score = 0
    
    for col in available_columns:
        col_clean = normalize_column_name(col)
        
        # Calculate similarity score
        score = calculate_similarity(target_clean, col_clean)
        
        if score > best_score and score > 0.7:  # Minimum similarity threshold
            best_score = score
            best_match = col
    
    return best_match

def normalize_column_name(name):
    """Normalize column name for comparison"""
    import re
    
    # Convert to lowercase
    clean_name = str(name).lower()
    
    # Remove common prefixes/suffixes
    prefixes_to_remove = ['feat_', 'feature_', 'col_', 'var_', 'eng_', 'final_', 'processed_', 'dbsnfp_']
    suffixes_to_remove = ['_feat', '_feature', '_col', '_var', '_eng', '_final', '_processed', '_score', '_pred']
    
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

def calculate_similarity(str1, str2):
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

def prepare_chunk_for_inference(chunk_df, feature_names, replace_nan_value=0, chunk_idx=0):
    """Prepare a chunk of data for inference using smart feature mapping"""
    
    # Keep essential columns for output format
    essential_cols = ['chr', 'pos', 'ref', 'alt']
    
    # Alternative column names that might be used
    alt_col_names = {
        'chr': ['chromosome', 'CHROM', '#CHROM'],
        'pos': ['position', 'POS', 'pos(1-based)', 'start'],
        'ref': ['reference', 'REF', 'ref_allele'],
        'alt': ['alternate', 'ALT', 'alt_allele']
    }
    
    # Initialize essential_data with all required keys
    essential_data = {col: None for col in essential_cols}
    
    # Find available essential columns with alternative names
    for col in essential_cols:
        found_data = None
        
        # Try exact match first
        if col in chunk_df.columns:
            found_data = chunk_df[col].copy()
        else:
            # Try alternative names
            for alt_name in alt_col_names.get(col, []):
                if alt_name in chunk_df.columns:
                    found_data = chunk_df[alt_name].copy()
                    break
        
        # Store the found data (or None if not found)
        essential_data[col] = found_data
        
        # Only warn for first chunk
        if found_data is None and chunk_idx == 0:
            print(f"    Warning: Column '{col}' not found in data")
    
    # Create smart feature mapping
    try:
        feature_mapping, missing_features = smart_feature_mapping(
            chunk_df.columns.tolist(), feature_names, chunk_idx
        )
    except Exception as e:
        if chunk_idx == 0:
            print(f"    Error in feature mapping: {e}")
        # Fallback: use available features
        feature_mapping = {f: f for f in feature_names if f in chunk_df.columns}
        missing_features = [f for f in feature_names if f not in chunk_df.columns]
    
    # Create feature matrix in correct order
    feature_data = {}
    
    for required_feature in feature_names:
        if required_feature in feature_mapping:
            # Use mapped column
            mapped_column = feature_mapping[required_feature]
            try:
                feature_data[required_feature] = chunk_df[mapped_column]
            except KeyError:
                feature_data[required_feature] = replace_nan_value
        else:
            # Feature is missing, use default value
            feature_data[required_feature] = replace_nan_value
    
    # Create DataFrame with features in correct order
    try:
        X = pd.DataFrame(feature_data, columns=feature_names)
    except Exception as e:
        if chunk_idx == 0:
            print(f"    Error creating feature DataFrame: {e}")
        # Fallback: create with default values
        X = pd.DataFrame({f: [replace_nan_value] * len(chunk_df) for f in feature_names})
    
    # Apply same preprocessing as training
    # Handle missing values
    missing_values = [np.inf, -np.inf, np.nan, 'nan', 'NaN', 'NA', 'na', 'NULL', 'null', '', ' ', 'n/a', 'N/A']
    X = X.replace(missing_values, np.nan)
    
    # Convert to numeric
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce').fillna(replace_nan_value)
    
    # Final verification
    if X.isnull().any().any():
        if chunk_idx == 0:  # Only warn for first chunk
            print(f"    Warning: Still have NaN values after preprocessing, filling with {replace_nan_value}")
        X = X.fillna(replace_nan_value)
    
    if np.isinf(X).any().any():
        if chunk_idx == 0:  # Only warn for first chunk
            print(f"    Warning: Still have infinite values after preprocessing, replacing with {replace_nan_value}")
        X = X.replace([np.inf, -np.inf], replace_nan_value)
    
    return X, essential_data

def make_chunk_predictions(model, X, label_encoder):
    """Make predictions for a chunk of data"""
    
    # Get predictions and probabilities
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)
    
    # Decode predictions to original class names
    predicted_classes = label_encoder.inverse_transform(predictions)
    
    # Create results DataFrame
    results = pd.DataFrame({
        'prediction': predicted_classes,
        'prediction_encoded': predictions,
        'probability_benign': probabilities[:, 0],
        'probability_pathogenic': probabilities[:, 1],
        'confidence': np.max(probabilities, axis=1)
    })
    
    return results

def format_output_chunk(predictions_df, essential_data, chunk_idx, chr_from_filename):
    """Format chunk predictions to dbNSFP-like format"""
    
    output_rows = []
    
    for i in range(len(predictions_df)):
        # Extract essential information safely using .get() method
        # Use chromosome from filename if chr column not found
        chr_data = essential_data.get('chr', None)
        if chr_data is not None:
            try:
                if hasattr(chr_data, 'iloc'):
                    chr_val = chr_data.iloc[i]
                elif hasattr(chr_data, '__getitem__'):
                    chr_val = chr_data[i]
                else:
                    chr_val = chr_data
            except (IndexError, KeyError, TypeError):
                chr_val = chr_from_filename
        else:
            chr_val = chr_from_filename
        
        # Handle position
        pos_data = essential_data.get('pos', None)
        if pos_data is not None:
            try:
                if hasattr(pos_data, 'iloc'):
                    pos_val = pos_data.iloc[i]
                elif hasattr(pos_data, '__getitem__'):
                    pos_val = pos_data[i]
                else:
                    pos_val = pos_data
            except (IndexError, KeyError, TypeError):
                pos_val = "*"
        else:
            pos_val = "*"
        
        # Handle reference allele
        ref_data = essential_data.get('ref', None)
        if ref_data is not None:
            try:
                if hasattr(ref_data, 'iloc'):
                    ref_val = ref_data.iloc[i]
                elif hasattr(ref_data, '__getitem__'):
                    ref_val = ref_data[i]
                else:
                    ref_val = ref_data
            except (IndexError, KeyError, TypeError):
                ref_val = "*"
        else:
            ref_val = "*"
        
        # Handle alternate allele
        alt_data = essential_data.get('alt', None)
        if alt_data is not None:
            try:
                if hasattr(alt_data, 'iloc'):
                    alt_val = alt_data.iloc[i]
                elif hasattr(alt_data, '__getitem__'):
                    alt_val = alt_data[i]
                else:
                    alt_val = alt_data
            except (IndexError, KeyError, TypeError):
                alt_val = "*"
        else:
            alt_val = "*"
        
        # Handle missing or NaN values
        if pd.isna(chr_val) or chr_val == "" or chr_val is None:
            chr_val = chr_from_filename
        if pd.isna(pos_val) or pos_val == "" or pos_val is None:
            pos_val = "*"
        if pd.isna(ref_val) or ref_val == "" or ref_val is None:
            ref_val = "*"
        if pd.isna(alt_val) or alt_val == "" or alt_val is None:
            alt_val = "*"
        
        # Get prediction info
        score = predictions_df.iloc[i]['probability_pathogenic']
        confidence = predictions_df.iloc[i]['confidence']
        pred_class = predictions_df.iloc[i]['prediction']
        
        # Create output row
        row = {
            'chr': str(chr_val),
            'pos(1-based)': str(pos_val),
            'ref': str(ref_val),
            'alt': str(alt_val),
            'score': f"{score:.6f}",
            'sd': f"{confidence:.6f}",  # Using confidence as standard deviation proxy
            'pred': str(pred_class),
            'comments': f"chunk_{chunk_idx}"
        }
        
        output_rows.append(row)
    
    return pd.DataFrame(output_rows)

def process_file_in_chunks(file_path, model, feature_names, label_encoder, 
                          output_dir, chunk_size=10000, replace_nan_value=0):
    """Process a single file in chunks and save predictions"""
    
    print(f"\nProcessing {file_path.name}...")
    
    # Get file info
    file_info = get_file_info(file_path)
    if file_info is None:
        print(f"Skipping {file_path.name} due to errors")
        return
    
    print(f"File info: {file_info['n_rows']:,} rows, {file_info['n_columns']} columns, {file_info['size_mb']:.1f} MB")
    
    # Calculate number of chunks
    n_chunks = (file_info['n_rows'] + chunk_size - 1) // chunk_size
    print(f"Processing in {n_chunks} chunks of {chunk_size:,} rows each")
    
    # Extract chromosome from filename
    chr_name = file_path.name.split('_')[0].replace('chr', '')
    
    # Create output file path
    output_file = output_dir / f"chr{chr_name}_predictions.txt"
    
    # Initialize output file with header
    header = "#chr\tpos(1-based)\tref\talt\tscore\tsd\tpred\tcomments\n"
    with open(output_file, 'w') as f:
        f.write(header)
    
    total_predictions = 0
    chunk_predictions_summary = []
    
    try:
        # Process file in chunks
        chunk_reader = pd.read_csv(file_path, chunksize=chunk_size)
        
        for chunk_idx, chunk_df in enumerate(chunk_reader):
            print(f"  Processing chunk {chunk_idx + 1}/{n_chunks} ({len(chunk_df):,} rows)...")
            
            try:
                # Prepare chunk for inference
                X_chunk, essential_data = prepare_chunk_for_inference(
                    chunk_df, feature_names, replace_nan_value, chunk_idx
                )
                
                # Make predictions
                predictions = make_chunk_predictions(model, X_chunk, label_encoder)
                
                # Format output
                output_chunk = format_output_chunk(predictions, essential_data, chunk_idx, chr_name)
                
                # Append to output file (without header)
                output_chunk.to_csv(output_file, mode='a', sep='\t', header=False, index=False)
                
                # Track statistics
                total_predictions += len(predictions)
                chunk_summary = {
                    'chunk': chunk_idx,
                    'rows': len(predictions),
                    'benign': sum(predictions['prediction'] == 'Benign'),
                    'pathogenic': sum(predictions['prediction'] == 'Pathogenic'),
                    'avg_confidence': predictions['confidence'].mean()
                }
                chunk_predictions_summary.append(chunk_summary)
                
                print(f"    Predictions: {chunk_summary['benign']} Benign, {chunk_summary['pathogenic']} Pathogenic")
                print(f"    Avg confidence: {chunk_summary['avg_confidence']:.3f}")
                
                # Clean up memory
                del chunk_df, X_chunk, predictions, output_chunk
                gc.collect()
                
            except Exception as e:
                print(f"    Error processing chunk {chunk_idx}: {e}")
                continue
    
    except Exception as e:
        print(f"Error reading file in chunks: {e}")
        return
    
    # Summary statistics
    print(f"\n✓ Completed {file_path.name}")
    print(f"  Total predictions: {total_predictions:,}")
    if chunk_predictions_summary:
        total_benign = sum(c['benign'] for c in chunk_predictions_summary)
        total_pathogenic = sum(c['pathogenic'] for c in chunk_predictions_summary)
        avg_confidence = np.mean([c['avg_confidence'] for c in chunk_predictions_summary])
        
        print(f"  Final summary: {total_benign:,} Benign ({total_benign/total_predictions*100:.1f}%), "
              f"{total_pathogenic:,} Pathogenic ({total_pathogenic/total_predictions*100:.1f}%)")
        print(f"  Average confidence: {avg_confidence:.3f}")
    
    print(f"  Output saved to: {output_file}")
    
    # Save chunk summary
    summary_file = output_dir / f"chr{chr_name}_chunk_summary.json"
    summary_data = {
        'file_name': file_path.name,
        'total_predictions': total_predictions,
        'chunks_processed': len(chunk_predictions_summary),
        'chunk_size': chunk_size,
        'chunk_summaries': chunk_predictions_summary,
        'processing_timestamp': datetime.now().isoformat()
    }
    
    with open(summary_file, 'w') as f:
        json.dump(summary_data, f, indent=2)
    
    return total_predictions

def validate_output_format(output_file, n_samples=10):
    """Validate that output file has correct format"""
    try:
        print(f"\nValidating output format for {output_file.name}...")
        
        # Read first few lines
        with open(output_file, 'r') as f:
            lines = [f.readline().strip() for _ in range(min(n_samples + 1, 11))]
        
        print("Sample output:")
        for i, line in enumerate(lines):
            if i == 0:
                print(f"Header: {line}")
            else:
                print(f"Row {i}: {line}")
        
        # Check header format
        expected_header = "#chr\tpos(1-based)\tref\talt\tscore\tsd\tpred\tcomments"
        if lines[0] == expected_header:
            print("✓ Header format is correct")
        else:
            print(f"⚠ Header format mismatch")
            print(f"Expected: {expected_header}")
            print(f"Got:      {lines[0]}")
        
        # Count total lines
        with open(output_file, 'r') as f:
            total_lines = sum(1 for _ in f)
        
        print(f"Total lines (including header): {total_lines:,}")
        print(f"Data rows: {total_lines - 1:,}")
        
    except Exception as e:
        print(f"Error validating output: {e}")

def main():
    """Main inference function"""
    
    parser = argparse.ArgumentParser(description='Chunked XGBoost ClinVar Model Inference')
    parser.add_argument('--input-dir', '-i', type=str, default='Final_Results',
                       help='Input directory containing chr*_features_engineered.csv files')
    parser.add_argument('--output-dir', '-o', type=str, default='Final_Dataset/Predictions',
                       help='Output directory for predictions')
    parser.add_argument('--model-dir', type=str, default='Final_Dataset/Features',
                       help='Directory containing the saved model files')
    parser.add_argument('--chunk-size', type=int, default=10000,
                       help='Number of rows to process per chunk (default: 10000)')
    parser.add_argument('--replace-nan', type=float, default=0,
                       help='Value to replace NaN with (default: 0)')
    parser.add_argument('--chromosomes', nargs='+', 
                       help='Specific chromosomes to process (e.g., --chromosomes 1 2 X)')
    
    args = parser.parse_args()
    
    print("="*80)
    print("CHUNKED XGBOOST CLINVAR MODEL INFERENCE")
    print("="*80)
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Model directory: {args.model_dir}")
    print(f"Chunk size: {args.chunk_size:,} rows")
    print(f"Replace NaN with: {args.replace_nan}")
    if args.chromosomes:
        print(f"Specific chromosomes: {args.chromosomes}")
    print("="*80)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model and features
    try:
        model, feature_names, label_encoder, metadata = load_model_and_features(args.model_dir)
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Display model info
    print(f"\nModel Information:")
    print(f"Model type: {metadata['model_info']['model_type']}")
    print(f"Training features: {metadata['data_info']['n_features']}")
    print(f"Classes: {metadata['data_info']['class_names']}")
    print(f"Training samples: {metadata['data_info']['n_samples']:,}")
    print(f"Model trained on: {metadata['training_info']['timestamp']}")
    
    # Find feature files
    feature_files = find_feature_files(args.input_dir)
    
    if not feature_files:
        print("No feature files found!")
        return
    
    # Filter files by chromosome if specified
    if args.chromosomes:
        filtered_files = []
        for file in feature_files:
            chr_name = file.name.split('_')[0].replace('chr', '')
            if chr_name in args.chromosomes:
                filtered_files.append(file)
        feature_files = filtered_files
        print(f"\nFiltered to {len(feature_files)} files for specified chromosomes")
    
    # Process each file
    total_processed = 0
    successful_files = 0
    
    print(f"\nProcessing {len(feature_files)} files...")
    
    for file_idx, file_path in enumerate(feature_files):
        print(f"\n{'='*60}")
        print(f"File {file_idx + 1}/{len(feature_files)}: {file_path.name}")
        print(f"{'='*60}")
        
        try:
            predictions_count = process_file_in_chunks(
                file_path=file_path,
                model=model,
                feature_names=feature_names,
                label_encoder=label_encoder,
                output_dir=output_dir,
                chunk_size=args.chunk_size,
                replace_nan_value=args.replace_nan
            )
            
            if predictions_count and predictions_count > 0:
                total_processed += predictions_count
                successful_files += 1
                
                # Validate output format for first file
                if file_idx == 0:
                    chr_name = file_path.name.split('_')[0].replace('chr', '')
                    output_file = output_dir / f"chr{chr_name}_predictions.txt"
                    validate_output_format(output_file)
            
        except Exception as e:
            print(f"Error processing {file_path.name}: {e}")
            continue
    
    # Final summary
    print(f"\n{'='*80}")
    print("INFERENCE COMPLETED")
    print(f"{'='*80}")
    print(f"Files processed successfully: {successful_files}/{len(feature_files)}")
    print(f"Total predictions made: {total_processed:,}")
    print(f"Output directory: {output_dir}")
    print(f"Chunk size used: {args.chunk_size:,}")
    
    print(f"\nOutput files:")
    output_files = list(output_dir.glob("chr*_predictions.txt"))
    for output_file in sorted(output_files):
        size_mb = output_file.stat().st_size / (1024 * 1024)
        print(f"  {output_file.name} ({size_mb:.1f} MB)")
    
    print(f"\nSummary files:")
    summary_files = list(output_dir.glob("chr*_chunk_summary.json"))
    for summary_file in sorted(summary_files):
        print(f"  {summary_file.name}")
    
    if successful_files == len(feature_files):
        print(f"\n✓ All files processed successfully!")
    else:
        print(f"\n⚠ {len(feature_files) - successful_files} files failed to process")
    
    print(f"\nPredictions are ready for use!")

if __name__ == "__main__":
    main()