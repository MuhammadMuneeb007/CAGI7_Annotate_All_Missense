#!/usr/bin/env python3
"""
Simple ClinVar Filter
Read features and predictions, keep only rows where prediction is NOT 'unknown'
Now supports numeric chromosome mapping: 23->X, 24->Y, 25->M
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
from tqdm import tqdm

def map_chromosome_id(chr_input):
    """
    Map chromosome input to standard chromosome identifier
    23 -> X, 24 -> Y, 25 -> M
    All other inputs are returned as-is
    """
    # Handle both string and numeric inputs
    if str(chr_input) == "23":
        return "X"
    elif str(chr_input) == "24":
        return "Y" 
    elif str(chr_input) == "25":
        return "M"
    else:
        return str(chr_input)

def process_chromosome_simple(chr_input, input_dir="Final_Results", chunk_size=50000):
    """Simple processing: read files, filter non-unknown, save result"""
    
    # Map chromosome input to actual chromosome ID
    chr_id = map_chromosome_id(chr_input)
    
    feature_file = Path(input_dir) / f"chr{chr_id}_features_engineered.csv"
    prediction_file = Path(input_dir) / f"chr{chr_id}_predictions.csv"
    output_file = Path(input_dir) / f"chr{chr_id}_features_engineered_clinvar.csv"
    
    print(f"Processing chromosome {chr_input} -> {chr_id}")
    print(f"Feature file: {feature_file}")
    print(f"Prediction file: {prediction_file}")
    
    # Check files exist
    if not feature_file.exists():
        print(f"Error: Feature file not found")
        return False
    
    if not prediction_file.exists():
        print(f"Error: Prediction file not found")
        return False
    
    # Load predictions (small file)
    print("Loading predictions...")
    try:
        predictions = pd.read_csv(prediction_file)
        print(f"Predictions loaded: {predictions.shape}")
    except Exception as e:
        print(f"Error loading predictions: {e}")
        return False
    
    # Check CLNSIG column
    if 'CLNSIG' not in predictions.columns:
        print(f"Error: CLNSIG column not found. Available columns: {predictions.columns.tolist()}")
        return False
    
    # Show prediction distribution
    print(f"CLNSIG distribution:")
    value_counts = predictions['CLNSIG'].value_counts()
    for val, count in value_counts.items():
        print(f"  '{val}': {count:,}")
    
    # Find non-unknown rows (case-insensitive)
    non_unknown_mask = predictions['CLNSIG'].str.lower() != 'unknown'
    non_unknown_indices = non_unknown_mask[non_unknown_mask].index.tolist()
    
    print(f"Total predictions: {len(predictions):,}")
    print(f"Non-unknown predictions: {len(non_unknown_indices):,}")
    print(f"Unknown predictions: {len(predictions) - len(non_unknown_indices):,}")
    
    if len(non_unknown_indices) == 0:
        print("No non-unknown predictions found!")
        return False
    
    # Get total rows in features file for progress
    print("Counting feature file rows...")
    try:
        with open(feature_file, 'r') as f:
            total_rows = sum(1 for _ in f) - 1  # subtract header
        print(f"Total feature rows: {total_rows:,}")
    except:
        total_rows = None
    
    # Process features in chunks and keep only non-unknown rows
    print("Processing features in chunks...")
    
    # Remove output file if exists
    if output_file.exists():
        output_file.unlink()
    
    saved_rows = 0
    processed_rows = 0
    first_write = True
    
    try:
        # Read features in chunks
        chunk_reader = pd.read_csv(feature_file, chunksize=chunk_size, on_bad_lines='skip')
        
        # Progress bar
        pbar = tqdm(total=total_rows, desc=f"Chr{chr_id}", unit="rows") if total_rows else None
        
        for chunk_num, chunk in enumerate(chunk_reader):
            # Get row indices for this chunk
            start_idx = processed_rows
            end_idx = start_idx + len(chunk)
            
            # Find which rows in this chunk have non-unknown predictions
            chunk_indices = list(range(start_idx, min(end_idx, len(predictions))))
            
            # Filter chunk indices to only non-unknown
            keep_indices = []
            for i, idx in enumerate(chunk_indices):
                if idx in non_unknown_indices:
                    keep_indices.append(i)
            
            if keep_indices:
                # Keep only the rows with non-unknown predictions
                filtered_chunk = chunk.iloc[keep_indices].copy()
                
                # Add corresponding predictions
                pred_indices = [chunk_indices[i] for i in keep_indices]
                filtered_chunk['CLNSIG'] = predictions.iloc[pred_indices]['CLNSIG'].values
                
                # Save chunk
                if first_write:
                    filtered_chunk.to_csv(output_file, index=False)
                    first_write = False
                else:
                    filtered_chunk.to_csv(output_file, mode='a', header=False, index=False)
                
                saved_rows += len(filtered_chunk)
            
            processed_rows += len(chunk)
            
            # Update progress
            if pbar:
                pbar.update(len(chunk))
                pbar.set_postfix({
                    'saved': saved_rows,
                    'rate': f"{saved_rows/processed_rows*100:.1f}%" if processed_rows > 0 else "0%"
                })
            
            # Stop if we've processed all predictions
            if processed_rows >= len(predictions):
                break
    
    finally:
        if pbar:
            pbar.close()
    
    print(f"Processing complete!")
    print(f"Total rows processed: {processed_rows:,}")
    print(f"Rows saved: {saved_rows:,}")
    
    if processed_rows > 0:
        print(f"Filter rate: {saved_rows/processed_rows*100:.2f}%")
    
    # Verify output
    if output_file.exists() and saved_rows > 0:
        print(f"Output saved to: {output_file}")
        
        # Quick verification
        try:
            result_df = pd.read_csv(output_file)
            print(f"Verification - Output shape: {result_df.shape}")
            
            if 'CLNSIG' in result_df.columns:
                final_counts = result_df['CLNSIG'].value_counts()
                print(f"Final CLNSIG distribution:")
                for val, count in final_counts.items():
                    print(f"  '{val}': {count:,}")
        except Exception as e:
            print(f"Verification error: {e}")
        
        return True
    else:
        print("No output file created")
        return False

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Simple ClinVar Filter with Chromosome Mapping")
    parser.add_argument('--chromosome', type=str, required=True,
                       help='Chromosome to process (1-22, 23=X, 24=Y, 25=M, or X, Y, M)')
    parser.add_argument('--input-dir', default='Final_Results',
                       help='Input directory (default: Final_Results)')
    parser.add_argument('--chunk-size', type=int, default=50000,
                       help='Chunk size (default: 50000)')
    
    args = parser.parse_args()
    
    print("="*60)
    print("SIMPLE CLINVAR FILTER")
    print("Keep only rows where CLNSIG is NOT 'unknown'")
    print("Chromosome mapping: 23->X, 24->Y, 25->M")
    print("="*60)
    
    success = process_chromosome_simple(args.chromosome, args.input_dir, args.chunk_size)
    
    # Get the mapped chromosome ID for display
    mapped_chr = map_chromosome_id(args.chromosome)
    
    if success:
        print(f"\n✅ Successfully processed chromosome {args.chromosome} -> {mapped_chr}")
        print(f"Output: chr{mapped_chr}_features_engineered_clinvar.csv")
    else:
        print(f"\n❌ Failed to process chromosome {args.chromosome} -> {mapped_chr}")

if __name__ == "__main__":
    main()