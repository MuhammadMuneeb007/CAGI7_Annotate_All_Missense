#!/usr/bin/env python3
"""
Two-Phase Chromosome Processor - Optimized for Speed
Phase 1: Global discovery across all chromosomes
Phase 2: Consistent processing using global mappings

Usage: 
  python script.py discover  # Phase 1: Discover all unique values
  python script.py process [chr_num]  # Phase 2: Process with global mappings
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
import warnings
import re
import sys
import pickle
from datetime import datetime
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from tqdm import tqdm
import gc

warnings.filterwarnings('ignore')

class GlobalChromosomeProcessor:
    def __init__(self, config_file="Step6.11-FeatureAnalysis-Processor.json", 
                 input_dir="Annovar_Merge", output_dir="Final_Results"):
        self.config_file = Path(config_file)
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Global mappings directory
        self.mappings_dir = self.output_dir / "global_mappings"
        self.mappings_dir.mkdir(exist_ok=True, parents=True)
        
        self.chr_mapping = self.get_chromosome_mapping()
        self.config = self.load_config()
        
        # Performance settings
        self.chunk_size = self.config.get('performance', {}).get('chunk_size', 50000)
        self.n_workers = self.config.get('performance', {}).get('n_workers', min(8, mp.cpu_count()))
        self.memory_limit = self.config.get('performance', {}).get('memory_limit_gb', 8) * 1024 * 1024 * 1024

    def get_chromosome_mapping(self):
        """Map chromosome numbers to identifiers"""
        mapping = {}
        for i in range(1, 23):
            mapping[i] = str(i)
        mapping[23] = 'X'
        mapping[24] = 'Y'
        mapping[25] = 'M'
        return mapping
        
    def load_config(self):
        """Load JSON configuration file"""
        with open(self.config_file, 'r') as f:
            config = json.load(f)
        return config
    
    def find_all_chromosome_files(self):
        """Find all chromosome files"""
        chromosome_files = {}
        
        for chr_num, chr_id in self.chr_mapping.items():
            file_pattern = f"chr{chr_id}_concatenated_annotations.csv"
            target_file = self.input_dir / file_pattern
            
            if target_file.exists():
                chromosome_files[chr_num] = target_file
            else:
                # Try alternative patterns
                alt_patterns = [
                    f"chr{chr_id}_*.csv",
                    f"*chr{chr_id}*.csv", 
                    f"chromosome{chr_id}*.csv"
                ]
                
                for pattern in alt_patterns:
                    files = list(self.input_dir.glob(pattern))
                    if files:
                        chromosome_files[chr_num] = files[0]
                        break
        
        print(f"Found {len(chromosome_files)} chromosome files:")
        for chr_num, file_path in sorted(chromosome_files.items()):
            chr_id = self.chr_mapping[chr_num]
            print(f"  Chr{chr_num} ({chr_id}): {file_path.name}")
        
        return chromosome_files
    
    def discover_global_patterns(self):
        """Phase 1: Discover all unique values across all chromosomes - OPTIMIZED"""
        print("PHASE 1: GLOBAL DISCOVERY (OPTIMIZED)")
        print("=" * 50)
        
        chromosome_files = self.find_all_chromosome_files()
        
        # Get columns to analyze
        columns_to_include = [col for col, settings in self.config['column_settings'].items() 
                             if settings.get('include', False)]
        
        print(f"Analyzing {len(columns_to_include)} columns across {len(chromosome_files)} chromosomes")
        print(f"Processing ALL ROWS with chunks of {self.chunk_size:,}")
        print(f"Using {self.n_workers} parallel workers")
        
        # Global discoveries
        global_unique_values = defaultdict(set)
        global_data_types = {}
        global_stats = defaultdict(dict)
        clnsig_all_values = set()
        
        # Process chromosomes in parallel
        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            # Submit all chromosome processing jobs
            future_to_chr = {
                executor.submit(self._process_chromosome_discovery, file_path, chr_num, columns_to_include): chr_num
                for chr_num, file_path in chromosome_files.items()
            }
            
            # Collect results with progress bar
            with tqdm(total=len(chromosome_files), desc="Processing chromosomes") as pbar:
                for future in as_completed(future_to_chr):
                    chr_num = future_to_chr[future]
                    chr_id = self.chr_mapping[chr_num]
                    
                    try:
                        chr_results = future.result()
                        
                        # Merge results
                        if chr_results:
                            # Merge unique values
                            for col_name, unique_vals in chr_results['unique_values'].items():
                                if col_name == 'CLNSIG':
                                    clnsig_all_values.update(unique_vals)
                                else:
                                    global_unique_values[col_name].update(unique_vals)
                            
                            # Merge data types
                            global_data_types.update(chr_results['data_types'])
                            
                            # Merge stats
                            for col_name, stats in chr_results['stats'].items():
                                if col_name not in global_stats:
                                    global_stats[col_name] = stats
                                else:
                                    # Update global min/max
                                    if stats['min'] is not None:
                                        global_stats[col_name]['min'] = min(
                                            global_stats[col_name]['min'] or float('inf'), 
                                            stats['min']
                                        )
                                    if stats['max'] is not None:
                                        global_stats[col_name]['max'] = max(
                                            global_stats[col_name]['max'] or float('-inf'), 
                                            stats['max']
                                        )
                            
                            pbar.set_postfix({
                                f'Chr{chr_num}': f"{chr_results['total_rows']:,} rows"
                            })
                        
                    except Exception as e:
                        print(f"    Error processing Chr{chr_num}: {e}")
                    
                    pbar.update(1)
        
        # Create global mappings
        print(f"\nCreating global mappings...")
        
        # 1. CLNSIG mapping (always 3 categories)
        clnsig_mapping = self.create_clnsig_mapping(clnsig_all_values)
        
        # 2. Categorical encoding mappings
        encoding_mappings = {}
        for column_name, unique_values in global_unique_values.items():
            if len(unique_values) <= self.config['processing_config'].get('max_categories_for_encoding', 1000):
                sorted_values = sorted(list(unique_values))
                encoding_mappings[column_name] = {
                    'values': sorted_values,
                    'mappings': {val: i+1 for i, val in enumerate(sorted_values)}
                }
                print(f"  {column_name}: {len(sorted_values)} categories for encoding")
            else:
                print(f"  {column_name}: Too many categories ({len(unique_values)}) - no encoding")
        
        # Save global mappings
        mappings = {
            'clnsig_mapping': clnsig_mapping,
            'encoding_mappings': encoding_mappings,
            'data_types': global_data_types,
            'global_stats': dict(global_stats),
            'creation_timestamp': datetime.now().isoformat()
        }
        
        mappings_file = self.mappings_dir / "global_mappings.json"
        with open(mappings_file, 'w') as f:
            json.dump(mappings, f, indent=2, default=str)
        
        # Also save as pickle for easier loading
        pickle_file = self.mappings_dir / "global_mappings.pkl"
        with open(pickle_file, 'wb') as f:
            pickle.dump(mappings, f)
        
        print(f"\n✓ Global mappings saved to: {mappings_file}")
        print(f"✓ Global mappings saved to: {pickle_file}")
        
        # Summary
        print(f"\nGLOBAL DISCOVERY SUMMARY:")
        print(f"  CLNSIG mapping: {len(clnsig_all_values)} → 3 categories")
        print(f"  Encoding mappings: {len(encoding_mappings)} columns")
        print(f"  Data types discovered: {len(global_data_types)} columns")
        
        return mappings
    
    def create_clnsig_mapping(self, all_clnsig_values):
        """Create comprehensive CLNSIG mapping"""
        print(f"  Creating CLNSIG mapping from {len(all_clnsig_values)} unique values")
        
        # Enhanced mapping based on all discovered values
        clnsig_mapping = {
            None: 'Unknown',
            '': 'Unknown',
            'not_provided': 'Unknown',
            
            # PATHOGENIC CLASS
            'Pathogenic': 'Pathogenic',
            'Likely_pathogenic': 'Pathogenic',
            'Pathogenic/Likely_pathogenic': 'Pathogenic',
            'Pathogenic|_risk_factor': 'Pathogenic',
            'Likely_pathogenic|_risk_factor': 'Pathogenic',
            'Pathogenic/Likely_pathogenic|_risk_factor': 'Pathogenic',
            'Pathogenic|_other': 'Pathogenic',
            'Pathogenic/Likely_pathogenic|_other': 'Pathogenic',
            'Likely_pathogenic|_other': 'Pathogenic',
            'Pathogenic/Likely_pathogenic|_drug_response': 'Pathogenic',
            'Pathogenic|_Affects': 'Pathogenic',
            'Pathogenic|_association': 'Pathogenic',
            'Pathogenic|_drug_response': 'Pathogenic',
            'Likely_pathogenic|_drug_response': 'Pathogenic',
            'Likely_pathogenic|_Affects': 'Pathogenic',
            'Pathogenic|_confers_sensitivity': 'Pathogenic',
            'Pathogenic|_protective': 'Pathogenic',
            
            # AFFECTS - These should be PATHOGENIC (you were right!)
            'Affects': 'Pathogenic',
            'Affects|_association': 'Pathogenic',
            'Affects|_risk_factor': 'Pathogenic',
            'Uncertain_significance|_Affects': 'Pathogenic',  # If it affects something, lean pathogenic
            'Conflicting_interpretations_of_pathogenicity|_Affects': 'Pathogenic',
            
            # BENIGN CLASS
            'Benign': 'Benign',
            'Likely_benign': 'Benign',
            'Benign/Likely_benign': 'Benign',
            'Benign/Likely_benign|_other': 'Benign',
            'Benign|_other': 'Benign',
            'Benign/Likely_benign|_other|_risk_factor': 'Benign',
            'Benign|_risk_factor': 'Benign',
            'Benign/Likely_benign|_risk_factor': 'Benign',
            'Likely_benign|_other': 'Benign',
            'Benign/Likely_benign|_drug_response': 'Benign',
            'Benign|_drug_response': 'Benign',
            'Benign/Likely_benign|_drug_response|_other': 'Benign',
            'Benign|_confers_sensitivity': 'Benign',
            'Benign|_association|_confers_sensitivity': 'Benign',
            'Likely_benign|_drug_response|_other': 'Benign',
            
            # PROTECTIVE - These are beneficial, so BENIGN
            'protective': 'Benign',
            'protective|_risk_factor': 'Benign',
            'Benign|_protective': 'Benign',
            
            # UNCERTAIN/UNKNOWN/CONFLICTING
            'Uncertain_significance': 'Unknown',
            'Conflicting_interpretations_of_pathogenicity': 'Unknown',
            'Conflicting_interpretations_of_pathogenicity|_other': 'Unknown',
            'Conflicting_interpretations_of_pathogenicity|_association': 'Unknown',
            'Uncertain_significance|_risk_factor': 'Unknown',
            'Conflicting_interpretations_of_pathogenicity|_drug_response': 'Unknown',
            'Uncertain_significance|_other': 'Unknown',
            'Uncertain_significance|_drug_response': 'Unknown',
            'Conflicting_interpretations_of_pathogenicity|_other|_risk_factor': 'Unknown',
            'Conflicting_interpretations_of_pathogenicity|_association|_risk_factor': 'Unknown',
            'Conflicting_interpretations_of_pathogenicity|_risk_factor': 'Unknown',
            
            # STANDALONE CATEGORIES - Lean toward Unknown unless clearly beneficial/harmful
            'association': 'Unknown',
            'risk_factor': 'Unknown', 
            'drug_response': 'Unknown',
            'other': 'Unknown',
            'confers_sensitivity': 'Unknown',
            'drug_response|_risk_factor': 'Unknown',
            
            # COMPLEX CASES - Keep Unknown for now, but these could be refined
            'Pathogenic|_drug_response|_other': 'Pathogenic',  # Too complex, conflicting info
        }
        
        # Check for unmapped values
        unmapped_values = []
        for value in all_clnsig_values:
            if value and str(value).strip() not in clnsig_mapping:
                unmapped_values.append(str(value).strip())
        
        if unmapped_values:
            print(f"    WARNING: {len(unmapped_values)} unmapped CLNSIG values will be set to 'Unknown':")
            for val in sorted(unmapped_values):
                print(f"      '{val}'")
        
        return clnsig_mapping
    
    def load_global_mappings(self):
        """Load saved global mappings"""
        pickle_file = self.mappings_dir / "global_mappings.pkl"
        
        if pickle_file.exists():
            with open(pickle_file, 'rb') as f:
                mappings = pickle.load(f)
            print(f"✓ Loaded global mappings from: {pickle_file}")
            return mappings
        else:
            raise FileNotFoundError(f"Global mappings not found. Run discovery phase first: python script.py discover")
    
    def process_chromosome_with_global_mappings(self, chromosome_num):
        """Phase 2: Process specific chromosome using global mappings - OPTIMIZED"""
        print(f"PHASE 2: PROCESSING CHROMOSOME {chromosome_num} (OPTIMIZED)")
        print("=" * 50)
        
        # Load global mappings
        global_mappings = self.load_global_mappings()
        
        # Find chromosome file
        chr_id = self.chr_mapping[chromosome_num]
        file_pattern = f"chr{chr_id}_concatenated_annotations.csv"
        target_file = self.input_dir / file_pattern
        
        if not target_file.exists():
            # Try alternative patterns
            alt_patterns = [f"chr{chr_id}_*.csv", f"*chr{chr_id}*.csv", f"chromosome{chr_id}*.csv"]
            found = False
            for pattern in alt_patterns:
                files = list(self.input_dir.glob(pattern))
                if files:
                    target_file = files[0]
                    found = True
                    break
            
            if not found:
                raise FileNotFoundError(f"No file found for chromosome {chromosome_num} ({chr_id})")
        
        print(f"Processing: {target_file.name}")
        
        columns_to_include = [col for col, settings in self.config['column_settings'].items() 
                             if settings.get('include', False)]
        
        # Process in chunks and save incrementally
        processed_chunks = []
        total_rows = 0
        
        chunk_iter = pd.read_csv(target_file, chunksize=self.chunk_size, low_memory=False)
        
        with tqdm(desc=f"Processing Chr{chromosome_num} chunks") as pbar:
            for chunk_num, chunk in enumerate(chunk_iter):
                total_rows += len(chunk)
                
                # Process chunk
                processed_chunk = self._process_chunk(chunk, columns_to_include, global_mappings)
                
                if processed_chunk is not None:
                    processed_chunks.append(processed_chunk)
                
                pbar.set_postfix({
                    'rows': f"{total_rows:,}",
                    'chunks': chunk_num + 1
                })
                pbar.update(1)
                
                # Memory management - save and clear every 20 chunks
                if len(processed_chunks) >= 20:
                    self._save_chunk_batch(processed_chunks, chromosome_num, chunk_num)
                    processed_chunks = []
                    gc.collect()
        
        # Save any remaining chunks
        if processed_chunks:
            self._save_chunk_batch(processed_chunks, chromosome_num, 'final')
        
        # Combine all saved chunks into final files
        final_df = self._combine_saved_chunks(chromosome_num)
        
        print(f"\nFinal dataset: {len(final_df):,} rows × {len(final_df.columns):,} columns")
        
        return final_df
    
    def _process_chromosome_discovery(self, file_path, chr_num, columns_to_include):
        """Process single chromosome for discovery - optimized worker function"""
        chr_id = self.chr_mapping[chr_num]
        
        try:
            results = {
                'unique_values': defaultdict(set),
                'data_types': {},
                'stats': {},
                'total_rows': 0
            }
            
            # Process file in chunks
            chunk_iter = pd.read_csv(file_path, chunksize=self.chunk_size, low_memory=False)
            
            for chunk_num, chunk in enumerate(chunk_iter):
                results['total_rows'] += len(chunk)
                
                for column_name in columns_to_include:
                    if column_name not in chunk.columns:
                        continue
                    
                    series = chunk[column_name]
                    
                    # Store data type (first chunk only)
                    if chunk_num == 0 and column_name not in results['data_types']:
                        results['data_types'][column_name] = str(series.dtype)
                    
                    # Collect unique values for specific columns
                    col_config = self.config['column_settings'].get(column_name, 
                                                                  self.config['default_settings'])
                    
                    if column_name == 'CLNSIG':
                        unique_vals = series.dropna().unique()
                        results['unique_values']['CLNSIG'].update(str(v) for v in unique_vals)
                    
                    elif col_config.get('encoding_required', False) and series.dtype == 'object':
                        unique_vals = series.dropna().unique()
                        # Limit unique values to prevent memory explosion
                        if len(results['unique_values'][column_name]) < 10000:
                            results['unique_values'][column_name].update(str(v) for v in unique_vals)
                    
                    # Collect statistics for numeric columns
                    if pd.api.types.is_numeric_dtype(series) and not series.empty:
                        chunk_stats = {
                            'min': float(series.min()),
                            'max': float(series.max()),
                            'mean': float(series.mean())
                        }
                        
                        if column_name not in results['stats']:
                            results['stats'][column_name] = chunk_stats
                        else:
                            # Update running stats
                            results['stats'][column_name]['min'] = min(
                                results['stats'][column_name]['min'], chunk_stats['min']
                            )
                            results['stats'][column_name]['max'] = max(
                                results['stats'][column_name]['max'], chunk_stats['max']
                            )
                
                # Memory cleanup every 10 chunks
                if chunk_num % 10 == 0:
                    gc.collect()
            
            return results
            
        except Exception as e:
            print(f"Error in worker processing Chr{chr_num}: {e}")
            return None

    def _process_chunk(self, chunk, columns_to_include, global_mappings):
        """Process a single chunk efficiently"""
        try:
            processed_columns = {}
            
            # Filter to only include relevant columns that exist
            available_columns = [col for col in columns_to_include if col in chunk.columns]
            
            for column_name in available_columns:
                series = chunk[column_name].copy()
                
                # Apply global processing
                if column_name == 'CLNSIG':
                    processed_series = self._apply_clnsig_mapping_fast(series, global_mappings['clnsig_mapping'])
                    processed_columns['CLNSIG_original'] = chunk[column_name].copy()
                    processed_columns['CLNSIG'] = processed_series
                    
                elif column_name in global_mappings['encoding_mappings']:
                    # Apply global encoding efficiently
                    encoded_cols = self._apply_encoding_fast(series, column_name, global_mappings['encoding_mappings'][column_name])
                    processed_columns.update(encoded_cols)
                    
                else:
                    # Standard cleaning
                    processed_columns[column_name] = self._clean_column_fast(series, column_name)
            
            return pd.DataFrame(processed_columns) if processed_columns else None
            
        except Exception as e:
            print(f"Error processing chunk: {e}")
            return None
    
    def _apply_clnsig_mapping_fast(self, series, clnsig_mapping):
        """Fast CLNSIG mapping using vectorized operations"""
        # Create mapping for all values at once
        result = pd.Series('Unknown', index=series.index, dtype='object')
        
        # Apply mapping for non-null values
        mask = series.notna() & (series != '')
        mapped_values = series[mask].astype(str).str.strip().map(clnsig_mapping)
        result[mask] = mapped_values.fillna('Unknown')
        
        return result
    
    def _apply_encoding_fast(self, series, column_name, encoding_info):
        """Fast encoding using vectorized operations"""
        encoded_columns = {}
        
        # Keep original column
        encoded_columns[column_name] = series
        
        # Create all encoded columns at once using vectorized operations
        for i, value in enumerate(encoding_info['values'], 1):
            encoded_col_name = f"{column_name}_encoded{i}"
            
            # Vectorized comparison
            result = pd.Series(0, index=series.index, dtype='Int64')
            mask = series == value
            result[mask] = 1
            result[series.isnull()] = np.nan
            
            encoded_columns[encoded_col_name] = result
        
        return encoded_columns
    
    def _clean_column_fast(self, series, column_name):
        """Fast column cleaning using vectorized operations"""
        # Special handling for specific columns
        if column_name == 'genomicSuperDups':
            return self._extract_score_fast(series)
        elif column_name in ['Aloft_pred', 'Aloft_Confidence']:
            return self._clean_aloft_fast(series)
        else:
            return self._clean_standard_fast(series)
    
    def _extract_score_fast(self, series):
        """Fast score extraction using vectorized string operations"""
        # Use vectorized string operations
        extracted = series.astype(str).str.extract(r'Score=([0-9\.]+)', expand=False)
        return pd.to_numeric(extracted, errors='coerce')
    
    def _clean_aloft_fast(self, series):
        """Fast Aloft cleaning using vectorized operations"""
        # Replace missing values first
        cleaned = series.replace(['', '.', 'NA'], np.nan)
        
        # For non-null values, apply string processing
        mask = cleaned.notna()
        if mask.any():
            # Vectorized string operations where possible
            cleaned[mask] = cleaned[mask].astype(str).str.replace(r';+', ';', regex=True)
            cleaned[mask] = cleaned[mask].str.strip(';')
        
        return cleaned
    
    def _clean_standard_fast(self, series):
        """Fast standard cleaning using vectorized operations"""
        missing_values = ['', '.', 'NA', 'N/A', 'null', 'NULL', '-', 'nan', 'NaN', 
                         'UNKNOWN', 'unknown', '?', 'None', 'none']
        
        # Vectorized replacement
        cleaned = series.replace(missing_values, np.nan)
        
        # Fast numeric conversion attempt
        if cleaned.dtype == 'object':
            numeric_series = pd.to_numeric(cleaned, errors='coerce')
            # Use numeric if mostly numeric
            if cleaned.count() > 0 and (numeric_series.count() / cleaned.count()) >= 0.8:
                cleaned = numeric_series
        
        return cleaned
    
    def _save_chunk_batch(self, chunks, chromosome_num, batch_id):
        """Save a batch of processed chunks"""
        if not chunks:
            return
        
        # Combine chunks
        combined = pd.concat(chunks, ignore_index=True)
        
        # Save temporary file
        temp_dir = self.output_dir / "temp_chunks"
        temp_dir.mkdir(exist_ok=True)
        
        chr_id = self.chr_mapping[chromosome_num]
        temp_file = temp_dir / f"chr{chr_id}_batch_{batch_id}.csv"
        combined.to_csv(temp_file, index=False)
        
        print(f"  Saved batch {batch_id}: {len(combined):,} rows")
    
    def _combine_saved_chunks(self, chromosome_num):
        """Combine all saved chunks into final files"""
        chr_id = self.chr_mapping[chromosome_num]
        temp_dir = self.output_dir / "temp_chunks"
        
        # Find all batch files for this chromosome
        batch_files = list(temp_dir.glob(f"chr{chr_id}_batch_*.csv"))
        
        if not batch_files:
            print("No batch files found to combine")
            return pd.DataFrame()
        
        print(f"Combining {len(batch_files)} batch files...")
        
        # Read and combine all batch files
        dfs = []
        for batch_file in sorted(batch_files):
            df = pd.read_csv(batch_file, low_memory=False)
            dfs.append(df)
        
        final_df = pd.concat(dfs, ignore_index=True)
        
        # Save final results
        self.save_chromosome_results(final_df, chromosome_num)
        
        # Clean up temporary files
        for batch_file in batch_files:
            batch_file.unlink()
        
        return final_df

    def save_chromosome_results(self, processed_df, chromosome_num):
        """Save chromosome results with consistent schema"""
        chr_id = self.chr_mapping[chromosome_num]
        
        # Separate features and predictions
        feature_columns = [col for col in processed_df.columns 
                          if col not in ['CLNSIG', 'CLNSIG_original']]
        
        # Save features
        if feature_columns:
            features_df = processed_df[feature_columns]
            features_file = self.output_dir / f"chr{chr_id}_features.csv"
            features_df.to_csv(features_file, index=False)
            print(f"✓ Features saved: {features_file} ({len(feature_columns)} columns)")
        
        # Save predictions
        if 'CLNSIG' in processed_df.columns:
            prediction_columns = [col for col in ['CLNSIG_original', 'CLNSIG'] 
                                if col in processed_df.columns]
            predictions_df = processed_df[prediction_columns]
            predictions_file = self.output_dir / f"chr{chr_id}_predictions.csv"
            predictions_df.to_csv(predictions_file, index=False)
            print(f"✓ Predictions saved: {predictions_file} ({len(prediction_columns)} columns)")


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python script.py discover                    # Phase 1: Global discovery")
        print("  python script.py process [chromosome_number] # Phase 2: Process chromosome")
        print("")
        print("Examples:")
        print("  python script.py discover")
        print("  python script.py process 1")
        print("  python script.py process 23  # X chromosome")
        return
    
    command = sys.argv[1].lower()
    
    processor = GlobalChromosomeProcessor(
        config_file="Step6.11-FeatureAnalysis-Processor.json",
        input_dir="Annovar_Merge",
        output_dir="Final_Results"
    )
    
    if command == 'discover':
        print("Starting global discovery phase...")
        mappings = processor.discover_global_patterns()
        print(f"\n✅ Discovery phase completed!")
        print("Next steps:")
        print("  python script.py process 1   # Process chromosome 1")
        print("  python script.py process 23  # Process chromosome X")
        
    elif command == 'process':
        if len(sys.argv) < 3:
            print("Error: Chromosome number required for process command")
            print("Usage: python script.py process [chromosome_number]")
            return
        
        try:
            chromosome_num = int(sys.argv[2])
        except ValueError:
            print("Error: Chromosome number must be an integer")
            return
        
        if chromosome_num < 1 or chromosome_num > 25:
            print(f"Error: Chromosome number must be between 1-25, got {chromosome_num}")
            return
        
        try:
            processed_df = processor.process_chromosome_with_global_mappings(chromosome_num)
            chr_id = processor.chr_mapping[chromosome_num]
            print(f"\n✅ Chromosome {chromosome_num} ({chr_id}) processing completed!")
            
        except FileNotFoundError as e:
            print(f"❌ Error: {e}")
        except Exception as e:
            print(f"❌ Processing failed: {e}")
            import traceback
            traceback.print_exc()
    
    else:
        print(f"Unknown command: {command}")
        print("Available commands: discover, process")


if __name__ == "__main__":
    main()