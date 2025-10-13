#!/usr/bin/env python3
"""
ARRAY JOB ANNOTATION + dbNSFP MERGER
Process single chromosome based on SLURM array task ID
Supports all chromosomes: 1-22, X, Y, M (mitochondrial)
Ensures consistent column schema across all chromosomes using chunked processing

Usage: 
  python Step5-dbNSFP-Merger.py ${SLURM_ARRAY_TASK_ID}  # SLURM array job
  python Step5-dbNSFP-Merger.py 1                       # Process chr1
  python Step5-dbNSFP-Merger.py 23                      # Process chrX  
  python Step5-dbNSFP-Merger.py 24                      # Process chrY
  python Step5-dbNSFP-Merger.py 25                      # Process chrM
  python Step5-dbNSFP-Merger.py                         # List available chromosomes

Chromosome Naming Conversion:
  - Handles various formats: chr1, 1, chrX, X, chrY, Y, chrM, M, chrMT, MT
  - Converts numeric codes: 23→chrX, 24→chrY, 25→chrM
  - Standardizes all to chr1, chr2, ..., chr22, chrX, chrY, chrM format
"""

import pandas as pd
import sys
import os
import time
from tqdm import tqdm
import gc
import tempfile
import pickle
import glob
import shutil
from datetime import datetime
from collections import defaultdict

# STANDARD SCHEMA - Ensures all chromosomes have identical columns
STANDARD_DBNSFP_COLUMNS = [
    'chr', 'pos(1-based)', 'ref', 'alt', 'aaref', 'aaalt', 
    'hg19_chr', 'hg19_pos(1-based)', 'hg18_chr', 'hg18_pos(1-based)', 
    'genename', 'cds_strand', 'refcodon', 'codonpos', 
    'Ensembl_geneid', 'Ensembl_transcriptid', 'Ensembl_proteinid', 'aapos'
]

# Columns to add to annotation files (consistent across all chromosomes)
DBNSFP_OUTPUT_COLUMNS = [
    'aaref', 'aaalt', 'aapos', 'genename', 'cds_strand', 'refcodon', 'codonpos',
    'Ensembl_geneid', 'Ensembl_transcriptid', 'Ensembl_proteinid',
    'hg19_chr', 'hg19_pos', 'hg18_chr', 'hg18_pos'
]

def normalize_chromosome_name(chr_name):
    """Convert various chromosome naming conventions to standard format"""
    chr_name = str(chr_name).strip()
    
    # Convert numeric to standard format
    if chr_name == "23":
        return "chrX"
    elif chr_name == "24":
        return "chrY"
    elif chr_name == "25" or chr_name.lower() in ["mt", "m"]:
        return "chrM"
    
    # Handle already formatted names
    if chr_name.lower() == "chrmt":
        return "chrM"
    
    # Add chr prefix if missing
    if chr_name.isdigit() or chr_name.upper() in ["X", "Y", "M"]:
        return f"chr{chr_name.upper()}"
    
    # Already has chr prefix
    if chr_name.startswith("chr"):
        return chr_name
    
    return f"chr{chr_name}"

def find_chromosome_pairs():
    """Find all available chromosome file pairs including X, Y, M"""
    print("🔍 DISCOVERING CHROMOSOME FILES")
    print("-" * 40)
    
    # Find annotation files with various patterns
    patterns = [
        "Annovar_Merge/chr*_concatenated_annotations.csv",
        "Annovar_Merge/chr*_*_concatenated_annotations.csv"  # In case of different naming
    ]
    
    all_annotation_files = []
    for pattern in patterns:
        all_annotation_files.extend(glob.glob(pattern))
    
    # Remove duplicates
    annotation_files = list(set(all_annotation_files))
    
    if not annotation_files:
        print(f"❌ No annotation files found")
        print("📁 Expected pattern: Annovar_Merge/chr*_concatenated_annotations.csv")
        return []
    
    available_pairs = []
    
    for ann_file in sorted(annotation_files):
        # Extract chromosome name
        filename = os.path.basename(ann_file)
        chr_name = filename.replace("_concatenated_annotations.csv", "")
        
        # Normalize chromosome name
        normalized_chr = normalize_chromosome_name(chr_name)
        
        # Look for corresponding dbNSFP file
        dbnsfp_file = f"dbNSFP5.1_nsSNV.{normalized_chr}"
        
        if os.path.exists(dbnsfp_file):
            available_pairs.append(normalized_chr)
            ann_size = os.path.getsize(ann_file) / (1024**2)
            dbnsfp_size = os.path.getsize(dbnsfp_file) / (1024**3)
            
            # Determine chromosome type
            if normalized_chr[3:].isdigit():
                chr_type = "Autosome"
            elif normalized_chr in ["chrX", "chrY"]:
                chr_type = "Sex"
            elif normalized_chr == "chrM":
                chr_type = "Mitochondrial"
            else:
                chr_type = "Other"
            
            print(f"   ✅ {normalized_chr} ({chr_type}): Ann {ann_size:.1f}MB, dbNSFP {dbnsfp_size:.1f}GB")
            
            if normalized_chr in ["chrX", "chrY", "chrM"]:
                print(f"      📍 Special chromosome detected: {filename}")
        else:
            print(f"   ❌ {normalized_chr}: Missing dbNSFP file ({dbnsfp_file})")
    
    print(f"\n📊 Found {len(available_pairs)} valid chromosome pairs")
    return available_pairs

def get_chromosome_list():
    """Get properly sorted list of available chromosomes (1-22, X, Y, M)"""
    chr_pairs = find_chromosome_pairs()
    
    # Custom sorting: chr1-chr22, then chrX, chrY, chrM
    def chr_sort_key(chr_name):
        if chr_name.startswith("chr"):
            chr_suffix = chr_name[3:]
            if chr_suffix.isdigit():
                return (0, int(chr_suffix))  # Autosomes first
            elif chr_suffix == "X":
                return (1, 0)  # X after autosomes
            elif chr_suffix == "Y":
                return (1, 1)  # Y after X
            elif chr_suffix == "M":
                return (1, 2)  # M after Y
            else:
                return (2, chr_suffix)  # Other chromosomes last
        return (3, chr_name)  # Fallback
    
    sorted_chromosomes = sorted(chr_pairs, key=chr_sort_key)
    
    return sorted_chromosomes

def print_array_header(array_id=None):
    """Print script header for array job"""
    print("=" * 70)
    print("🚀 ARRAY JOB CHROMOSOME-dbNSFP MERGER")
    print("🧬 Including X, Y, M chromosomes")
    print("🎯 Consistent Schema | Chunked Processing | Memory Efficient")
    print("=" * 70)
    print(f"📅 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if array_id:
        print(f"🎯 Array Task ID: {array_id}")
    print(f"📋 Standard dbNSFP columns: {len(DBNSFP_OUTPUT_COLUMNS)}")
    print(f"🧬 Target chunk size: 1GB per chunk")
    print()

def create_genomic_key(chr_val, pos_val, ref_val, alt_val):
    """Create standardized genomic coordinate key"""
    if pd.isna(chr_val) or pd.isna(pos_val) or pd.isna(ref_val) or pd.isna(alt_val):
        return None
    
    # Standardize chromosome format (remove 'chr' prefix if present)
    chr_clean = str(chr_val).replace('chr', '')
    return f"{chr_clean}:{pos_val}:{ref_val}:{alt_val}"

def read_annotation_file(annotation_file):
    """Read chromosome annotation file and create genomic keys"""
    print(f"📂 Reading annotation file: {os.path.basename(annotation_file)}")
    
    try:
        # Read with pandas
        df = pd.read_csv(annotation_file, low_memory=False)
        print(f"   ✓ Loaded {len(df):,} rows, {len(df.columns)} columns")
        
        # Create genomic keys
        print("   Creating genomic keys...")
        tqdm.pandas(desc="   🔑 Keys")
        df['genomic_key'] = df.progress_apply(
            lambda row: create_genomic_key(row['Chr'], row['Start'], row['Ref'], row['Alt']), 
            axis=1
        )
        
        valid_keys = df['genomic_key'].notna().sum()
        print(f"   ✓ Created {valid_keys:,} valid genomic keys ({valid_keys/len(df)*100:.1f}%)")
        
        return df
        
    except Exception as e:
        print(f"   ❌ Error reading annotation file: {e}")
        return None

def estimate_chunk_size(file_path, target_size_gb=1):
    """Estimate optimal chunk size for target memory usage"""
    print(f"   📐 Estimating chunk size for {target_size_gb} GB chunks...")
    
    file_size_bytes = os.path.getsize(file_path)
    file_size_gb = file_size_bytes / (1024**3)
    
    print(f"   📁 File size: {file_size_gb:.2f} GB")
    
    # Sample lines to estimate row size
    with open(file_path, 'r') as f:
        header = f.readline()
        sample_lines = []
        for i in range(min(1000, int(file_size_bytes / 1000))):
            line = f.readline()
            if not line:
                break
            sample_lines.append(line)
    
    if not sample_lines:
        return 10000, 100
    
    # Calculate average line size
    avg_line_size = sum(len(line) for line in sample_lines) / len(sample_lines)
    estimated_total_lines = file_size_bytes / avg_line_size
    
    # Calculate chunk size for target GB
    target_bytes = target_size_gb * 1024**3
    chunk_size = int(target_bytes / avg_line_size)
    chunk_size = max(1000, min(chunk_size, 100000))  # Reasonable bounds
    
    estimated_chunks = int(estimated_total_lines / chunk_size) + 1
    
    print(f"   ✓ Estimated {estimated_total_lines:,.0f} total rows")
    print(f"   ✓ Chunk size: {chunk_size:,} rows (~{chunk_size * avg_line_size / (1024**3):.2f} GB)")
    print(f"   ✓ Estimated chunks: {estimated_chunks}")
    
    return chunk_size, estimated_chunks

def ensure_standard_dbnsfp_columns(df):
    """Ensure dbNSFP dataframe has all standard columns"""
    missing_cols = []
    for col in STANDARD_DBNSFP_COLUMNS:
        if col not in df.columns:
            df[col] = None
            missing_cols.append(col)
    
    if missing_cols:
        print(f"   ⚠️  Added {len(missing_cols)} missing columns")
    
    return df

def process_dbnsfp_chunks(dbnsfp_file, annotation_keys, temp_dir):
    """Process dbNSFP file in chunks and extract matching variants"""
    print(f"🧬 Processing dbNSFP file: {os.path.basename(dbnsfp_file)}")
    
    try:
        file_size_gb = os.path.getsize(dbnsfp_file) / (1024**3)
        print(f"   📁 File size: {file_size_gb:.1f} GB")
        
        # Estimate chunk size
        chunk_size, estimated_chunks = estimate_chunk_size(dbnsfp_file, target_size_gb=1)
        
        # Convert annotation keys to set for fast lookup
        annotation_keys_set = set(key for key in annotation_keys if pd.notna(key))
        print(f"   🎯 Looking for {len(annotation_keys_set):,} annotation variants")
        
        # Process chunks
        chunk_files = []
        total_matches = 0
        chunk_num = 0
        
        print("   📚 Processing in chunks...")
        
        # Create chunk iterator
        try:
            chunk_iterator = pd.read_csv(
                dbnsfp_file,
                sep='\t',
                skiprows=1,  # Skip header line starting with #
                names=STANDARD_DBNSFP_COLUMNS,
                low_memory=False,
                chunksize=chunk_size
            )
        except Exception as e:
            print(f"   ❌ Error creating chunk iterator: {e}")
            return [], 0
        
        with tqdm(total=estimated_chunks, desc="   🔄 Chunks", unit="chunk") as pbar:
            for chunk in chunk_iterator:
                try:
                    # Ensure standard columns
                    chunk = ensure_standard_dbnsfp_columns(chunk)
                    
                    # Create genomic keys for this chunk
                    chunk['genomic_key'] = chunk.apply(
                        lambda row: create_genomic_key(row['chr'], row['pos(1-based)'], row['ref'], row['alt']), 
                        axis=1
                    )
                    
                    # Filter to only matching variants
                    matching_chunk = chunk[chunk['genomic_key'].isin(annotation_keys_set)]
                    
                    if len(matching_chunk) > 0:
                        # Remove duplicates within chunk
                        matching_chunk = matching_chunk.drop_duplicates(subset=['genomic_key'], keep='first')
                        
                        # Save chunk to temp file
                        chunk_file = os.path.join(temp_dir, f"dbnsfp_chunk_{chunk_num:06d}.pkl")
                        with open(chunk_file, 'wb') as f:
                            pickle.dump(matching_chunk, f)
                        
                        chunk_files.append(chunk_file)
                        total_matches += len(matching_chunk)
                    
                    chunk_num += 1
                    pbar.update(1)
                    pbar.set_postfix({'matches': f"{total_matches:,}"})
                    
                    # Clean up memory
                    del chunk
                    if 'matching_chunk' in locals():
                        del matching_chunk
                    gc.collect()
                    
                except Exception as e:
                    print(f"   ⚠️  Error in chunk {chunk_num}: {e}")
                    chunk_num += 1
                    pbar.update(1)
                    continue
        
        print(f"   ✓ Processed {chunk_num} chunks")
        print(f"   ✓ Found {total_matches:,} matching variants")
        
        return chunk_files, total_matches
        
    except Exception as e:
        print(f"   ❌ Error processing dbNSFP file: {e}")
        return [], 0

def merge_dbnsfp_chunks(chunk_files):
    """Merge all dbNSFP chunk files into single dataframe"""
    print(f"🔄 Merging {len(chunk_files)} dbNSFP chunks...")
    
    if not chunk_files:
        print("   ⚠️  No chunk files to merge")
        return pd.DataFrame()
    
    all_chunks = []
    
    with tqdm(chunk_files, desc="   📦 Loading") as pbar:
        for chunk_file in pbar:
            try:
                with open(chunk_file, 'rb') as f:
                    chunk_data = pickle.load(f)
                    all_chunks.append(chunk_data)
                pbar.set_postfix({'rows': f"{sum(len(c) for c in all_chunks):,}"})
            except Exception as e:
                print(f"   ⚠️  Error loading {chunk_file}: {e}")
                continue
    
    if not all_chunks:
        print("   ❌ No valid chunks loaded")
        return pd.DataFrame()
    
    # Combine chunks
    print("   🔗 Concatenating chunks...")
    combined_df = pd.concat(all_chunks, ignore_index=True)
    
    # Ensure standard schema
    combined_df = ensure_standard_dbnsfp_columns(combined_df)
    
    # Remove cross-chunk duplicates
    print("   🗑️  Removing duplicates...")
    original_count = len(combined_df)
    combined_df = combined_df.drop_duplicates(subset=['genomic_key'], keep='first')
    final_count = len(combined_df)
    
    if original_count != final_count:
        print(f"   ✓ Removed {original_count - final_count:,} duplicates")
    
    print(f"   ✓ Final dbNSFP data: {final_count:,} unique variants")
    
    # Clean up
    del all_chunks
    gc.collect()
    
    return combined_df

def create_standardized_output(annotation_df, dbnsfp_df):
    """Create output dataframe with standardized schema"""
    print(f"🎯 Creating standardized output...")
    
    # Start with annotation data
    result_df = annotation_df.copy()
    
    # Add all standard dbNSFP columns with None defaults
    for col in DBNSFP_OUTPUT_COLUMNS:
        result_df[col] = None
    
    print(f"   ✓ Added {len(DBNSFP_OUTPUT_COLUMNS)} standard dbNSFP columns")
    
    if len(dbnsfp_df) > 0:
        # Create lookup dictionary for fast annotation
        print("   🗂️  Creating lookup dictionary...")
        dbnsfp_dict = {}
        
        for _, row in tqdm(dbnsfp_df.iterrows(), total=len(dbnsfp_df), desc="   📚 Building lookup"):
            key = row['genomic_key']
            if pd.notna(key):
                dbnsfp_dict[key] = {
                    'aaref': row.get('aaref'),
                    'aaalt': row.get('aaalt'),
                    'aapos': row.get('aapos'),
                    'genename': row.get('genename'),
                    'cds_strand': row.get('cds_strand'),
                    'refcodon': row.get('refcodon'),
                    'codonpos': row.get('codonpos'),
                    'Ensembl_geneid': row.get('Ensembl_geneid'),
                    'Ensembl_transcriptid': row.get('Ensembl_transcriptid'),
                    'Ensembl_proteinid': row.get('Ensembl_proteinid'),
                    'hg19_chr': row.get('hg19_chr'),
                    'hg19_pos': row.get('hg19_pos(1-based)'),
                    'hg18_chr': row.get('hg18_chr'),
                    'hg18_pos': row.get('hg18_pos(1-based)')
                }
        
        # Fill in dbNSFP annotations
        print("   🔄 Filling annotations...")
        matched_count = 0
        
        for idx, row in tqdm(result_df.iterrows(), total=len(result_df), desc="   ✍️  Annotating"):
            key = row['genomic_key']
            if pd.notna(key) and key in dbnsfp_dict:
                for col, value in dbnsfp_dict[key].items():
                    result_df.at[idx, col] = value
                matched_count += 1
        
        print(f"   ✓ Successfully annotated {matched_count:,} variants")
        
        # Clean up
        del dbnsfp_dict
        gc.collect()
    
    return result_df

def merge_chromosome_data(annotation_df, dbnsfp_df):
    """Main merge function with validation"""
    print(f"🔄 Merging chromosome data...")
    
    original_count = len(annotation_df)
    
    # Report data summary
    annotation_keys = set(annotation_df['genomic_key'].dropna())
    if len(dbnsfp_df) > 0:
        dbnsfp_keys = set(dbnsfp_df['genomic_key'].dropna())
        common_keys = annotation_keys.intersection(dbnsfp_keys)
        overlap_pct = (len(common_keys) / len(annotation_keys)) * 100 if annotation_keys else 0
        
        print(f"   📊 Data summary:")
        print(f"     • Annotation variants: {len(annotation_df):,}")
        print(f"     • dbNSFP variants: {len(dbnsfp_df):,}")
        print(f"     • Overlapping variants: {len(common_keys):,} ({overlap_pct:.1f}%)")
    else:
        print(f"   ⚠️  No dbNSFP data available for this chromosome")
        common_keys = set()
    
    # Create standardized output
    merged_df = create_standardized_output(annotation_df, dbnsfp_df)
    
    # Validate row count preservation
    final_count = len(merged_df)
    if final_count != original_count:
        print(f"   ❌ ERROR: Row count mismatch!")
        print(f"      Expected: {original_count:,}, Got: {final_count:,}")
        return None
    
    print(f"   ✅ Row count preserved: {original_count:,}")
    
    # Calculate annotation success
    annotated_count = merged_df['aaref'].notna().sum()
    annotation_rate = (annotated_count / final_count) * 100
    
    print(f"   📈 Annotation results:")
    print(f"     • Total variants: {final_count:,}")
    print(f"     • Annotated variants: {annotated_count:,}")
    print(f"     • Success rate: {annotation_rate:.1f}%")
    print(f"     • Final columns: {len(merged_df.columns)}")
    
    return merged_df

def save_result(df, output_file):
    """Save merged result with backup"""
    print(f"💾 Saving results...")
    
    # Create backup of existing file
    if os.path.exists(output_file):
        backup_file = output_file + ".backup"
        try:
            shutil.copy2(output_file, backup_file)
            print(f"   ✅ Backup created: {os.path.basename(backup_file)}")
        except Exception as e:
            print(f"   ⚠️  Warning: Could not create backup: {e}")
    
    # Prepare data for saving (remove helper column)
    df_to_save = df.copy()
    if 'genomic_key' in df_to_save.columns:
        df_to_save = df_to_save.drop('genomic_key', axis=1)
    
    # Save to CSV
    start_time = time.time()
    df_to_save.to_csv(output_file, index=False)
    save_time = time.time() - start_time
    
    file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
    print(f"   ✓ Saved: {file_size_mb:.1f} MB in {save_time:.1f}s")
    print(f"   ✓ File: {output_file}")
    print(f"   ✓ Columns: {len(df_to_save.columns)}")

def process_single_chromosome(chr_name):
    """Process a single chromosome with full pipeline"""
    annotation_file = f"Annovar_Merge/{chr_name}_concatenated_annotations.csv"
    dbnsfp_file = f"dbNSFP5.1_nsSNV.{chr_name}"
    
    # Determine chromosome type
    if chr_name[3:].isdigit():
        chr_type = "Autosome"
    elif chr_name in ["chrX", "chrY"]:
        chr_type = "Sex chromosome"
    elif chr_name == "chrM":
        chr_type = "Mitochondrial"
    else:
        chr_type = "Other"
    
    print(f"\n{'='*70}")
    print(f"PROCESSING {chr_name.upper()} ({chr_type.upper()})")
    print("="*70)
    
    # Validate input files
    if not os.path.exists(annotation_file):
        print(f"❌ Annotation file not found: {annotation_file}")
        return False, 0, 0
    
    if not os.path.exists(dbnsfp_file):
        print(f"❌ dbNSFP file not found: {dbnsfp_file}")
        return False, 0, 0
    
    # Show file sizes
    ann_size = os.path.getsize(annotation_file) / (1024**2)
    dbnsfp_size = os.path.getsize(dbnsfp_file) / (1024**3)
    print(f"📁 Input files:")
    print(f"   • Annotation: {ann_size:.1f} MB")
    print(f"   • dbNSFP: {dbnsfp_size:.1f} GB")
    
    # Special notes for different chromosome types
    if chr_name == "chrM":
        print(f"   🔬 Processing mitochondrial chromosome")
    elif chr_name in ["chrX", "chrY"]:
        print(f"   🚻 Processing sex chromosome")
    
    temp_dir = None
    try:
        start_time = time.time()
        
        # Create temporary directory
        temp_dir = tempfile.mkdtemp(prefix=f"dbnsfp_{chr_name}_")
        
        print(f"\nSTEP 1: READING ANNOTATION DATA")
        print("-" * 40)
        annotation_df = read_annotation_file(annotation_file)
        if annotation_df is None:
            return False, 0, 0
        
        annotation_keys = annotation_df['genomic_key'].dropna().tolist()
        
        print(f"\nSTEP 2: PROCESSING dbNSFP DATA")
        print("-" * 40)
        chunk_files, total_matches = process_dbnsfp_chunks(dbnsfp_file, annotation_keys, temp_dir)
        
        print(f"\nSTEP 3: MERGING dbNSFP CHUNKS")
        print("-" * 40)
        dbnsfp_df = merge_dbnsfp_chunks(chunk_files)
        
        print(f"\nSTEP 4: CREATING FINAL OUTPUT")
        print("-" * 40)
        merged_df = merge_chromosome_data(annotation_df, dbnsfp_df)
        if merged_df is None:
            return False, 0, 0
        
        print(f"\nSTEP 5: SAVING RESULTS")
        print("-" * 40)
        save_result(merged_df, annotation_file)
        
        # Summary
        total_time = time.time() - start_time
        annotated_count = merged_df['aaref'].notna().sum()
        result_count = len(merged_df)
        
        print(f"\n✅ {chr_name.upper()} COMPLETED!")
        print(f"   🧬 Chromosome type: {chr_type}")
        print(f"   ⏱️  Processing time: {total_time:.1f}s ({total_time/60:.1f} min)")
        print(f"   📊 Final result: {result_count:,} variants")
        print(f"   🧬 Annotated: {annotated_count:,} variants")
        print(f"   📋 Columns: {len(merged_df.columns)}")
        
        # Special completion messages
        if chr_name in ["chrX", "chrY", "chrM"]:
            print(f"   🌟 Special chromosome {chr_name} processed successfully!")
        
        # Memory cleanup
        del annotation_df, dbnsfp_df, merged_df
        gc.collect()
        
        return True, result_count, annotated_count
        
    except Exception as e:
        print(f"❌ Error processing {chr_name}: {e}")
        import traceback
        traceback.print_exc()
        return False, 0, 0
        
    finally:
        # Clean up temporary directory
        if temp_dir and os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
            except Exception as e:
                print(f"   ⚠️  Warning: Could not clean temp dir: {e}")

def list_available_chromosomes():
    """List all available chromosomes and their file counts"""
    print("📋 AVAILABLE CHROMOSOMES FOR ARRAY PROCESSING")
    print("=" * 70)
    
    chromosomes = get_chromosome_list()
    
    if not chromosomes:
        print("❌ No chromosome pairs found!")
        print("📁 Expected files:")
        print("   Annovar_Merge/chr*_concatenated_annotations.csv")
        print("   dbNSFP5.1_nsSNV.chr*")
        return
    
    print(f"Found {len(chromosomes)} chromosomes ready for dbNSFP merging")
    print()
    print("Array ID | Chromosome | Type         | Files")
    print("-" * 50)
    
    for i, chr_name in enumerate(chromosomes, 1):
        # Determine chromosome type
        if chr_name[3:].isdigit():
            chr_type = "Autosome"
        elif chr_name in ["chrX", "chrY"]:
            chr_type = "Sex"
        elif chr_name == "chrM":
            chr_type = "Mitochondrial"
        else:
            chr_type = "Other"
        
        # Check file sizes
        ann_file = f"Annovar_Merge/{chr_name}_concatenated_annotations.csv"
        dbnsfp_file = f"dbNSFP5.1_nsSNV.{chr_name}"
        
        ann_size = os.path.getsize(ann_file) / (1024**2) if os.path.exists(ann_file) else 0
        dbnsfp_size = os.path.getsize(dbnsfp_file) / (1024**3) if os.path.exists(dbnsfp_file) else 0
        
        print(f"{i:8d} | {chr_name:10s} | {chr_type:12s} | Ann:{ann_size:5.0f}MB, dbNSFP:{dbnsfp_size:4.1f}GB")
    
    print()
    print("USAGE EXAMPLES:")
    print(f"  # SLURM Array Job (process all chromosomes including X, Y, M):")
    print(f"  sbatch --array=1-{len(chromosomes)} script.sh")
    print(f"  ")
    print(f"  # In script.sh:")
    print(f"  python Step5-dbNSFP-Merger.py ${{SLURM_ARRAY_TASK_ID}}")
    print(f"  ")
    print(f"  # Single chromosome examples:")
    print(f"  python Step5-dbNSFP-Merger.py 1      # Process {chromosomes[0]}")
    
    # Show examples for sex and mitochondrial chromosomes
    for i, chr_name in enumerate(chromosomes, 1):
        if chr_name == "chrX":
            print(f"  python Step5-dbNSFP-Merger.py {i}     # Process chrX")
        elif chr_name == "chrY":
            print(f"  python Step5-dbNSFP-Merger.py {i}     # Process chrY")
        elif chr_name == "chrM":
            print(f"  python Step5-dbNSFP-Merger.py {i}     # Process chrM")
    
    if len(chromosomes) > 0:
        print(f"  python Step5-dbNSFP-Merger.py {len(chromosomes)}     # Process {chromosomes[-1]}")
    
    print(f"\n📊 Chromosome Summary:")
    autosomes = [c for c in chromosomes if c[3:].isdigit()]
    sex_chroms = [c for c in chromosomes if c in ["chrX", "chrY"]]
    mito = [c for c in chromosomes if c == "chrM"]
    
    print(f"  • Autosomes (1-22): {len(autosomes)} found")
    print(f"  • Sex chromosomes (X,Y): {len(sex_chroms)} found {sex_chroms}")
    print(f"  • Mitochondrial (M): {len(mito)} found {mito}")

def process_array_chromosome(array_id):
    """Process a single chromosome based on array ID"""
    print_array_header(array_id)
    
    # Convert array ID to chromosome name
    if 1 <= array_id <= 22:
        target_chr = f"chr{array_id}"
    elif array_id == 23:
        target_chr = "chrX"
    elif array_id == 24:
        target_chr = "chrY"
    elif array_id == 25:
        target_chr = "chrM"
    else:
        print(f"❌ Array ID {array_id} is invalid!")
        print("Valid array IDs:")
        print("  1-22: Autosomes (chr1-chr22)")
        print("  23: Sex chromosome X (chrX)")
        print("  24: Sex chromosome Y (chrY)")
        print("  25: Mitochondrial (chrM)")
        sys.exit(1)
    
    # Check if files exist for this chromosome
    annotation_file = f"Annovar_Merge/{target_chr}_concatenated_annotations.csv"
    dbnsfp_file = f"dbNSFP5.1_nsSNV.{target_chr}"
    
    if not os.path.exists(annotation_file):
        print(f"❌ Annotation file not found for {target_chr}!")
        print(f"   Expected: {annotation_file}")
        sys.exit(1)
    
    if not os.path.exists(dbnsfp_file):
        print(f"❌ dbNSFP file not found for {target_chr}!")
        print(f"   Expected: {dbnsfp_file}")
        sys.exit(1)
    
    # Determine chromosome type for reporting
    if target_chr[3:].isdigit():
        chr_type = "Autosome"
    elif target_chr in ["chrX", "chrY"]:
        chr_type = "Sex chromosome"
    elif target_chr == "chrM":
        chr_type = "Mitochondrial"
    else:
        chr_type = "Other"
    
    print(f"🎯 Target: {target_chr} ({chr_type})")
    print(f"📍 Array ID: {array_id}")
    print()
    
    # Process the chromosome
    start_time = time.time()
    success, variant_count, annotated_count = process_single_chromosome(target_chr)
    total_time = time.time() - start_time
    
    # Final summary
    print(f"\n{'='*70}")
    print("🏆 ARRAY JOB COMPLETE")
    print("=" * 70)
    
    if success:
        ann_file = f"Annovar_Merge/{target_chr}_concatenated_annotations.csv"
        file_size = os.path.getsize(ann_file) / (1024*1024) if os.path.exists(ann_file) else 0
        
        print(f"✅ Successfully processed {target_chr} ({chr_type})")
        print(f"📁 Output: {ann_file}")
        print(f"📊 Variants: {variant_count:,} total, {annotated_count:,} annotated")
        print(f"📄 Size: {file_size:.1f} MB")
        print(f"⏱️ Time: {total_time:.1f} seconds")
        print(f"🎯 Array ID {array_id} SUCCESSFUL")
        
        # Special message for sex and mitochondrial chromosomes
        if target_chr in ["chrX", "chrY", "chrM"]:
            print(f"🌟 Special chromosome {target_chr} merged with dbNSFP successfully!")
            
    else:
        print(f"❌ Failed to process {target_chr} ({chr_type})")
        print(f"🎯 Array ID {array_id} FAILED")
        sys.exit(1)

def main():
    """Main function - handle command line arguments for all chromosomes including X, Y, M"""
    
    # Check command line arguments
    if len(sys.argv) < 2:
        # No arguments - list available chromosomes
        print("📋 CHROMOSOME DISCOVERY MODE")
        print("=" * 40)
        list_available_chromosomes()
        print("\n💡 TIP: The script automatically handles chromosome name conversion:")
        print("   • chr1, chr2, ..., chr22 (autosomes)")
        print("   • chrX, chrY (sex chromosomes)")  
        print("   • chrM (mitochondrial)")
        print("   • Converts: 23→chrX, 24→chrY, 25→chrM, MT→chrM")
        print("\n🔗 This script merges annotation files with dbNSFP for functional predictions")
        return
    
    # Get array ID from command line
    try:
        array_id = int(sys.argv[1])
    except ValueError:
        print(f"❌ Error: '{sys.argv[1]}' is not a valid integer")
        print("Usage: python Step5-dbNSFP-Merger.py <array_id>")
        print("       python Step5-dbNSFP-Merger.py              # List chromosomes")
        print("\nExamples:")
        print("  python Step5-dbNSFP-Merger.py 1     # chr1")
        print("  python Step5-dbNSFP-Merger.py 23    # chrX") 
        print("  python Step5-dbNSFP-Merger.py 24    # chrY")
        print("  python Step5-dbNSFP-Merger.py 25    # chrM")
        sys.exit(1)
    
    # Process single chromosome
    process_array_chromosome(array_id)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n\n⚠️  Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)