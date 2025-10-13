#!/usr/bin/env python3
"""
ESM Predictions Matcher - SIMPLIFIED RIPGREP VERSION
Simplified, robust approach to match ESM predictions with annotation data
"""

import polars as pl
import os
import sys
import time
import argparse
import subprocess
import tempfile
from pathlib import Path
from tqdm import tqdm
import gc
import shutil

# Amino acid mapping
AA_MAP = {
    'A': 'Ala', 'R': 'Arg', 'N': 'Asn', 'D': 'Asp', 'C': 'Cys',
    'Q': 'Gln', 'E': 'Glu', 'G': 'Gly', 'H': 'His', 'I': 'Ile',
    'L': 'Leu', 'K': 'Lys', 'M': 'Met', 'F': 'Phe', 'P': 'Pro',
    'S': 'Ser', 'T': 'Thr', 'W': 'Trp', 'Y': 'Tyr', 'V': 'Val',
    'X': 'Ter'
}

def convert_aa_to_three_letter(aa):
    """Convert single letter amino acid to three letter code"""
    if aa is None or aa == '' or str(aa).upper() == 'NAN':
        return None
    return AA_MAP.get(str(aa).upper(), str(aa))

def extract_hgvs_protein_change(hgvs_str):
    """Extract protein change from HGVS string"""
    if hgvs_str is None or not isinstance(hgvs_str, str) or hgvs_str == '':
        return None
    
    if ':p.' in hgvs_str:
        try:
            return hgvs_str.split(':p.')[1]
        except:
            return None
    return None

def create_protein_change_key(aaref, aaalt, aapos, gene_id):
    """Create protein change key from amino acid information"""
    if any(x is None or str(x) == 'nan' or str(x) == '' for x in [aaref, aaalt, aapos, gene_id]):
        return None
    
    try:
        aa_ref_3 = convert_aa_to_three_letter(aaref)
        aa_alt_3 = convert_aa_to_three_letter(aaalt)
        
        if not aa_ref_3 or not aa_alt_3:
            return None
        
        try:
            pos_int = int(float(aapos))
        except (ValueError, TypeError):
            return None
        
        protein_change = f"p.{aa_ref_3}{pos_int}{aa_alt_3}"
        gene_clean = str(gene_id).strip()
        
        return f"{gene_clean}:{protein_change}"
        
    except Exception as e:
        return None

def extract_target_genes_from_annotation(annotation_file):
    """Extract unique gene IDs from annotation file"""
    print(f"Reading annotation file: {os.path.basename(annotation_file)}")
    
    try:
        # Read with Polars - handle ragged lines from previous ESM runs
        df = pl.read_csv(
            annotation_file,
            null_values=[".", "NA", "nan", "NULL", "null", ""],
            ignore_errors=True,
            infer_schema_length=10000,
            truncate_ragged_lines=True  # Handle inconsistent column counts
        )
        print(f"✓ Loaded {len(df):,} rows, {len(df.columns)} columns")
        
        # Check for required columns
        required_cols = ['aaref', 'aaalt', 'aapos', 'Ensembl_geneid']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            print(f"❌ Missing required columns: {missing_cols}")
            return None, None
        
        print(f"✓ Found required dbNSFP columns: {required_cols}")
        
        # Filter valid rows
        try:
            valid_df = df.filter(
                (pl.col('aaref').is_not_null()) &
                (pl.col('aaalt').is_not_null()) &
                (pl.col('aapos').is_not_null()) &
                (pl.col('Ensembl_geneid').is_not_null()) &
                (pl.col('aaref').cast(pl.Utf8) != '.') &
                (pl.col('aaalt').cast(pl.Utf8) != '.') &
                (pl.col('aapos').cast(pl.Utf8) != '.') &
                (pl.col('Ensembl_geneid').cast(pl.Utf8) != '.')
            )
        except Exception as e:
            print(f"⚠️ Polars filtering failed: {e}")
            df_pandas = df.to_pandas()
            valid_mask = (
                df_pandas['aaref'].notna() & 
                df_pandas['aaalt'].notna() & 
                df_pandas['aapos'].notna() & 
                df_pandas['Ensembl_geneid'].notna() &
                (df_pandas['aaref'].astype(str) != '.') & 
                (df_pandas['aaalt'].astype(str) != '.') &
                (df_pandas['aapos'].astype(str) != '.') & 
                (df_pandas['Ensembl_geneid'].astype(str) != '.')
            )
            valid_df = pl.from_pandas(df_pandas[valid_mask])
        
        print(f"✓ Found {len(valid_df):,} rows with amino acid changes")
        
        if len(valid_df) == 0:
            print("⚠️ No valid amino acid changes found")
            return df.with_columns(pl.lit(None).alias('esm_match_key')), set()
        
        # Extract unique gene IDs
        gene_ids = set()
        gene_list = valid_df.select('Ensembl_geneid').to_series().to_list()
        
        for gene_str in gene_list:
            if gene_str and str(gene_str) != 'nan':
                genes = str(gene_str).split(';')
                for gene in genes:
                    gene_clean = gene.strip()
                    if gene_clean and gene_clean != '.':
                        # Remove version number if present
                        if '.' in gene_clean:
                            gene_clean = gene_clean.split('.')[0]
                        gene_ids.add(gene_clean)
        
        print(f"✓ Extracted {len(gene_ids):,} unique gene IDs")
        
        # Create ESM keys
        print("Creating ESM match keys...")
        valid_pandas = valid_df.to_pandas()
        esm_keys = []
        
        with tqdm(desc="Creating keys", total=len(valid_pandas)) as pbar:
            for _, row in valid_pandas.iterrows():
                genes = str(row['Ensembl_geneid']).split(';')
                aa_positions = str(row['aapos']).split(';')
                
                gene_id = genes[0].strip() if genes and genes[0] != 'nan' else None
                aa_pos = aa_positions[0].strip() if aa_positions and aa_positions[0] != 'nan' else None
                
                if gene_id and aa_pos:
                    # Remove version number
                    if '.' in gene_id:
                        gene_id = gene_id.split('.')[0]
                    
                    key = create_protein_change_key(
                        row['aaref'], row['aaalt'], aa_pos, gene_id
                    )
                else:
                    key = None
                
                esm_keys.append(key)
                pbar.update(1)
        
        # Add keys back to the original dataframe
        df_pandas = df.to_pandas()
        df_pandas['esm_match_key'] = None
        
        # Map keys to valid rows
        valid_indices = valid_df.to_pandas().index
        for i, key in enumerate(esm_keys):
            if i < len(valid_indices):
                df_pandas.loc[valid_indices[i], 'esm_match_key'] = key
        
        final_df = pl.from_pandas(df_pandas)
        
        valid_keys = final_df.filter(pl.col('esm_match_key').is_not_null()).height
        print(f"✓ Created {valid_keys:,} ESM match keys")
        
        # Print sample keys
        sample_keys = final_df.filter(pl.col('esm_match_key').is_not_null()).select('esm_match_key').head(5).to_series().to_list()
        if sample_keys:
            print(f"Sample annotation keys:")
            for i, key in enumerate(sample_keys, 1):
                print(f"  {i}. {key}")
        
        return final_df, gene_ids
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None, None

def get_esm_file_header(esm_file):
    """Get the header from the ESM file"""
    try:
        with open(esm_file, 'r') as f:
            header_line = f.readline().strip()
        
        header = header_line.split('\t')
        print(f"ESM file header: {len(header)} columns")
        print(f"First 5 columns: {header[:5]}")
        
        # Verify expected columns
        expected = ['GENCODE.v43.ENSG', 'GENCODE.v43.ENST', 'GENCODE.v43.ENSP', 'HGVS.p']
        found = [col for col in expected if col in header]
        print(f"Expected columns found: {found}")
        
        # Find ESM score columns
        esm_cols = [col for col in header if 'esm' in col.lower()]
        print(f"ESM score columns: {len(esm_cols)} found")
        
        return header_line
        
    except Exception as e:
        print(f"❌ Could not read ESM file header: {e}")
        return None

def filter_esm_with_ripgrep(esm_file, gene_ids, output_file):
    """Filter ESM file using ripgrep"""
    print(f"Filtering ESM file for {len(gene_ids):,} genes...")
    
    if not gene_ids:
        print("⚠️ No gene IDs to filter")
        return False
    
    # Create pattern for ripgrep
    gene_list = sorted(list(gene_ids))
    pattern = '|'.join(gene_list)
    
    print(f"Pattern length: {len(pattern)} characters")
    
    try:
        # Get the original header
        header = get_esm_file_header(esm_file)
        if not header:
            return False
        
        # Check if ripgrep is available
        try:
            subprocess.run(["rg", "--version"], capture_output=True, check=True)
            use_rg = True
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("ripgrep not found, using grep...")
            use_rg = False
        
        # Run the filtering command
        if use_rg:
            cmd = ["rg", pattern, esm_file]
        else:
            cmd = ["grep", "-E", pattern, esm_file]
        
        print(f"Running {'ripgrep' if use_rg else 'grep'}...")
        start_time = time.time()
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        filter_time = time.time() - start_time
        
        # Write output file
        with open(output_file, 'w') as outf:
            # Write header
            outf.write(header + '\n')
            # Write filtered data
            if result.stdout:
                outf.write(result.stdout)
        
        if os.path.exists(output_file):
            file_size = os.path.getsize(output_file)
            line_count = sum(1 for _ in open(output_file)) - 1
            print(f"✓ Filtered file created: {file_size / (1024*1024):.1f} MB, {line_count:,} lines")
            print(f"  Filter time: {filter_time:.1f}s")
            return True
        else:
            print("❌ Failed to create filtered file")
            return False
            
    except Exception as e:
        print(f"❌ Filtering error: {e}")
        return False

def process_esm_matches(filtered_esm_file, annotation_keys_set):
    """Process filtered ESM file to find matches"""
    print(f"Processing filtered ESM file...")
    
    try:
        # Read the filtered file
        df = pl.read_csv(
            filtered_esm_file,
            separator='\t',
            null_values=["", "NA", "NaN", "nan", "NULL", "null", "."],
            ignore_errors=True
        )
        
        print(f"✓ Loaded ESM data: {len(df):,} rows")
        
        # Check columns
        print(f"ESM columns: {list(df.columns)[:5]} ...")
        
        # Convert to pandas for processing
        df_pandas = df.to_pandas()
        
        matches = {}
        esm_keys_created = []
        
        print("Processing ESM rows...")
        for idx, row in tqdm(df_pandas.iterrows(), total=len(df_pandas), desc="Processing ESM"):
            try:
                # Get gene ID and strip version
                gene_id = str(row.get('GENCODE.v43.ENSG', ''))
                if '.' in gene_id:
                    gene_id_clean = gene_id.split('.')[0]
                else:
                    gene_id_clean = gene_id
                
                # Get protein change
                hgvs = str(row.get('HGVS.p', ''))
                protein_change = extract_hgvs_protein_change(hgvs)
                
                if gene_id_clean and protein_change:
                    esm_key = f"{gene_id_clean}:p.{protein_change}"
                    
                    # Store sample keys for debugging
                    if len(esm_keys_created) < 5:
                        esm_keys_created.append(esm_key)
                    
                    # Check if this key matches annotation
                    if esm_key in annotation_keys_set:
                        # Create match data
                        match_data = {
                            'GENCODE.v43.ENSG': gene_id,
                            'GENCODE.v43.ENST': row.get('GENCODE.v43.ENST', ''),
                            'GENCODE.v43.ENSP': row.get('GENCODE.v43.ENSP', ''),
                            'HGVS.p': hgvs
                        }
                        
                        # Add ESM score columns
                        for col in df_pandas.columns:
                            if 'esm' in col.lower() or 'score' in col.lower():
                                try:
                                    value = row.get(col)
                                    if value is not None and str(value).strip():
                                        match_data[col] = float(value)
                                    else:
                                        match_data[col] = None
                                except:
                                    match_data[col] = str(value) if value is not None else None
                        
                        matches[esm_key] = match_data
                        
            except Exception as e:
                continue
        
        # Show debugging info
        print(f"Sample ESM keys created:")
        for i, key in enumerate(esm_keys_created, 1):
            print(f"  {i}. {key}")
        
        print(f"Sample annotation keys:")
        sample_ann_keys = list(annotation_keys_set)[:5]
        for i, key in enumerate(sample_ann_keys, 1):
            print(f"  {i}. {key}")
        
        print(f"✓ Found {len(matches):,} ESM matches")
        
        return matches
        
    except Exception as e:
        print(f"❌ Error processing ESM file: {e}")
        return {}

def merge_esm_data(annotation_df, esm_matches):
    """Merge ESM data with annotation dataframe"""
    print(f"Merging ESM data...")
    
    if not esm_matches:
        print("⚠️ No ESM matches found - adding empty ESM columns")
        esm_columns = [
            'GENCODE.v43.ENSG', 'GENCODE.v43.ENST', 'GENCODE.v43.ENSP', 'HGVS.p',
            'esm1v_t33_650M_UR90S_1', 'esm1v_t33_650M_UR90S_2', 'esm1v_t33_650M_UR90S_3',
            'esm1v_t33_650M_UR90S_4', 'esm1v_t33_650M_UR90S_5'
        ]
        return annotation_df.with_columns([pl.lit(None).alias(col) for col in esm_columns])
    
    try:
        print(f"Processing {len(esm_matches):,} ESM matches for merging...")
        
        # Create standardized records with explicit type control
        standardized_data = []
        
        # Get all unique columns from matches
        all_columns = set()
        for data in esm_matches.values():
            all_columns.update(data.keys())
        
        # Remove esm_match_key from data columns (we'll add it separately)
        data_columns = [col for col in all_columns if col != 'esm_match_key']
        
        print(f"ESM data columns found: {len(data_columns)}")
        
        # Process each match with careful type handling
        for key, data in esm_matches.items():
            record = {}
            
            # Add the match key as string (this is critical)
            record['esm_match_key'] = str(key)
            
            # Add all data columns with appropriate type conversion
            for col in data_columns:
                value = data.get(col)
                
                # Handle different column types
                if 'esm' in col.lower() or 'score' in col.lower():
                    # ESM score columns - convert to float
                    try:
                        if value is not None and str(value).strip() and str(value).lower() not in ['nan', 'na', 'null']:
                            record[col] = float(value)
                        else:
                            record[col] = None
                    except (ValueError, TypeError):
                        record[col] = None
                else:
                    # Text columns - keep as string
                    record[col] = str(value) if value is not None else None
            
            standardized_data.append(record)
        
        # Create Polars DataFrame directly (skip pandas to avoid type confusion)
        print(f"Creating Polars DataFrame with {len(standardized_data):,} records...")
        
        # Define explicit schema to avoid inference issues
        schema = {'esm_match_key': pl.Utf8}  # Ensure this stays as string
        
        # Add schema for other columns based on first record
        if standardized_data:
            sample_record = standardized_data[0]
            for col, value in sample_record.items():
                if col == 'esm_match_key':
                    continue
                elif 'esm' in col.lower() or 'score' in col.lower():
                    schema[col] = pl.Float64
                else:
                    schema[col] = pl.Utf8
        
        print(f"Using schema with {len(schema)} columns")
        
        # Create DataFrame with explicit schema
        esm_df = pl.DataFrame(standardized_data, schema=schema)
        print(f"✓ ESM DataFrame created: {len(esm_df):,} rows, {len(esm_df.columns)} columns")
        
        # Verify the esm_match_key column type
        esm_key_dtype = esm_df.select('esm_match_key').dtypes[0]
        print(f"ESM esm_match_key dtype: {esm_key_dtype}")
        
        # Check annotation esm_match_key dtype
        ann_key_dtype = annotation_df.select('esm_match_key').dtypes[0]
        print(f"Annotation esm_match_key dtype: {ann_key_dtype}")
        
        # Ensure both are strings
        if esm_key_dtype != pl.Utf8:
            esm_df = esm_df.with_columns(pl.col('esm_match_key').cast(pl.Utf8))
            print("✓ Converted ESM esm_match_key to string")
        
        if ann_key_dtype != pl.Utf8:
            annotation_df = annotation_df.with_columns(pl.col('esm_match_key').cast(pl.Utf8))
            print("✓ Converted annotation esm_match_key to string")
        
        # Check for existing ESM columns and remove them to avoid conflicts
        # BUT preserve the esm_match_key column we just created
        existing_esm_cols = [col for col in annotation_df.columns 
                           if ('esm' in col.lower() or 'GENCODE.v43.' in col or 'HGVS.p' in col) 
                           and col != 'esm_match_key']  # Keep our join key!
        
        if existing_esm_cols:
            print(f"Removing {len(existing_esm_cols)} existing ESM columns to avoid conflicts...")
            print(f"Preserving 'esm_match_key' column for joining...")
            annotation_df = annotation_df.drop(existing_esm_cols)
            print(f"✓ Cleaned annotation dataframe now has {len(annotation_df.columns)} columns")
        
        # Verify esm_match_key still exists
        if 'esm_match_key' not in annotation_df.columns:
            print("❌ ERROR: esm_match_key column was accidentally removed!")
            return None
        
        print(f"✓ esm_match_key column preserved for joining")
        
        # Perform the merge with debugging
        original_count = len(annotation_df)
        print(f"Performing left join on {original_count:,} annotation rows...")
        
        # Debug: Check some sample keys before join
        ann_sample_keys = annotation_df.filter(pl.col('esm_match_key').is_not_null()).select('esm_match_key').head(3).to_series().to_list()
        esm_sample_keys = esm_df.select('esm_match_key').head(3).to_series().to_list()
        
        print(f"Sample annotation keys: {ann_sample_keys}")
        print(f"Sample ESM keys: {esm_sample_keys}")
        
        # Check for actual overlapping keys
        ann_keys_set = set(annotation_df.filter(pl.col('esm_match_key').is_not_null()).select('esm_match_key').to_series().to_list())
        esm_keys_set = set(esm_df.select('esm_match_key').to_series().to_list())
        
        overlap = ann_keys_set.intersection(esm_keys_set)
        print(f"Actual key overlap: {len(overlap):,} keys match between annotation and ESM")
        print(f"Sample overlapping keys: {list(overlap)[:5]}")
        
        if len(overlap) == 0:
            print("❌ NO OVERLAPPING KEYS FOUND! This explains the 0% annotation rate.")
            print("Annotation keys sample:", list(ann_keys_set)[:5])
            print("ESM keys sample:", list(esm_keys_set)[:5])
            return None
        
        merged_df = annotation_df.join(esm_df, on='esm_match_key', how='left')
        
        if len(merged_df) != original_count:
            print(f"❌ ERROR: Row count changed from {original_count:,} to {len(merged_df):,}")
            return None
        
        print(f"✓ Row count preserved: {original_count:,}")
        
        # Check what columns exist after join
        esm_related_cols = [col for col in merged_df.columns if 'GENCODE' in col or 'esm' in col.lower()]
        print(f"ESM-related columns after join: {esm_related_cols}")
        
        # Count matches - Debug version
        print("Debugging merge results...")
        
        # Check if ESM columns exist
        esm_cols_in_result = [col for col in merged_df.columns if 'GENCODE.v43.ENSG' in col or 'esm' in col.lower()]
        print(f"ESM-related columns in result: {esm_cols_in_result[:10]}")
        
        # Check for any non-null ESM data
        gencode_col = None
        for col in merged_df.columns:
            if 'GENCODE.v43.ENSG' in col:
                gencode_col = col
                break
        
        if gencode_col:
            print(f"Using column '{gencode_col}' to count ESM matches...")
            esm_annotated = merged_df.filter(pl.col(gencode_col).is_not_null()).height
            
            # Also check how many have the esm_match_key populated
            esm_with_keys = merged_df.filter(pl.col('esm_match_key').is_not_null()).height
            print(f"Rows with esm_match_key: {esm_with_keys:,}")
            
            # Sample the merged data to see what happened
            sample_merged = merged_df.filter(pl.col('esm_match_key').is_not_null()).head(5)
            if len(sample_merged) > 0:
                print("Sample merged rows with ESM keys:")
                sample_df = sample_merged.select(['esm_match_key', gencode_col] + 
                                               [col for col in merged_df.columns if 'esm1v' in col][:3]).to_pandas()
                for i, row in sample_df.iterrows():
                    print(f"  {i+1}. Key: {row['esm_match_key']}, ENSG: {row[gencode_col]}")
            else:
                print("No rows found with populated ESM keys after merge!")
                
                # Check a few random annotation keys to see if they exist in ESM matches
                print("Checking if annotation keys exist in ESM matches...")
                ann_sample_keys = merged_df.filter(pl.col('esm_match_key').is_not_null()).select('esm_match_key').head(5).to_series().to_list()
                print(f"Sample annotation keys from merged data: {ann_sample_keys}")
        else:
            print("No GENCODE.v43.ENSG column found in merged result!")
            esm_annotated = 0
        
        esm_rate = (esm_annotated / original_count) * 100 if original_count > 0 else 0
        
        print(f"ESM Results:")
        print(f"  • Variants with ESM predictions: {esm_annotated:,}")
        print(f"  • ESM annotation rate: {esm_rate:.1f}%")
        
        return merged_df
        
    except Exception as e:
        print(f"❌ Error in merge_esm_data: {e}")
        import traceback
        traceback.print_exc()
        return None

def save_result(df, output_file):
    """Save the merged result"""
    print(f"Saving result...")
    
    # Create backup
    if os.path.exists(output_file):
        backup_file = output_file + ".backup"
        try:
            shutil.copy2(output_file, backup_file)
            print(f"✓ Backup created: {backup_file}")
        except Exception as e:
            print(f"⚠️ Warning: Could not create backup: {e}")
    
    # Remove helper column
    if 'esm_match_key' in df.columns:
        df = df.drop('esm_match_key')
    
    # Save
    start_time = time.time()
    df.write_csv(output_file)
    save_time = time.time() - start_time
    
    file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
    print(f"✓ Saved: {file_size_mb:.1f} MB in {save_time:.1f}s")

def parse_chromosome_input(chromosome_input):
    """Parse chromosome input and convert to standard format"""
    chromosome_map = {'23': 'chrX', '24': 'chrY', '25': 'chrM'}
    
    chr_input = str(chromosome_input).strip().lower()
    
    if chr_input.startswith('chr'):
        chr_input = chr_input[3:]
    
    if chr_input in chromosome_map:
        return chromosome_map[chr_input]
    
    try:
        chr_num = int(chr_input)
        if 1 <= chr_num <= 22:
            return f"chr{chr_num}"
        elif chr_num == 23:
            return 'chrX'
        elif chr_num == 24:
            return 'chrY'
        elif chr_num == 25:
            return 'chrM'
        else:
            raise ValueError(f"Invalid chromosome number: {chr_num}")
    except ValueError:
        pass
    
    text_map = {'x': 'chrX', 'y': 'chrY', 'm': 'chrM', 'mt': 'chrM', 'mito': 'chrM'}
    
    if chr_input in text_map:
        return text_map[chr_input]
    
    if chr_input.startswith('chr'):
        return chromosome_input
    
    return f"chr{chr_input}"

def process_chromosome_esm(chr_name, esm_file, annotation_dir="Annovar_Merge"):
    """Process a single chromosome for ESM matching"""
    annotation_file = f"{annotation_dir}/{chr_name}_concatenated_annotations.csv"
    
    print(f"\n{'='*60}")
    print(f"PROCESSING {chr_name.upper()} + ESM MATCHING")
    print(f"{'='*60}")
    
    if not os.path.exists(annotation_file):
        print(f"❌ Annotation file not found: {annotation_file}")
        return False
    
    if not os.path.exists(esm_file):
        print(f"❌ ESM file not found: {esm_file}")
        return False
    
    temp_dir = None
    try:
        start_time = time.time()
        
        # Create temp directory
        temp_dir = tempfile.mkdtemp(prefix=f"esm_{chr_name}_")
        filtered_esm_file = os.path.join(temp_dir, "filtered_esm.tsv")
        
        # Step 1: Extract target genes from annotation
        print("STEP 1: EXTRACT TARGET GENES")
        print("-" * 40)
        annotation_df, gene_ids = extract_target_genes_from_annotation(annotation_file)
        if annotation_df is None or gene_ids is None:
            return False
        
        # Step 2: Filter ESM file
        print("\nSTEP 2: FILTER ESM FILE")
        print("-" * 40)
        if not filter_esm_with_ripgrep(esm_file, gene_ids, filtered_esm_file):
            print("⚠️ Continuing with empty ESM data...")
            esm_matches = {}
        else:
            # Step 3: Process filtered ESM file
            print("\nSTEP 3: PROCESS ESM MATCHES")
            print("-" * 40)
            annotation_keys = annotation_df.filter(
                pl.col('esm_match_key').is_not_null()
            ).select('esm_match_key').to_series().to_list()
            annotation_keys_set = set(annotation_keys)
            print(f"Ready for matching: {len(annotation_keys):,} annotation keys")
            
            esm_matches = process_esm_matches(filtered_esm_file, annotation_keys_set)
        
        # Step 4: Merge data
        print("\nSTEP 4: MERGE DATA")
        print("-" * 40)
        merged_df = merge_esm_data(annotation_df, esm_matches)
        if merged_df is None:
            return False
        
        # Step 5: Save result
        print("\nSTEP 5: SAVE RESULT")
        print("-" * 40)
        save_result(merged_df, annotation_file)
        
        # Summary
        total_time = time.time() - start_time
        print(f"\n✓ {chr_name.upper()} + ESM COMPLETED!")
        print(f"  Time: {total_time:.1f} seconds")
        print(f"  Final: {len(merged_df):,} rows, {len(merged_df.columns)} columns")
        
        # Clean up
        del annotation_df, merged_df, esm_matches
        gc.collect()
        
        return True
        
    except Exception as e:
        print(f"❌ Error processing {chr_name}: {e}")
        return False
        
    finally:
        # Cleanup temp directory
        if temp_dir and os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
                print(f"✓ Cleaned up temp directory")
            except:
                print(f"⚠️ Could not clean up: {temp_dir}")

def main():
    """Main execution"""
    parser = argparse.ArgumentParser(
        description='ESM Predictions Matcher - Simplified Version',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python esm_matcher.py 1
  python esm_matcher.py 22
  python esm_matcher.py X
        """
    )
    
    parser.add_argument(
        'chromosome',
        help='Chromosome identifier (1-25, X, Y, M, chrX, etc.)'
    )
    
    parser.add_argument(
        '--esm-file',
        default='annovar/humandb/IGVFFI8105TNNO.tsv',
        help='Path to ESM predictions file'
    )
    
    parser.add_argument(
        '--annotation-dir',
        default='Annovar_Merge',
        help='Directory containing annotation files'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("ESM PREDICTIONS MATCHER - SIMPLIFIED")
    print("From Annotation Files → ESM Data")
    print("=" * 60)
    
    # Parse chromosome input
    try:
        target_chromosome = parse_chromosome_input(args.chromosome)
        print(f"Input: {args.chromosome} -> {target_chromosome}")
    except ValueError as e:
        print(f"❌ {e}")
        sys.exit(1)
    
    # Configuration
    esm_file = args.esm_file
    annotation_dir = args.annotation_dir
    
    # Check ESM file exists
    if not os.path.exists(esm_file):
        alt_paths = [
            "IGVFFI8105TNNO.tsv", 
            "humandb/IGVFFI8105TNNO.tsv",
            os.path.join(annotation_dir, "IGVFFI8105TNNO.tsv")
        ]
        found = False
        for path in alt_paths:
            if os.path.exists(path):
                esm_file = path
                found = True
                break
        if not found:
            print(f"❌ ESM file not found: {args.esm_file}")
            sys.exit(1)
    
    print(f"Target chromosome: {target_chromosome}")
    print(f"ESM file: {esm_file}")
    print(f"Annotation directory: {annotation_dir}")
    print()
    
    # Process the chromosome
    success = process_chromosome_esm(target_chromosome, esm_file, annotation_dir)
    
    if success:
        print(f"\n🎉 SUCCESS! {target_chromosome} annotation + ESM merge completed!")
        print(f"📁 Output: {annotation_dir}/{target_chromosome}_concatenated_annotations.csv")
    else:
        print(f"\n❌ FAILED to process {target_chromosome}")
        sys.exit(1)

if __name__ == "__main__":
    main()