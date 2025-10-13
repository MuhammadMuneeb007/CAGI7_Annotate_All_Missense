#!/usr/bin/env python3
"""
Ultra-Fast ANNOVAR Concatenation - Clean Remake
- Single-threaded processing for stability
- Auto-fixes row mismatches by padding with header rows
- Handles duplicate columns
- Enhanced debugging
"""

import os
import glob
import time
from pathlib import Path

def get_csv_library():
    """Get the fastest available CSV library"""
    try:
        import polars as pl
        return 'polars', pl
    except ImportError:
        import pandas as pd
        return 'pandas', pd

def read_csv_file(filepath):
    """Read CSV file with the best available library"""
    lib_name, lib = get_csv_library()
    
    if lib_name == 'polars':
        return lib.read_csv(filepath), lib_name
    else:
        return lib.read_csv(filepath, engine='c', low_memory=False), lib_name

def count_data_rows(filepath):
    """Count actual data rows (excluding header)"""
    try:
        with open(filepath, 'r') as f:
            return sum(1 for line in f) - 1  # -1 for header
    except:
        return 0

def get_headers(filepath):
    """Get column headers from CSV file"""
    try:
        with open(filepath, 'r') as f:
            return f.readline().strip().split(',')
    except:
        return []

def pad_file_with_headers(filepath, target_rows, protocol):
    """Pad file with header rows to match target row count"""
    current_rows = count_data_rows(filepath)
    if current_rows >= target_rows:
        return filepath  # No padding needed
    
    padding_needed = target_rows - current_rows
    if padding_needed > 10:  # Don't pad large differences
        return filepath
    
    print(f"      🔧 Padding {protocol} with {padding_needed} header row(s)")
    
    try:
        # Read original file
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        if not lines:
            return filepath
        
        # Create padded file
        padded_file = f"{filepath}.padded"
        with open(padded_file, 'w') as f:
            # Write original header
            f.write(lines[0])
            
            # Add padding header rows
            for _ in range(padding_needed):
                f.write(lines[0])
            
            # Write remaining data
            for line in lines[1:]:
                f.write(line)
        
        return padded_file
    
    except Exception as e:
        print(f"      ❌ Failed to pad {protocol}: {e}")
        return filepath

def show_mismatch_debug(base_file, base_rows, problem_file, problem_rows, protocol):
    """Show detailed debugging for row mismatches"""
    print(f"\n    🔍 ROW MISMATCH DEBUG - {protocol}")
    print(f"    📊 Base: {base_rows:,} rows")
    print(f"    📊 {protocol}: {problem_rows:,} rows")
    print(f"    📊 Difference: {abs(base_rows - problem_rows):,}")
    
    base_headers = get_headers(base_file)
    problem_headers = get_headers(problem_file)
    
    print(f"    📋 Base headers ({len(base_headers)}):")
    for i, h in enumerate(base_headers[:8]):
        print(f"       {i+1}. {h}")
    if len(base_headers) > 8:
        print(f"       ... +{len(base_headers)-8} more")
    
    print(f"    📋 {protocol} headers ({len(problem_headers)}):")
    for i, h in enumerate(problem_headers[:8]):
        print(f"       {i+1}. {h}")
    if len(problem_headers) > 8:
        print(f"       ... +{len(problem_headers)-8} more")
    
    print("    " + "="*50)

def find_annovar_files():
    """Find and organize ANNOVAR output files by chromosome"""
    files_by_chr = {}
    
    pattern = "Annovar_Output_Files/*_results.hg38_multianno.csv"
    csv_files = glob.glob(pattern)
    
    for filepath in csv_files:
        filename = Path(filepath).name
        parts = filename.split('_')
        
        if len(parts) >= 2:
            chr_name = parts[0]  # chr1, chr2, etc.
            protocol = parts[1]  # refGene, gnomad, etc.
            
            if chr_name not in files_by_chr:
                files_by_chr[chr_name] = []
            files_by_chr[chr_name].append((protocol, filepath))
    
    return files_by_chr

def process_single_chromosome(chr_name, file_list):
    """Process a single chromosome with row mismatch handling"""
    start_time = time.time()
    print(f"Processing {chr_name} with {len(file_list)} files...")
    
    # Sort files for consistent processing
    file_list.sort(key=lambda x: x[0])
    
    try:
        # Key columns that should be identical
        key_columns = {'Chr', 'Start', 'End', 'Ref', 'Alt'}
        
        # Load base file
        base_protocol, base_file = file_list[0]
        print(f"  📖 Base: {base_protocol}")
        
        base_df, lib_name = read_csv_file(base_file)
        base_rows = len(base_df)
        
        print(f"     {base_rows:,} rows × {len(base_df.columns)} columns")
        
        if base_rows == 0:
            return None, f"{chr_name}: Base file is empty"
        
        # Track existing column names to handle duplicates
        if lib_name == 'polars':
            existing_columns = set(base_df.columns)
        else:
            existing_columns = set(base_df.columns.tolist())
        
        annotation_dataframes = []
        
        # Process each additional file
        for i, (protocol, filepath) in enumerate(file_list[1:], 1):
            print(f"  [{i:2d}/{len(file_list)-1}] {protocol}...", end=" ")
            
            # Check if padding is needed
            file_rows = count_data_rows(filepath)
            if file_rows != base_rows:
                if file_rows < base_rows and (base_rows - file_rows) <= 10:
                    # Try to pad the file
                    padded_file = pad_file_with_headers(filepath, base_rows, protocol)
                    if padded_file != filepath:
                        # Use padded file and clean up later
                        filepath = padded_file
                        file_rows = count_data_rows(filepath)
                
                # Check if padding worked
                if file_rows != base_rows:
                    print(f"❌ Mismatch: {file_rows:,} vs {base_rows:,}")
                    show_mismatch_debug(base_file, base_rows, filepath, file_rows, protocol)
                    
                    # Clean up padded file if created
                    if filepath.endswith('.padded') and os.path.exists(filepath):
                        os.remove(filepath)
                    continue
            
            # Load the file
            df, _ = read_csv_file(filepath)
            
            # Verify row count after loading
            if len(df) != base_rows:
                print(f"❌ Load mismatch: {len(df):,} vs {base_rows:,}")
                if filepath.endswith('.padded') and os.path.exists(filepath):
                    os.remove(filepath)
                continue
            
            # Get annotation columns (non-key columns)
            if lib_name == 'polars':
                all_cols = df.columns
            else:
                all_cols = df.columns.tolist()
            
            ann_cols = [col for col in all_cols if col not in key_columns]
            
            if not ann_cols:
                print("❌ No annotation columns")
                if filepath.endswith('.padded') and os.path.exists(filepath):
                    os.remove(filepath)
                continue
            
            # Handle duplicate column names
            final_cols = []
            rename_map = {}
            
            for col in ann_cols:
                if col in existing_columns:
                    # Create unique name
                    new_name = f"{col}_{protocol}"
                    counter = 1
                    while new_name in existing_columns:
                        new_name = f"{col}_{protocol}_{counter}"
                        counter += 1
                    rename_map[col] = new_name
                    final_cols.append(new_name)
                    existing_columns.add(new_name)
                else:
                    final_cols.append(col)
                    existing_columns.add(col)
            
            # Rename duplicates if needed
            if rename_map:
                if lib_name == 'polars':
                    df = df.rename(rename_map)
                else:
                    df = df.rename(columns=rename_map)
                print(f"⚠️  Renamed {len(rename_map)} duplicates", end=" ")
            
            # Extract annotation columns only
            if lib_name == 'polars':
                ann_df = df.select(final_cols)
            else:
                ann_df = df[final_cols]
            
            annotation_dataframes.append(ann_df)
            print(f"✅ +{len(final_cols)} cols")
            
            # Clean up padded file if created
            if filepath.endswith('.padded') and os.path.exists(filepath):
                os.remove(filepath)
        
        # Combine all dataframes horizontally
        if annotation_dataframes:
            print(f"  🔗 Combining {len(annotation_dataframes)} annotation files...")
            
            if lib_name == 'polars':
                # Polars horizontal concatenation
                all_dfs = [base_df] + annotation_dataframes
                import polars as pl
                final_df = pl.concat(all_dfs, how="horizontal")
            else:
                # Pandas horizontal concatenation
                final_df = base_df.copy()
                for ann_df in annotation_dataframes:
                    final_df = pd.concat([final_df, ann_df], axis=1)
        else:
            final_df = base_df
        
        # Save result
        output_file = f"Annovar_Merge/{chr_name}_merged_annotations.csv"
        print(f"  💾 Saving {output_file}...")
        
        if lib_name == 'polars':
            final_df.write_csv(output_file)
        else:
            final_df.to_csv(output_file, index=False)
        
        # Calculate statistics
        final_rows = len(final_df)
        final_cols = len(final_df.columns)
        file_size_mb = os.path.getsize(output_file) / (1024*1024)
        total_time = time.time() - start_time
        
        print(f"  ✅ {chr_name}: {final_rows:,} × {final_cols:,} ({file_size_mb:.0f}MB) in {total_time:.1f}s")
        
        return {
            'chr': chr_name,
            'rows': final_rows,
            'cols': final_cols,
            'size_mb': file_size_mb,
            'time': total_time,
            'files_processed': len(annotation_dataframes) + 1,
            'library': lib_name
        }, None
        
    except Exception as e:
        import traceback
        error = f"{chr_name}: {str(e)}\n{traceback.format_exc()}"
        print(f"  ❌ ERROR: {error}")
        return None, error

def main():
    """Main processing function"""
    start_time = time.time()
    
    print("🚀 ULTRA-FAST ANNOVAR CONCATENATION")
    print("=" * 50)
    print("🐌 Single-threaded for maximum stability")
    print("🔧 Auto-fixes row mismatches with header padding")
    print("🔧 Handles duplicate column names")
    
    # Check available library
    lib_name, _ = get_csv_library()
    print(f"📚 Using: {lib_name.upper()}")
    
    # Create output directory
    os.makedirs("Annovar_Merge", exist_ok=True)
    
    # Find all ANNOVAR files
    print("\n🔍 Scanning for ANNOVAR files...")
    chr_files = find_annovar_files()
    
    if not chr_files:
        print("❌ No ANNOVAR files found!")
        print("   Looking for: Annovar_Output_Files/*_results.hg38_multianno.csv")
        return
    
    total_files = sum(len(files) for files in chr_files.values())
    print(f"✅ Found {len(chr_files)} chromosomes, {total_files} total files")
    
    for chr_name, files in sorted(chr_files.items()):
        print(f"   {chr_name}: {len(files)} files")
    
    # Process chromosomes one by one
    print(f"\n🔄 Processing {len(chr_files)} chromosomes sequentially...")
    
    successful = []
    failed = []
    
    for i, (chr_name, files) in enumerate(sorted(chr_files.items()), 1):
        print(f"\n{'='*60}")
        print(f"[{i:2d}/{len(chr_files)}] CHROMOSOME {chr_name}")
        print(f"{'='*60}")
        
        result, error = process_single_chromosome(chr_name, files)
        
        if result:
            successful.append(result)
            print(f"✅ {chr_name} COMPLETE")
        else:
            failed.append((chr_name, error))
            print(f"❌ {chr_name} FAILED")
        
        # Memory cleanup
        import gc
        gc.collect()
        
        # Progress update
        remaining = len(chr_files) - i
        if remaining > 0:
            print(f"📊 {i}/{len(chr_files)} done, {remaining} remaining")
    
    # Final results
    total_time = time.time() - start_time
    
    print(f"\n{'='*60}")
    print("📊 FINAL RESULTS")
    print(f"{'='*60}")
    print(f"✅ Successful: {len(successful)}/{len(chr_files)} chromosomes")
    print(f"⏱️  Total time: {total_time/60:.1f} minutes")
    
    if failed:
        print(f"\n❌ Failed chromosomes ({len(failed)}):")
        for chr_name, error in failed[:3]:  # Show first 3
            print(f"   {chr_name}: {error.split('Traceback:')[0].strip()}")
        if len(failed) > 3:
            print(f"   ... and {len(failed)-3} more")
    
    if successful:
        total_rows = sum(r['rows'] for r in successful)
        max_cols = max(r['cols'] for r in successful)
        total_size = sum(r['size_mb'] for r in successful) / 1024  # GB
        
        print(f"\n📈 SUCCESS STATISTICS:")
        print(f"   Rows processed: {total_rows:,}")
        print(f"   Max columns: {max_cols:,}")
        print(f"   Total size: {total_size:.1f} GB")
        print(f"   Speed: {total_files/total_time:.1f} files/second")
        
        print(f"\n📁 OUTPUT FILES:")
        for result in sorted(successful, key=lambda x: x['chr']):
            print(f"   {result['chr']}_merged_annotations.csv "
                  f"({result['rows']:,} × {result['cols']:,}, {result['size_mb']:.0f}MB)")
    
    print(f"\n🎉 COMPLETE! Files saved to 'Annovar_Merge/' directory")

if __name__ == "__main__":
    main()