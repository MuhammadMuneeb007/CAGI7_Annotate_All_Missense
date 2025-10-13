#!/usr/bin/env python3
"""
ARRAY JOB ANNOVAR CONCATENATION (POLARS)
Process single chromosome based on SLURM array task ID
Supports all chromosomes: 1-22, X, Y, M (mitochondrial)

Usage: 
  python Step4-MergeFiles.py ${SLURM_ARRAY_TASK_ID}  # SLURM array job
  python Step4-MergeFiles.py 1                       # Process chr1
  python Step4-MergeFiles.py 23                      # Process chrX  
  python Step4-MergeFiles.py 24                      # Process chrY
  python Step4-MergeFiles.py 25                      # Process chrM
  python Step4-MergeFiles.py                         # List available chromosomes

Chromosome Naming Conversion:
  - Handles various formats: chr1, 1, chrX, X, chrY, Y, chrM, M, chrMT, MT
  - Converts numeric codes: 23→chrX, 24→chrY, 25→chrM
  - Standardizes all to chr1, chr2, ..., chr22, chrX, chrY, chrM format
"""

import os
import glob
import polars as pl
from collections import defaultdict
import time
import gc
import sys

# Create merge output directory
os.makedirs("Annovar_Merge", exist_ok=True)


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


def get_chromosome_files():
    """Get all chromosome files organized by chromosome, including X, Y, M"""
    chr_files = defaultdict(list)
    
    # Find all CSV output files with various patterns
    patterns = [
        "Annovar_Output_Files/chr*_results.hg38_multianno.csv",
        "Annovar_Output_Files/chr*_*_results.hg38_multianno.csv",  # In case of protocol in name
    ]
    
    all_csv_files = []
    for pattern in patterns:
        all_csv_files.extend(glob.glob(pattern))
    
    # Remove duplicates
    csv_files = list(set(all_csv_files))
    
    print(f"🔍 Found {len(csv_files)} total annotation files")
    
    for csv_file in csv_files:
        filename = os.path.basename(csv_file)
        
        # Extract chromosome from filename like "chr1_refGene_results.hg38_multianno.csv"
        parts = filename.split('_')
        if len(parts) >= 2:
            chr_part = parts[0]  # Gets "chr1", "chr2", "chrX", etc.
            protocol = parts[1]  # Gets protocol name
            
            # Normalize chromosome name
            normalized_chr = normalize_chromosome_name(chr_part)
            
            chr_files[normalized_chr].append((protocol, csv_file))
            
            # Debug output for X, Y, M chromosomes
            if normalized_chr in ["chrX", "chrY", "chrM"]:
                print(f"  📍 Found {normalized_chr}: {filename}")
    
    return chr_files


def get_chromosome_list():
    """Get properly sorted list of available chromosomes (1-22, X, Y, M)"""
    chr_files = get_chromosome_files()
    chromosomes = list(chr_files.keys())
    
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
    
    sorted_chromosomes = sorted(chromosomes, key=chr_sort_key)
    
    # Show chromosome summary
    print(f"\n📋 Available chromosomes: {len(sorted_chromosomes)}")
    autosomes = [c for c in sorted_chromosomes if c.startswith("chr") and c[3:].isdigit()]
    sex_chroms = [c for c in sorted_chromosomes if c in ["chrX", "chrY"]]
    mito = [c for c in sorted_chromosomes if c == "chrM"]
    
    print(f"  🧬 Autosomes: {len(autosomes)} ({', '.join(autosomes[:3])}...{', '.join(autosomes[-2:]) if len(autosomes) > 3 else ''})")
    if sex_chroms:
        print(f"  🚻 Sex chromosomes: {', '.join(sex_chroms)}")
    if mito:
        print(f"  🔬 Mitochondrial: {', '.join(mito)}")
    
    return sorted_chromosomes


def concatenate_chromosome_files(chr_name, file_list):
    """Simple horizontal concatenation - optimized for all chromosome types"""
    start_time = time.time()
    
    # Special handling for different chromosome types
    if chr_name[3:].isdigit():
        chr_type = "autosome"
    elif chr_name in ["chrX", "chrY"]:
        chr_type = "sex chromosome"
    elif chr_name == "chrM":
        chr_type = "mitochondrial"
    else:
        chr_type = "chromosome"
    
    print(f"🧬 Processing {chr_name} ({chr_type})...")
    print(f"📁 Found {len(file_list)} protocol files")
    
    if not file_list:
        print(f"❌ No files found for {chr_name}")
        return None
    
    # Sort files by protocol for consistent processing
    file_list.sort(key=lambda x: x[0])
    
    # Key columns that should be identical across all files
    key_columns = ['Chr', 'Start', 'End', 'Ref', 'Alt']
    
    # Load the first file as base
    first_protocol, first_file = file_list[0]
    print(f"  📂 Loading base: {first_protocol}...", end=" ")
    
    try:
        base_df = pl.read_csv(
            first_file,
            comment_prefix="#",
            ignore_errors=True,
            schema_overrides={"Chr": pl.Utf8}
        )
        base_rows = base_df.height
        base_cols = len(base_df.columns)
        print(f"✓ {base_rows:,} rows, {base_cols} cols")
        
        if base_df.is_empty():
            print(f"    ❌ ERROR: Base file is empty")
            return None
        
        # Special validation for mitochondrial chromosome
        if chr_name == "chrM" and base_rows > 50000:
            print(f"    ⚠️ Warning: Unusually high variant count for mitochondrial chromosome")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return None
    
    # Track column names to handle duplicates
    used_columns = set(base_df.columns)
    current_df = base_df
    
    # Process remaining files
    successful_concatenations = 0
    row_mismatches = 0
    
    for i, (protocol, csv_file) in enumerate(file_list[1:], 1):
        print(f"  🔄 [{i:2d}/{len(file_list)-1}] Loading {protocol}...", end=" ")
        
        try:
            # Load annotation file
            ann_df = pl.read_csv(
                csv_file,
                comment_prefix="#",
                ignore_errors=True,
                schema_overrides={"Chr": pl.Utf8}
            )
            
            # Check row count match
            if ann_df.height != base_rows:
                print(f"❌ Row mismatch ({ann_df.height:,} vs {base_rows:,})")
                row_mismatches += 1
                continue
            
            # Get annotation columns (exclude key columns for concatenation)
            annotation_cols = [col for col in ann_df.columns if col not in key_columns]
            
            if not annotation_cols:
                print("❌ No annotation columns")
                continue
            
            # Handle duplicate column names by adding protocol prefix
            rename_map = {}
            final_cols = []
            
            for col in annotation_cols:
                if col in used_columns:
                    # Column already exists, rename with protocol prefix
                    new_col_name = f"{protocol}_{col}"
                    counter = 1
                    while new_col_name in used_columns:
                        new_col_name = f"{protocol}_{col}_{counter}"
                        counter += 1
                    rename_map[col] = new_col_name
                    final_cols.append(new_col_name)
                    used_columns.add(new_col_name)
                else:
                    final_cols.append(col)
                    used_columns.add(col)
            
            # Select and rename annotation columns
            ann_subset = ann_df.select(annotation_cols)
            if rename_map:
                ann_subset = ann_subset.rename(rename_map)
            
            # Simple horizontal concatenation
            current_df = current_df.hstack(ann_subset)
            successful_concatenations += 1
            print(f"✓ +{len(annotation_cols)} cols")
            
        except Exception as e:
            print(f"❌ Error: {e}")
            continue
    
    # Save results
    print(f"  💾 Saving...", end=" ")
    save_start = time.time()
    output_file = f"Annovar_Merge/{chr_name}_concatenated_annotations.csv"
    current_df.write_csv(output_file)
    save_time = time.time() - save_start
    
    # Calculate file size and timing
    file_size = os.path.getsize(output_file) / (1024*1024)  # MB
    total_time = time.time() - start_time
    
    print(f"✓ {save_time:.1f}s")
    print(f"  ✅ {output_file}")
    print(f"  📊 {current_df.height:,} rows × {len(current_df.columns):,} columns ({file_size:.0f} MB)")
    print(f"  ✅ Successful concatenations: {successful_concatenations}")
    if row_mismatches > 0:
        print(f"  ⚠️ Row mismatches skipped: {row_mismatches}")
    print(f"  ⏱️ Total time: {total_time:.1f}s")
    
    # Special reporting for different chromosome types
    if chr_name == "chrM":
        print(f"  🔬 Mitochondrial chromosome processing complete")
    elif chr_name in ["chrX", "chrY"]:
        print(f"  🚻 Sex chromosome processing complete")
    
    return output_file


def list_available_chromosomes():
    """List all available chromosomes and their file counts"""
    chr_files = get_chromosome_files()
    
    if not chr_files:
        print("❌ No ANNOVAR output files found in Annovar_Output_Files/")
        print("📁 Expected file pattern: chr*_results.hg38_multianno.csv")
        return
    
    chromosomes = get_chromosome_list()
    total_files = sum(len(files) for files in chr_files.values())
    
    print("📋 AVAILABLE CHROMOSOMES FOR ARRAY PROCESSING")
    print("=" * 60)
    print(f"Found {len(chromosomes)} chromosomes with {total_files} total files")
    print()
    print("Array ID | Chromosome | Files | Type")
    print("-" * 40)
    
    for i, chr_name in enumerate(chromosomes, 1):
        file_count = len(chr_files[chr_name])
        
        # Determine chromosome type
        if chr_name[3:].isdigit():
            chr_type = "Autosome"
        elif chr_name in ["chrX", "chrY"]:
            chr_type = "Sex"
        elif chr_name == "chrM":
            chr_type = "Mitochondrial"
        else:
            chr_type = "Other"
        
        print(f"{i:8d} | {chr_name:10s} | {file_count:5d} | {chr_type}")
    
    print()
    print("USAGE EXAMPLES:")
    print(f"  # SLURM Array Job (process all chromosomes including X, Y, M):")
    print(f"  sbatch --array=1-{len(chromosomes)} script.sh")
    print(f"  ")
    print(f"  # In script.sh:")
    print(f"  python Step4-MergeFiles.py ${{SLURM_ARRAY_TASK_ID}}")
    print(f"  ")
    print(f"  # Single chromosome examples:")
    print(f"  python Step4-MergeFiles.py 1      # Process {chromosomes[0]}")
    
    # Show examples for sex and mitochondrial chromosomes
    for i, chr_name in enumerate(chromosomes, 1):
        if chr_name == "chrX":
            print(f"  python Step4-MergeFiles.py {i}     # Process chrX")
        elif chr_name == "chrY":
            print(f"  python Step4-MergeFiles.py {i}     # Process chrY")
        elif chr_name == "chrM":
            print(f"  python Step4-MergeFiles.py {i}     # Process chrM")
    
    if len(chromosomes) > 0:
        print(f"  python Step4-MergeFiles.py {len(chromosomes)}     # Process {chromosomes[-1]}")
    
    print(f"\n📊 Chromosome Summary:")
    autosomes = [c for c in chromosomes if c[3:].isdigit()]
    sex_chroms = [c for c in chromosomes if c in ["chrX", "chrY"]]
    mito = [c for c in chromosomes if c == "chrM"]
    
    print(f"  • Autosomes (1-22): {len(autosomes)} found")
    print(f"  • Sex chromosomes (X,Y): {len(sex_chroms)} found {sex_chroms}")
    print(f"  • Mitochondrial (M): {len(mito)} found {mito}")


def process_single_chromosome(array_id):
    """Process a single chromosome based on array ID"""
    print("🚀 ARRAY JOB ANNOVAR CONCATENATION")
    print("🧬 Including X, Y, M chromosomes")
    print("=" * 60)
    print(f"Array Task ID: {array_id}")
    
    # Get available chromosomes (properly sorted with X, Y, M)
    chromosomes = get_chromosome_list()
    
    if not chromosomes:
        print("❌ No chromosomes found!")
        print("📁 Check that files exist in Annovar_Output_Files/")
        sys.exit(1)
    
    # Validate array ID
    if array_id < 1 or array_id > len(chromosomes):
        print(f"❌ Array ID {array_id} is out of range!")
        print(f"Valid range: 1-{len(chromosomes)}")
        print(f"Available chromosomes: {', '.join(chromosomes)}")
        print("\nDid you include X, Y, M chromosomes?")
        sys.exit(1)
    
    # Get target chromosome (array IDs start from 1)
    target_chr = chromosomes[array_id - 1]
    chr_files = get_chromosome_files()
    
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
    print(f"📍 Array index: {array_id}/{len(chromosomes)}")
    print(f"📁 Files to process: {len(chr_files[target_chr])}")
    
    # Show protocols for this chromosome
    protocols = [protocol for protocol, _ in chr_files[target_chr]]
    print(f"🧪 Protocols: {', '.join(sorted(set(protocols)))}")
    print()
    
    # Process the chromosome
    start_time = time.time()
    output_file = concatenate_chromosome_files(target_chr, chr_files[target_chr])
    total_time = time.time() - start_time
    
    # Final summary
    print(f"\n{'='*60}")
    print("🏆 ARRAY JOB COMPLETE")
    print("=" * 60)
    
    if output_file:
        file_size = os.path.getsize(output_file) / (1024*1024)  # MB
        print(f"✅ Successfully processed {target_chr} ({chr_type})")
        print(f"📁 Output: {output_file}")
        print(f"📊 Size: {file_size:.1f} MB")
        print(f"⏱️ Time: {total_time:.1f} seconds")
        print(f"🎯 Array ID {array_id} SUCCESSFUL")
        
        # Special message for sex and mitochondrial chromosomes
        if target_chr in ["chrX", "chrY", "chrM"]:
            print(f"🌟 Special chromosome {target_chr} processed successfully!")
            
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
        return
    
    # Get array ID from command line
    try:
        array_id = int(sys.argv[1])
    except ValueError:
        print(f"❌ Error: '{sys.argv[1]}' is not a valid integer")
        print("Usage: python Step4-MergeFiles.py <array_id>")
        print("       python Step4-MergeFiles.py              # List chromosomes")
        print("\nExamples:")
        print("  python Step4-MergeFiles.py 1     # chr1")
        print("  python Step4-MergeFiles.py 23    # chrX") 
        print("  python Step4-MergeFiles.py 24    # chrY")
        print("  python Step4-MergeFiles.py 25    # chrM")
        sys.exit(1)
    
    # Process single chromosome
    process_single_chromosome(array_id)


if __name__ == "__main__":
    main()