#!/usr/bin/env python3
"""
ARRAY JOB ALPHAMISSENSE ANNOTATION
Process single chromosome based on SLURM array task ID
Supports all chromosomes: 1-22, X, Y, M (mitochondrial)
Memory-efficient chunked processing for large chromosomes

Usage: 
  python Step6-AlphaMissense-Annotator.py ${SLURM_ARRAY_TASK_ID}  # SLURM array job
  python Step6-AlphaMissense-Annotator.py 1                       # Process chr1
  python Step6-AlphaMissense-Annotator.py 23                      # Process chrX  
  python Step6-AlphaMissense-Annotator.py 24                      # Process chrY
  python Step6-AlphaMissense-Annotator.py 25                      # Process chrM
  python Step6-AlphaMissense-Annotator.py                         # List available chromosomes

Chromosome Naming Conversion:
  - Handles various formats: chr1, 1, chrX, X, chrY, Y, chrM, M, chrMT, MT
  - Converts numeric codes: 23→chrX, 24→chrY, 25→chrM
  - Standardizes all to chr1, chr2, ..., chr22, chrX, chrY, chrM format
"""

import polars as pl
import os
import glob
import gc
import sys
import time
from pathlib import Path
from datetime import datetime

# ===============================
# Configuration
# ===============================
ANNOVAR_DIR = "Annovar_Merge"
ALPHAMISSENSE_FILE = "AlphaMissense_hg38.tsv"
CHUNK_SIZE = 1000000  # Process 1M variants at a time for large chromosomes
MEMORY_THRESHOLD = 3000000  # Use chunking for chromosomes > 3M variants

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

def find_chromosome_files():
    """Find all available chromosome annotation files including X, Y, M"""
    print("🔍 DISCOVERING CHROMOSOME FILES")
    print("-" * 40)
    
    # Find annotation files with various patterns
    patterns = [
        os.path.join(ANNOVAR_DIR, "chr*_concatenated_annotations.csv"),
        os.path.join(ANNOVAR_DIR, "chr*_*_concatenated_annotations.csv")
    ]
    
    all_files = []
    for pattern in patterns:
        all_files.extend(glob.glob(pattern))
    
    # Remove duplicates and process
    annotation_files = list(set(all_files))
    
    if not annotation_files:
        print(f"❌ No annotation files found in {ANNOVAR_DIR}/")
        print("📁 Expected pattern: chr*_concatenated_annotations.csv")
        return []
    
    available_chromosomes = []
    
    for ann_file in sorted(annotation_files):
        # Extract chromosome name
        filename = os.path.basename(ann_file)
        chr_name = filename.replace("_concatenated_annotations.csv", "")
        
        # Normalize chromosome name
        normalized_chr = normalize_chromosome_name(chr_name)
        
        if normalized_chr not in available_chromosomes:
            available_chromosomes.append(normalized_chr)
            
            # Get file size
            file_size = os.path.getsize(ann_file) / (1024**2)
            
            # Determine chromosome type
            if normalized_chr[3:].isdigit():
                chr_type = "Autosome"
            elif normalized_chr in ["chrX", "chrY"]:
                chr_type = "Sex"
            elif normalized_chr == "chrM":
                chr_type = "Mitochondrial"
            else:
                chr_type = "Other"
            
            print(f"   ✅ {normalized_chr} ({chr_type}): {file_size:.1f} MB")
            
            if normalized_chr in ["chrX", "chrY", "chrM"]:
                print(f"      📍 Special chromosome detected: {filename}")
    
    print(f"\n📊 Found {len(available_chromosomes)} chromosome files")
    return available_chromosomes

def get_chromosome_list():
    """Get properly sorted list of available chromosomes (1-22, X, Y, M)"""
    chr_files = find_chromosome_files()
    
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
    
    sorted_chromosomes = sorted(chr_files, key=chr_sort_key)
    
    return sorted_chromosomes

def print_array_header(array_id=None):
    """Print script header for array job"""
    print("=" * 70)
    print("🚀 ARRAY JOB ALPHAMISSENSE ANNOTATION")
    print("🧬 Including X, Y, M chromosomes")
    print("🎯 Memory-Efficient | Chunked Processing")
    print("=" * 70)
    print(f"📅 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if array_id:
        print(f"🎯 Array Task ID: {array_id}")
    print(f"📋 Chunk size: {CHUNK_SIZE:,} variants")
    print(f"🧠 Memory threshold: {MEMORY_THRESHOLD:,} variants")
    print()

def load_alphamissense_data():
    """Load and prepare AlphaMissense data"""
    print(f"📂 Reading AlphaMissense file: {ALPHAMISSENSE_FILE}")
    
    if not os.path.exists(ALPHAMISSENSE_FILE):
        print(f"❌ AlphaMissense file not found: {ALPHAMISSENSE_FILE}")
        return None
    
    try:
        # Read AlphaMissense with memory optimizations
        alphamiss_df = pl.read_csv(
            ALPHAMISSENSE_FILE,
            separator="\t",
            skip_rows=3,
            ignore_errors=True,
            try_parse_dates=False,
            truncate_ragged_lines=True
        )
        
        # Standardize column names
        alphamiss_df = alphamiss_df.rename({
            "#CHROM": "Chr",
            "POS": "Start",
            "REF": "Ref", 
            "ALT": "Alt"
        })
        
        print(f"   ✓ AlphaMissense total variants: {len(alphamiss_df):,}")
        print(f"   ✓ Memory usage: ~{alphamiss_df.estimated_size('mb'):.1f} MB")
        
        return alphamiss_df
        
    except Exception as e:
        print(f"   ❌ Error reading AlphaMissense file: {e}")
        return None

def process_chromosome(chr_name, alphamiss_df):
    """Process a single chromosome with AlphaMissense annotation"""
    
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
    
    # Find annotation file
    annovar_file = os.path.join(ANNOVAR_DIR, f"{chr_name}_concatenated_annotations.csv")
    
    if not os.path.exists(annovar_file):
        print(f"❌ Annotation file not found: {annovar_file}")
        return False, 0, 0
    
    try:
        start_time = time.time()
        
        # ===============================
        # Read Annovar file
        # ===============================
        print(f"📂 Reading: {os.path.basename(annovar_file)}")
        annovar_df = pl.read_csv(annovar_file, separator=",", ignore_errors=True)
        print(f"   ✓ Loaded {len(annovar_df):,} variants")
        
        if len(annovar_df) == 0:
            print(f"   ⚠️  Empty file - skipping {chr_name}")
            return True, 0, 0
        
        original_count = len(annovar_df)
        use_chunking = original_count > MEMORY_THRESHOLD
        
        if use_chunking:
            print(f"   🧠 Large chromosome detected - using chunked processing")
            print(f"   📦 Chunk size: {CHUNK_SIZE:,} variants")
        
        # Special notes for different chromosome types
        if chr_name == "chrM":
            print(f"   🔬 Processing mitochondrial chromosome")
        elif chr_name in ["chrX", "chrY"]:
            print(f"   🚻 Processing sex chromosome")
        
        # ===============================
        # Filter AlphaMissense for this chromosome
        # ===============================
        print("   🎯 Filtering AlphaMissense data for chromosome...")
        
        # Check chromosome format in annotation file
        annovar_chr_values = annovar_df.select("Chr").unique().to_series().to_list()
        uses_chr_prefix = any(str(val).startswith("chr") for val in annovar_chr_values if val is not None)
        
        if uses_chr_prefix:
            target_chr = chr_name
            alphamiss_chr_df = alphamiss_df.filter(pl.col("Chr") == target_chr)
        else:
            target_chr = chr_name.replace("chr", "")
            alphamiss_no_chr = alphamiss_df.with_columns(
                pl.col("Chr").str.replace("^chr", "").alias("Chr")
            )
            alphamiss_chr_df = alphamiss_no_chr.filter(pl.col("Chr") == target_chr)
        
        print(f"   ✓ AlphaMissense variants for {target_chr}: {len(alphamiss_chr_df):,}")
        
        # ===============================
        # Handle case with no AlphaMissense data
        # ===============================
        if len(alphamiss_chr_df) == 0:
            print(f"   ⚠️  No AlphaMissense data for {chr_name}")
            annovar_df = annovar_df.with_columns([
                pl.lit(None).cast(pl.Utf8).alias("genome"),
                pl.lit(None).cast(pl.Utf8).alias("uniprot_id"),
                pl.lit(None).cast(pl.Utf8).alias("transcript_id"),
                pl.lit(None).cast(pl.Utf8).alias("protein_variant"),
                pl.lit(None).cast(pl.Float64).alias("am_pathogenicity"),
                pl.lit(None).cast(pl.Utf8).alias("am_class")
            ])
            
            print("   💾 Saving results...")
            annovar_df.write_csv(annovar_file)
            
            total_time = time.time() - start_time
            print(f"   ✅ {chr_name} processed (no annotations added)")
            print(f"   ⏱️  Processing time: {total_time:.1f}s")
            
            return True, len(annovar_df), 0
        
        # ===============================
        # Prepare AlphaMissense data
        # ===============================
        print("   🔧 Preparing AlphaMissense data...")
        
        # Remove duplicates
        original_alphamiss_count = len(alphamiss_chr_df)
        alphamiss_chr_df = alphamiss_chr_df.unique(subset=["Chr", "Start", "Ref", "Alt"], keep="first")
        dedup_count = len(alphamiss_chr_df)
        
        if original_alphamiss_count != dedup_count:
            print(f"   🗑️  Removed {original_alphamiss_count - dedup_count} duplicate AlphaMissense entries")
        
        # Standardize data types
        alphamiss_chr_df = alphamiss_chr_df.with_columns([
            pl.col("Chr").cast(pl.Utf8),
            pl.col("Start").cast(pl.Utf8),
            pl.col("Ref").cast(pl.Utf8),
            pl.col("Alt").cast(pl.Utf8)
        ])
        
        # ===============================
        # Process with or without chunking
        # ===============================
        if not use_chunking:
            # Small chromosome - process normally
            print("   🔄 Processing in single operation...")
            
            annovar_df = annovar_df.with_columns([
                pl.col("Chr").cast(pl.Utf8),
                pl.col("Start").cast(pl.Utf8),
                pl.col("Ref").cast(pl.Utf8),
                pl.col("Alt").cast(pl.Utf8)
            ])
            
            merged_df = annovar_df.join(
                alphamiss_chr_df.select([
                    "Chr", "Start", "Ref", "Alt", 
                    "genome", "uniprot_id", "transcript_id", "protein_variant", 
                    "am_pathogenicity", "am_class"
                ]),
                on=["Chr", "Start", "Ref", "Alt"],
                how="left"
            )
            
        else:
            # Large chromosome - process in chunks
            print("   📦 Processing in chunks...")
            
            annovar_df = annovar_df.with_columns([
                pl.col("Chr").cast(pl.Utf8),
                pl.col("Start").cast(pl.Utf8),
                pl.col("Ref").cast(pl.Utf8),
                pl.col("Alt").cast(pl.Utf8)
            ])
            
            merged_chunks = []
            total_chunks = (len(annovar_df) + CHUNK_SIZE - 1) // CHUNK_SIZE
            
            for chunk_idx in range(total_chunks):
                start_idx = chunk_idx * CHUNK_SIZE
                end_idx = min(start_idx + CHUNK_SIZE, len(annovar_df))
                
                print(f"     🔄 Chunk {chunk_idx + 1}/{total_chunks} (variants {start_idx:,}-{end_idx:,})")
                
                # Get chunk
                chunk = annovar_df.slice(start_idx, end_idx - start_idx)
                
                # Merge chunk
                merged_chunk = chunk.join(
                    alphamiss_chr_df.select([
                        "Chr", "Start", "Ref", "Alt", 
                        "genome", "uniprot_id", "transcript_id", "protein_variant", 
                        "am_pathogenicity", "am_class"
                    ]),
                    on=["Chr", "Start", "Ref", "Alt"],
                    how="left"
                )
                
                merged_chunks.append(merged_chunk)
                
                # Clean up memory periodically
                if (chunk_idx + 1) % 3 == 0:
                    gc.collect()
            
            # Combine all chunks
            print("   🔗 Combining chunks...")
            merged_df = pl.concat(merged_chunks)
            
            # Clean up chunk list
            del merged_chunks
            gc.collect()
        
        # ===============================
        # Verify merge integrity
        # ===============================
        final_count = len(merged_df)
        
        if final_count != original_count:
            print(f"   ❌ ERROR: Row count mismatch!")
            print(f"      Expected: {original_count:,}")
            print(f"      Got: {final_count:,}")
            return False, 0, 0
        else:
            print(f"   ✅ Row count preserved: {original_count:,}")
        
        # Calculate statistics
        annotated_count = merged_df.filter(pl.col('am_pathogenicity').is_not_null()).height
        annotation_rate = (annotated_count/final_count)*100 if final_count > 0 else 0
        
        print(f"   📊 Variants with AlphaMissense annotations: {annotated_count:,}")
        print(f"   📈 Annotation rate: {annotation_rate:.1f}%")
        
        # ===============================
        # Save results
        # ===============================
        print("   💾 Saving results...")
        merged_df.write_csv(annovar_file)
        
        total_time = time.time() - start_time
        
        print(f"\n✅ {chr_name.upper()} COMPLETED!")
        print(f"   🧬 Chromosome type: {chr_type}")
        print(f"   ⏱️  Processing time: {total_time:.1f}s ({total_time/60:.1f} min)")
        print(f"   📊 Final result: {final_count:,} variants")
        print(f"   🧬 Annotated: {annotated_count:,} variants")
        print(f"   📋 Columns: {len(merged_df.columns)}")
        
        # Special completion messages
        if chr_name in ["chrX", "chrY", "chrM"]:
            print(f"   🌟 Special chromosome {chr_name} annotated successfully!")
        
        # Clean up memory
        del annovar_df, alphamiss_chr_df, merged_df
        gc.collect()
        
        return True, final_count, annotated_count
        
    except Exception as e:
        print(f"❌ Error processing {chr_name}: {e}")
        import traceback
        traceback.print_exc()
        return False, 0, 0

def list_available_chromosomes():
    """List all available chromosomes for annotation"""
    print("📋 AVAILABLE CHROMOSOMES FOR ALPHAMISSENSE ANNOTATION")
    print("=" * 70)
    
    # Check AlphaMissense file
    if not os.path.exists(ALPHAMISSENSE_FILE):
        print(f"❌ AlphaMissense file not found: {ALPHAMISSENSE_FILE}")
        print("   Please ensure AlphaMissense_hg38.tsv is in the current directory")
        return
    
    chromosomes = get_chromosome_list()
    
    if not chromosomes:
        print("❌ No chromosome files found!")
        print("📁 Expected files:")
        print(f"   {ANNOVAR_DIR}/chr*_concatenated_annotations.csv")
        return
    
    print(f"Found {len(chromosomes)} chromosomes ready for AlphaMissense annotation")
    print(f"AlphaMissense file: {ALPHAMISSENSE_FILE}")
    print()
    print("Array ID | Chromosome | Type         | Size")
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
        
        # Check file size
        ann_file = os.path.join(ANNOVAR_DIR, f"{chr_name}_concatenated_annotations.csv")
        file_size = os.path.getsize(ann_file) / (1024**2) if os.path.exists(ann_file) else 0
        
        print(f"{i:8d} | {chr_name:10s} | {chr_type:12s} | {file_size:6.1f} MB")
    
    print()
    print("USAGE EXAMPLES:")
    print(f"  # SLURM Array Job (process all chromosomes including X, Y, M):")
    print(f"  sbatch --array=1-{len(chromosomes)} script.sh")
    print(f"  ")
    print(f"  # In script.sh:")
    print(f"  python Step6-AlphaMissense-Annotator.py ${{SLURM_ARRAY_TASK_ID}}")
    print(f"  ")
    print(f"  # Single chromosome examples:")
    print(f"  python Step6-AlphaMissense-Annotator.py 1      # Process {chromosomes[0]}")
    
    # Show examples for sex and mitochondrial chromosomes
    for i, chr_name in enumerate(chromosomes, 1):
        if chr_name == "chrX":
            print(f"  python Step6-AlphaMissense-Annotator.py {i}     # Process chrX")
        elif chr_name == "chrY":
            print(f"  python Step6-AlphaMissense-Annotator.py {i}     # Process chrY")
        elif chr_name == "chrM":
            print(f"  python Step6-AlphaMissense-Annotator.py {i}     # Process chrM")
    
    if len(chromosomes) > 0:
        print(f"  python Step6-AlphaMissense-Annotator.py {len(chromosomes)}     # Process {chromosomes[-1]}")
    
    print(f"\n📊 Chromosome Summary:")
    autosomes = [c for c in chromosomes if c[3:].isdigit()]
    sex_chroms = [c for c in chromosomes if c in ["chrX", "chrY"]]
    mito = [c for c in chromosomes if c == "chrM"]
    
    print(f"  • Autosomes (1-22): {len(autosomes)} found")
    print(f"  • Sex chromosomes (X,Y): {len(sex_chroms)} found {sex_chroms}")
    print(f"  • Mitochondrial (M): {len(mito)} found {mito}")
    
    # Configuration info
    print(f"\n⚙️  Configuration:")
    print(f"  • Chunk size for large chromosomes: {CHUNK_SIZE:,} variants")
    print(f"  • Memory threshold for chunking: {MEMORY_THRESHOLD:,} variants")

def process_array_chromosome(array_id):
    """Process a single chromosome based on array ID"""
    print_array_header(array_id)
    
    # Load AlphaMissense data first
    alphamiss_df = load_alphamissense_data()
    if alphamiss_df is None:
        sys.exit(1)
    
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
    
    # Check if annotation file exists for this chromosome
    annotation_file = os.path.join(ANNOVAR_DIR, f"{target_chr}_concatenated_annotations.csv")
    
    if not os.path.exists(annotation_file):
        print(f"❌ Annotation file not found for {target_chr}!")
        print(f"   Expected: {annotation_file}")
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
    success, variant_count, annotated_count = process_chromosome(target_chr, alphamiss_df)
    total_time = time.time() - start_time
    
    # Final summary
    print(f"\n{'='*70}")
    print("🏆 ARRAY JOB COMPLETE")
    print("=" * 70)
    
    if success:
        ann_file = os.path.join(ANNOVAR_DIR, f"{target_chr}_concatenated_annotations.csv")
        file_size = os.path.getsize(ann_file) / (1024*1024) if os.path.exists(ann_file) else 0
        
        print(f"✅ Successfully processed {target_chr} ({chr_type})")
        print(f"📁 Output: {ann_file}")
        print(f"📊 Variants: {variant_count:,} total, {annotated_count:,} annotated")
        print(f"📄 Size: {file_size:.1f} MB")
        print(f"⏱️ Time: {total_time:.1f} seconds")
        print(f"🎯 Array ID {array_id} SUCCESSFUL")
        
        # Special message for sex and mitochondrial chromosomes
        if target_chr in ["chrX", "chrY", "chrM"]:
            print(f"🌟 Special chromosome {target_chr} annotated with AlphaMissense successfully!")
            
    else:
        print(f"❌ Failed to process {target_chr} ({chr_type})")
        print(f"🎯 Array ID {array_id} FAILED")
        sys.exit(1)
    
    # Clean up final memory
    del alphamiss_df
    gc.collect()

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
        print("\n🧬 This script adds AlphaMissense pathogenicity predictions to annotations")
        return
    
    # Get array ID from command line
    try:
        array_id = int(sys.argv[1])
    except ValueError:
        print(f"❌ Error: '{sys.argv[1]}' is not a valid integer")
        print("Usage: python Step6-AlphaMissense-Annotator.py <array_id>")
        print("       python Step6-AlphaMissense-Annotator.py              # List chromosomes")
        print("\nExamples:")
        print("  python Step6-AlphaMissense-Annotator.py 1     # chr1")
        print("  python Step6-AlphaMissense-Annotator.py 23    # chrX") 
        print("  python Step6-AlphaMissense-Annotator.py 24    # chrY")
        print("  python Step6-AlphaMissense-Annotator.py 25    # chrM")
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