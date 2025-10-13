#!/bin/bash
"""
BASH GENOMIC DATA MERGER
Memory-efficient processing using standard Unix tools
Processes files line-by-line without loading into RAM
Uses sort + join for fast, low-memory merging
"""

set -euo pipefail  # Exit on errors, undefined vars, pipe failures

# Configuration
CORES=$(nproc)
TEMP_DIR="/tmp/genomic_merge_$$"
SORT_MEMORY="2G"  # Memory per sort process

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Create genomic key function (used in awk)
create_genomic_key() {
    # This will be used in awk scripts
    # Format: chr:pos:ref:alt (removing 'chr' prefix)
    echo 'function create_key(chr, pos, ref, alt) {
        gsub(/^chr/, "", chr)
        if (chr == "" || pos == "" || ref == "" || alt == "") return ""
        return chr ":" pos ":" ref ":" alt
    }'
}

cleanup() {
    log_info "Cleaning up temporary files..."
    rm -rf "$TEMP_DIR"
}

trap cleanup EXIT

setup_temp_dir() {
    mkdir -p "$TEMP_DIR"
    log_info "Using temp directory: $TEMP_DIR"
}

prepare_annotation_file() {
    local annotation_file="$1"
    local chr_name="$2"
    local output_file="$TEMP_DIR/annotation_prepared.txt"
    
    log_info "📂 Preparing annotation file: $(basename "$annotation_file")"
    
    # Check if file exists
    if [[ ! -f "$annotation_file" ]]; then
        log_error "Annotation file not found: $annotation_file"
        return 1
    fi
    
    # Get file info
    local rows=$(wc -l < "$annotation_file")
    local size_mb=$(du -m "$annotation_file" | cut -f1)
    log_info "   File: $rows rows, ${size_mb}MB"
    
    # Create prepared file with genomic key as first column
    log_info "   Creating genomic keys..."
    
    {
        # Header line
        echo -e "genomic_key\t$(head -n1 "$annotation_file")"
        
        # Data lines - create genomic key and prepend it
        tail -n +2 "$annotation_file" | awk -F',' -v OFS='\t' '
        function create_key(chr, pos, ref, alt) {
            gsub(/^chr/, "", chr)
            gsub(/["'"'"']/, "", chr); gsub(/["'"'"']/, "", pos); gsub(/["'"'"']/, "", ref); gsub(/["'"'"']/, "", alt)
            if (chr == "" || pos == "" || ref == "" || alt == "") return ""
            return chr ":" pos ":" ref ":" alt
        }
        NR > 0 {
            # Find Chr, Start, Ref, Alt columns (adjust based on your CSV structure)
            # Assuming standard column order - adjust if needed
            chr = $1; start = $2; ref = $4; alt = $5  # Modify these column numbers as needed
            key = create_key(chr, start, ref, alt)
            if (key != "") {
                print key "\t" $0
            }
        }'
    } > "$output_file"
    
    local prepared_rows=$(wc -l < "$output_file")
    log_success "   Prepared $((prepared_rows-1)) data rows with genomic keys"
    
    echo "$output_file"
}

prepare_dbnsfp_file() {
    local dbnsfp_file="$1"
    local output_file="$TEMP_DIR/dbnsfp_prepared.txt"
    
    log_info "🧬 Preparing dbNSFP file: $(basename "$dbnsfp_file")"
    
    # Check if file exists
    if [[ ! -f "$dbnsfp_file" ]]; then
        log_error "dbNSFP file not found: $dbnsfp_file"
        return 1
    fi
    
    # Get file info
    local size_gb=$(du --apparent-size -BG "$dbnsfp_file" | cut -f1 | tr -d 'G')
    log_info "   File size: ${size_gb}GB"
    
    # Create prepared file with genomic key as first column
    log_info "   Creating genomic keys and preparing data..."
    
    {
        # Skip the #chr header line, create our own header
        echo -e "genomic_key\tchr\tpos_1_based\tref\talt\taaref\taaalt\thg19_chr\thg19_pos_1_based\thg18_chr\thg18_pos_1_based\tgenename\tcds_strand\trefcodon\tcodonpos\tEnsembl_geneid\tEnsembl_transcriptid\tEnsembl_proteinid\taapos"
        
        # Process data lines (skip header that starts with #)
        grep -v '^#' "$dbnsfp_file" | awk -F'\t' -v OFS='\t' '
        function create_key(chr, pos, ref, alt) {
            gsub(/^chr/, "", chr)
            if (chr == "" || pos == "" || ref == "" || alt == "") return ""
            return chr ":" pos ":" ref ":" alt
        }
        {
            # dbNSFP columns: chr, pos(1-based), ref, alt, ...
            chr = $1; pos = $2; ref = $3; alt = $4
            key = create_key(chr, pos, ref, alt)
            if (key != "") {
                print key "\t" $0
            }
        }'
    } > "$output_file"
    
    local prepared_rows=$(wc -l < "$output_file")
    log_success "   Prepared $((prepared_rows-1)) data rows with genomic keys"
    
    echo "$output_file"
}

sort_file() {
    local input_file="$1"
    local output_file="$2"
    local description="$3"
    
    log_info "🔄 Sorting $description..."
    
    # Sort by genomic key (first column) using multiple cores
    LC_ALL=C sort \
        --parallel="$CORES" \
        --buffer-size="$SORT_MEMORY" \
        --temporary-directory="$TEMP_DIR" \
        -k1,1 \
        "$input_file" > "$output_file"
    
    local rows=$(wc -l < "$output_file")
    log_success "   Sorted $rows rows"
    
    echo "$output_file"
}

merge_files() {
    local annotation_sorted="$1"
    local dbnsfp_sorted="$2"
    local output_file="$3"
    
    log_info "🔗 Merging annotation + dbNSFP data..."
    
    # Use join for LEFT JOIN (keep all annotation records)
    # -t $'\t' = tab delimiter
    # -1 1 -2 1 = join on column 1 of both files
    # -a 1 = include unpairable lines from file 1 (LEFT JOIN)
    # -o auto = output all columns
    LC_ALL=C join \
        -t $'\t' \
        -1 1 -2 1 \
        -a 1 \
        -o auto \
        "$annotation_sorted" \
        "$dbnsfp_sorted" > "$output_file"
    
    local merged_rows=$(wc -l < "$output_file")
    log_success "   Merged result: $((merged_rows-1)) data rows"
    
    # Count how many got dbNSFP annotations (have more than just annotation columns)
    local annotation_cols=$(head -n1 "$annotation_sorted" | tr '\t' '\n' | wc -l)
    local total_cols=$(head -n1 "$output_file" | tr '\t' '\n' | wc -l)
    
    if [[ $total_cols -gt $annotation_cols ]]; then
        local annotated_count=$(tail -n +2 "$output_file" | awk -F'\t' -v min_cols="$annotation_cols" 'NF > min_cols' | wc -l)
        local annotation_rate=$(echo "scale=1; $annotated_count * 100 / ($merged_rows - 1)" | bc -l 2>/dev/null || echo "N/A")
        log_success "   Variants with dbNSFP data: $annotated_count ($annotation_rate%)"
    fi
    
    echo "$output_file"
}

finalize_output() {
    local merged_file="$1"
    local final_output="$2"
    
    log_info "📝 Finalizing output..."
    
    # Remove the genomic_key column and convert back to CSV format
    awk -F'\t' 'BEGIN {OFS=","} 
    NR==1 {
        # Remove first column (genomic_key) from header
        for(i=2; i<=NF; i++) {
            if(i>2) printf ","
            printf "%s", $i
        }
        printf "\n"
    }
    NR>1 {
        # Remove first column from data
        for(i=2; i<=NF; i++) {
            if(i>2) printf ","
            # Escape commas in data if needed
            gsub(/,/, "\\,", $i)
            printf "%s", $i
        }
        printf "\n"
    }' "$merged_file" > "$final_output"
    
    local final_rows=$(wc -l < "$final_output")
    local final_size_mb=$(du -m "$final_output" | cut -f1)
    
    log_success "   Final output: $((final_rows-1)) rows, ${final_size_mb}MB"
    log_success "   Saved: $final_output"
}

process_chromosome() {
    local chr_name="$1"
    
    echo "========================================"
    echo "🧬 PROCESSING $chr_name"
    echo "========================================"
    
    local annotation_file="Annovar_Merge/${chr_name}_merged_annotations.csv"
    local dbnsfp_file="dbNSFP5.1_nsSNV.${chr_name}"
    
    # Check files exist
    if [[ ! -f "$annotation_file" ]]; then
        log_error "Annotation file not found: $annotation_file"
        return 1
    fi
    
    if [[ ! -f "$dbnsfp_file" ]]; then
        log_error "dbNSFP file not found: $dbnsfp_file"
        return 1
    fi
    
    local start_time=$(date +%s)
    
    # Setup
    setup_temp_dir
    
    # Step 1: Prepare files
    log_info "STEP 1: Preparing files with genomic keys"
    local annotation_prepared=$(prepare_annotation_file "$annotation_file" "$chr_name")
    local dbnsfp_prepared=$(prepare_dbnsfp_file "$dbnsfp_file")
    
    # Step 2: Sort files
    log_info "STEP 2: Sorting files"
    local annotation_sorted=$(sort_file "$annotation_prepared" "$TEMP_DIR/annotation_sorted.txt" "annotation data")
    local dbnsfp_sorted=$(sort_file "$dbnsfp_prepared" "$TEMP_DIR/dbnsfp_sorted.txt" "dbNSFP data")
    
    # Step 3: Merge
    log_info "STEP 3: Merging data"
    local merged_file=$(merge_files "$annotation_sorted" "$dbnsfp_sorted" "$TEMP_DIR/merged.txt")
    
    # Step 4: Finalize
    log_info "STEP 4: Creating final output"
    finalize_output "$merged_file" "$annotation_file"
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    log_success "✅ $chr_name completed in ${duration}s"
    
    return 0
}

main() {
    echo "========================================"
    echo "🚀 BASH GENOMIC DATA MERGER"
    echo "🧬 Memory-Efficient Processing"
    echo "========================================"
    echo "💻 CPU cores: $CORES"
    echo "💾 Sort memory: $SORT_MEMORY per process"
    echo "========================================"
    
    # Find available chromosomes
    local chromosomes=()
    for annotation_file in Annovar_Merge/chr*_merged_annotations.csv; do
        if [[ -f "$annotation_file" ]]; then
            local chr_name=$(basename "$annotation_file" "_merged_annotations.csv")
            local dbnsfp_file="dbNSFP5.1_nsSNV.${chr_name}"
            if [[ -f "$dbnsfp_file" ]]; then
                chromosomes+=("$chr_name")
            else
                log_warn "Skipping $chr_name: dbNSFP file not found"
            fi
        fi
    done
    
    if [[ ${#chromosomes[@]} -eq 0 ]]; then
        log_error "No matching chromosome files found!"
        exit 1
    fi
    
    log_info "Found ${#chromosomes[@]} chromosome pairs: ${chromosomes[*]}"
    
    # Process each chromosome
    local successful=0
    local failed=0
    local total_start=$(date +%s)
    
    for chr_name in "${chromosomes[@]}"; do
        echo
        if process_chromosome "$chr_name"; then
            ((successful++))
        else
            ((failed++))
            log_error "Failed to process $chr_name"
        fi
    done
    
    local total_end=$(date +%s)
    local total_duration=$((total_end - total_start))
    
    echo
    echo "========================================"
    echo "🏆 FINAL SUMMARY"
    echo "========================================"
    echo "⏱️  Total time: ${total_duration}s ($((total_duration/60))m)"
    echo "✅ Successful: $successful"
    echo "❌ Failed: $failed"
    echo "📊 Total chromosomes: ${#chromosomes[@]}"
    
    if [[ $successful -eq ${#chromosomes[@]} ]]; then
        log_success "🎉 ALL CHROMOSOMES COMPLETED SUCCESSFULLY!"
    else
        log_warn "⚠️  $failed chromosome(s) failed"
    fi
}

# Check dependencies
check_dependencies() {
    local missing=()
    for cmd in awk sort join bc wc du; do
        if ! command -v "$cmd" &> /dev/null; then
            missing+=("$cmd")
        fi
    done
    
    if [[ ${#missing[@]} -gt 0 ]]; then
        log_error "Missing required commands: ${missing[*]}"
        exit 1
    fi
}

# Run main function
check_dependencies
main "$@"