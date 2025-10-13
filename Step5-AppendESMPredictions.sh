#!/bin/bash
# FIXED SIMPLE bash merger - no complex awk variables
set -e

echo "🚀 FIXED SIMPLE BASH MERGER"

# Files  
DBNSFP="dbNSFP5.1_nsSNV.chr22"
ESM="annovar/humandb/IGVFFI8105TNNO.tsv"
OUTPUT="merged_fixed.tsv"

if [[ ! -f "$DBNSFP" ]] || [[ ! -f "$ESM" ]]; then
    echo "❌ Files not found"
    exit 1
fi

echo "✅ Files found"
start_time=$(date +%s)

echo "🔵 Step 1: Processing dbNSFP..."
# Simple dbNSFP processing - hardcode column positions for reliability
awk -F'\t' 'BEGIN{OFS="\t"}
NR==1 {
    print $0, "match_key"
    next
}
{
    # Convert amino acids (columns 5=aaref, 6=aaalt, 15=transcript, 18=aapos)
    aaref = $5
    aaalt = $6
    transcript = $15  
    aapos = $18
    
    # Convert single letters to 3-letter codes
    if (aaref == "A") aaref = "Ala"
    else if (aaref == "R") aaref = "Arg"  
    else if (aaref == "N") aaref = "Asn"
    else if (aaref == "D") aaref = "Asp"
    else if (aaref == "C") aaref = "Cys"
    else if (aaref == "Q") aaref = "Gln"
    else if (aaref == "E") aaref = "Glu"
    else if (aaref == "G") aaref = "Gly"
    else if (aaref == "H") aaref = "His"
    else if (aaref == "I") aaref = "Ile"
    else if (aaref == "L") aaref = "Leu"
    else if (aaref == "K") aaref = "Lys"
    else if (aaref == "M") aaref = "Met"
    else if (aaref == "F") aaref = "Phe"
    else if (aaref == "P") aaref = "Pro"
    else if (aaref == "S") aaref = "Ser"
    else if (aaref == "T") aaref = "Thr"
    else if (aaref == "W") aaref = "Trp"
    else if (aaref == "Y") aaref = "Tyr"
    else if (aaref == "V") aaref = "Val"
    
    if (aaalt == "A") aaalt = "Ala"
    else if (aaalt == "R") aaalt = "Arg"
    else if (aaalt == "N") aaalt = "Asn"
    else if (aaalt == "D") aaalt = "Asp"
    else if (aaalt == "C") aaalt = "Cys"
    else if (aaalt == "Q") aaalt = "Gln"
    else if (aaalt == "E") aaalt = "Glu"
    else if (aaalt == "G") aaalt = "Gly"
    else if (aaalt == "H") aaalt = "His"
    else if (aaalt == "I") aaalt = "Ile"
    else if (aaalt == "L") aaalt = "Leu"
    else if (aaalt == "K") aaalt = "Lys"
    else if (aaalt == "M") aaalt = "Met"
    else if (aaalt == "F") aaalt = "Phe"
    else if (aaalt == "P") aaalt = "Pro"
    else if (aaalt == "S") aaalt = "Ser"
    else if (aaalt == "T") aaalt = "Thr"
    else if (aaalt == "W") aaalt = "Trp"
    else if (aaalt == "Y") aaalt = "Tyr"
    else if (aaalt == "V") aaalt = "Val"
    
    match_key = transcript "_" aaref aapos aaalt
    print $0, match_key
}' "$DBNSFP" > dbnsfp_with_keys.tsv

dbnsfp_rows=$(wc -l < dbnsfp_with_keys.tsv)
echo "✅ dbNSFP: $((dbnsfp_rows-1)) rows"

echo "🔵 Step 2: Extract unique keys..."
tail -n +2 dbnsfp_with_keys.tsv | awk -F'\t' '{print $(NF)}' | sort -u > unique_keys.txt
unique_count=$(wc -l < unique_keys.txt)
echo "✅ Unique keys: $unique_count"

echo "🔴 Step 3: Processing ESM file (this is the slow part)..."
# Simple ESM processing - hardcode columns 2 and 4
awk -F'\t' 'BEGIN{OFS="\t"}
NR==1 {
    print $0, "match_key"
    next  
}
$4 ~ /:p\./ {
    # Column 2 = GENCODE.v43.ENST, Column 4 = HGVS.p
    enst = $2
    hgvs = $4
    
    # Remove version from transcript
    split(enst, trans_parts, ".")
    transcript = trans_parts[1]
    
    # Extract amino acid change
    split(hgvs, hgvs_parts, ":p.")
    if (length(hgvs_parts) >= 2) {
        aa_change = hgvs_parts[2]
        match_key = transcript "_" aa_change
        print $0, match_key
    }
}' "$ESM" > esm_with_keys.tsv

esm_rows=$(wc -l < esm_with_keys.tsv)
echo "✅ ESM processed: $((esm_rows-1)) rows"

echo "🔵 Step 4: Finding matches..."
# Extract match keys from ESM
tail -n +2 esm_with_keys.tsv | awk -F'\t' '{print $(NF)}' | sort -u > esm_keys.txt

# Find common keys
comm -12 unique_keys.txt esm_keys.txt > common_keys.txt
common_count=$(wc -l < common_keys.txt)
echo "✅ Common keys: $common_count"

if [[ $common_count -eq 0 ]]; then
    echo "❌ No matches found!"
    echo "dbNSFP sample keys:"
    head -5 unique_keys.txt
    echo "ESM sample keys:"  
    head -5 esm_keys.txt
    exit 1
fi

echo "🔵 Step 5: Creating final output..."
# Create lookup from ESM matches
awk -F'\t' 'NR>1 {print $(NF), $0}' esm_with_keys.tsv | sort -k1,1 > esm_lookup.txt

# Merge with dbNSFP
awk -F'\t' 'BEGIN{OFS="\t"}
NR==1 {
    # Print combined header
    print $0, "ESM_GENCODE.v43.ENSG", "ESM_GENCODE.v43.ENST", "ESM_GENCODE.v43.ENSP", "ESM_HGVS.p", "esm1v_t33_650M_UR90S_1", "esm1v_t33_650M_UR90S_2", "esm1v_t33_650M_UR90S_3", "esm1v_t33_650M_UR90S_4", "esm1v_t33_650M_UR90S_5", "combined_score"
    next
}
{
    match_key = $(NF)
    
    # Look up in ESM data
    cmd = "grep \"^" match_key "\" esm_lookup.txt | head -1"
    if ((cmd | getline esm_line) > 0) {
        close(cmd)
        # Parse ESM line
        split(esm_line, esm_fields, "\t")
        # Add ESM columns (skip first field which is the key)
        esm_data = esm_fields[2] "\t" esm_fields[3] "\t" esm_fields[4] "\t" esm_fields[5] "\t" esm_fields[6] "\t" esm_fields[7] "\t" esm_fields[8] "\t" esm_fields[9] "\t" esm_fields[10] "\t" esm_fields[16]
        print $0, esm_data
    } else {
        # No match - add empty columns
        print $0, ".", ".", ".", ".", ".", ".", ".", ".", ".", "."
    }
}' dbnsfp_with_keys.tsv > "$OUTPUT"

# Final stats
end_time=$(date +%s)
total_time=$((end_time - start_time))
minutes=$((total_time / 60))
seconds=$((total_time % 60))

output_rows=$(wc -l < "$OUTPUT")
output_size=$(du -h "$OUTPUT" | cut -f1)

echo
echo "🎉 FIXED BASH MERGER COMPLETED! 🎉"
echo "⏱️  Total time: ${minutes}m ${seconds}s"
echo "📊 Results:"
echo "   • Input rows: $((dbnsfp_rows-1))"  
echo "   • Output rows: $((output_rows-1))"
echo "   • Matches found: $common_count"
echo "   • Output file: $OUTPUT ($output_size)"

# Cleanup
rm -f dbnsfp_with_keys.tsv unique_keys.txt esm_with_keys.tsv esm_keys.txt common_keys.txt esm_lookup.txt

echo "✅ Done! Check $OUTPUT"