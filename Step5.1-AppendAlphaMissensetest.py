import polars as pl
import re

# Configuration
annovar_file = "Annovar_Merge/chrX_concatenated_annotations.csv"
alphamiss_file = "AlphaMissense_hg38.tsv"

def extract_first_enst(text):
    """Extract first ENST transcript ID"""
    if not text or text == '.' or text == 'UNKNOWN':
        return None
    match = re.search(r'ENST\d+\.\d+', str(text))
    return match.group(0) if match else None

print("FIRST 10 VARIANTS - CHR:POS:REF:ALT:TRANSCRIPT")
print("=" * 60)

# ===============================
# Annovar data
# ===============================
print("\nANNOVAR (first 10 with transcripts):")
print("-" * 40)

annovar_df = pl.read_csv(annovar_file, separator=",", ignore_errors=True)

# Extract transcripts and show first 10 with transcript data
annovar_with_transcript = annovar_df.with_columns(
    pl.col("AAChange.ensGene").map_elements(extract_first_enst, return_dtype=pl.Utf8).alias("transcript_id")
).filter(
    pl.col("transcript_id").is_not_null()
).head(10)

for i, row in enumerate(annovar_with_transcript.select(["Chr", "Start", "Ref", "Alt", "transcript_id"]).iter_rows(), 1):
    print(f"{i:2d}. {row[0]}:{row[1]}:{row[2]}:{row[3]}:{row[4]}")

# ===============================
# AlphaMissense data
# ===============================
print(f"\nALPHAMISSENSE (first 10 for chrX):")
print("-" * 40)

alphamiss_df = pl.read_csv(alphamiss_file, separator="\t", skip_rows=3, ignore_errors=True)
alphamiss_df = alphamiss_df.rename({"#CHROM": "Chr", "POS": "Start", "REF": "Ref", "ALT": "Alt"})

# Filter for X chromosome and show first 10
alphamiss_x = alphamiss_df.filter(pl.col("Chr") == "chrX").head(10)

for i, row in enumerate(alphamiss_x.select(["Chr", "Start", "Ref", "Alt", "transcript_id"]).iter_rows(), 1):
    print(f"{i:2d}. {row[0]}:{row[1]}:{row[2]}:{row[3]}:{row[4]}")

print(f"\nNote: AlphaMissense uses 'chrX' format, Annovar uses 'X' format")