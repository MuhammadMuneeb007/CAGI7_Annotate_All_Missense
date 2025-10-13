import polars as pl
import os

print("="*70)
print("MERGING PREDICTIONS WITH CAGI TEMPLATE FILES")
print("="*70)

# List of all chromosomes
chromosomes = [f'chr{i}' for i in range(1, 23)] + ['chrX', 'chrY', 'chrM']

# Read your predictions - force score and sd to be strings
print("\nLoading your predictions...")
predictions = pl.read_csv(
    'Final_Submission1/MetaModelPredictions1.tsv',
    separator='\t',
    dtypes={'#chr': pl.Utf8, 'score': pl.Utf8, 'sd': pl.Utf8}  # Keep as strings
)
print(f"Loaded {len(predictions):,} predictions")

# Create a key for matching
print("\nCreating lookup key for predictions...")
predictions = predictions.with_columns(
    (pl.col('#chr') + '_' + 
     pl.col('pos(1-based)').cast(pl.Utf8) + '_' + 
     pl.col('ref') + '_' + 
     pl.col('alt')).alias('key')
)

# Convert to dictionary
pred_dict = {}
for row in predictions.iter_rows(named=True):
    key = row['key']
    pred_dict[key] = {
        'score': row['score'],
        'sd': row['sd'],
        'pred': row['pred'],
        'comments': row['comments']
    }

print(f"Created lookup dictionary with {len(pred_dict):,} entries")

# Process each chromosome template
all_filled_templates = []
total_matched = 0
total_unmatched = 0

print("\n" + "="*70)
print("PROCESSING TEMPLATE FILES")
print("="*70)

for chrom in chromosomes:
    template_file = f'templates/dbNSFP4_nsSNV.{chrom}.template'
    
    if not os.path.exists(template_file):
        print(f"\n{chrom}: Template file not found, skipping...")
        continue
    
    print(f"\nProcessing {chrom}...")
    
    try:
        # Read template
        template = pl.read_csv(
            template_file,
            separator='\t',
            null_values=['.'],
            dtypes={'#chr': pl.Utf8}
        )
        
        print(f"  Template rows: {len(template):,}")
        
        # Create key for template
        template = template.with_columns(
            (pl.col('#chr') + '_' + 
             pl.col('pos(1-based)').cast(pl.Utf8) + '_' + 
             pl.col('ref') + '_' + 
             pl.col('alt')).alias('key')
        )
        
        # Match and fill values
        matched = 0
        unmatched = 0
        
        scores = []
        sds = []
        preds = []
        comments = []
        
        for row in template.iter_rows(named=True):
            key = row['key']
            if key in pred_dict:
                scores.append(pred_dict[key]['score'])
                sds.append(pred_dict[key]['sd'])
                preds.append(pred_dict[key]['pred'])
                comments.append(pred_dict[key]['comments'])
                matched += 1
            else:
                scores.append('0')
                sds.append('0')
                preds.append('U')
                comments.append('No prediction available')
                unmatched += 1
        
        filled_template = template.select(['#chr', 'pos(1-based)', 'ref', 'alt']).with_columns([
            pl.Series('score', scores, dtype=pl.Utf8),
            pl.Series('sd', sds, dtype=pl.Utf8),
            pl.Series('pred', preds, dtype=pl.Utf8),
            pl.Series('comments', comments, dtype=pl.Utf8)
        ])
        
        all_filled_templates.append(filled_template)
        total_matched += matched
        total_unmatched += unmatched
        
        print(f"  Matched: {matched:,}")
        print(f"  Unmatched (filled with U): {unmatched:,}")
        
    except Exception as e:
        print(f"  Error: {e}")

# Merge all chromosomes
print("\n" + "="*70)
print("MERGING ALL CHROMOSOMES")
print("="*70)

final_submission = pl.concat(all_filled_templates)

print(f"\nTotal rows: {len(final_submission):,}")
print(f"Total matched: {total_matched:,}")
print(f"Total unmatched (U): {total_unmatched:,}")

# Save final submission
output_file = 'Final_Submission1/CAGI7_FinalSubmission_model_1.tsv'
final_submission.write_csv(output_file, separator='\t')

print(f"\n{'='*70}")
print(f"FINAL SUBMISSION SAVED: {output_file}")
print(f"{'='*70}")

# Count predictions
d_count = final_submission.filter(pl.col('pred') == 'D').shape[0]
t_count = final_submission.filter(pl.col('pred') == 'T').shape[0]
u_count = final_submission.filter(pl.col('pred') == 'U').shape[0]

print(f"\nDeleterious (D): {d_count:,}")
print(f"Tolerated (T):   {t_count:,}")
print(f"Unknown (U):     {u_count:,}")
print(f"TOTAL:           {len(final_submission):,}")