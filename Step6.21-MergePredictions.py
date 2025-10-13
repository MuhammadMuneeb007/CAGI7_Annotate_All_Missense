import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create output directory
os.makedirs('Final_Submission1', exist_ok=True)

# List of all chromosomes
chromosomes = [f'chr{i}' for i in range(1, 23)] + ['chrX', 'chrY', 'chrM']

# Store all dataframes for merging
all_predictions = []

print("Processing chromosomes...")
print("="*60)

for chrom in chromosomes:
    try:
        print(f"\nProcessing {chrom}...")
        
        # Read only the first 4 columns from dbNSFP file that we need
        dbnsfp = pl.read_csv(
            f'dbNSFP5.1_nsSNV.{chrom}',
            separator='\t',
            null_values=['.'],
            columns=['#chr', 'pos(1-based)', 'ref', 'alt'],  # Only read what we need
            infer_schema_length=0  # Read all as strings first
        )
        
        # Read the predictions file
        predictions = pl.read_csv(
            f'Final_Dataset/Predictions/{chrom}_predictions.txt',
            separator='\t'
        )
        
        # Add the actual chr, pos, ref, alt from dbNSFP to predictions
        predictions_with_coords = predictions.with_columns([
            dbnsfp['#chr'].alias('#chr'),
            dbnsfp['pos(1-based)'].alias('pos(1-based)'),
            dbnsfp['ref'].alias('ref'),
            dbnsfp['alt'].alias('alt')
        ])
        
        # Map predictions: Benign -> T, Pathogenic -> D
        predictions_with_coords = predictions_with_coords.with_columns(
            pl.col('pred').map_elements(
                lambda x: 'T' if x == 'Benign' else 'D' if x == 'Pathogenic' else x,
                return_dtype=pl.Utf8
            )
        )
        
        # Create comment field combining score, pred, and sd
        predictions_with_coords = predictions_with_coords.with_columns(
            pl.format(
                "score={}, pred={}, sd={}",
                pl.col('score'),
                pl.col('pred'),
                pl.col('sd')
            ).alias('comments')
        )
        
        # Reorder columns to match desired format
        # Make sure all columns are strings for consistency
        predictions_with_coords = predictions_with_coords.select([
            pl.col('#chr').cast(pl.Utf8),
            pl.col('pos(1-based)').cast(pl.Utf8),
            pl.col('ref').cast(pl.Utf8),
            pl.col('alt').cast(pl.Utf8),
            pl.col('score').cast(pl.Utf8),
            pl.col('sd').cast(pl.Utf8),
            pl.col('pred').cast(pl.Utf8),
            pl.col('comments').cast(pl.Utf8)
        ])
        
        # Save individual chromosome file
        output_file = f'Final_Submission1/{chrom}_predictions.txt'
        predictions_with_coords.write_csv(output_file, separator='\t')
        print(f"  Saved: {output_file} ({len(predictions_with_coords):,} variants)")
        
        # Add to list for merging
        all_predictions.append(predictions_with_coords)
        
    except Exception as e:
        print(f"  Error processing {chrom}: {e}")

# Merge all chromosomes
print("\n" + "="*60)
print("Merging all chromosomes...")
merged_predictions = pl.concat(all_predictions)

# Save merged file
merged_file = 'Final_Submission1/MetaModelPredictions1.tsv'
merged_predictions.write_csv(merged_file, separator='\t')
print(f"Saved merged file: {merged_file}")
print(f"Total variants: {len(merged_predictions):,}")

# Count predictions (convert pred to string for filtering)
d_count = merged_predictions.filter(pl.col('pred') == 'D').shape[0]
t_count = merged_predictions.filter(pl.col('pred') == 'T').shape[0]

print(f"\nDeleterious (D): {d_count:,}")
print(f"Tolerated (T): {t_count:,}")
print(f"Percentage D: {d_count/(d_count+t_count)*100:.2f}%")
print(f"Percentage T: {t_count/(d_count+t_count)*100:.2f}%")

# ============================================
# CREATE DISTRIBUTION PLOTS
# ============================================
print("\n" + "="*60)
print("Creating distribution plots...")

# Convert to pandas for plotting - convert score back to float
df = merged_predictions.to_pandas()
df['score'] = df['score'].astype(float)

# Create figure with multiple subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Histogram with overlapping distributions
ax1 = axes[0, 0]
df[df['pred'] == 'D']['score'].hist(bins=50, alpha=0.6, color='red', label='Deleterious (D)', ax=ax1)
df[df['pred'] == 'T']['score'].hist(bins=50, alpha=0.6, color='blue', label='Tolerated (T)', ax=ax1)
ax1.set_xlabel('Score', fontsize=12)
ax1.set_ylabel('Frequency', fontsize=12)
ax1.set_title('Score Distribution - Histogram (All Chromosomes)', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(alpha=0.3)

# 2. Density plot
ax2 = axes[0, 1]
df[df['pred'] == 'D']['score'].plot(kind='density', color='red', label='Deleterious (D)', ax=ax2, linewidth=2)
df[df['pred'] == 'T']['score'].plot(kind='density', color='blue', label='Tolerated (T)', ax=ax2, linewidth=2)
ax2.set_xlabel('Score', fontsize=12)
ax2.set_ylabel('Density', fontsize=12)
ax2.set_title('Score Distribution - Density Plot (All Chromosomes)', fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(alpha=0.3)

# 3. Box plot
ax3 = axes[1, 0]
data_to_plot = [df[df['pred'] == 'T']['score'], df[df['pred'] == 'D']['score']]
bp = ax3.boxplot(data_to_plot, labels=['Tolerated (T)', 'Deleterious (D)'], 
                  patch_artist=True, showfliers=False)
bp['boxes'][0].set_facecolor('blue')
bp['boxes'][1].set_facecolor('red')
ax3.set_ylabel('Score', fontsize=12)
ax3.set_title('Score Distribution - Box Plot (All Chromosomes)', fontsize=14, fontweight='bold')
ax3.grid(alpha=0.3)

# 4. Violin plot
ax4 = axes[1, 1]
sns.violinplot(data=df, x='pred', y='score', palette={'T': 'blue', 'D': 'red'}, ax=ax4)
ax4.set_xlabel('Prediction', fontsize=12)
ax4.set_ylabel('Score', fontsize=12)
ax4.set_title('Score Distribution - Violin Plot (All Chromosomes)', fontsize=14, fontweight='bold')
ax4.set_xticklabels(['Tolerated (T)', 'Deleterious (D)'])
ax4.grid(alpha=0.3)

plt.tight_layout()
plot_file = 'Final_Submission1/score_distribution_all_chromosomes.png'
plt.savefig(plot_file, dpi=300, bbox_inches='tight')
print(f"Plot saved to: {plot_file}")

# Print statistics
print("\n" + "="*60)
print("STATISTICS (All Chromosomes)")
print("="*60)
print(f"\nDeleterious (D):")
print(f"  Count: {len(df[df['pred'] == 'D']):,}")
print(f"  Mean score: {df[df['pred'] == 'D']['score'].mean():.4f}")
print(f"  Median score: {df[df['pred'] == 'D']['score'].median():.4f}")
print(f"  Std score: {df[df['pred'] == 'D']['score'].std():.4f}")
print(f"  Min score: {df[df['pred'] == 'D']['score'].min():.4f}")
print(f"  Max score: {df[df['pred'] == 'D']['score'].max():.4f}")

print(f"\nTolerated (T):")
print(f"  Count: {len(df[df['pred'] == 'T']):,}")
print(f"  Mean score: {df[df['pred'] == 'T']['score'].mean():.4f}")
print(f"  Median score: {df[df['pred'] == 'T']['score'].median():.4f}")
print(f"  Std score: {df[df['pred'] == 'T']['score'].std():.4f}")
print(f"  Min score: {df[df['pred'] == 'T']['score'].min():.4f}")
print(f"  Max score: {df[df['pred'] == 'T']['score'].max():.4f}")

print("\n" + "="*60)
print("DONE! All files saved in Final_Submission1/")
print("="*60)

plt.show()