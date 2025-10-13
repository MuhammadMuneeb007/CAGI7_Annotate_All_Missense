import polars as pl
import matplotlib.pyplot as plt
import numpy as np

print("="*70)
print("ORIGINAL PREDICTION COUNTS")
print("="*70)

# Files to check
files = {
    'Model 2': 'submission2/CAGI7_FinalSubmission_model_2.tsv',
    'Model 3': 'submission3/CAGI7_FinalSubmission_model_3.tsv',
    'Model 4': 'optimzed/CAGI7_FinalSubmission_model_4.tsv'
}

# Mean thresholds for each model (from your analysis)
mean_thresholds = {
    'Model 2': 0.839145,
    'Model 3': 0.839445,
    'Model 4': 0.859800
}

# Store data for plotting
model_data = {}

for model_name, filepath in files.items():
    print(f"\n{model_name}: {filepath}")
    
    try:
        df = pl.read_csv(
            filepath,
            separator='\t',
            dtypes={'#chr': pl.Utf8}
        )
        
        model_data[model_name] = df
        
        total = len(df)
        d_count = df.filter(pl.col('pred') == 'D').shape[0]
        t_count = df.filter(pl.col('pred') == 'T').shape[0]
        u_count = df.filter(pl.col('pred') == 'U').shape[0]
        
        print(f"  Total variants:      {total:,}")
        print(f"  Deleterious (D):     {d_count:,} ({d_count/total*100:.2f}%)")
        print(f"  Tolerated (T):       {t_count:,} ({t_count/total*100:.2f}%)")
        print(f"  Unknown (U):         {u_count:,} ({u_count/total*100:.2f}%)")
        
    except Exception as e:
        print(f"  Error: {e}")

# Reclassification using MEAN as threshold
print("\n" + "="*70)
print("RECLASSIFIED PREDICTION COUNTS (Using Mean Threshold)")
print("="*70)
print("\nClassification Rules:")
print("  - score = 0 AND sd = 0  → U (Unknown)")
print("  - 0 < score < mean      → T (Tolerated/Benign)")
print("  - score >= mean         → D (Deleterious/Pathogenic)")
print("="*70)

reclassified_data = {}

for model_name, df in model_data.items():
    mean_threshold = mean_thresholds[model_name]
    print(f"\n{model_name} (Mean threshold: {mean_threshold:.6f}):")
    
    # Reclassify based on score, SD, and mean threshold
    df = df.with_columns(
        pl.when((pl.col('score') == 0) & (pl.col('sd') == 0))
        .then(pl.lit('U'))
        .when((pl.col('score') > 0) & (pl.col('score') < mean_threshold))
        .then(pl.lit('T'))
        .when(pl.col('score') >= mean_threshold)
        .then(pl.lit('D'))
        .otherwise(pl.lit('U'))  # Edge cases default to U
        .alias('pred_new')
    )
    
    reclassified_data[model_name] = df
    
    total = len(df)
    d_count = df.filter(pl.col('pred_new') == 'D').shape[0]
    t_count = df.filter(pl.col('pred_new') == 'T').shape[0]
    u_count = df.filter(pl.col('pred_new') == 'U').shape[0]
    
    print(f"  Total variants:      {total:,}")
    print(f"  Deleterious (D):     {d_count:,} ({d_count/total*100:.2f}%)")
    print(f"  Tolerated (T):       {t_count:,} ({t_count/total*100:.2f}%)")
    print(f"  Unknown (U):         {u_count:,} ({u_count/total*100:.2f}%)")
    
    # Show changes from original
    orig_d = model_data[model_name].filter(pl.col('pred') == 'D').shape[0]
    orig_t = model_data[model_name].filter(pl.col('pred') == 'T').shape[0]
    orig_u = model_data[model_name].filter(pl.col('pred') == 'U').shape[0]
    
    print(f"  Changes from original:")
    print(f"    D: {orig_d:,} → {d_count:,} ({d_count - orig_d:+,})")
    print(f"    T: {orig_t:,} → {t_count:,} ({t_count - orig_t:+,})")
    print(f"    U: {orig_u:,} → {u_count:,} ({u_count - orig_u:+,})")

# Create score distribution plots with threshold lines
fig, axes = plt.subplots(len(files), 1, figsize=(14, 5*len(files)))
if len(files) == 1:
    axes = [axes]

for idx, (model_name, df) in enumerate(model_data.items()):
    ax = axes[idx]
    
    # Get scores
    scores = df['score'].to_numpy()
    non_zero_scores = scores[scores > 0]
    zero_count = np.sum(scores == 0)
    mean_threshold = mean_thresholds[model_name]
    
    # Plot histogram
    ax.hist(non_zero_scores, bins=100, alpha=0.7, edgecolor='black', color='steelblue')
    ax.set_xlabel('Score', fontsize=12, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax.set_title(f'{model_name} - Score Distribution\n(Excluding {zero_count:,} zero scores)', 
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add vertical line at mean threshold
    ax.axvline(x=mean_threshold, color='red', linestyle='--', linewidth=2.5, 
               label=f'Mean Threshold ({mean_threshold:.4f})')
    
    # Add shaded regions
    ax.axvspan(0, mean_threshold, alpha=0.2, color='green', label='Tolerated (T)')
    ax.axvspan(mean_threshold, 1, alpha=0.2, color='red', label='Deleterious (D)')
    
    ax.legend(fontsize=11, loc='upper left')
    ax.set_xlim([0, 1])
    
plt.tight_layout()
plt.savefig('score_distributions_with_mean_threshold.png', dpi=300, bbox_inches='tight')
print("\n[Saved: score_distributions_with_mean_threshold.png]")

# Create comparison bar plots
fig, axes = plt.subplots(len(files), 2, figsize=(16, 5*len(files)))
if len(files) == 1:
    axes = axes.reshape(1, -1)

for idx, model_name in enumerate(model_data.keys()):
    df_orig = model_data[model_name]
    df_new = reclassified_data[model_name]
    
    # Original distribution
    orig_counts = [
        df_orig.filter(pl.col('pred') == 'D').shape[0],
        df_orig.filter(pl.col('pred') == 'T').shape[0],
        df_orig.filter(pl.col('pred') == 'U').shape[0]
    ]
    
    # Reclassified distribution
    new_counts = [
        df_new.filter(pl.col('pred_new') == 'D').shape[0],
        df_new.filter(pl.col('pred_new') == 'T').shape[0],
        df_new.filter(pl.col('pred_new') == 'U').shape[0]
    ]
    
    labels = ['D\n(Deleterious)', 'T\n(Tolerated)', 'U\n(Unknown)']
    colors = ['#e74c3c', '#27ae60', '#95a5a6']
    
    # Original
    bars1 = axes[idx, 0].bar(labels, orig_counts, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    axes[idx, 0].set_title(f'{model_name} - Original Distribution', fontsize=14, fontweight='bold')
    axes[idx, 0].set_ylabel('Count', fontsize=12, fontweight='bold')
    axes[idx, 0].grid(axis='y', alpha=0.3)
    for i, (bar, v) in enumerate(zip(bars1, orig_counts)):
        height = bar.get_height()
        axes[idx, 0].text(bar.get_x() + bar.get_width()/2., height,
                         f'{v:,}\n({v/sum(orig_counts)*100:.1f}%)',
                         ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Reclassified
    bars2 = axes[idx, 1].bar(labels, new_counts, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    axes[idx, 1].set_title(f'{model_name} - Mean-Based Reclassification\n(Threshold: {mean_thresholds[model_name]:.4f})', 
                          fontsize=14, fontweight='bold')
    axes[idx, 1].set_ylabel('Count', fontsize=12, fontweight='bold')
    axes[idx, 1].grid(axis='y', alpha=0.3)
    for i, (bar, v) in enumerate(zip(bars2, new_counts)):
        height = bar.get_height()
        axes[idx, 1].text(bar.get_x() + bar.get_width()/2., height,
                         f'{v:,}\n({v/sum(new_counts)*100:.1f}%)',
                         ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('prediction_comparison_mean_threshold.png', dpi=300, bbox_inches='tight')
print("[Saved: prediction_comparison_mean_threshold.png]")

# Summary table
print("\n" + "="*70)
print("SUMMARY TABLE")
print("="*70)
print(f"\n{'Model':<15} {'Threshold':<12} {'D (Original)':<15} {'D (New)':<15} {'Change':<10}")
print("-" * 70)
for model_name in model_data.keys():
    df_orig = model_data[model_name]
    df_new = reclassified_data[model_name]
    
    orig_d = df_orig.filter(pl.col('pred') == 'D').shape[0]
    new_d = df_new.filter(pl.col('pred_new') == 'D').shape[0]
    change = new_d - orig_d
    
    print(f"{model_name:<15} {mean_thresholds[model_name]:<12.6f} {orig_d:<15,} {new_d:<15,} {change:+,}")

print("\n" + "="*70)