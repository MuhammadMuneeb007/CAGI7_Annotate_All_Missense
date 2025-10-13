#!/usr/bin/env python3
"""
Simple ClinVar Data Analysis with PCA, t-SNE, and UMAP
Simplified version that loads all samples and performs dimensionality reduction
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler, LabelEncoder
import umap.umap_ as umap
import warnings
warnings.filterwarnings('ignore')

def load_all_clinvar_files(input_dir="Final_Results"):
    """Load and merge all ClinVar files"""
    print("Loading ClinVar files...")
    
    # Find all ClinVar files
    input_path = Path(input_dir)
    clinvar_files = list(input_path.glob("chr*_features_engineered_clinvar.csv"))
    
    print(f"Found {len(clinvar_files)} files")
    
    if not clinvar_files:
        print("No ClinVar files found!")
        return None
    
    # Load and concatenate all files
    dataframes = []
    for file in clinvar_files:
        print(f"Loading {file.name}...")
        df = pd.read_csv(file)
        df['chromosome'] = file.name.split('_')[0].replace('chr', '')
        dataframes.append(df)
    
    # Merge all dataframes
    merged_df = pd.concat(dataframes, ignore_index=True)
    print(f"Total samples loaded: {len(merged_df):,}")
    print(f"Total columns: {len(merged_df.columns)}")
    
    return merged_df

def prepare_data(df):
    """Prepare data for analysis"""
    print("\nPreparing data...")
    
    # Check if target column exists
    if 'CLNSIG' not in df.columns:
        print("Error: CLNSIG column not found!")
        return None, None, None
    
    # Encode target variable
    le = LabelEncoder()
    y = le.fit_transform(df['CLNSIG'])
    class_names = le.classes_
    
    print(f"Classes found: {list(class_names)}")
    
    # Get feature columns (exclude target and metadata)
    exclude_cols = ['CLNSIG', 'chromosome']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    X = df[feature_cols].copy()
    
    print(f"Initial features: {len(feature_cols)}")
    
    # Convert to numeric and handle missing values
    print("Converting features to numeric...")
    for col in X.columns:
        if X[col].dtype == 'object':
            X[col] = pd.to_numeric(X[col], errors='coerce')
    
    # Replace infinities and NaN with 0
    X = X.replace([np.inf, -np.inf, np.nan], 0)
    
    # Remove constant features
    constant_features = X.columns[X.var() == 0].tolist()
    if constant_features:
        print(f"Removing {len(constant_features)} constant features")
        X = X.drop(columns=constant_features)
    
    print(f"Final feature matrix: {X.shape}")
    print(f"Class distribution:")
    for i, class_name in enumerate(class_names):
        count = sum(y == i)
        print(f"  {class_name}: {count:,} ({count/len(y)*100:.1f}%)")
    
    return X, y, class_names

def perform_pca(X, n_components=2):
    """Perform PCA analysis"""
    print(f"\nPerforming PCA with {n_components} components...")
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply PCA
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    
    # Print explained variance
    print("PCA Results:")
    for i, ratio in enumerate(pca.explained_variance_ratio_):
        print(f"  PC{i+1}: {ratio:.3f} ({ratio*100:.1f}%)")
    print(f"  Total: {pca.explained_variance_ratio_.sum():.3f} ({pca.explained_variance_ratio_.sum()*100:.1f}%)")
    
    return X_pca, pca

def perform_tsne(X, n_components=2):
    """Perform t-SNE analysis on ALL samples"""
    print(f"\nPerforming t-SNE with {n_components} components on ALL {len(X):,} samples...")
    print("Note: This may take a while for large datasets...")
    
    # Use ALL samples - no sampling
    X_sample = X
    indices = np.arange(len(X))
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_sample)
    
    # Apply t-SNE with optimized parameters for large datasets
    perplexity = min(50, max(5, len(X_sample) // 100))  # Better perplexity for large datasets
    tsne = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        random_state=42,
        n_iter=1000,
        verbose=1,  # Show progress
        n_jobs=-1   # Use all CPU cores
    )
    X_tsne = tsne.fit_transform(X_scaled)
    
    print(f"t-SNE completed on ALL {len(X_tsne):,} samples")
    
    return X_tsne, indices

def perform_umap(X, n_components=2):
    """Perform UMAP analysis"""
    print(f"\nPerforming UMAP with {n_components} components...")
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply UMAP
    n_neighbors = min(50, len(X) // 100)
    umap_reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=0.1,
        random_state=42
    )
    X_umap = umap_reducer.fit_transform(X_scaled)
    
    print(f"UMAP completed on {len(X_umap):,} samples")
    
    return X_umap, umap_reducer

def create_visualization(X_pca, X_tsne, X_umap, y, class_names, tsne_indices=None):
    """Create visualization plots"""
    print("\nCreating visualizations...")
    
    # Set up colors
    colors = plt.cm.Set3(np.linspace(0, 1, len(class_names)))
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # PCA Plot
    ax1 = axes[0]
    for i, class_name in enumerate(class_names):
        mask = y == i
        if mask.sum() > 0:
            ax1.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                       c=[colors[i]], label=f'{class_name} (n={mask.sum():,})', 
                       alpha=0.3, s=15)
    ax1.set_title('PCA Analysis')
    ax1.set_xlabel('PC1')
    ax1.set_ylabel('PC2')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # t-SNE Plot
    ax2 = axes[1]
    if tsne_indices is not None:
        y_tsne = y[tsne_indices]
    else:
        y_tsne = y
    
    for i, class_name in enumerate(class_names):
        mask = y_tsne == i
        if mask.sum() > 0:
            ax2.scatter(X_tsne[mask, 0], X_tsne[mask, 1], 
                       c=[colors[i]], label=f'{class_name} (n={mask.sum():,})', 
                       alpha=0.3, s=15)
    ax2.set_title('t-SNE Analysis')
    ax2.set_xlabel('t-SNE 1')
    ax2.set_ylabel('t-SNE 2')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # UMAP Plot
    ax3 = axes[2]
    for i, class_name in enumerate(class_names):
        mask = y == i
        if mask.sum() > 0:
            ax3.scatter(X_umap[mask, 0], X_umap[mask, 1], 
                       c=[colors[i]], label=f'{class_name} (n={mask.sum():,})', 
                       alpha=0.3, s=15)
    ax3.set_title('UMAP Analysis')
    ax3.set_xlabel('UMAP 1')
    ax3.set_ylabel('UMAP 2')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.suptitle('ClinVar Data - Dimensionality Reduction Comparison', fontsize=16)
    plt.tight_layout()
    
    # Save plot
    output_dir = Path("plots")
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / "clinvar_analysis.png", dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / "clinvar_analysis.pdf", bbox_inches='tight')
    
    print("Plots saved to: plots/clinvar_analysis.png and .pdf")
    plt.show()

def main():
    """Main analysis function"""
    print("=" * 60)
    print("SIMPLE CLINVAR DATA ANALYSIS")
    print("PCA, t-SNE, and UMAP")
    print("=" * 60)
    
    # Set random seed
    np.random.seed(42)
    
    # Load all data
    df = load_all_clinvar_files()
    if df is None:
        return
    
    # Prepare data
    X, y, class_names = prepare_data(df)
    if X is None:
        return
    
    # Perform dimensionality reduction
    print("\n" + "=" * 40)
    print("DIMENSIONALITY REDUCTION")
    print("=" * 40)
    
    if len(X) > 100000:
        print(f"WARNING: Large dataset detected ({len(X):,} samples)")
        print("t-SNE may take considerable time. Consider using a subset if needed.")
        print("PCA and UMAP will be faster.\n")
    
    # PCA
    X_pca, pca = perform_pca(X, n_components=2)
    
    # t-SNE
    X_tsne, tsne_indices = perform_tsne(X, n_components=2)
    
    # UMAP
    X_umap, umap_reducer = perform_umap(X, n_components=2)
    
    # Create visualizations
    create_visualization(X_pca, X_tsne, X_umap, y, class_names, tsne_indices)
    
    # Print summary
    print("\n" + "=" * 60)
    print("ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"Total samples: {len(y):,}")
    print(f"Features used: {X.shape[1]}")
    print(f"Classes: {len(class_names)}")
    print(f"PCA variance explained: {pca.explained_variance_ratio_.sum():.1%}")
    print("Analysis completed successfully!")

if __name__ == "__main__":
    main()