#!/usr/bin/env python3
"""
Streamlined Genomics Feature Analyzer
Reads first 10,000 rows from each chromosome file, merges, and classifies features for ML
"""

import pandas as pd
import numpy as np
from pathlib import Path
import os
from datetime import datetime
import warnings
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.feature_extraction.text import HashingVectorizer
import re
import gc

warnings.filterwarnings('ignore')

class GenomicsFeatureAnalyzer:
    def __init__(self, input_dir="Annovar_Merge", output_dir="Final_Results"):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Create Allfiles subdirectory
        self.allfiles_dir = self.output_dir / "Allfiles"
        self.allfiles_dir.mkdir(exist_ok=True, parents=True)
        
        self.merged_data = None
        self.feature_analysis = []
        
        # Define feature categories
        self.conservation_patterns = [
            'gerp', 'phylop', 'phastcons', 'siphy', 'conservation'
        ]
        
        self.functional_patterns = [
            'func', 'exonic', 'intronic', 'utr', 'splice', 'upstream', 'downstream',
            'gene', 'transcript', 'protein', 'aachange', 'refgene', 'ensgene', 'knowngene'
        ]
        
        self.frequency_patterns = [
            'af', 'freq', 'gnomad', '1000g', 'esp', 'exac', 'allele_freq', 'maf'
        ]
        
        self.prediction_patterns = [
            'sift', 'polyphen', 'cadd', 'revel', 'vest', 'fathmm', 'provean',
            'metasvm', 'metalr', 'metarnn', 'mcap', 'primateai', 'deogen',
            'bayesdel', 'clinpred', 'dann', 'lrt', 'mutationassessor', 
            'mutationtaster', 'mutpred', 'mvp', 'mpc', 'eigen', 'alphamissense'
        ]
        
        self.clinical_patterns = [
            'clinvar', 'clnalleleid', 'clndn', 'clnsig', 'clnrevstat', 'clndisdb',
            'clnhgvs', 'omim', 'hgmd', 'cosmic'
        ]
        
        self.variant_id_patterns = [
            'chr', 'pos', 'start', 'end', 'ref', 'alt', 'rsid', 'dbsnp', 'avsnp'
        ]
    
    def find_csv_files(self):
        """Find all CSV files in input directory"""
        print(f"Searching for CSV files in: {self.input_dir}")
        csv_files = list(self.input_dir.glob("*.csv"))
        
        if not csv_files:
            # Try alternative patterns
            csv_files = list(self.input_dir.glob("*annotation*.csv"))
            csv_files.extend(list(self.input_dir.glob("*chr*.csv")))
            csv_files.extend(list(self.input_dir.glob("*merge*.csv")))
        
        # Remove duplicates and sort
        csv_files = sorted(list(set(csv_files)))
        print(f"Found {len(csv_files)} CSV files")
        
        return csv_files
    
    def read_and_merge_data(self, csv_files, max_rows_per_file=100000):
        """Read first N rows from each CSV file and merge them"""
        print(f"\nReading first {max_rows_per_file:,} rows from each chromosome file...")
        
        all_chunks = []
        total_rows = 0
        
        for i, file_path in enumerate(csv_files, 1):
            print(f"Processing file {i}/{len(csv_files)}: {file_path.name}")
            
            try:
                # Read only first N rows from each file
                file_data = pd.read_csv(file_path, nrows=max_rows_per_file, low_memory=False)
                
                if not file_data.empty:
                    all_chunks.append(file_data)
                    total_rows += len(file_data)
                    print(f"  Loaded {len(file_data):,} rows")
                    
                    # Clean memory
                    del file_data
                    gc.collect()
                    
            except Exception as e:
                print(f"  ERROR reading {file_path.name}: {e}")
                continue
        
        if not all_chunks:
            raise ValueError("No data could be loaded from CSV files")
        
        # Merge all data
        print(f"\nMerging {len(all_chunks)} datasets...")
        self.merged_data = pd.concat(all_chunks, ignore_index=True, sort=False)
        
        # Clean memory
        del all_chunks
        gc.collect()
        
        print(f"Merged dataset: {len(self.merged_data):,} rows × {len(self.merged_data.columns):,} columns")
        return True
    
    def categorize_feature(self, column_name):
        """Categorize feature based on name patterns"""
        col_lower = column_name.lower()
        
        if any(pattern in col_lower for pattern in self.conservation_patterns):
            return "Conservation Score"
        elif any(pattern in col_lower for pattern in self.functional_patterns):
            return "Functional Annotation"
        elif any(pattern in col_lower for pattern in self.frequency_patterns):
            return "Population Frequency"
        elif any(pattern in col_lower for pattern in self.prediction_patterns):
            return "Pathogenicity Prediction"
        elif any(pattern in col_lower for pattern in self.clinical_patterns):
            return "Clinical Database"
        elif any(pattern in col_lower for pattern in self.variant_id_patterns):
            return "Variant Identifier"
        else:
            return "Other"
    
    def clean_and_infer_dtype(self, series):
        """Clean column and infer proper data type"""
        # Handle missing values
        series = series.replace(['', '.', 'NA', 'N/A', 'null', 'NULL', '-', 'nan', 'NaN'], np.nan)
        
        # Try to convert to numeric
        numeric_series = pd.to_numeric(series, errors='coerce')
        
        # If most values can be converted to numeric, treat as numeric
        non_null_count = series.count()
        numeric_count = numeric_series.count()
        
        if non_null_count == 0:
            return series, 'empty'
        
        if numeric_count / non_null_count > 0.8:  # 80% numeric
            # Determine if int or float
            non_null_numeric = numeric_series.dropna()
            if len(non_null_numeric) > 0 and (non_null_numeric % 1 == 0).all():
                return numeric_series.astype('Int64'), 'int'
            else:
                return numeric_series, 'float'
        else:
            return series, 'string'
    
    def estimate_encoding_columns(self, series, max_categories=50):
        """Estimate number of columns needed for encoding"""
        if series.dtype in ['int64', 'Int64', 'float64']:
            return 0, "No encoding needed (numeric)"
        
        unique_values = series.dropna().unique()
        n_unique = len(unique_values)
        
        if n_unique <= 1:
            return 0, "Constant feature (no encoding needed)"
        elif n_unique == 2:
            return 1, "Binary encoding (1 column)"
        elif n_unique <= max_categories:
            return n_unique, f"One-hot encoding ({n_unique} columns)"
        else:
            return 10, f"Hash encoding (10 columns) - too many categories ({n_unique})"
    
    def is_ml_suitable(self, column_name, missingness, n_unique, data_type, category):
        """Determine if feature is suitable for ML"""
        col_lower = column_name.lower()
        
        # Exclude based on missingness
        if missingness > 90:
            return False, "Too much missing data (>90%)"
        
        # Exclude clinical database features (data leakage)
        if category == "Clinical Database":
            return False, "Data leakage risk (clinical database)"
        
        # Exclude some variant identifiers
        if category == "Variant Identifier":
            if any(term in col_lower for term in ['chr', 'pos', 'start', 'end']):
                return False, "Genomic coordinates (not predictive)"
            elif any(term in col_lower for term in ['rsid', 'dbsnp', 'avsnp']):
                return False, "Variant ID (not generalizable)"
        
        # Exclude features with no variation
        if n_unique <= 1:
            return False, "No variation (constant)"
        
        # Exclude categorical features with too many categories
        if data_type == 'string' and n_unique > 1000:
            return False, f"Too many categories ({n_unique})"
        
        # Include good features
        return True, "Suitable for ML"
    
    def analyze_all_features(self):
        """Analyze all features in the merged dataset"""
        print("\nAnalyzing all features...")
        
        total_columns = len(self.merged_data.columns)
        
        for i, column in enumerate(self.merged_data.columns, 1):
            print(f"Analyzing feature {i}/{total_columns}: {column}")
            
            try:
                # Get column data
                series = self.merged_data[column].copy()
                
                # Clean and infer data type
                cleaned_series, inferred_type = self.clean_and_infer_dtype(series)
                
                # Basic statistics
                total_count = len(series)
                missing_count = series.isnull().sum()
                missingness = (missing_count / total_count) * 100
                
                # Unique values analysis
                non_null_series = series.dropna()
                n_unique = len(non_null_series.unique())
                
                # Get first 10 unique values
                unique_values = non_null_series.unique()[:10]
                unique_values_str = [str(v)[:30] for v in unique_values]  # Truncate long values
                
                # Feature categorization
                category = self.categorize_feature(column)
                
                # Encoding requirements
                encoding_cols, encoding_type = self.estimate_encoding_columns(cleaned_series)
                
                # ML suitability
                ml_suitable, ml_reason = self.is_ml_suitable(
                    column, missingness, n_unique, inferred_type, category
                )
                
                # Store analysis results
                feature_info = {
                    'Feature_Name': column,
                    'Missingness_Percent': round(missingness, 2),
                    'Unique_Values_Count': n_unique,
                    'First_10_Unique_Values': unique_values_str,
                    'ML_Suitable': 'Yes' if ml_suitable else 'No',
                    'ML_Reason': ml_reason,
                    'Feature_Category': category,
                    'Requires_Encoding': 'Yes' if encoding_cols > 0 else 'No',
                    'Encoding_Type': encoding_type,
                    'New_Columns_If_Encoded': encoding_cols,
                    'Data_Type': inferred_type,
                    'Original_Type': str(series.dtype)
                }
                
                self.feature_analysis.append(feature_info)
                
            except Exception as e:
                print(f"  ERROR analyzing {column}: {e}")
                # Add error entry
                error_info = {
                    'Feature_Name': column,
                    'Missingness_Percent': 100.0,
                    'Unique_Values_Count': 0,
                    'First_10_Unique_Values': ['ERROR'],
                    'ML_Suitable': 'No',
                    'ML_Reason': f'Analysis error: {str(e)}',
                    'Feature_Category': 'Error',
                    'Requires_Encoding': 'No',
                    'Encoding_Type': 'Error',
                    'New_Columns_If_Encoded': 0,
                    'Data_Type': 'error',
                    'Original_Type': str(self.merged_data[column].dtype)
                }
                self.feature_analysis.append(error_info)
        
        print(f"Completed analysis of {len(self.feature_analysis)} features")
    
    def save_results(self):
        """Save analysis results to CSV file"""
        print("\nSaving results...")
        
        # Convert to DataFrame
        results_df = pd.DataFrame(self.feature_analysis)
        
        # Sort by ML suitability and missingness
        results_df['ML_Suitable_Sort'] = results_df['ML_Suitable'].map({'Yes': 0, 'No': 1})
        results_df = results_df.sort_values(['ML_Suitable_Sort', 'Missingness_Percent'])
        results_df = results_df.drop('ML_Suitable_Sort', axis=1)
        
        # Save main results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.allfiles_dir / f"feature_analysis_{timestamp}.csv"
        results_df.to_csv(output_file, index=False)
        
        # Save summary statistics
        summary_file = self.allfiles_dir / f"analysis_summary_{timestamp}.txt"
        
        with open(summary_file, 'w') as f:
            f.write("GENOMICS FEATURE ANALYSIS SUMMARY\n")
            f.write("=" * 50 + "\n")
            f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Sampling: First 10,000 rows from each chromosome file\n")
            f.write(f"Total Features Analyzed: {len(results_df)}\n")
            f.write(f"Total Sample Variants: {len(self.merged_data):,}\n")
            f.write(f"CSV Files Processed: {len(self.find_csv_files())}\n\n")
            
            # ML Suitability Summary
            ml_suitable_count = (results_df['ML_Suitable'] == 'Yes').sum()
            ml_unsuitable_count = (results_df['ML_Suitable'] == 'No').sum()
            
            f.write("ML SUITABILITY:\n")
            f.write(f"  Suitable for ML: {ml_suitable_count}\n")
            f.write(f"  Not suitable: {ml_unsuitable_count}\n\n")
            
            # Category breakdown
            f.write("FEATURE CATEGORIES:\n")
            category_counts = results_df['Feature_Category'].value_counts()
            for category, count in category_counts.items():
                f.write(f"  {category}: {count}\n")
            
            f.write(f"\nData Type Distribution:\n")
            type_counts = results_df['Data_Type'].value_counts()
            for dtype, count in type_counts.items():
                f.write(f"  {dtype}: {count}\n")
            
            f.write(f"\nEncoding Requirements:\n")
            encoding_counts = results_df['Requires_Encoding'].value_counts()
            for req, count in encoding_counts.items():
                f.write(f"  {req}: {count}\n")
            
            # Missingness statistics
            f.write(f"\nMissingness Statistics:\n")
            f.write(f"  Average: {results_df['Missingness_Percent'].mean():.2f}%\n")
            f.write(f"  Median: {results_df['Missingness_Percent'].median():.2f}%\n")
            f.write(f"  Min: {results_df['Missingness_Percent'].min():.2f}%\n")
            f.write(f"  Max: {results_df['Missingness_Percent'].max():.2f}%\n")
        
        print(f"Results saved:")
        print(f"  📊 {output_file}")
        print(f"  📋 {summary_file}")
        
        return output_file, summary_file
    
    def generate_ml_ready_recommendations(self):
        """Generate recommendations for ML-ready features"""
        results_df = pd.DataFrame(self.feature_analysis)
        ml_ready = results_df[results_df['ML_Suitable'] == 'Yes'].copy()
        
        if len(ml_ready) == 0:
            return
        
        # Sort by category and missingness
        ml_ready = ml_ready.sort_values(['Feature_Category', 'Missingness_Percent'])
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        recommendations_file = self.allfiles_dir / f"ml_recommendations_{timestamp}.txt"
        
        with open(recommendations_file, 'w') as f:
            f.write("MACHINE LEARNING READY FEATURES\n")
            f.write("=" * 50 + "\n")
            f.write(f"Total ML-Ready Features: {len(ml_ready)}\n\n")
            
            f.write("FEATURES BY CATEGORY:\n")
            f.write("-" * 30 + "\n")
            
            for category in ml_ready['Feature_Category'].unique():
                cat_features = ml_ready[ml_ready['Feature_Category'] == category]
                f.write(f"\n{category.upper()} ({len(cat_features)} features):\n")
                
                for _, row in cat_features.iterrows():
                    encoding_info = f" [+{row['New_Columns_If_Encoded']} cols]" if row['New_Columns_If_Encoded'] > 0 else ""
                    f.write(f"  • {row['Feature_Name']} ({row['Missingness_Percent']:.1f}% missing, {row['Data_Type']}){encoding_info}\n")
            
            f.write(f"\nENCODING SUMMARY:\n")
            f.write("-" * 20 + "\n")
            total_new_cols = ml_ready['New_Columns_If_Encoded'].sum()
            f.write(f"Total additional columns from encoding: {total_new_cols}\n")
            f.write(f"Final feature count after encoding: {len(ml_ready) + total_new_cols - len(ml_ready[ml_ready['New_Columns_If_Encoded'] > 0])}\n")
        
        print(f"  📋 {recommendations_file}")
    
    def run_analysis(self):
        """Run the complete analysis pipeline"""
        print("GENOMICS FEATURE ANALYZER")
        print("=" * 50)
        
        try:
            # Step 1: Find CSV files
            csv_files = self.find_csv_files()
            if not csv_files:
                print("ERROR: No CSV files found!")
                return False
            
            # Step 2: Read and merge data (first 10,000 rows per file)
            self.read_and_merge_data(csv_files)
            
            # Step 3: Analyze all features
            self.analyze_all_features()
            
            # Step 4: Save results
            output_file, summary_file = self.save_results()
            
            # Step 5: Generate ML recommendations
            self.generate_ml_ready_recommendations()
            
            # Final summary
            ml_suitable_count = sum(1 for f in self.feature_analysis if f['ML_Suitable'] == 'Yes')
            
            print(f"\n" + "=" * 50)
            print("ANALYSIS COMPLETE!")
            print("=" * 50)
            print(f"Total features analyzed: {len(self.feature_analysis)}")
            print(f"ML-suitable features: {ml_suitable_count}")
            print(f"Sample size: {len(self.merged_data):,} variants (first 10k from each file)")
            print(f"\nResults saved in: {self.allfiles_dir}")
            
            return True
            
        except Exception as e:
            print(f"ERROR: Analysis failed - {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    """Main function to run the analysis"""
    analyzer = GenomicsFeatureAnalyzer(
        input_dir="Annovar_Merge",  # Your CSV files directory
        output_dir="Final_Results"   # Output directory
    )
    
    success = analyzer.run_analysis()
    
    if success:
        print("\n✅ Analysis completed successfully!")
        print("Check the Final_Results/Allfiles directory for output files.")
    else:
        print("\n❌ Analysis failed!")


if __name__ == "__main__":
    main()