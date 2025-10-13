#!/usr/bin/env python3
"""
SIMPLE TEST SCRIPT - Schema Consistency Check
Tests 100 variants from 2 chromosomes to verify identical column schemas
"""

import pandas as pd
import os
import sys

# Standard dbNSFP columns to add (same as in main script)
DBNSFP_OUTPUT_COLUMNS = [
    'aaref', 'aaalt', 'aapos', 'genename', 'cds_strand', 'refcodon', 'codonpos',
    'Ensembl_geneid', 'Ensembl_transcriptid', 'Ensembl_proteinid',
    'hg19_chr', 'hg19_pos', 'hg18_chr', 'hg18_pos'
]

def create_genomic_key(chr_val, pos_val, ref_val, alt_val):
    """Create standardized genomic coordinate key"""
    if pd.isna(chr_val) or pd.isna(pos_val) or pd.isna(ref_val) or pd.isna(alt_val):
        return None
    chr_clean = str(chr_val).replace('chr', '')
    return f"{chr_clean}:{pos_val}:{ref_val}:{alt_val}"

def read_annotation_sample(annotation_file, n_rows=10000):
    """Read first n rows from annotation file"""
    print(f"📂 Reading {n_rows} rows from: {os.path.basename(annotation_file)}")
    
    try:
        # Read first n rows
        df = pd.read_csv(annotation_file, nrows=n_rows, low_memory=False)
        print(f"   ✓ Loaded {len(df):,} rows, {len(df.columns)} columns")
        
        # Create genomic keys
        df['genomic_key'] = df.apply(
            lambda row: create_genomic_key(row['Chr'], row['Start'], row['Ref'], row['Alt']), 
            axis=1
        )
        
        valid_keys = df['genomic_key'].notna().sum()
        print(f"   ✓ Created {valid_keys:,} valid genomic keys")
        
        return df
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return None

def read_dbnsfp_sample(dbnsfp_file, annotation_keys, n_rows=100000):
    """Read sample from dbNSFP file and filter for annotation keys"""
    print(f"🧬 Reading sample from: {os.path.basename(dbnsfp_file)}")
    
    try:
        # Define column names
        dbnsfp_columns = [
            'chr', 'pos(1-based)', 'ref', 'alt', 'aaref', 'aaalt', 
            'hg19_chr', 'hg19_pos(1-based)', 'hg18_chr', 'hg18_pos(1-based)', 
            'genename', 'cds_strand', 'refcodon', 'codonpos', 
            'Ensembl_geneid', 'Ensembl_transcriptid', 'Ensembl_proteinid', 'aapos'
        ]
        
        # Read first n_rows (skip header)
        df = pd.read_csv(
            dbnsfp_file,
            sep='\t',
            skiprows=1,
            names=dbnsfp_columns,
            nrows=n_rows,
            low_memory=False
        )
        
        print(f"   ✓ Read {len(df):,} rows from dbNSFP")
        
        # Create genomic keys
        df['genomic_key'] = df.apply(
            lambda row: create_genomic_key(row['chr'], row['pos(1-based)'], row['ref'], row['alt']), 
            axis=1
        )
        
        # Filter for annotation keys
        annotation_keys_set = set(key for key in annotation_keys if pd.notna(key))
        matching_df = df[df['genomic_key'].isin(annotation_keys_set)]
        
        print(f"   ✓ Found {len(matching_df):,} matching variants")
        
        return matching_df
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return pd.DataFrame()

def create_standardized_output(annotation_df, dbnsfp_df):
    """Create output with standardized schema"""
    print(f"🔄 Creating standardized output...")
    
    # Start with annotation data
    result_df = annotation_df.copy()
    
    # Add all standard dbNSFP columns with None defaults
    for col in DBNSFP_OUTPUT_COLUMNS:
        result_df[col] = None
    
    print(f"   ✓ Added {len(DBNSFP_OUTPUT_COLUMNS)} standard dbNSFP columns")
    
    if len(dbnsfp_df) > 0:
        # Create mapping dictionary
        dbnsfp_dict = {}
        for _, row in dbnsfp_df.iterrows():
            key = row['genomic_key']
            if pd.notna(key):
                dbnsfp_dict[key] = {
                    'aaref': row.get('aaref'),
                    'aaalt': row.get('aaalt'),
                    'aapos': row.get('aapos'),
                    'genename': row.get('genename'),
                    'cds_strand': row.get('cds_strand'),
                    'refcodon': row.get('refcodon'),
                    'codonpos': row.get('codonpos'),
                    'Ensembl_geneid': row.get('Ensembl_geneid'),
                    'Ensembl_transcriptid': row.get('Ensembl_transcriptid'),
                    'Ensembl_proteinid': row.get('Ensembl_proteinid'),
                    'hg19_chr': row.get('hg19_chr'),
                    'hg19_pos': row.get('hg19_pos(1-based)'),
                    'hg18_chr': row.get('hg18_chr'),
                    'hg18_pos': row.get('hg18_pos(1-based)')
                }
        
        # Fill in dbNSFP data
        matched_count = 0
        for idx, row in result_df.iterrows():
            key = row['genomic_key']
            if pd.notna(key) and key in dbnsfp_dict:
                for col, value in dbnsfp_dict[key].items():
                    result_df.at[idx, col] = value
                matched_count += 1
        
        print(f"   ✓ Filled {matched_count:,} variants with dbNSFP data")
    
    return result_df

def test_chromosome(chr_name, n_variants=100):
    """Test a single chromosome"""
    annotation_file = f"Annovar_Merge/{chr_name}_concatenated_annotations.csv"
    dbnsfp_file = f"dbNSFP5.1_nsSNV.{chr_name}"
    
    print(f"\n{'='*50}")
    print(f"TESTING {chr_name.upper()} ({n_variants} variants)")
    print("="*50)
    
    # Check files exist
    if not os.path.exists(annotation_file):
        print(f"❌ Annotation file not found: {annotation_file}")
        return None
    
    if not os.path.exists(dbnsfp_file):
        print(f"❌ dbNSFP file not found: {dbnsfp_file}")
        return None
    
    try:
        # Read annotation sample
        annotation_df = read_annotation_sample(annotation_file, n_variants)
        if annotation_df is None:
            return None
        
        # Get annotation keys
        annotation_keys = annotation_df['genomic_key'].dropna().tolist()
        
        # Read dbNSFP sample
        dbnsfp_df = read_dbnsfp_sample(dbnsfp_file, annotation_keys)
        
        # Create standardized output
        result_df = create_standardized_output(annotation_df, dbnsfp_df)
        
        # Remove helper column for final result
        final_df = result_df.drop('genomic_key', axis=1)
        
        print(f"\n📊 {chr_name.upper()} RESULTS:")
        print(f"   • Input variants: {len(annotation_df):,}")
        print(f"   • dbNSFP matches: {result_df['aaref'].notna().sum():,}")
        print(f"   • Final columns: {len(final_df.columns)}")
        print(f"   • Column names: {list(final_df.columns)}")
        
        return {
            'chromosome': chr_name,
            'variants': len(final_df),
            'annotated': result_df['aaref'].notna().sum(),
            'columns': len(final_df.columns),
            'column_names': list(final_df.columns),
            'dataframe': final_df
        }
        
    except Exception as e:
        print(f"❌ Error testing {chr_name}: {e}")
        return None

def compare_schemas(results):
    """Compare schemas between chromosomes"""
    print(f"\n{'='*60}")
    print("🔍 SCHEMA COMPARISON")
    print("="*60)
    
    if len(results) < 2:
        print("❌ Need at least 2 successful chromosome results to compare")
        return
    
    # Check column counts
    column_counts = [r['columns'] for r in results]
    print(f"📊 Column counts:")
    for r in results:
        print(f"   • {r['chromosome']}: {r['columns']} columns")
    
    # Check if all counts are the same
    if len(set(column_counts)) == 1:
        print(f"✅ All chromosomes have identical column counts: {column_counts[0]}")
    else:
        print(f"❌ Column count mismatch detected!")
        return
    
    # Check column names
    print(f"\n📋 Column name comparison:")
    first_columns = set(results[0]['column_names'])
    
    all_identical = True
    for r in results[1:]:
        other_columns = set(r['column_names'])
        if first_columns != other_columns:
            all_identical = False
            print(f"❌ {r['chromosome']} has different columns!")
            
            missing = first_columns - other_columns
            extra = other_columns - first_columns
            
            if missing:
                print(f"   Missing: {missing}")
            if extra:
                print(f"   Extra: {extra}")
        else:
            print(f"✅ {r['chromosome']}: identical to {results[0]['chromosome']}")
    
    if all_identical:
        print(f"\n🎉 ALL SCHEMAS ARE IDENTICAL!")
        print(f"📊 Standard column count: {column_counts[0]}")
        print(f"📋 Standard columns added: {len(DBNSFP_OUTPUT_COLUMNS)}")
        
        # Show annotation rates
        print(f"\n📈 Annotation rates:")
        for r in results:
            rate = (r['annotated'] / r['variants']) * 100 if r['variants'] > 0 else 0
            print(f"   • {r['chromosome']}: {r['annotated']}/{r['variants']} ({rate:.1f}%)")
    else:
        print(f"\n❌ SCHEMA INCONSISTENCIES DETECTED!")

def main():
    """Main test function"""
    print("=" * 60)
    print("🧪 SCHEMA CONSISTENCY TEST")
    print("🎯 Testing 100 variants from 2 chromosomes")
    print("=" * 60)
    
    # Test parameters
    n_variants = 10000
    test_chromosomes = ['chr1', 'chr10', 'chr11', 'chr12']  # Change these if needed

    print(f"📋 Standard dbNSFP columns to add: {len(DBNSFP_OUTPUT_COLUMNS)}")
    print(f"🧬 Testing chromosomes: {', '.join(test_chromosomes)}")
    print(f"📊 Sample size: {n_variants} variants per chromosome")
    
    # Test each chromosome
    results = []
    
    for chr_name in test_chromosomes:
        result = test_chromosome(chr_name, n_variants)
        if result:
            results.append(result)
    
    # Compare schemas
    if results:
        compare_schemas(results)
        
        # Save test outputs for inspection
        print(f"\n💾 Saving test outputs...")
        for r in results:
            test_file = f"test_{r['chromosome']}_sample.csv"
            r['dataframe'].to_csv(test_file, index=False)
            print(f"   ✓ {test_file}")
            
        print(f"\n🔍 You can inspect the test files to verify column consistency")
    else:
        print(f"\n❌ No successful test results")

if __name__ == "__main__":
    main()