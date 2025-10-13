#!/usr/bin/env python3
"""
Maximally Inclusive Genomics Feature Analyzer
Includes almost all features except obvious data leakage
Provides detailed guidance on how to handle challenging features
"""

import polars as pl
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import os
from datetime import datetime
import warnings
from collections import Counter
import seaborn as sns

warnings.filterwarnings('ignore')
plt.style.use('default')

class MaximalInclusiveAnalyzer:
    def __init__(self, input_dir="Annovar_Merge", output_dir="Feature_Analysis"):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.feature_stats = {}
        self.feature_decisions = {}
        self.total_variants = 0
        self.total_files = 0
        
    def find_all_csv_files(self):
        """Find all CSV files in the input directory"""
        print(f"Looking for CSV files in: {self.input_dir.absolute()}")
        
        patterns = ["*.csv", "*annotation*.csv", "*chr*.csv", "*merge*.csv"]
        all_files = []
        
        for pattern in patterns:
            files = list(self.input_dir.glob(pattern))
            all_files.extend(files)
        
        unique_files = list(set(all_files))
        unique_files.sort()
        
        if not unique_files:
            print(f"ERROR: No CSV files found in {self.input_dir}")
            if self.input_dir.exists():
                all_items = list(self.input_dir.iterdir())
                print(f"Files found in directory: {[f.name for f in all_items[:10]]}")
        
        return unique_files
    
    def analyze_single_csv_file(self, file_path, max_rows=50000):
        """Analyze features in a single CSV file"""
        print(f"  Processing: {file_path.name}")
        
        try:
            df = pl.read_csv(
                str(file_path),
                infer_schema_length=1000,
                null_values=["", ".", "NA", "N/A", "null", "NULL", "-", "nan", "NaN"],
                ignore_errors=True,
                n_rows=max_rows
            )
            
            rows = df.height
            cols = df.width
            print(f"    Loaded: {rows:,} rows × {cols:,} columns")
            self.total_variants += rows
            
            file_feature_stats = {}
            
            for column in df.columns:
                try:
                    total_count = rows
                    null_count = df[column].null_count()
                    unique_count = df[column].n_unique()
                    missing_percentage = (null_count / total_count) * 100 if total_count > 0 else 100
                    
                    dtype_str = str(df[column].dtype)
                    is_numeric = any(t in dtype_str for t in ['Int', 'Float', 'Decimal'])
                    
                    try:
                        sample_data = df.select(pl.col(column).drop_nulls().head(5)).to_pandas()[column].tolist()
                        sample_values = [str(v)[:50] for v in sample_data] if sample_data else []
                    except:
                        sample_values = []
                    
                    file_feature_stats[column] = {
                        'missing_pct': missing_percentage,
                        'unique_count': unique_count,
                        'total_count': total_count,
                        'is_numeric': is_numeric,
                        'data_type': dtype_str,
                        'sample_values': sample_values
                    }
                    
                except Exception as e:
                    print(f"    Warning: Error analyzing column {column}: {e}")
                    continue
            
            return file_feature_stats
            
        except Exception as e:
            print(f"    ERROR: Failed to read {file_path.name}: {e}")
            return {}
    
    def combine_stats_across_files(self, all_file_stats):
        """Combine feature statistics from all files"""
        print(f"\nCombining statistics across {len(all_file_stats)} files...")
        
        all_feature_names = set()
        for file_stats in all_file_stats:
            all_feature_names.update(file_stats.keys())
        
        print(f"Found {len(all_feature_names)} unique features across all files")
        
        for feature_name in all_feature_names:
            feature_data_list = []
            files_containing_feature = 0
            
            for file_stats in all_file_stats:
                if feature_name in file_stats:
                    feature_data_list.append(file_stats[feature_name])
                    files_containing_feature += 1
            
            if feature_data_list:
                avg_missing = np.mean([data['missing_pct'] for data in feature_data_list])
                min_missing = np.min([data['missing_pct'] for data in feature_data_list])
                max_missing = np.max([data['missing_pct'] for data in feature_data_list])
                max_unique = max([data['unique_count'] for data in feature_data_list])
                is_numeric = any([data['is_numeric'] for data in feature_data_list])
                sample_values = feature_data_list[0]['sample_values']
                coverage_pct = (files_containing_feature / len(all_file_stats)) * 100
                
                self.feature_stats[feature_name] = {
                    'avg_missing_pct': avg_missing,
                    'min_missing_pct': min_missing,
                    'max_missing_pct': max_missing,
                    'max_unique_count': max_unique,
                    'files_present_in': files_containing_feature,
                    'total_files': len(all_file_stats),
                    'file_coverage_pct': coverage_pct,
                    'is_numeric': is_numeric,
                    'sample_values': sample_values
                }
    
    def make_maximal_decision(self, feature_name, stats):
        """Make maximally inclusive decision - only exclude obvious problems"""
        fname = feature_name.lower()
        missing_pct = stats['avg_missing_pct']
        unique_count = stats['max_unique_count']
        is_numeric = stats['is_numeric']
        coverage_pct = stats['file_coverage_pct']
        
        # EXCLUDE - Only the most obvious data leakage (ClinVar clinical annotations)
        clinvar_leakage = ['clnsig', 'clndn', 'clnrevstat', 'clndisdb']  # Reduced list
        if any(term in fname for term in clinvar_leakage):
            return 'EXCLUDE', 'Data leakage risk - ClinVar clinical outcome annotations'
        
        # EXCLUDE - Only completely useless constant features
        if unique_count <= 1:
            return 'EXCLUDE', 'No variation - constant values'
        
        # CRITICAL - Core computational prediction scores
        critical_scores = [
            'sift_score', 'sift_converted_rankscore', 'sift4g_score',
            'alphamissense', 'am_pathogenicity', 'am_class',
            'cadd_phred', 'cadd_raw', 'cadd_raw_rankscore',
            'revel_score', 'revel_rankscore'
        ]
        
        if any(term in fname for term in critical_scores):
            return 'CRITICAL', f'Core prediction score (missing: {missing_pct:.1f}%)'
        
        # CRITICAL - Conservation scores (including your specific ones)
        conservation_scores = [
            'gerp++_rs', 'gerp++_nr', 'gerp++_rs_rankscore',
            'phylop100way_vertebrate', 'phylop30way_mammalian',
            'phastcons100way_vertebrate', 'phastcons30way_mammalian',
            'phastcons100way'  # Specifically include this one you mentioned
        ]
        
        if any(term in fname for term in conservation_scores):
            return 'CRITICAL', f'Conservation score (missing: {missing_pct:.1f}%)'
        
        # IMPORTANT - Other computational prediction scores
        other_scores = [
            'vest4_score', 'vest4_rankscore',
            'provean_score', 'provean_converted_rankscore', 
            'fathmm_score', 'fathmm_converted_rankscore',
            'metasvm_score', 'metasvm_rankscore',
            'metalr_score', 'metalr_rankscore', 
            'metarnn_score', 'metarnn_rankscore',
            'm-cap_score', 'm-cap_rankscore',
            'primateai_score', 'primateai_rankscore',
            'deogen2_score', 'deogen2_rankscore',
            'bayesdel_addaf_score', 'bayesdel_noaf_score',
            'clinpred_score', 'clinpred_rankscore',
            'list-s2_score', 'list-s2_rankscore',
            'dann_score', 'dann_rankscore',
            'lrt_score', 'lrt_converted_rankscore',
            'mutationassessor_score', 'mutationassessor_rankscore',
            'mutationtaster_score', 'mutationtaster_converted_rankscore',
            'mutpred_score', 'mutpred_rankscore',
            'mvp_score', 'mvp_rankscore',
            'mpc_score', 'mpc_rankscore',
            'eigen-raw_coding', 'eigen-pc-raw_coding',
            'fathmm-mkl_coding_score', 'fathmm-xf_coding_score',
            'genocanyon_score', 'integrated_fitcons_score',
            'linsight', 'linsight_rankscore',  # Include these
            'aloft_pred', 'aloft_confidence'  # Include Aloft features
        ]
        
        if any(term in fname for term in other_scores):
            if missing_pct < 80:
                return 'IMPORTANT', f'Computational score (missing: {missing_pct:.1f}%)'
            else:
                return 'MAYBE', f'Computational score with high missingness (missing: {missing_pct:.1f}%)'
        
        # IMPORTANT - Prediction categories
        prediction_categories = [
            'sift_pred', 'sift4g_pred',
            'polyphen2_hdiv_pred', 'polyphen2_hvar_pred',
            'fathmm_pred', 'provean_pred', 'metasvm_pred', 'metalr_pred',
            'metarnn_pred', 'm-cap_pred', 'primateai_pred', 'deogen2_pred',
            'lrt_pred', 'mutationassessor_pred', 'mutationtaster_pred'
        ]
        
        if any(term in fname for term in prediction_categories):
            return 'IMPORTANT', f'Prediction category (missing: {missing_pct:.1f}%, {unique_count} classes)'
        
        # IMPORTANT - Amino acid and sequence features
        sequence_features = ['aaref', 'aaalt', 'ref', 'alt']
        if fname in sequence_features:
            return 'IMPORTANT', f'Sequence feature (missing: {missing_pct:.1f}%)'
        
        # MAYBE - Codon features (specifically include refcodon you mentioned)
        if fname in ['refcodon', 'codonpos', 'aapos']:
            return 'MAYBE', f'Sequence position feature (missing: {missing_pct:.1f}%, {unique_count} unique) - use with proper encoding/grouping'
        
        # IMPORTANT - Basic functional annotations
        functional_features = [
            'func.refgene', 'exonicfunc.refgene', 'func.ensgene', 'func.knowngene',
            'cytoband', 'cds_strand'
        ]
        
        if any(term in fname for term in functional_features):
            return 'IMPORTANT', f'Functional annotation (missing: {missing_pct:.1f}%, {unique_count} categories)'
        
        # MAYBE - Population frequencies (include all the ones you mentioned, even with 99%+ missing)
        population_freq_terms = [
            'af_', 'gnomad', 'exac_', 'freq', '1000g', 'sites.2015',
            '_af', 'popmax', 'controls_af', 'af', 'af_raw', 'af_male', 'af_female',
            'af_afr', 'af_amr', 'af_eas', 'af_eur', 'af_nfe', 'af_fin', 'af_asj', 'af_oth', 'af_sas',
            'afr.sites.2015', 'sas.sites.2015', 'eas.sites.2015', 'amr.sites.2015', 'eur.sites.2015',
            'all.sites.2015'  # Specifically include this pattern
        ]
        
        if any(term in fname for term in population_freq_terms) and 'bayesdel' not in fname:
            if missing_pct < 99.9:  # Very permissive threshold to include your 99.8% features
                return 'MAYBE', f'Population frequency (missing: {missing_pct:.1f}%) - valuable for rare variant analysis despite high missingness'
            else:
                return 'EXCLUDE', f'Population frequency with extreme missingness ({missing_pct:.1f}%)'
        
        # MAYBE - Splicing scores
        if 'dbscsnv' in fname:
            return 'MAYBE', f'Splicing prediction score (missing: {missing_pct:.1f}%)'
        
        # MAYBE - ESM protein language model features (but exclude the problematic "next" features)
        if 'esm1v' in fname:
            if 'next' not in fname:
                return 'MAYBE', f'Protein language model feature (missing: {missing_pct:.1f}%)'
            else:
                return 'EXCLUDE', f'Protein language model derivative feature with extreme missingness'
        
        # MAYBE - Genomic context (specifically handle Chr)
        if fname == 'chr':
            return 'MAYBE', f'Chromosome context (missing: {missing_pct:.1f}%, {unique_count} categories) - could help with population structure/batch effects'
        
        # MAYBE - Repeat elements and genomic annotations
        if fname in ['simplerepeats', 'rmsk']:
            return 'MAYBE', f'Genomic repeat element (missing: {missing_pct:.1f}%) - could indicate mutagenic regions'
        
        # EXCLUDE - Only true identifiers that are completely non-generalizable
        true_identifiers = [
            'gene.', 'genename', 'gene_name', 'ensembl_geneid', 'ensembl_transcriptid', 
            'ensembl_proteinid', 'uniprot_id', 'transcript_id',
            'aachange.', 'protein_variant', 'hgvs.p', 'hgvs.c', 'genedetail.'
        ]
        
        if any(term in fname for term in true_identifiers):
            return 'EXCLUDE', 'Gene/protein/variant identifier - not generalizable'
        
        # EXCLUDE - Database variant IDs (specifically handle cosmic70)
        if fname in ['avsnp150', 'cosmic70'] or (fname.startswith('rs') and len(fname) > 2):
            return 'EXCLUDE', 'Database variant identifier - not generalizable'
        
        # MAYBE - Rankscore features (include the GERP rankscore you mentioned)
        if 'rankscore' in fname:
            return 'MAYBE', f'Normalized score (missing: {missing_pct:.1f}%) - useful ranking metric'
        
        # MAYBE - Everything else numeric with reasonable coverage
        if is_numeric and missing_pct < 99.95 and coverage_pct > 30:
            return 'MAYBE', f'Numeric feature (missing: {missing_pct:.1f}%) - could provide signal with proper handling'
        
        # MAYBE - Everything else categorical with reasonable properties
        elif not is_numeric and unique_count <= 10000 and missing_pct < 99.5 and coverage_pct > 30:
            if unique_count <= 50:
                return 'MAYBE', f'Categorical feature (missing: {missing_pct:.1f}%, {unique_count} categories) - manageable with standard encoding'
            elif unique_count <= 1000:
                return 'MAYBE', f'High-cardinality categorical (missing: {missing_pct:.1f}%, {unique_count} categories) - needs advanced encoding'
            else:
                return 'MAYBE', f'Very high-cardinality categorical (missing: {missing_pct:.1f}%, {unique_count} categories) - needs special handling'
        
        # Default: EXCLUDE only if truly unusable
        return 'EXCLUDE', f'Unusable feature (missing: {missing_pct:.1f}%, unique: {unique_count}, coverage: {coverage_pct:.0f}%)'
    
    def categorize_all_features(self):
        """Make maximal inclusion decisions"""
        print("\nMaking maximally inclusive categorization decisions...")
        
        for feature_name, stats in self.feature_stats.items():
            decision, reason = self.make_maximal_decision(feature_name, stats)
            
            self.feature_decisions[feature_name] = {
                'decision': decision,
                'reason': reason,
                'missing_pct': stats['avg_missing_pct'],
                'unique_count': stats['max_unique_count'],
                'is_numeric': stats['is_numeric'],
                'sample_values': stats['sample_values'],
                'file_coverage': stats['file_coverage_pct']
            }
    
    def create_comprehensive_report(self):
        """Create comprehensive report with implementation guidance"""
        print("\nCreating comprehensive maximal inclusion report...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f'MAXIMAL_INCLUSIVE_ANALYSIS_{timestamp}.txt'
        report_path = self.output_dir / report_filename
        
        # Separate features by decision
        critical_features = {k: v for k, v in self.feature_decisions.items() if v['decision'] == 'CRITICAL'}
        important_features = {k: v for k, v in self.feature_decisions.items() if v['decision'] == 'IMPORTANT'}
        maybe_features = {k: v for k, v in self.feature_decisions.items() if v['decision'] == 'MAYBE'}
        excluded_features = {k: v for k, v in self.feature_decisions.items() if v['decision'] == 'EXCLUDE'}
        
        # Sort each category by missing percentage
        critical_sorted = sorted(critical_features.items(), key=lambda x: x[1]['missing_pct'])
        important_sorted = sorted(important_features.items(), key=lambda x: x[1]['missing_pct'])
        maybe_sorted = sorted(maybe_features.items(), key=lambda x: x[1]['missing_pct'])
        excluded_sorted = sorted(excluded_features.items(), key=lambda x: x[1]['missing_pct'])
        
        with open(report_path, 'w', encoding='utf-8') as f:
            # Header
            f.write("MAXIMAL INCLUSIVE GENOMICS FEATURE ANALYSIS\n")
            f.write("=" * 90 + "\n")
            f.write("PHILOSOPHY: Include everything possible except obvious data leakage\n")
            f.write("APPROACH: Provides detailed guidance for handling challenging features\n")
            f.write("=" * 90 + "\n")
            f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Files Processed: {self.total_files}\n")
            f.write(f"Total Variants Analyzed: {self.total_variants:,}\n")
            f.write(f"Total Features Found: {len(self.feature_decisions)}\n")
            f.write("=" * 90 + "\n\n")
            
            # Executive Summary
            f.write("MAXIMAL INCLUSION SUMMARY\n")
            f.write("-" * 50 + "\n")
            f.write(f"🔴 CRITICAL features:     {len(critical_features)} (core predictors)\n")
            f.write(f"🟡 IMPORTANT features:    {len(important_features)} (valuable additions)\n")
            f.write(f"🔵 MAYBE features:        {len(maybe_features)} (experimental/special handling)\n")
            f.write(f"⚫ EXCLUDE features:      {len(excluded_features)} (only obvious problems)\n")
            f.write(f"\nTOTAL USABLE FEATURES: {len(critical_features) + len(important_features) + len(maybe_features)}\n")
            f.write(f"EXCLUSION RATE: {(len(excluded_features) / len(self.feature_decisions)) * 100:.1f}% (very permissive)\n\n")
            
            # Usage philosophy
            f.write("MAXIMAL INCLUSION PHILOSOPHY:\n")
            f.write("• Start with CRITICAL features for baseline model\n")
            f.write("• Add IMPORTANT features for enhanced performance\n")
            f.write("• Experiment with MAYBE features based on your specific use case\n")
            f.write("• Only exclude features with obvious data leakage or complete lack of value\n")
            f.write("• Use advanced preprocessing techniques to handle challenging features\n")
            f.write("• Let your ML algorithm and cross-validation guide final feature selection\n\n")
            
            # Write each category
            categories = [
                ("CRITICAL", critical_sorted, "Must-use core predictors"),
                ("IMPORTANT", important_sorted, "Valuable features to include"),
                ("MAYBE", maybe_sorted, "Experimental features - handle with care"),
                ("EXCLUDE", excluded_sorted, "Avoid these features")
            ]
            
            for cat_name, cat_features, cat_desc in categories:
                emoji = {"CRITICAL": "🔴", "IMPORTANT": "🟡", "MAYBE": "🔵", "EXCLUDE": "⚫"}[cat_name]
                
                f.write(f"{emoji} {cat_name} FEATURES\n")
                f.write("=" * 60 + "\n")
                f.write(f"Total: {len(cat_features)} features\n")
                f.write(f"Description: {cat_desc}\n\n")
                
                if cat_name == "MAYBE":
                    f.write("SPECIAL HANDLING NOTES FOR MAYBE FEATURES:\n")
                    f.write("- High-cardinality categoricals: Use target encoding, embeddings, or feature hashing\n")
                    f.write("- High-missingness features: Add missing indicators, use tree-based models\n")
                    f.write("- Population frequencies: Valuable for rare variants despite high missingness\n")
                    f.write("- Sequence features: Consider biochemical property encoding\n")
                    f.write("- Genomic annotations: May provide population structure information\n\n")
                
                f.write("Format: # | Feature Name | Missing% | Type | Unique Values | Notes\n")
                f.write("-" * 90 + "\n")
                
                for i, (feature_name, info) in enumerate(cat_features, 1):
                    data_type = "Numeric" if info['is_numeric'] else "Categorical"
                    unique_display = f"{info['unique_count']:,}" if info['unique_count'] < 1000 else f"{info['unique_count']:,.0e}"
                    f.write(f"{i:3d} | {feature_name:<40} | {info['missing_pct']:6.1f}% | "
                           f"{data_type:<11} | {unique_display:>10} | {info['reason']}\n")
                
                f.write("\n\n")
            
            # Advanced implementation guide
            f.write("ADVANCED IMPLEMENTATION GUIDE\n")
            f.write("=" * 60 + "\n")
            f.write("PHASE 1 - BASELINE MODEL:\n")
            f.write(f"• Use all {len(critical_features)} CRITICAL features\n")
            f.write("• Simple preprocessing: StandardScaler for numeric, LabelEncoder for low-cardinality categorical\n")
            f.write("• Model: Random Forest or XGBoost\n")
            f.write("• Expected performance: Good baseline\n\n")
            
            f.write("PHASE 2 - ENHANCED MODEL:\n")
            f.write(f"• Add {len(important_features)} IMPORTANT features\n")
            f.write("• Advanced preprocessing: Handle missing values with indicators\n")
            f.write("• Model: XGBoost or LightGBM (handles mixed data types well)\n")
            f.write("• Expected performance: Significant improvement\n\n")
            
            f.write("PHASE 3 - EXPERIMENTAL MODEL:\n")
            f.write(f"• Selectively add MAYBE features ({len(maybe_features)} available)\n")
            f.write("• Advanced techniques required:\n")
            f.write("  - Target encoding for high-cardinality categoricals\n")
            f.write("  - Feature embeddings for very high-cardinality features\n")
            f.write("  - Careful missing value handling\n")
            f.write("  - Feature selection based on importance scores\n")
            f.write("• Model: Neural networks or advanced ensemble methods\n")
            f.write("• Expected performance: Potential for best performance with careful tuning\n\n")
            
            f.write("HANDLING SPECIFIC CHALLENGING FEATURES:\n")
            f.write("=" * 50 + "\n")
            f.write("1. HIGH-CARDINALITY CATEGORICALS (>100 categories):\n")
            f.write("   - Target encoding with cross-validation\n")
            f.write("   - Feature hashing\n")
            f.write("   - Embedding layers in neural networks\n")
            f.write("   - Group rare categories into 'Other'\n\n")
            
            f.write("2. HIGH-MISSINGNESS FEATURES (>80% missing):\n")
            f.write("   - Add binary missing indicator\n")
            f.write("   - Use tree-based models (handle missing values naturally)\n")
            f.write("   - Consider that absence of data might be informative\n")
            f.write("   - For population frequencies: missing often means very rare\n\n")
            
            f.write("3. POPULATION FREQUENCY FEATURES:\n")
            f.write("   - Transform with log(AF + epsilon) for better distribution\n")
            f.write("   - Missing values often indicate rare variants (informative)\n")
            f.write("   - Create combined features (e.g., max_pop_AF)\n")
            f.write("   - Use different imputation strategies for different populations\n\n")
            
            f.write("4. SEQUENCE-BASED FEATURES:\n")
            f.write("   - Codon features: Group by biochemical properties\n")
            f.write("   - Amino acid features: Use physicochemical property encoding\n")
            f.write("   - Position features: Consider positional embeddings\n\n")
            
            f.write("FEATURE SELECTION STRATEGIES:\n")
            f.write("=" * 40 + "\n")
            f.write("1. Start with domain knowledge (use CRITICAL features)\n")
            f.write("2. Add features incrementally and validate with cross-validation\n")
            f.write("3. Use feature importance from tree-based models\n")
            f.write("4. Apply statistical tests (mutual information, chi-square)\n")
            f.write("5. Use L1 regularization for automatic feature selection\n")
            f.write("6. Consider feature interactions and polynomial features\n")
            f.write("7. Monitor for overfitting with learning curves\n\n")
            
            f.write("VALIDATION RECOMMENDATIONS:\n")
            f.write("=" * 40 + "\n")
            f.write("• Use stratified cross-validation or gene-based splits\n")
            f.write("• Test generalization to different chromosomes/genes\n")
            f.write("• Validate on different variant types (synonymous vs missense)\n")
            f.write("• Check performance across different allele frequencies\n")
            f.write("• Monitor feature importance rankings for biological plausibility\n\n")
            
            # Statistics
            usable_features = len(critical_features) + len(important_features) + len(maybe_features)
            f.write("FINAL STATISTICS\n")
            f.write("=" * 30 + "\n")
            f.write(f"Total features available for ML: {usable_features}\n")
            f.write(f"Exclusion rate: {(len(excluded_features) / len(self.feature_decisions)) * 100:.1f}%\n")
            f.write(f"Recommended starting features: {len(critical_features) + len(important_features)}\n")
            f.write(f"Advanced experimental features: {len(maybe_features)}\n")
            f.write(f"\nThis maximal inclusion approach gives you the most flexibility\n")
            f.write(f"to experiment and find the optimal feature set for your specific use case.\n")
        
        print(f"Comprehensive maximal inclusion report saved: {report_path}")
        return report_path
    
    def create_maximal_visualization(self):
        """Create visualization for maximal inclusion analysis"""
        print("Creating maximal inclusion visualization...")
        
        # Prepare data
        feature_names = list(self.feature_decisions.keys())
        decisions = [self.feature_decisions[f]['decision'] for f in feature_names]
        missing_pcts = [self.feature_decisions[f]['missing_pct'] for f in feature_names]
        
        # Calculate excluded features count from decisions
        excluded_count = sum(1 for d in decisions if d == 'EXCLUDE')
        
        # Sort by category priority then by missing rate
        category_order = {'CRITICAL': 0, 'IMPORTANT': 1, 'MAYBE': 2, 'EXCLUDE': 3}
        sorted_data = sorted(
            zip(feature_names, decisions, missing_pcts), 
            key=lambda x: (category_order[x[1]], x[2])
        )
        
        sorted_names = [x[0] for x in sorted_data]
        sorted_decisions = [x[1] for x in sorted_data]
        sorted_missing = [x[2] for x in sorted_data]
        
        # Create figure
        n_features = len(feature_names)
        fig_height = max(18, n_features * 0.25)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, fig_height))
        
        # Color mapping - make MAYBE more prominent since it's a large category
        color_map = {
            'CRITICAL': '#DC143C',    # Crimson
            'IMPORTANT': '#FF8C00',   # Dark Orange  
            'MAYBE': '#4169E1',       # Royal Blue
            'EXCLUDE': '#708090'      # Slate Gray
        }
        
        colors = [color_map[d] for d in sorted_decisions]
        y_positions = range(n_features)
        
        # Left plot: All features with categories
        bars = ax1.barh(y_positions, [1] * n_features, color=colors, alpha=0.8, height=0.8)
        
        ax1.set_yticks(y_positions)
        display_names = [name[:55] + "..." if len(name) > 55 else name for name in sorted_names]
        ax1.set_yticklabels(display_names, fontsize=6)
        ax1.set_xlabel('Feature Categories', fontweight='bold', fontsize=12)
        
        usable_count = sum(1 for d in decisions if d in ['CRITICAL', 'IMPORTANT', 'MAYBE'])
        ax1.set_title(f'Maximal Inclusion Analysis: All Features\n'
                     f'({usable_count} Usable, {sum(1 for d in decisions if d == "EXCLUDE")} Excluded)', 
                     fontweight='bold', fontsize=14)
        ax1.set_xlim(0, 1)
        ax1.set_xticks([])
        ax1.invert_yaxis()
        
        # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=color_map['CRITICAL'], label=f'CRITICAL ({sum(1 for d in decisions if d == "CRITICAL")})'),
            Patch(facecolor=color_map['IMPORTANT'], label=f'IMPORTANT ({sum(1 for d in decisions if d == "IMPORTANT")})'),
            Patch(facecolor=color_map['MAYBE'], label=f'MAYBE ({sum(1 for d in decisions if d == "MAYBE")})'),
            Patch(facecolor=color_map['EXCLUDE'], label=f'EXCLUDE ({sum(1 for d in decisions if d == "EXCLUDE")})')
        ]
        ax1.legend(handles=legend_elements, loc='lower right', fontsize=10)
        
        # Right plot: Category distribution with detailed breakdown
        category_counts = Counter(decisions)
        categories = ['CRITICAL', 'IMPORTANT', 'MAYBE', 'EXCLUDE']
        counts = [category_counts[cat] for cat in categories]
        colors_pie = [color_map[cat] for cat in categories]
        
        # Create pie chart
        wedges, texts, autotexts = ax2.pie(
            counts, 
            labels=[f'{cat}\n{count} features\n({count/sum(counts)*100:.1f}%)' for cat, count in zip(categories, counts)],
            colors=colors_pie, 
            autopct='',
            startangle=90,
            textprops={'fontsize': 9, 'fontweight': 'bold'}
        )
        
        ax2.set_title('Maximal Inclusion Distribution\n(Very Low Exclusion Rate)', fontweight='bold', fontsize=14)
        
        # Add text box with key statistics
        usable_pct = (usable_count / len(feature_names)) * 100
        textstr = f'Usable Features: {usable_count} ({usable_pct:.1f}%)\nExcluded: {excluded_count} ({100-usable_pct:.1f}%)\nPhilosophy: Include everything possible'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax2.text(0.02, 0.98, textstr, transform=ax2.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
        
        plt.tight_layout()
        
        # Save diagram
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        diagram_filename = f'maximal_inclusive_analysis_{timestamp}.png'
        diagram_path = self.output_dir / diagram_filename
        plt.savefig(diagram_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"Maximal inclusion diagram saved: {diagram_path}")
        return diagram_path
    
    def run_maximal_analysis(self):
        """Run the maximal inclusion analysis"""
        print("MAXIMAL INCLUSIVE GENOMICS FEATURE ANALYZER")
        print("=" * 70)
        print("Philosophy: Include everything possible except obvious data leakage")
        print("=" * 70)
        
        # Steps 1-4: Same as before
        csv_files = self.find_all_csv_files()
        if not csv_files:
            return False
        
        self.total_files = len(csv_files)
        print(f"\nFound {len(csv_files)} CSV files:")
        for i, file in enumerate(csv_files, 1):
            file_size = file.stat().st_size / (1024*1024)
            print(f"  {i:2d}. {file.name} ({file_size:.1f} MB)")
        
        print(f"\nAnalyzing features...")
        all_file_stats = []
        
        for file_path in csv_files:
            file_stats = self.analyze_single_csv_file(file_path)
            if file_stats:
                all_file_stats.append(file_stats)
        
        if not all_file_stats:
            return False
        
        self.combine_stats_across_files(all_file_stats)
        self.categorize_all_features()
        
        # Summary
        critical_count = sum(1 for v in self.feature_decisions.values() if v['decision'] == 'CRITICAL')
        important_count = sum(1 for v in self.feature_decisions.values() if v['decision'] == 'IMPORTANT')
        maybe_count = sum(1 for v in self.feature_decisions.values() if v['decision'] == 'MAYBE')
        exclude_count = sum(1 for v in self.feature_decisions.values() if v['decision'] == 'EXCLUDE')
        
        usable_count = critical_count + important_count + maybe_count
        total_count = len(self.feature_decisions)
        exclusion_rate = (exclude_count / total_count) * 100
        
        print(f"\nMAXIMAL INCLUSION RESULTS:")
        print(f"  🔴 CRITICAL:  {critical_count:3d} features")
        print(f"  🟡 IMPORTANT: {important_count:3d} features") 
        print(f"  🔵 MAYBE:     {maybe_count:3d} features")
        print(f"  ⚫ EXCLUDE:   {exclude_count:3d} features")
        print(f"\n  📊 USABLE:    {usable_count:3d} features ({100-exclusion_rate:.1f}%)")
        print(f"  📊 EXCLUDED:  {exclude_count:3d} features ({exclusion_rate:.1f}%)")
        
        # Create outputs
        report_path = self.create_comprehensive_report()
        diagram_path = self.create_maximal_visualization()
        
        print(f"\n" + "=" * 70)
        print("MAXIMAL INCLUSION ANALYSIS COMPLETE!")
        print("=" * 70)
        print(f"Files created:")
        print(f"  📄 {report_path.name}")
        print(f"  📊 {diagram_path.name}")
        print(f"\nYOU NOW HAVE ACCESS TO {usable_count} FEATURES!")
        print(f"Start with {critical_count + important_count} core features, experiment with {maybe_count} additional features")
        
        return True


def main():
    """Main function"""
    analyzer = MaximalInclusiveAnalyzer()
    
    try:
        success = analyzer.run_maximal_analysis()
        if success:
            print(f"\n✅ SUCCESS! Maximal inclusion analysis complete.")
        else:
            print(f"\n❌ Analysis failed.")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()