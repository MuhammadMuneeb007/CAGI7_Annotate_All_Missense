#!/usr/bin/env python3

import requests
import pandas as pd
import time
import json
from pathlib import Path
import sys

class MultiDatabasePredictor:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'CAGI7-MultiPredictor/1.0'
        })

    def load_data(self, input_file, limit=200):
        """Load and clean dbNSFP data"""
        print(f"Loading data from {input_file}...")
        
        with open(input_file, 'r') as f:
            header = f.readline().strip().lstrip('#').split('\t')
        
        df = pd.read_csv(input_file, sep='\t', comment='#', names=header, low_memory=False)
        
        if limit:
            df = df.head(limit)
            print(f"Limited to first {len(df)} variants")
        
        # Clean data
        df = df.dropna(subset=['chr', 'pos(1-based)', 'ref', 'alt'])
        df = df[~df['chr'].isin(['.', ''])]
        df = df[~df['ref'].isin(['.', ''])]
        df = df[~df['alt'].isin(['.', ''])]
        
        df['pos'] = pd.to_numeric(df['pos(1-based)'], errors='coerce')
        df = df.dropna(subset=['pos'])
        df['chr'] = df['chr'].astype(str).str.replace('chr', '', regex=False)
        
        print(f"After cleaning: {len(df)} variants")
        return df[['chr', 'pos', 'ref', 'alt']].reset_index(drop=True)

    def save_results(self, results, output_file, method_name):
        """Save results in CAGI format"""
        if not results:
            print(f"No results for {method_name}")
            return None
        
        df = pd.DataFrame(results)
        
        with open(output_file, 'w') as f:
            f.write("#chr\tpos(1-based)\tref\talt\tscore\tsd\tpred\tcomments\n")
            for _, row in df.iterrows():
                f.write(f"{row['chr']}\t{row['pos(1-based)']}\t{row['ref']}\t{row['alt']}\t"
                       f"{row['score']}\t{row['sd']}\t{row['pred']}\t{row['comments']}\n")
        
        print(f"✓ {method_name}: {len(results)} predictions saved to {output_file}")
        return df

    def print_summary(self, df, method_name):
        """Print summary statistics"""
        if df is None or len(df) == 0:
            return
            
        damaging = sum(df['pred'] == 'D')
        tolerated = sum(df['pred'] == 'T')
        unknown = sum(df['pred'] == 'U')
        
        print(f"  {method_name} Summary:")
        print(f"    Damaging (D): {damaging} ({100*damaging/len(df):.1f}%)")
        print(f"    Tolerated (T): {tolerated} ({100*tolerated/len(df):.1f}%)")
        print(f"    Unknown (U): {unknown} ({100*unknown/len(df):.1f}%)")
        print(f"    Average score: {df['score'].mean():.3f}")

    # Method 1: Ensembl VEP (SIFT + PolyPhen)
    def predict_with_vep(self, df):
        """Use Ensembl VEP API"""
        print("\n=== Method 1: Ensembl VEP (SIFT + PolyPhen) ===")
        
        results = []
        batch_size = 25
        
        for i in range(0, len(df), batch_size):
            batch = df.iloc[i:i+batch_size]
            print(f"VEP batch {i//batch_size + 1}/{(len(df)-1)//batch_size + 1}")
            
            # Format variants for VEP
            vep_variants = []
            for _, row in batch.iterrows():
                variant_str = f"{row['chr']} {row['pos']} {row['pos']} {row['ref']}/{row['alt']} +"
                vep_variants.append(variant_str)
            
            # Query VEP
            url = "https://rest.ensembl.org/vep/human/region"
            data = {
                "variants": vep_variants,
                "sift": "b",
                "polyphen": "b",
                "canonical": 1
            }
            
            try:
                response = self.session.post(url, json=data, timeout=30)
                if response.status_code == 200:
                    vep_results = response.json()
                    
                    for j, vep_result in enumerate(vep_results):
                        if j >= len(batch):
                            break
                            
                        original_row = batch.iloc[j]
                        score, sd, method = self._extract_vep_prediction(vep_result)
                        
                        pred = 'D' if score >= 0.5 else ('T' if score <= 0.3 else 'U')
                        
                        results.append({
                            'chr': original_row['chr'],
                            'pos(1-based)': original_row['pos'],
                            'ref': original_row['ref'],
                            'alt': original_row['alt'],
                            'score': round(score, 3),
                            'sd': round(sd, 3),
                            'pred': pred,
                            'comments': f"VEP-{method}:{score:.3f}"
                        })
            except Exception as e:
                print(f"VEP error: {e}")
                break
            
            time.sleep(1.5)
        
        return results

    def _extract_vep_prediction(self, vep_result):
        """Extract prediction from VEP result"""
        if not vep_result or 'transcript_consequences' not in vep_result:
            return 0.5, 0.4, "No_prediction"
        
        for consequence in vep_result['transcript_consequences']:
            if 'polyphen_score' in consequence and consequence['polyphen_score'] is not None:
                try:
                    score = float(consequence['polyphen_score'])
                    pred = consequence.get('polyphen_prediction', '')
                    return score, 0.2, f"PolyPhen({pred})"
                except:
                    continue
            
            if 'sift_score' in consequence and consequence['sift_score'] is not None:
                try:
                    sift_score = float(consequence['sift_score'])
                    sift_pred = consequence.get('sift_prediction', '')
                    score = 1 - sift_score  # Invert SIFT
                    return score, 0.25, f"SIFT({sift_pred})"
                except:
                    continue
        
        return 0.5, 0.4, "No_prediction"

    # Method 2: CADD Scores
    def predict_with_cadd(self, df):
        """Use CADD web service"""
        print("\n=== Method 2: CADD (Combined Annotation Dependent Depletion) ===")
        
        results = []
        batch_size = 10
        
        for i in range(0, min(len(df), 50), batch_size):  # Limit to 50 for CADD testing
            batch = df.iloc[i:i+batch_size]
            print(f"CADD batch {i//batch_size + 1}")
            
            # Format for CADD (chr, pos, ref, alt)
            cadd_input = []
            for _, row in batch.iterrows():
                cadd_input.append(f"{row['chr']}\t{row['pos']}\t{row['ref']}\t{row['alt']}")
            
            # Try CADD API (note: this might be limited or require registration)
            try:
                cadd_url = "https://cadd.gs.washington.edu/api/v1.0/score"
                cadd_data = "\n".join(cadd_input)
                
                response = requests.post(cadd_url, data=cadd_data, timeout=30)
                if response.status_code == 200:
                    cadd_results = response.text.split('\n')
                    
                    for j, result_line in enumerate(cadd_results[1:]):  # Skip header
                        if j >= len(batch) or not result_line.strip():
                            break
                            
                        fields = result_line.split('\t')
                        if len(fields) >= 5:
                            try:
                                cadd_score = float(fields[4])  # CADD Phred score
                                normalized_score = min(cadd_score / 30, 1.0)
                                
                                original_row = batch.iloc[j]
                                pred = 'D' if normalized_score >= 0.5 else ('T' if normalized_score <= 0.3 else 'U')
                                
                                results.append({
                                    'chr': original_row['chr'],
                                    'pos(1-based)': original_row['pos'],
                                    'ref': original_row['ref'],
                                    'alt': original_row['alt'],
                                    'score': round(normalized_score, 3),
                                    'sd': 0.3,
                                    'pred': pred,
                                    'comments': f"CADD:{cadd_score:.1f}"
                                })
                            except:
                                continue
                else:
                    print(f"CADD API returned status {response.status_code}")
                    break
                    
            except Exception as e:
                print(f"CADD error: {e}")
                # Use placeholder predictions
                for _, row in batch.iterrows():
                    results.append({
                        'chr': row['chr'],
                        'pos(1-based)': row['pos'],
                        'ref': row['ref'],
                        'alt': row['alt'],
                        'score': 0.5,
                        'sd': 0.4,
                        'pred': 'U',
                        'comments': "CADD_unavailable"
                    })
                break
            
            time.sleep(2)  # Be nice to CADD servers
        
        # Fill remaining variants with defaults if CADD didn't process all
        while len(results) < len(df):
            idx = len(results)
            row = df.iloc[idx]
            results.append({
                'chr': row['chr'],
                'pos(1-based)': row['pos'],
                'ref': row['ref'],
                'alt': row['alt'],
                'score': 0.5,
                'sd': 0.4,
                'pred': 'U',
                'comments': "CADD_not_queried"
            })
        
        return results

    # Method 3: ClinVar
    def predict_with_clinvar(self, df):
        """Use ClinVar API"""
        print("\n=== Method 3: ClinVar (Clinical Significance) ===")
        
        results = []
        
        for i, (_, row) in enumerate(df.iterrows()):
            if i % 10 == 0:
                print(f"ClinVar progress: {i+1}/{len(df)}")
            
            # Query ClinVar
            try:
                clinvar_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
                params = {
                    'db': 'clinvar',
                    'term': f"{row['chr']}[chr] AND {row['pos']}[chrpos] AND {row['ref']}>{row['alt']}[variant_name]",
                    'retmode': 'json'
                }
                
                response = self.session.get(clinvar_url, params=params, timeout=10)
                
                score = 0.5  # Default
                comment = "ClinVar_not_found"
                
                if response.status_code == 200:
                    result = response.json()
                    if result.get('esearchresult', {}).get('count', '0') != '0':
                        # Found in ClinVar - assume pathogenic for now
                        score = 0.8
                        comment = "ClinVar_found"
                
                pred = 'D' if score >= 0.5 else ('T' if score <= 0.3 else 'U')
                
                results.append({
                    'chr': row['chr'],
                    'pos(1-based)': row['pos'],
                    'ref': row['ref'],
                    'alt': row['alt'],
                    'score': round(score, 3),
                    'sd': 0.3 if score != 0.5 else 0.4,
                    'pred': pred,
                    'comments': comment
                })
                
            except Exception as e:
                # Default prediction
                results.append({
                    'chr': row['chr'],
                    'pos(1-based)': row['pos'],
                    'ref': row['ref'],
                    'alt': row['alt'],
                    'score': 0.5,
                    'sd': 0.4,
                    'pred': 'U',
                    'comments': "ClinVar_error"
                })
            
            time.sleep(0.5)  # Rate limiting
        
        return results

    # Method 4: Amino Acid Properties Based
    def predict_with_aa_properties(self, df):
        """Use amino acid physicochemical properties"""
        print("\n=== Method 4: Amino Acid Properties Based ===")
        
        # Load AA info from original file to get aaref and aaalt
        with open("dbNSFP5.1_nsSNV.chr22", 'r') as f:
            header = f.readline().strip().lstrip('#').split('\t')
        
        full_df = pd.read_csv("dbNSFP5.1_nsSNV.chr22", sep='\t', comment='#', names=header, low_memory=False)
        full_df = full_df.head(200)  # Match our test set
        
        # Amino acid properties
        aa_properties = {
            'A': {'hydrophobic': 1, 'aromatic': 0, 'polar': 0, 'charged': 0, 'size': 1},
            'R': {'hydrophobic': 0, 'aromatic': 0, 'polar': 1, 'charged': 1, 'size': 4},
            'N': {'hydrophobic': 0, 'aromatic': 0, 'polar': 1, 'charged': 0, 'size': 2},
            'D': {'hydrophobic': 0, 'aromatic': 0, 'polar': 1, 'charged': -1, 'size': 2},
            'C': {'hydrophobic': 1, 'aromatic': 0, 'polar': 0, 'charged': 0, 'size': 1},
            'Q': {'hydrophobic': 0, 'aromatic': 0, 'polar': 1, 'charged': 0, 'size': 3},
            'E': {'hydrophobic': 0, 'aromatic': 0, 'polar': 1, 'charged': -1, 'size': 3},
            'G': {'hydrophobic': 1, 'aromatic': 0, 'polar': 0, 'charged': 0, 'size': 0},
            'H': {'hydrophobic': 0, 'aromatic': 1, 'polar': 1, 'charged': 0, 'size': 3},
            'I': {'hydrophobic': 1, 'aromatic': 0, 'polar': 0, 'charged': 0, 'size': 3},
            'L': {'hydrophobic': 1, 'aromatic': 0, 'polar': 0, 'charged': 0, 'size': 3},
            'K': {'hydrophobic': 0, 'aromatic': 0, 'polar': 1, 'charged': 1, 'size': 4},
            'M': {'hydrophobic': 1, 'aromatic': 0, 'polar': 0, 'charged': 0, 'size': 3},
            'F': {'hydrophobic': 1, 'aromatic': 1, 'polar': 0, 'charged': 0, 'size': 4},
            'P': {'hydrophobic': 1, 'aromatic': 0, 'polar': 0, 'charged': 0, 'size': 2},
            'S': {'hydrophobic': 0, 'aromatic': 0, 'polar': 1, 'charged': 0, 'size': 1},
            'T': {'hydrophobic': 0, 'aromatic': 0, 'polar': 1, 'charged': 0, 'size': 2},
            'W': {'hydrophobic': 1, 'aromatic': 1, 'polar': 0, 'charged': 0, 'size': 5},
            'Y': {'hydrophobic': 1, 'aromatic': 1, 'polar': 1, 'charged': 0, 'size': 4},
            'V': {'hydrophobic': 1, 'aromatic': 0, 'polar': 0, 'charged': 0, 'size': 2},
            'X': {'hydrophobic': 0, 'aromatic': 0, 'polar': 0, 'charged': 0, 'size': 0}
        }
        
        results = []
        
        for i, (_, row) in enumerate(df.iterrows()):
            # Get corresponding amino acid change
            full_row = full_df.iloc[i] if i < len(full_df) else None
            
            if full_row is not None:
                aa_ref = full_row.get('aaref', 'X')
                aa_alt = full_row.get('aaalt', 'X')
                
                # Calculate property differences
                if aa_ref in aa_properties and aa_alt in aa_properties:
                    ref_props = aa_properties[aa_ref]
                    alt_props = aa_properties[aa_alt]
                    
                    # Calculate difference score
                    differences = []
                    weights = [0.2, 0.3, 0.2, 0.3, 0.15]
                    
                    for prop, weight in zip(['hydrophobic', 'aromatic', 'polar', 'charged', 'size'], weights):
                        diff = abs(ref_props[prop] - alt_props[prop])
                        differences.append(diff * weight)
                    
                    score = min(sum(differences), 1.0)
                    comment = f"AA_change({aa_ref}->{aa_alt})"
                else:
                    score = 0.5
                    comment = f"Unknown_AA({aa_ref}->{aa_alt})"
            else:
                score = 0.5
                comment = "No_AA_data"
            
            pred = 'D' if score >= 0.5 else ('T' if score <= 0.3 else 'U')
            
            results.append({
                'chr': row['chr'],
                'pos(1-based)': row['pos'],
                'ref': row['ref'],
                'alt': row['alt'],
                'score': round(score, 3),
                'sd': 0.25,
                'pred': pred,
                'comments': comment
            })
        
        return results

def main():
    input_file = "dbNSFP5.1_nsSNV.chr22"
    
    print("=== Multiple Database Variant Predictor Suite ===")
    print("Testing first 200 variants with multiple prediction methods\n")
    
    if not Path(input_file).exists():
        print(f"Error: {input_file} not found!")
        return
    
    predictor = MultiDatabasePredictor()
    
    # Load data once
    df = predictor.load_data(input_file, limit=200)
    
    if len(df) == 0:
        print("No variants to process!")
        return
    
    # Dictionary to store all results for comparison
    all_results = {}
    
    # Method 1: Ensembl VEP
    try:
        vep_results = predictor.predict_with_vep(df)
        vep_df = predictor.save_results(vep_results, "chr22_predictions_VEP.tsv", "VEP")
        all_results['VEP'] = vep_df
    except Exception as e:
        print(f"VEP failed: {e}")
    
    # Method 2: CADD
    try:
        cadd_results = predictor.predict_with_cadd(df)
        cadd_df = predictor.save_results(cadd_results, "chr22_predictions_CADD.tsv", "CADD")
        all_results['CADD'] = cadd_df
    except Exception as e:
        print(f"CADD failed: {e}")
    
    # Method 3: ClinVar
    try:
        clinvar_results = predictor.predict_with_clinvar(df)
        clinvar_df = predictor.save_results(clinvar_results, "chr22_predictions_ClinVar.tsv", "ClinVar")
        all_results['ClinVar'] = clinvar_df
    except Exception as e:
        print(f"ClinVar failed: {e}")
    
    # Method 4: Amino Acid Properties
    try:
        aa_results = predictor.predict_with_aa_properties(df)
        aa_df = predictor.save_results(aa_results, "chr22_predictions_AAProperties.tsv", "AA_Properties")
        all_results['AA_Properties'] = aa_df
    except Exception as e:
        print(f"AA Properties failed: {e}")
    
    # Summary comparison
    print(f"\n=== FINAL SUMMARY COMPARISON ===")
    for method, result_df in all_results.items():
        predictor.print_summary(result_df, method)
        print()

if __name__ == "__main__":
    main()