#!/usr/bin/env python3

import pandas as pd
import numpy as np
from pathlib import Path

class MissensePredictor:
    def __init__(self):
        # Amino acid properties for prediction
        # Based on physicochemical properties and evolutionary conservation
        self.aa_properties = {
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
            'X': {'hydrophobic': 0, 'aromatic': 0, 'polar': 0, 'charged': 0, 'size': 0}  # Stop codon
        }
        
        # Conservative amino acid substitutions (lower impact)
        self.conservative_subs = {
            ('A', 'V'), ('V', 'A'), ('A', 'I'), ('I', 'A'),
            ('V', 'I'), ('I', 'V'), ('V', 'L'), ('L', 'V'),
            ('I', 'L'), ('L', 'I'), ('F', 'Y'), ('Y', 'F'),
            ('K', 'R'), ('R', 'K'), ('D', 'E'), ('E', 'D'),
            ('Q', 'N'), ('N', 'Q'), ('S', 'T'), ('T', 'S')
        }

    def calculate_aa_difference_score(self, aa_ref, aa_alt):
        """Calculate amino acid substitution impact score"""
        if aa_ref == aa_alt:
            return 0.0  # No change
        
        if aa_ref not in self.aa_properties or aa_alt not in self.aa_properties:
            return 0.8  # Unknown amino acid, assume high impact
        
        # Check if it's a conservative substitution
        if (aa_ref, aa_alt) in self.conservative_subs:
            return 0.2
        
        # Calculate property differences
        ref_props = self.aa_properties[aa_ref]
        alt_props = self.aa_properties[aa_alt]
        
        differences = []
        for prop in ['hydrophobic', 'aromatic', 'polar', 'charged', 'size']:
            diff = abs(ref_props[prop] - alt_props[prop])
            differences.append(diff)
        
        # Weighted scoring
        weights = [0.2, 0.3, 0.2, 0.3, 0.15]  # Charge and aromatic changes are more impactful
        score = sum(d * w for d, w in zip(differences, weights))
        
        return min(score, 1.0)

    def predict_variant_effect(self, row):
        """Predict the effect of a single variant"""
        aa_ref = row['aaref']
        aa_alt = row['aaalt']
        gene_name = row['genename']
        aa_pos = row.get('aapos', 1)
        codon_pos = row.get('codonpos', 1)
        
        # Base score from amino acid change
        base_score = self.calculate_aa_difference_score(aa_ref, aa_alt)
        
        # Position-based adjustments
        position_factor = 1.0
        
        # Start codon changes are typically more severe
        if aa_pos == 1 and aa_ref == 'M':
            position_factor = 1.3
        
        # Stop gain/loss variants
        if aa_ref == 'X' or aa_alt == 'X':
            base_score = 0.9  # High impact for stop changes
            
        # Gene function based adjustments (simplified)
        gene_factor = 1.0
        if gene_name:
            if 'OR' in gene_name:  # Olfactory receptors - often less critical
                gene_factor = 0.8
            elif any(x in gene_name.upper() for x in ['TP53', 'BRCA', 'MLH1', 'MSH']):
                gene_factor = 1.2  # Cancer-related genes
        
        # Final score calculation
        final_score = base_score * position_factor * gene_factor
        final_score = np.clip(final_score, 0.0, 1.0)
        
        # Standard deviation (confidence) - lower for more confident predictions
        if final_score < 0.2 or final_score > 0.8:
            sd = 0.1  # High confidence
        else:
            sd = 0.3  # Lower confidence for borderline cases
        
        # Categorical prediction
        if final_score >= 0.5:
            pred = 'D'  # Damaging
        elif final_score <= 0.3:
            pred = 'T'  # Tolerated
        else:
            pred = 'U'  # Unknown
        
        # Comment
        comment = f"AA_change_score:{base_score:.2f}"
        
        return final_score, sd, pred, comment

def process_chr22_data(input_file):
    """Process chromosome 22 data and generate predictions"""
    
    predictor = MissensePredictor()
    results = []
    
    print("Reading chromosome 22 data...")
    
    # Read the text file
    with open(input_file, 'r') as f:
        # Read header
        header = f.readline().strip().lstrip('#').split('\t')
        
        # Process each line
        for line_num, line in enumerate(f, 1):
            if line_num % 10000 == 0:
                print(f"Processed {line_num} variants...")
            
            fields = line.strip().split('\t')
            
            # Create row dictionary
            row = dict(zip(header, fields))
            
            # Convert position to int
            try:
                pos = int(row['pos(1-based)'])
            except:
                continue
                
            # Convert aa_pos if available
            try:
                if 'aapos' in row and row['aapos'] != '.':
                    row['aapos'] = int(row['aapos'])
                else:
                    row['aapos'] = 1
            except:
                row['aapos'] = 1
            
            # Convert codon_pos if available
            try:
                if 'codonpos' in row and row['codonpos'] != '.':
                    row['codonpos'] = int(row['codonpos'])
                else:
                    row['codonpos'] = 1
            except:
                row['codonpos'] = 1
            
            # Predict variant effect
            score, sd, pred, comment = predictor.predict_variant_effect(row)
            
            # Append result
            result = {
                'chr': row['chr'],
                'pos(1-based)': pos,
                'ref': row['ref'],
                'alt': row['alt'],
                'score': round(score, 3),
                'sd': round(sd, 3),
                'pred': pred,
                'comments': comment
            }
            results.append(result)
    
    return results

def main():
    # Input file
    input_file = "dbNSFP5.1_nsSNV.chr22"
    output_file = "chr22_predictions.tsv"
    
    if not Path(input_file).exists():
        print(f"Error: {input_file} not found!")
        print("Make sure the file exists in the current directory.")
        return
    
    # Process data
    results = process_chr22_data(input_file)
    
    print(f"\nProcessed {len(results)} variants")
    
    # Convert to DataFrame and save
    df = pd.DataFrame(results)
    
    # Save with proper format
    with open(output_file, 'w') as f:
        # Write header with # prefix as required
        f.write("#chr\tpos(1-based)\tref\talt\tscore\tsd\tpred\tcomments\n")
        
        # Write data
        for _, row in df.iterrows():
            f.write(f"{row['chr']}\t{row['pos(1-based)']}\t{row['ref']}\t{row['alt']}\t"
                   f"{row['score']}\t{row['sd']}\t{row['pred']}\t{row['comments']}\n")
    
    print(f"Results saved to {output_file}")
    
    # Show sample results
    print("\nSample predictions:")
    print(df.head(10).to_string(index=False))
    
    # Summary statistics
    print(f"\nPrediction Summary:")
    print(f"Total variants: {len(df)}")
    print(f"Damaging (D): {sum(df['pred'] == 'D')} ({100*sum(df['pred'] == 'D')/len(df):.1f}%)")
    print(f"Tolerated (T): {sum(df['pred'] == 'T')} ({100*sum(df['pred'] == 'T')/len(df):.1f}%)")
    print(f"Unknown (U): {sum(df['pred'] == 'U')} ({100*sum(df['pred'] == 'U')/len(df):.1f}%)")
    print(f"Average score: {df['score'].mean():.3f}")

if __name__ == "__main__":
    main()