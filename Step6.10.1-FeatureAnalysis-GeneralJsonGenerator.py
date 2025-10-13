#!/usr/bin/env python3
"""
JSON Configuration Generator for Genomics Feature Processing
Reads feature analysis CSV and generates Step6.11-FeatureAnalysis-Processor.json
"""

import pandas as pd
import json
from pathlib import Path
from datetime import datetime

def generate_json_config(analysis_csv_path, output_json_path="Step6.11-FeatureAnalysis-Processor.json"):
    """
    Generate JSON configuration file from feature analysis CSV
    """
    print("JSON CONFIGURATION GENERATOR")
    print("=" * 50)
    
    # Read the feature analysis CSV
    print(f"Reading analysis file: {analysis_csv_path}")
    df = pd.read_csv(analysis_csv_path)
    
    print(f"Found {len(df)} features to process")
    
    # Initialize the JSON structure
    config = {
        "processing_config": {
            "sample_size": 10000,
            "input_file_pattern": "*chr1*.csv",
            "output_file": "processed_chr1_features.csv",
            "remove_constant_columns": True,
            "missing_value_threshold": 95,
            "max_categories_for_encoding": 50
        },
        "column_settings": {},
        "default_settings": {
            "include": False,
            "cleaning_required": True,
            "encoding_required": False,
            "reason": "Default - needs review"
        }
    }
    
    # Process each feature
    print("\nProcessing features...")
    
    for index, row in df.iterrows():
        feature_name = row['Feature_Name']
        ml_suitable = row['ML_Suitable'] == 'Yes'
        ml_reason = row['ML_Reason']
        requires_encoding = row['Requires_Encoding'] == 'Yes'
        data_type = row['Data_Type']
        missingness = row['Missingness_Percent']
        unique_count = row['Unique_Values_Count']
        category = row['Feature_Category']
        
        # Determine include/exclude based on ML suitability and other factors
        include = ml_suitable
        
        # Override include decision for specific cases
        if missingness > 95:
            include = False
            reason = f"Too much missing data ({missingness:.1f}%)"
        elif 'constant' in ml_reason.lower() or unique_count <= 1:
            include = False
            reason = "Constant feature - no variation"
        elif 'coordinate' in ml_reason.lower() or 'position' in ml_reason.lower():
            include = False
            reason = "Genomic coordinates - not predictive"
        elif 'leakage' in ml_reason.lower():
            include = False
            reason = "Data leakage risk"
        elif 'identifier' in ml_reason.lower() and 'not generalizable' in ml_reason.lower():
            include = False
            reason = "Identifier - not generalizable"
        elif 'too many categories' in ml_reason.lower():
            include = False
            reason = f"Too many categories ({unique_count}) for encoding"
        else:
            reason = ml_reason
        
        # Determine cleaning requirement
        cleaning_required = True  # Generally true for genomics data
        
        # Determine encoding requirement
        encoding_required = requires_encoding and include and data_type == 'string'
        
        # Special handling for some feature types
        if category == "Clinical Database":
            include = False
            reason = "Data leakage risk - clinical annotation"
            encoding_required = False
        elif category == "Variant Identifier" and any(term in feature_name.lower() 
                                                      for term in ['chr', 'start', 'end', 'pos']):
            include = False
            reason = "Genomic coordinates - not predictive"
            encoding_required = False
        
        # Create column configuration with ALL CSV fields
        config["column_settings"][feature_name] = {
            "include": include,
            "cleaning_required": cleaning_required,
            "encoding_required": encoding_required,
            "reason": reason,
            # Include all original CSV fields
            "missingness_percent": float(missingness),
            "unique_values_count": int(unique_count),
            "first_10_unique_values": eval(row['First_10_Unique_Values']) if pd.notna(row['First_10_Unique_Values']) else [],
            "ml_suitable": ml_suitable,
            "ml_reason": str(ml_reason),
            "feature_category": str(category),
            "requires_encoding": requires_encoding,
            "encoding_type": str(row['Encoding_Type']) if pd.notna(row['Encoding_Type']) else "No encoding needed",
            "new_columns_if_encoded": int(row['New_Columns_If_Encoded']) if pd.notna(row['New_Columns_If_Encoded']) else 0,
            "data_type": str(data_type),
            "original_type": str(row['Original_Type']) if pd.notna(row['Original_Type']) else "unknown"
        }
        
        # Print progress with more details
        status = "INCLUDE" if include else "EXCLUDE"
        encoding_info = f" + ENCODE({row['New_Columns_If_Encoded']})" if encoding_required else ""
        missing_info = f" [{missingness:.1f}% missing]"
        print(f"  {feature_name:<35} | {status:<7}{encoding_info:<15}{missing_info:<15} | {reason}")
    
    # Generate summary statistics
    total_features = len(config["column_settings"])
    included_features = sum(1 for settings in config["column_settings"].values() 
                           if settings["include"])
    excluded_features = total_features - included_features
    encoding_features = sum(1 for settings in config["column_settings"].values() 
                           if settings["encoding_required"])
    
    print(f"\nSUMMARY:")
    print(f"Total features: {total_features}")
    print(f"To INCLUDE: {included_features}")
    print(f"To EXCLUDE: {excluded_features}")
    print(f"Need encoding: {encoding_features}")
    
    # Save JSON configuration
    output_path = Path(output_json_path)
    
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=4)
    
    print(f"\nJSON configuration saved: {output_path}")
    print(f"File size: {output_path.stat().st_size / 1024:.1f} KB")
    
    # Create a summary report
    report_path = output_path.parent / f"config_generation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    
    with open(report_path, 'w') as f:
        f.write("JSON CONFIGURATION GENERATION REPORT\n")
        f.write("=" * 50 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Source CSV: {analysis_csv_path}\n")
        f.write(f"Output JSON: {output_path}\n\n")
        
        f.write("SUMMARY STATISTICS:\n")
        f.write(f"Total features analyzed: {total_features}\n")
        f.write(f"Features to INCLUDE: {included_features}\n")
        f.write(f"Features to EXCLUDE: {excluded_features}\n")
        f.write(f"Features requiring encoding: {encoding_features}\n\n")
        
        # Breakdown by category
        f.write("BREAKDOWN BY ACTION:\n")
        
        include_reasons = {}
        exclude_reasons = {}
        
        for feature, settings in config["column_settings"].items():
            reason = settings["reason"]
            if settings["include"]:
                include_reasons[reason] = include_reasons.get(reason, 0) + 1
            else:
                exclude_reasons[reason] = exclude_reasons.get(reason, 0) + 1
        
        f.write("\nINCLUDE REASONS:\n")
        for reason, count in sorted(include_reasons.items()):
            f.write(f"  {reason}: {count}\n")
        
        f.write("\nEXCLUDE REASONS:\n")
        for reason, count in sorted(exclude_reasons.items()):
            f.write(f"  {reason}: {count}\n")
        
        f.write(f"\nRECOMMENDATION:\n")
        f.write(f"Review the generated JSON file and adjust settings as needed.\n")
        f.write(f"Pay special attention to features with high unique counts or unusual patterns.\n")
    
    print(f"Generation report saved: {report_path}")
    
    return output_path, included_features, excluded_features, encoding_features

def main():
    """
    Main function - modify the CSV path as needed
    """
    # Path to your feature analysis CSV file
    # UPDATE THIS PATH to match your actual file
    analysis_csv = "Final_Results/Allfiles/feature_analysis_20250926_212555.csv"  # Update with your actual file path
    
    # Check if file exists
    if not Path(analysis_csv).exists():
        print("ERROR: Analysis CSV file not found!")
        print(f"Please update the 'analysis_csv' path in the script.")
        print(f"Looking for: {analysis_csv}")
        
        # Try to find CSV files in common locations
        possible_paths = [
            "Final_Results/Allfiles/",
            "Final_Results/",
            "./"
        ]
        
        print("\nSearching for CSV files...")
        found_csvs = []
        
        for path in possible_paths:
            path_obj = Path(path)
            if path_obj.exists():
                csvs = list(path_obj.glob("feature_analysis_*.csv"))
                found_csvs.extend(csvs)
        
        if found_csvs:
            print("Found these analysis CSV files:")
            for i, csv_file in enumerate(found_csvs, 1):
                print(f"  {i}. {csv_file}")
            print(f"\nUpdate the 'analysis_csv' variable with one of these paths.")
        else:
            print("No feature analysis CSV files found.")
        
        return False
    
    try:
        # Generate the JSON configuration
        json_path, included, excluded, encoded = generate_json_config(analysis_csv)
        
        print(f"\n✅ SUCCESS!")
        print(f"JSON configuration generated successfully.")
        print(f"You can now review and edit: {json_path}")
        print(f"Then run your feature processor with this configuration.")
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    main()