import glob
import gzip
import os

# Create AnnovarInputFiles directory if it doesn't exist
os.makedirs("AnnovarInputFiles", exist_ok=True)

files = sorted(glob.glob("dbNSFP5.1_nsSNV.chr*.gz"))
for file in files:
    # Extract chromosome number from filename
    chr_num = file.split("chr")[1].split(".")[0]
    annovar_output = f"AnnovarInputFiles/chr{chr_num}.annovar_input"
    
    if os.path.exists(annovar_output):
        print(f"Skipping {file}, {annovar_output} already exists")
        continue
    
    print(f"Converting {file} to ANNOVAR format...")
    
    with gzip.open(file, "rt") as f_in, open(annovar_output, "w") as f_out:
        # Write ANNOVAR header
        f_out.write("#Chr\tStart\tEnd\tRef\tAlt\n")
        
        for line_num, line in enumerate(f_in):
            line = line.strip()
            
            # Skip dbNSFP header lines (start with #)
            if line.startswith("#"):
                continue
            
            # Skip empty lines
            if not line:
                continue
            
            # Split the line into columns
            columns = line.split("\t")
            
            # Extract required fields for ANNOVAR
            if len(columns) >= 4:
                chr_col = columns[0]
                pos = columns[1]
                ref = columns[2]
                alt = columns[3]
                
                # ANNOVAR format: chr start_pos end_pos ref alt
                annovar_line = f"{chr_col}\t{pos}\t{pos}\t{ref}\t{alt}\n"
                f_out.write(annovar_line)
    
    print(f"Converted {file} -> {annovar_output}")

print("All files processed successfully!")
