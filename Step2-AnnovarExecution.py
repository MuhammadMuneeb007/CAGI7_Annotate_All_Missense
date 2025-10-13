import os
import glob

# Create output directory
os.makedirs("Annovar_Output_Files", exist_ok=True)

# Define all ANNOVAR protocols and their operations
annovar_protocols = [
    # Gene-based databases (operation = g)
    {"protocol": "refGene", "operation": "g"},
    {"protocol": "ensGene", "operation": "g"},
    {"protocol": "knownGene", "operation": "g"},
    {"protocol": "ccdsGene", "operation": "g"},
    {"protocol": "wgEncodeGencodeBasicV44", "operation": "g"},
    
    # Region-based databases (operation = r)
    {"protocol": "cytoBand", "operation": "r"},
    {"protocol": "genomicSuperDups", "operation": "r"},
    {"protocol": "rmsk", "operation": "r"},
    {"protocol": "simpleRepeat", "operation": "r"},
    {"protocol": "phastCons100way", "operation": "r"},
    
    # Filter-based databases (operation = f)
    {"protocol": "clinvar_20220320", "operation": "f"},
    {"protocol": "cosmic70", "operation": "f"},
    {"protocol": "dbnsfp42a", "operation": "f"},
    {"protocol": "dbnsfp42c", "operation": "f"},
    {"protocol": "dbscsnv11", "operation": "f"},
    {"protocol": "avsnp150", "operation": "f"},
    {"protocol": "gnomad211_exome", "operation": "f"},
    {"protocol": "gnomad211_genome", "operation": "f"},
    {"protocol": "gnomad30_genome", "operation": "f"},
    {"protocol": "exac03", "operation": "f"},
    {"protocol": "AFR.sites.2015_08", "operation": "f"},
    {"protocol": "ALL.sites.2015_08", "operation": "f"},
    {"protocol": "AMR.sites.2015_08", "operation": "f"},
    {"protocol": "EAS.sites.2015_08", "operation": "f"},
    {"protocol": "EUR.sites.2015_08", "operation": "f"},
    {"protocol": "SAS.sites.2015_08", "operation": "f"},
    
    # Additional custom databases from document
    {"protocol": "AlphaMissense", "operation": "f"},
    {"protocol": "gnomad_constraint", "operation": "f"},
    {"protocol": "gene_ontology", "operation": "g"},
    {"protocol": "interpro_domains", "operation": "g"},
    {"protocol": "reactome_pathways", "operation": "g"},
    {"protocol": "string_interactions", "operation": "g"},
    {"protocol": "pfam_domains", "operation": "g"},
    {"protocol": "uniprot_mapping", "operation": "g"},
    {"protocol": "esm_scores", "operation": "g"}
]

# Define multi-protocol combinations from the document
multi_protocol_commands = [
    {"protocols": "refGene,gnomad_constraint", "operations": "g,f", "suffix": "constraint"},
    {"protocols": "refGene,gene_ontology", "operations": "g,g", "suffix": "go"},
    {"protocols": "refGene,interpro_domains", "operations": "g,g", "suffix": "interpro"},
    {"protocols": "refGene,reactome_pathways", "operations": "g,g", "suffix": "reactome"},
    {"protocols": "refGene,string_interactions", "operations": "g,g", "suffix": "string"},
    {"protocols": "refGene,pfam_domains", "operations": "g,g", "suffix": "pfam"},
    {"protocols": "refGene,uniprot_mapping", "operations": "g,g", "suffix": "uniprot"},
    {"protocols": "refGene,esm_scores", "operations": "g,g", "suffix": "esm"}
]

# Get input files
input_files = sorted(glob.glob("AnnovarInputFiles/chr*.annovar_input"))

if not input_files:
    print("No ANNOVAR input files found in AnnovarInputFiles/")
    exit(1)

print(f"Found {len(input_files)} input files")
print("Generating commands.txt...")

# Generate commands
with open("commands.txt", "w") as cmd_file:
    for input_file in input_files:
        file_id = os.path.basename(input_file).split('.')[0]
        
        # Single protocol commands
        for protocol_info in annovar_protocols:
            protocol = protocol_info["protocol"]
            operation = protocol_info["operation"]
            output_prefix = f"Annovar_Output_Files/{file_id}_{protocol}_results"
            
            command = f"perl annovar/table_annovar.pl {input_file} annovar/humandb/ -buildver hg38 -outfile {output_prefix} -remove -protocol {protocol} -operation {operation} -nastring . -csvout"
            cmd_file.write(f"{command}\n")
        
        # Multi-protocol combination commands
        for multi_cmd in multi_protocol_commands:
            protocols = multi_cmd["protocols"]
            operations = multi_cmd["operations"]
            suffix = multi_cmd["suffix"]
            output_prefix = f"Annovar_Output_Files/{file_id}_results_with_{suffix}"
            
            command = f"perl annovar/table_annovar.pl {input_file} annovar/humandb/ -buildver hg38 -outfile {output_prefix} -remove -protocol {protocols} -operation {operations} -nastring . -csvout"
            cmd_file.write(f"{command}\n")

print("✓ Commands saved to commands.txt")

# Count total commands
total_single_commands = len(input_files) * len(annovar_protocols)
total_multi_commands = len(input_files) * len(multi_protocol_commands)
total_commands = total_single_commands + total_multi_commands
print(f"Total single protocol commands: {total_single_commands}")
print(f"Total multi-protocol commands: {total_multi_commands}")
print(f"Total commands generated: {total_commands}")