perl annovar/table_annovar.pl input_variants.avinput annovar/humandb/ -buildver hg38 -outfile alphaMissense_results -remove -protocol AlphaMissense -operation f -nastring . -csvout

perl annovar/table_annovar.pl input_variants.avinput annovar/humandb/ \
-buildver hg38 \
-outfile results_with_constraint \
-remove \
-protocol refGene,gnomad_constraint \
-operation g,f \
-nastring . \
-csvout

perl annovar/table_annovar.pl input_variants.avinput annovar/humandb/ \
-buildver hg38 \
-outfile results_with_go \
-remove \
-protocol refGene,gene_ontology \
-operation g,g \
-nastring . \
-csvout

# Format InterPro domains (assuming it has gene/protein IDs)
cat /data/ascher02/uqmmune1/CAGI7/Data/Annotate\ All\ Loss-of-Function\ Variants/updated_lof_variants/interpro_protein_domains.dat | \
awk -F'\t' 'BEGIN{OFS="\t"} {print $1, $4, $5}' | \
sort -k1,1 > annovar/humandb/hg38_interpro_domains.txt

# Use in ANNOVAR
perl annovar/table_annovar.pl input_variants.avinput annovar/humandb/ \
-buildver hg38 \
-outfile results_with_interpro \
-remove \
-protocol refGene,interpro_domains \
-operation g,g \
-nastring . \
-csvout


# Format Reactome pathways
cat /data/ascher02/uqmmune1/CAGI7/Data/Annotate\ All\ Loss-of-Function\ Variants/updated_lof_variants/reactome_pathways.txt | \
awk -F'\t' 'BEGIN{OFS="\t"} {print $1, $2, $3}' | \
sort -k1,1 > annovar/humandb/hg38_reactome_pathways.txt

perl annovar/table_annovar.pl input_variants.avinput annovar/humandb/ \
-buildver hg38 \
-outfile results_with_reactome \
-remove \
-protocol refGene,reactome_pathways \
-operation g,g \
-nastring . \
-csvout

# Format STRING interactions
zcat /data/ascher02/uqmmune1/CAGI7/Data/Annotate\ All\ Loss-of-Function\ Variants/updated_lof_variants/string_human_interactions.txt.gz | \
awk -F'\t' 'BEGIN{OFS="\t"} NR>1 {print $1, $2, $3}' | \
sort -k1,1 > annovar/humandb/hg38_string_interactions.txt

# Use in ANNOVAR
perl annovar/table_annovar.pl input_variants.avinput annovar/humandb/ \
-buildver hg38 \
-outfile results_with_string \
-remove \
-protocol refGene,string_interactions \
-operation g,g \
-nastring . \
-csvout

# Format Pfam domains
zcat /data/ascher02/uqmmune1/CAGI7/Data/Annotate\ All\ Loss-of-Function\ Variants/updated_lof_variants/pfam_human_domains.tsv.gz | \
awk -F'\t' 'BEGIN{OFS="\t"} NR>1 {print $1, $2, $3}' | \
sort -k1,1 > annovar/humandb/hg38_pfam_domains.txt

# Use in ANNOVAR
perl annovar/table_annovar.pl input_variants.avinput annovar/humandb/ \
-buildver hg38 \
-outfile results_with_pfam \
-remove \
-protocol refGene,pfam_domains \
-operation g,g \
-nastring . \
-csvout


# Format UniProt annotations (you have multiple UniProt files)
# Using the ID mapping file to create gene-to-UniProt mappings
zcat /data/ascher02/uqmmune1/CAGI7/Data/Annotate\ All\ Loss-of-Function\ Variants/updated_lof_variants/uniprot_human_idmapping.dat.gz | \
awk -F'\t' '$2=="Gene_Name" {print $3, $1, "UniProt_ID"}' | \
sort -k1,1 > annovar/humandb/hg38_uniprot_mapping.txt

# Use in ANNOVAR
perl annovar/table_annovar.pl input_variants.avinput annovar/humandb/ \
-buildver hg38 \
-outfile results_with_uniprot \
-remove \
-protocol refGene,uniprot_mapping \
-operation g,g \
-nastring . \
-csvout



# First, create a gene symbol mapping (you'll need this)
# Extract unique gene IDs and get gene symbols
# Create detailed ESM annotation with all scores
cat annovar/humandb/IGVFFI8105TNNO.tsv | \
tail -n +2 | \
awk -F'\t' 'BEGIN{OFS="\t"} {
    gsub(/\.[0-9]*$/, "", $1); 
    gsub(/\.[0-9]*$/, "", $2); 
    gsub(/\.[0-9]*$/, "", $3); 
    avg_esm = ($5 + $6 + $7 + $8 + $9) / 5;
    print $1, $2, $3, $4, avg_esm, $15
}' | \
sort -k1,1 > annovar/humandb/hg38_esm_comprehensive.txt

# Using gene-based ESM scores
perl annovar/table_annovar.pl input_variants.avinput annovar/humandb/ \
-buildver hg38 \
-outfile results_with_esm \
-remove \
-protocol refGene,esm_scores \
-operation g,g \
-nastring . \
-csvout
