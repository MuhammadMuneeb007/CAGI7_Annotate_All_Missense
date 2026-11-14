# Transcript handling and genomic-allele representation

## What the code establishes

- The research resource is based on hg38 missense **genomic single-nucleotide alleles** (chromosome, one-based position, reference and alternate allele).
- `Step6.21-MergePredictions.py` takes `#chr`, `pos(1-based)`, `ref`, `alt` from a dbNSFP chromosome file and attaches predictions in the corresponding row order. It writes a variant-level intermediate output, not one predicted value explicitly calculated independently for every transcript.
- Different annotation providers (ANNOVAR refGene/knownGene/ensGene, dbNSFP, AlphaMissense, ESM) can attach transcript- or protein-dependent annotations. The training code also uses merged amino-acid/codon-context fields (`aaref_merged`, `aaalt_merged`, `refcodon_merged`).

## What is not yet established

**The public scripts and deposit metadata do not conclusively identify one canonical transcript selection rule (such as MANE Select or longest transcript) used for every released allele, or demonstrate that every genomic allele occurs exactly once in the DuckDB tables.**

Therefore the documentation **does not claim**:

- that the data are MANE Select-only;
- that all transcript annotations were averaged, maximised or taken from the first transcript;
- that the published DuckDB table has exactly one row per allele; or
- that a given model score can automatically be interpreted for every protein isoform.

These assertions require an explicit lineage audit against the actual final database and final model feature-processing scripts.

## Validate allele multiplicity

After identifying the real table and coordinate column names using [DATABASE.md](DATABASE.md), substitute them into:

```sql
SELECT
    "YOUR_CHR_COL" AS chr,
    "YOUR_POS_COL" AS pos,
    "YOUR_REF_COL" AS ref,
    "YOUR_ALT_COL" AS alt,
    COUNT(*) AS n_rows
FROM "YOUR_TABLE_NAME"
GROUP BY 1, 2, 3, 4
HAVING COUNT(*) > 1
LIMIT 20;
```

To validate an allele-level score, also check whether duplicated keys have identical scores and whether transcript identifiers are present; do **not** drop duplicates silently. Save the record counts and selection rule after verification. In the meantime, describe the outputs as **genomic-allele-keyed predictions with potentially transcript-derived inputs**, not transcript-specific clinical predictions.

All biological interpretations are for **research prioritisation**, not independent clinical classification.
