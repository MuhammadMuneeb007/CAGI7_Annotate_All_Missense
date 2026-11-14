# Database and schema

## Released files

Zenodo record [19981867](https://doi.org/10.5281/zenodo.19981867), version 1.0.0, published 2026-05-02:

| File | Bytes (compressed) | MD5 | Contents |
|---|---:|---|---|
| `variants.duckdb.gz` | 20,819,921,303 (20.8 GB) | `578ab4e9c988c2cf78d8aff9f7ceb63b` | DuckDB database, table `variants`, 337 columns: annotations and the AnnotateMissense prediction |
| `UQ_BioSig_model_Final.tsv.gz` | 1,351,827,163 (1.4 GB) | `417f2649058118bb3b576e693ba75344` | genome-wide prediction table: `#chr, pos(1-based), ref, alt, score, sd, pred, comments` (90,643,830 rows) |

```bash
md5sum variants.duckdb.gz          # 578ab4e9c988c2cf78d8aff9f7ceb63b
gzip -d variants.duckdb.gz         # or: gzip -dk to keep the archive
```

- **DuckDB version:** the file header reports storage written by **DuckDB v1.4.3**. Use DuckDB ≥ 1.4.3; older releases such as 0.10 cannot open it.
- **Extracted size:** not yet measured on the full file. The chromosome-22 extract (`variants_dev.duckdb`, identical layout) is 1,090,269,184 bytes for 1,681,296 rows (≈648 bytes per row). Extrapolated to the ≈82.2 million annotation records audited in the manuscript, that gives **≈50–55 GB**. Plan for ~80 GB free while extracting (archive plus database), or ~55 GB if you stream (`gzip -dc variants.duckdb.gz > variants.duckdb && rm variants.duckdb.gz`).
- **Row counts:** the manuscript's audit of the 25 merged chromosome annotation files found 82,198,516 records. The prediction table has 90,643,830 rows. Confirm the counts of your copy with `annotatemissense stats --db variants.duckdb`. Alleles with a prediction but no annotation row can be served from the [score index](cli.md#score-index-from-the-prediction-table).

## Table `variants`

The layout below was read directly from the database with `annotatemissense schema` (on the chromosome-22 extract, which shares the release's table definition):

- **Key:** `Chr` (`'chr1'` … `'chr22'`, `'chrX'`, `'chrY'`, `'chrM'`), `Start` (BIGINT, 1-based GRCh38), `Ref`, `Alt`. These are the ANNOVAR input columns; `End` = `Start` for SNVs.
- **Prediction:** `UQ_score` (DOUBLE, 0–1), `UQ_pred` (`D` if score ≥ 0.5, else `T`; NULL when not scored), `UQ_sd` (DOUBLE; the probability of the predicted class, max(score, 1 − score), not a standard deviation).
- **Types:** 287 VARCHAR, 49 DOUBLE, 1 BIGINT. Most predictor and frequency columns are stored as **text**, exactly as ANNOVAR wrote them, so use `TRY_CAST(col AS DOUBLE)` for numeric filters.
- **Column naming:** when two ANNOVAR databases produce the same column name, the pipeline (`Step4-MergeFiles.py`) prefixes the later one with its database name. Hence `dbnsfp42c_*` (dbNSFP v4.2c), `gnomad211_*` (gnomAD v2.1.1 *exome*) and `gnomad30_*` (gnomAD v3.0), while the unprefixed `AF*` columns are gnomAD v2.1.1 *genome*. The data confirm this: unprefixed `AF_sas` is empty, as v2.1.1 genomes contain no South Asian samples.
- **Rows per allele:** usually one. 1.0% of alleles have 2–4 rows, from protein-level joins; the AnnotateMissense score is identical across them. See [Transcripts](transcripts.md).

### Columns by source

| Source | Columns | Examples |
|---|---:|---|
| Genomic allele key (ANNOVAR input, GRCh38) | 5 | `Chr`, `Start`, `End`, `Ref`, `Alt` |
| 1000 Genomes (2015-08) | 6 | `ALL.sites.2015_08`, `AFR.sites.2015_08`, … |
| dbSNP 150 | 1 | `avsnp150` |
| ANNOVAR gene-based: refGene, ensGene, knownGene, ccdsGene | 4 × 5 | `Func.refGene`, `Gene.refGene`, `ExonicFunc.refGene`, `AAChange.refGene`, … |
| ANNOVAR region-based | 5 | `cytoBand`, `genomicSuperDups`, `rmsk`, `simpleRepeat`, `phastCons100way` |
| ClinVar (ANNOVAR filter-based) | 5 | `CLNSIG`, `CLNDN`, `CLNREVSTAT`, `CLNALLELEID`, `CLNDISDB` |
| COSMIC v70 | 1 | `cosmic70` |
| dbNSFP v4.2a (academic) | 107 | `SIFT_score`, `Polyphen2_HDIV_score`, `CADD_phred`, `REVEL_score`, `MetaRNN_score`, `GERP++_RS`, `Interpro_domain`, `GTEx_V8_gene` … |
| dbNSFP v4.2c (commercial-use subset) | 90 | `dbnsfp42c_SIFT_score`, `dbnsfp42c_MetaRNN_score`, … |
| dbscSNV splicing | 2 | `dbscSNV_ADA_SCORE`, `dbscSNV_RF_SCORE` |
| ExAC | 8 | `ExAC_ALL`, `ExAC_AFR`, … |
| gnomAD v2.1.1 genome (ANNOVAR `gnomad211_genome`; unprefixed) | 17 | `AF`, `AF_popmax`, `AF_nfe`, … |
| gnomAD v2.1.1 exome (ANNOVAR `gnomad211_exome`) | 17 | `gnomad211_AF`, `gnomad211_AF_popmax`, … |
| gnomAD v3.0 genome (ANNOVAR `gnomad30_genome`) | 13 | `gnomad30_AF`, `gnomad30_AF_nfe`, `AF_ami`, … |
| gnomAD gene constraint | 1 | `gnomad_constraint` |
| dbNSFP v5.1 transcript/protein context (challenge input) | 15 | `genename`, `aaref`, `aaalt`, `aapos`, `Ensembl_transcriptid`, `Ensembl_proteinid`, `hg19_pos`, … |
| AlphaMissense | 6 | `am_pathogenicity`, `am_class`, `transcript_id`, `uniprot_id`, `protein_variant`, `AlphaMissense` |
| ESM1v | 15 | `esm1v_t33_650M_UR90S_1…5`, `…_next`, `combined_score`, `HGVS.p`, `GENCODE.v43.ENST/ENSP/ENSG` |
| AnnotateMissense prediction | 3 | `UQ_score`, `UQ_sd`, `UQ_pred` |

??? abstract "All 337 columns (position, name, DuckDB type, source)"

    --8<-- "schema_columns.md"

The same list is served live by `GET /api/v1/schema` and printed by `annotatemissense schema`.

## Example queries

All queries below were run against the chromosome-22 extract; the output shown is real. With the full file they work unchanged.

```bash
duckdb -readonly variants.duckdb          # DuckDB CLI, or use Python: duckdb.connect("variants.duckdb", read_only=True)
```

**One allele**

```sql
SELECT Chr, Start, Ref, Alt, "Gene.refGene" AS gene,
       UQ_score AS annotatemissense_score, UQ_pred AS annotatemissense_pred, CLNSIG
FROM variants
WHERE Chr = 'chr22' AND Start = 36305985 AND Ref = 'G' AND Alt = 'A';
```
```text
  Chr    Start Ref Alt gene  annotatemissense_score annotatemissense_pred     CLNSIG
chr22 36305985   G   A MYH9                0.999574                     D Pathogenic
```

**One record per allele** (rows can repeat for protein-level joins; the score cannot differ)

```sql
SELECT Chr, Start, Ref, Alt, COUNT(*) AS n_rows, COUNT(DISTINCT UQ_score) AS n_scores,
       any_value(UQ_score) AS annotatemissense_score
FROM variants
WHERE Chr = 'chr22' AND Start = 21549460
GROUP BY ALL ORDER BY Alt;
```
```text
  Chr    Start Ref Alt  n_rows  n_scores  annotatemissense_score
chr22 21549460   C   A       2         1                0.049170
chr22 21549460   C   G       2         1                0.018423
chr22 21549460   C   T       2         1                0.023328
```

**High-scoring ClinVar VUS in a gene** (note `TRY_CAST` for text-typed predictor columns)

```sql
SELECT Start, Ref, Alt, split_part("AAChange.refGene", ',', 1) AS aa_change,
       UQ_score AS annotatemissense_score,
       TRY_CAST(am_pathogenicity AS DOUBLE) AS alphamissense,
       TRY_CAST(REVEL_score AS DOUBLE) AS revel
FROM variants
WHERE "Gene.refGene" = 'MYH9' AND CLNSIG LIKE 'Uncertain%' AND UQ_score >= 0.9
QUALIFY row_number() OVER (PARTITION BY Chr, Start, Ref, Alt) = 1
ORDER BY annotatemissense_score DESC LIMIT 5;
```
```text
   Start Ref Alt                               aa_change  annotatemissense_score  alphamissense  revel
36322475   G   A    MYH9:NM_002473:exon6:c.C659T:p.A220V                0.998705         0.9807  0.860
36319600   C   T  MYH9:NM_002473:exon10:c.G1048A:p.G350S                0.998321         0.9456  0.970
36349115   A   G     MYH9:NM_002473:exon2:c.T122C:p.F41S                0.997783         0.9976  0.963
36320833   T   C    MYH9:NM_002473:exon8:c.A833G:p.Y278C                0.997662         0.9245  0.967
36286729   G   A MYH9:NM_002473:exon35:c.C5050T:p.Q1684X                0.997368            NaN    NaN
```

The last row is a stop-gain. The release is dbNSFP's *nonsynonymous* SNV set, and on chromosome 22, 91.0% of rows are `nonsynonymous SNV` in refGene, 4.7% `stopgain`, 0.3% `startloss`, 0.2% `stoploss`, 0.2% `synonymous SNV` (missense in another transcript), and 3.5% unannotated or unknown. Filter on `"ExonicFunc.refGene" = 'nonsynonymous SNV'` if you need strict missense.

**Join your own variant table**

```sql
SELECT m.chrom, m.pos, m.ref, m.alt, v.gene, v.annotatemissense_score, v.annotatemissense_pred
FROM read_csv('my_variants.tsv', delim = '\t', header = true) AS m
LEFT JOIN (
    SELECT DISTINCT Chr, Start, Ref, Alt, "Gene.refGene" AS gene,
           UQ_score AS annotatemissense_score, UQ_pred AS annotatemissense_pred
    FROM variants
) AS v ON v.Chr = 'chr' || m.chrom AND v.Start = m.pos AND v.Ref = m.ref AND v.Alt = m.alt
ORDER BY m.pos;
```
```text
 chrom      pos ref alt                   gene  annotatemissense_score annotatemissense_pred
    22 10000000   A   G                   None                     NaN                  None
    22 19963748   G   A                   COMT                0.000972                     T
    22 21549460   C   A RIMBP3B,RIMBP3C,UBE2L3                0.049170                     T
    22 36305985   G   A                   MYH9                0.999574                     D
```

For VCF input, `annotatemissense annotate-vcf` does this join for you and handles multi-allelic and padded records ([VCF guide](vcf.md)).

**Prediction summary**

```sql
SELECT UQ_pred AS prediction, COUNT(*) AS rows, ROUND(AVG(UQ_score), 3) AS mean_score
FROM variants GROUP BY 1 ORDER BY 1;
```
```text
prediction   rows  mean_score            (chromosome 22)
         D 990435       0.869
         T 669316       0.178
      None  21545         NaN
```

## Score index schema

`annotatemissense build-score-index` creates `annotatemissense_scores.duckdb`:

| Table | Column | Type | Notes |
|---|---|---|---|
| `scores` | `chrom` | VARCHAR | `chr1` … `chrM` |
| | `pos` | INTEGER | 1-based GRCh38 |
| | `ref`, `alt` | VARCHAR | |
| | `score` | FLOAT | NULL where the TSV has `pred = U` |
| | `confidence` | FLOAT | TSV `sd` = max(score, 1 − score) |
| | `pred` | VARCHAR | `D` / `T` / NULL |
| | `source_rows` | USMALLINT | number of identical TSV rows collapsed |
| `meta` | `key`, `value` | VARCHAR | source file, row and allele counts, build |
