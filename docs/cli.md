# Command line

The `annotatemissense` command ships with the [server repository](https://github.com/MuhammadMuneeb007/AnnotateMissense-Server). Its core needs only Python ≥ 3.9 and `duckdb`.

```bash
pip install "git+https://github.com/MuhammadMuneeb007/AnnotateMissense-Server"
annotatemissense --version
```

Each command takes the annotation database with `--db variants.duckdb` and/or the score index with `--scores annotatemissense_scores.duckdb`. The environment variables `ANNOTATEMISSENSE_DB` and `ANNOTATEMISSENSE_SCORES` work too.

| Command | What it does |
|---|---|
| `lookup` | look up alleles given as `CHR:POS:REF:ALT` (or `--file`) and print TSV, or JSON with `--json` |
| `annotate-vcf` | annotate a VCF, VCF.gz or variant list → VCF, TSV or JSON ([details](vcf.md)) |
| `schema` | list every column with its type and annotation source |
| `stats` | row, allele and prediction counts per chromosome |
| `audit` | allele multiplicity: rows per allele, alleles with discordant scores |
| `build-score-index` | build the compact score index from `UQ_BioSig_model_Final.tsv.gz` |
| `serve` | run the web server and REST API |

## Examples

```console
$ annotatemissense lookup --db variants.duckdb 22:36305985:G:A chr22-19963748-G-A
query_chr  query_pos  query_ref  query_alt  found  annotatemissense_score  annotatemissense_confidence  annotatemissense_pred  Gene.refGene  ...
chr22      36305985   G          A          True   0.999574                0.999574                     D                      MYH9          ...
chr22      19963748   G          A          True   0.000972                0.999028                     T                      COMT          ...

$ annotatemissense annotate-vcf sample.vcf -o sample.annotatemissense.vcf --db variants.duckdb
AnnotateMissense: 5 records, 6 ALT alleles -> found 4, not_in_release 1, not_snv 1, no_score 0, invalid 0 (D=1, T=3) in 0.07s

$ annotatemissense audit --db variants_dev.duckdb          # chromosome-22 extract
{
  "rows": 1681296,
  "alleles": 1664217,
  "alleles_with_multiple_rows": 16718,
  "max_rows_per_allele": 4,
  "alleles_with_discordant_scores": 0,
  "seconds": 1.27
}

$ annotatemissense schema --db variants.duckdb | head -4
## variants (337 columns)
   1  Chr        VARCHAR    Genomic allele key (ANNOVAR input, GRCh38)
   2  Start      BIGINT     Genomic allele key (ANNOVAR input, GRCh38)
   3  End        VARCHAR    Genomic allele key (ANNOVAR input, GRCh38)
```

Run `annotatemissense audit --db variants.duckdb` on the full file to reproduce the [transcript/allele audit](transcripts.md) genome-wide. It is a single grouped scan; allow several GB of RAM and some minutes.

## Score index from the prediction table

```bash
annotatemissense build-score-index UQ_BioSig_model_Final.tsv.gz -o annotatemissense_scores.duckdb
annotatemissense lookup --scores annotatemissense_scores.duckdb 1:65565:A:T
```

The builder:

- turns `#chr` into `chr1` … `chrX`/`chrY`/`chrM`;
- stores rows with `pred = U` (comment "No prediction available") with a NULL score, because their `score` column holds a placeholder 0.0 that must not be read as benign;
- collapses exact duplicate allele rows. In the first 149,574 rows of the TSV, 3,915 alleles appeared twice, always with identical values;
- renames `sd` to `confidence`, since it equals max(score, 1 − score);
- sorts by chromosome and position so lookups skip irrelevant blocks.
