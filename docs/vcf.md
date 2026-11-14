# Annotating a VCF

AnnotateMissense annotates user-supplied VCFs **by lookup**. Each ALT allele is normalised and matched to the precomputed GRCh38 release; no model is re-run. The same code runs in three places:

| Where | Size limit | Data leaves your machine? |
|---|---|---|
| Web page `/annotate` | 10,000 ALT alleles (public instance) | yes, processed in memory and not stored |
| API `POST /api/v1/annotate` | same as the server's `MAX_VCF_ALLELES` | yes, to the server you call |
| CLI `annotatemissense annotate-vcf` | none | **no**, fully offline |

!!! tip "Patient data"
    For identifiable or clinical data, use the offline CLI or a server inside your institution. The CLI makes no network requests.

## Commands

```bash
# offline
annotatemissense annotate-vcf sample.vcf.gz -o sample.annotatemissense.vcf --db variants.duckdb
annotatemissense annotate-vcf sample.vcf.gz -o sample.annotatemissense.tsv --db variants.duckdb   # TSV, one row per ALT
annotatemissense annotate-vcf sample.vcf.gz --format json --db variants.duckdb > sample.json

# via a server
curl -X POST "$SERVER/api/v1/annotate?format=vcf" -F file=@sample.vcf.gz -o sample.annotatemissense.vcf
```

Accepted input: VCF v4.x (plain, gzip or bgzip), or a plain list of `chrom pos ref alt` (separated by tabs, spaces or commas, or written as `22:36305985:G:A`).

## What happens to each record

1. **Split** multi-allelic ALTs; each ALT is handled separately.
2. **Normalise** the chromosome (`22`, `chr22` and `CHR22` all become `chr22`; `MT` becomes `chrM`), upper-case the alleles, and trim shared leading/trailing bases (`CG>CA` at 36305984 becomes `G>A` at 36305985).
3. **Classify:** non-SNVs (indels, MNVs, `*`, symbolic `<DEL>`, breakends) → `not_snv`; unparseable values → `invalid`.
4. **Look up** the remaining SNVs: `found` (score available), `no_score` (allele present but without a prediction), or `not_in_release`.
5. **Write** your original records unchanged, with three INFO fields added.

A header that mentions GRCh37/hg19/b37 produces a warning, because positions would not match.

## Output

```text
##INFO=<ID=AnnotateMissense_score,Number=A,Type=Float,Description="AnnotateMissense pathogenicity score (0-1; >=0.5 predicted damaging). Precomputed GRCh38 lookup; . when unavailable">
##INFO=<ID=AnnotateMissense_pred,Number=A,Type=String,Description="AnnotateMissense prediction: D = damaging (score >= 0.5), T = tolerated">
##INFO=<ID=AnnotateMissense_status,Number=A,Type=String,Description="AnnotateMissense lookup status: found | no_score | not_in_release | not_snv | invalid">
##AnnotateMissense=<Version="1.1.0",Build="GRCh38",Date="2026-11-14T08:50:24Z",Source="https://doi.org/10.5281/zenodo.19981867">
#CHROM  POS       ID      REF  ALT  QUAL  FILTER  INFO
chr22   19963748  rs4680  G    A    .     PASS    DP=31;AnnotateMissense_score=0.000972;AnnotateMissense_pred=T;AnnotateMissense_status=found
chr22   21549460  .       C    A,G  .     PASS    DP=18;AnnotateMissense_score=0.04917,0.018423;AnnotateMissense_pred=T,T;AnnotateMissense_status=found,found
chr22   36305984  .       CG   CA   .     PASS    DP=22;AnnotateMissense_score=0.999574;AnnotateMissense_pred=D;AnnotateMissense_status=found
chr22   36305985  .       GG   G    .     PASS    DP=40;AnnotateMissense_score=.;AnnotateMissense_pred=.;AnnotateMissense_status=not_snv
chr22   10000000  .       A    G    .     PASS    DP=12;AnnotateMissense_score=.;AnnotateMissense_pred=.;AnnotateMissense_status=not_in_release
```

The fields are `Number=A` (one value per ALT), so they stay correct after `bcftools norm -m-` splitting and can be extracted with `bcftools query -f '%CHROM\t%POS\t%REF\t%ALT\t%INFO/AnnotateMissense_score\n'`. Existing INFO keys are preserved; re-annotating replaces earlier AnnotateMissense fields.

The TSV output has one row per ALT allele, with the normalised query and extra annotations:

```text
line chrom pos      id     ref alt query               status         annotatemissense_score annotatemissense_confidence annotatemissense_pred gene                   clinvar_clnsig ...
4    chr22 19963748 rs4680 G   A   chr22:19963748:G:A  found          0.000972               0.999028                    T                     COMT                   Benign
5    chr22 21549460 .      C   A   chr22:21549460:C:A  found          0.04917                0.95083                     T                     RIMBP3B,RIMBP3C,UBE2L3
5    chr22 21549460 .      C   G   chr22:21549460:C:G  found          0.018423               0.981577                    T                     RIMBP3B,RIMBP3C,UBE2L3
6    chr22 36305984 .      CG  CA  chr22:36305985:G:A  found          0.999574               0.999574                    D                     MYH9                   Pathogenic
7    chr22 36305985 .      GG  G                       not_snv
8    chr22 10000000 .      A   G   chr22:10000000:A:G  not_in_release
```

TSV columns: `line, chrom, pos, id, ref, alt, query, status, annotatemissense_score, annotatemissense_confidence, annotatemissense_pred, gene, exonic_function, aa_change_refgene, clinvar_clnsig, alphamissense, cadd_phred, revel, metarnn, gnomad30_af, annotation_rows`.

## Performance

These timings were measured on a Windows 11 laptop against the chromosome-22 extract (1.68 M rows):

| Input | Path | Time |
|---|---|---|
| 8-record VCF | web page / API | 0.07 s server-side |
| 99,949-allele VCF | `annotate-vcf` CLI | 10.8 s (≈9,300 alleles/s) |

Large inputs are joined against the database in a single pass (a temporary query table), so runtime grows with input size and one scan of the database. Expect longer times on the full genome-wide file; this has not been benchmarked here.

## Interpreting statuses

- `found`: use `AnnotateMissense_score` / `_pred`.
- `no_score`: the allele is in the release but the model produced no prediction (1.3% of chromosome-22 rows).
- `not_in_release`: not a nonsynonymous SNV in the dbNSFP v5.1 set (e.g. intronic, synonymous, or a wrong genome build). Absence is **not** evidence of benignity.
- `not_snv`: AnnotateMissense covers single-nucleotide substitutions only.
