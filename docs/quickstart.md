# Quick start

Four ways in, from no installation at all to a full local copy.

## 1. Web browser (no installation)

Open the public demo at <https://annotatemissense.vercel.app> (chromosome 22), enter `22 36305985 G A` (MYH9 p.R702C) and press **Search**. The page shows the AnnotateMissense score, ClinVar, predictors, conservation, gnomAD and the transcript context. See [Web server](server.md).

## 2. REST API (no installation)

```bash
SERVER=https://annotatemissense.vercel.app        # public demo (chr22); or your own server
curl "$SERVER/api/v1/variant/22:36305985:G:A"
```

```json
{
  "status": "success",
  "genome_build": "GRCh38",
  "variant": {"chr": "chr22", "pos": 36305985, "ref": "G", "alt": "A", "rsid": "rs80338826", "hgvs_p": "R702C"},
  "annotatemissense": {"score": 0.999574, "prediction": "D", "label": "Damaging", "confidence": 0.999574, "threshold": 0.5},
  "clinical": {"CLNSIG": "Pathogenic", "...": "..."},
  "transcripts": {"annotation_rows": 1, "annotatemissense_score_is_per_allele": true, "...": "..."}
}
```

Annotate a whole VCF and get a VCF back:

```bash
curl -X POST "$SERVER/api/v1/annotate?format=vcf" -F file=@my_variants.vcf.gz -o my_variants.annotatemissense.vcf
```

See [REST API](api.md) and [Annotating a VCF](vcf.md).

## 3. Local command line (offline, any size, nothing uploaded)

```bash
pip install "git+https://github.com/MuhammadMuneeb007/AnnotateMissense-Server"

# the database: 20.8 GB download, ~50-55 GB once extracted
curl -L -o variants.duckdb.gz "https://zenodo.org/records/19981867/files/variants.duckdb.gz?download=1"
gzip -d variants.duckdb.gz

annotatemissense lookup 22:36305985:G:A --db variants.duckdb
annotatemissense annotate-vcf my_variants.vcf.gz -o my_variants.annotatemissense.vcf --db variants.duckdb
```

See [Command line](cli.md).

## 4. SQL on the DuckDB file

```python
import duckdb
con = duckdb.connect("variants.duckdb", read_only=True)
con.sql("""
    SELECT Chr, Start, Ref, Alt, "Gene.refGene", UQ_score AS annotatemissense_score, UQ_pred, CLNSIG
    FROM variants
    WHERE Chr = 'chr22' AND Start = 36305985 AND Ref = 'G' AND Alt = 'A'
""").show()
```

See [Database and schema](database.md) for the 337 columns and more queries.

## Input conventions

- **Genome build:** GRCh38/hg38 only. Lift GRCh37 data over first; the VCF annotator warns when a header mentions GRCh37/hg19.
- **Coordinates:** 1-based VCF positions. Chromosomes may be written `22` or `chr22`; in the database they are `chr22`.
- **Alleles:** single-nucleotide REF/ALT. In VCFs, multi-allelic records are split per ALT, and padded SNVs (`CG>CA`) are trimmed to `G>A`.
- **Not found ≠ benign:** an allele without a score has no AnnotateMissense evidence either way.
