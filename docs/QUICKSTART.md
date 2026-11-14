# Quick start: precomputed lookup

## What users can do

Download the published **AnnotateMissense** database and look up **hg38/GRCh38 genomic alleles** that are already in the release. This is an **allele lookup**, not prediction of new variants; the release does **not** offer a ready-made VCF-to-new-score command or API. The accompanying research scripts are for reproducing annotation and model development.

## Download

See the authoritative [Zenodo record](https://zenodo.org/records/19981867):

- `variants.duckdb.gz` — compressed DuckDB database
- `UQ_BioSig_model_Final.tsv.gz` — original compressed final prediction table (legacy filename; **AnnotateMissense** is the product name)

The downloaded database is compressed: allow space **greater than** the reported 20.8 GB compressed size for extraction, plus working headroom. Its exact extracted size has not yet been verified.

```bash
# In the directory where you downloaded the published files:
gzip -dk variants.duckdb.gz  # keeps the archive; creates variants.duckdb

# Lightweight lookup installation (NOT the research-training environment):
python -m venv .venv
source .venv/bin/activate
python -m pip install duckdb

# Get the real schema first — do not guess a table or column name.
python - <<'PY'
import duckdb
con = duckdb.connect('variants.duckdb', read_only=True)
print(con.execute("SELECT table_schema, table_name FROM information_schema.tables WHERE table_type='BASE TABLE' ORDER BY 1, 2").df().to_string(index=False))
print(con.execute("SELECT table_schema, table_name, column_name, data_type FROM information_schema.columns ORDER BY table_schema, table_name, ordinal_position").df().to_string(index=False))
con.close()
PY
```

If `pandas` is absent, replace each `.df().to_string(index=False)` with `.fetchall()`; DuckDB itself is the only essential Python dependency for lookup.

## Lookup query

Once you have identified the **actual** table and genomic-allele column names from the schema inspection, substitute them in this SQL template. These are placeholders, **not claims about the deposited schema**.

```sql
SELECT *
FROM "YOUR_TABLE_NAME"
WHERE CAST("YOUR_CHROMOSOME_COLUMN" AS VARCHAR) IN ('1', 'chr1')
  AND TRY_CAST("YOUR_POSITION_COLUMN" AS BIGINT) = 123456
  AND UPPER(CAST("YOUR_REF_COLUMN" AS VARCHAR)) = 'A'
  AND UPPER(CAST("YOUR_ALT_COLUMN" AS VARCHAR)) = 'G'
LIMIT 20;
```

Run the query with Python (`duckdb.connect(...).execute(query).fetchall()`) or an installed DuckDB CLI. Check that assembly, one-based position and alleles match the actual deposited fields. **Do not collapse multiple returned rows without validating transcript and allele multiplicity.**

## Interpreting results

A score is a **research prioritisation aid**, not a clinical assertion. A variant absent from the precomputed resource cannot be scored by this lookup. Missing result = **not present**; it is not evidence of benignity. See [database schema](DATABASE.md), [transcripts](TRANSCRIPTS.md), and [licensing](LICENSING.md).
