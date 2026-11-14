# Database, deposited files, sizes and schema

## Verified publication metadata (2026-11-14)

Published Zenodo record: https://zenodo.org/records/19981867 (DOI `10.5281/zenodo.19981867`, version `1.0.0`).

| Deposited name | Compressed size | Exact deposited bytes | MD5 in Zenodo metadata |
|---|---:|---:|---|
| `variants.duckdb.gz` | 20.8 GB | 20,819,921,303 | `578ab4e9c988c2cf78d8aff9f7ceb63b` |
| `UQ_BioSig_model_Final.tsv.gz` | 1.4 GB | 1,351,827,163 | `417f2649058118bb3b576e693ba75344` |

Combined reported compressed size: approximately **22.2 GB**. The *uncompressed DuckDB size* has **not been measured**; do not quote 20.8 GB as the on-disk uncompressed size.

The deposit describes genome-wide annotations and AnnotateMissense prediction outputs for a study of 90,643,830 hg38 missense variants. That study count is not independently confirmed as the current `COUNT(*)` or `COUNT(DISTINCT allele)` of each database table.

## Authoritative schema inspection

**No table-level or column-level schema was published in the Zenodo record description.** Run these queries against the extracted `variants.duckdb` to record the exact table names, column names, types and row counts.

```python
import duckdb
con = duckdb.connect("variants.duckdb", read_only=True)

print("TABLES")
for row in con.execute("""
    SELECT table_schema, table_name
    FROM information_schema.tables
    WHERE table_type = 'BASE TABLE'
    ORDER BY 1, 2
""").fetchall():
    print(row)

print("COLUMNS")
for row in con.execute("""
    SELECT table_schema, table_name, ordinal_position, column_name, data_type
    FROM information_schema.columns
    ORDER BY table_schema, table_name, ordinal_position
""").fetchall():
    print(row)

con.close()
```

To inspect a confirmed table (replace the placeholder):

```sql
DESCRIBE SELECT * FROM "YOUR_TABLE_NAME";
SELECT * FROM "YOUR_TABLE_NAME" LIMIT 5;
SELECT COUNT(*) AS row_count FROM "YOUR_TABLE_NAME";
```

The last query can scan a large dataset; run it with adequate local resources.

## Variant key and example query

The research submission pipeline uses a GRCh38 genomic-allele key comprising `#chr`, `pos(1-based)`, `ref`, `alt` in its intermediate TSV. **The DuckDB table names and the exact names of its key/score columns require inspection**; they may differ from the intermediate TSV. A safe parameterised lookup after identifying your table and columns is:

```python
import duckdb
con = duckdb.connect("variants.duckdb", read_only=True)
# Replace quoted identifiers with the verified table/column names.
sql = """
    SELECT *
    FROM "YOUR_TABLE_NAME"
    WHERE CAST("YOUR_CHR_COL" AS VARCHAR) IN (?, ?)
      AND TRY_CAST("YOUR_POS_COL" AS BIGINT) = ?
      AND UPPER(CAST("YOUR_REF_COL" AS VARCHAR)) = ?
      AND UPPER(CAST("YOUR_ALT_COL" AS VARCHAR)) = ?
    LIMIT 20
"""
rows = con.execute(sql, ["1", "chr1", 123456, "A", "G"]).fetchall()
print(rows)
con.close()
```

Inspect [TRANSCRIPTS.md](TRANSCRIPTS.md) before assuming a unique score per allele. When measured, publish an explicit schema snapshot and query outputs in a subsequent version of this document; do not invent field names.

The original TSV filename remains unchanged for compatibility with the published archive. Branding uses **AnnotateMissense** consistently.
