#!/usr/bin/env python3
"""Inspect the released AnnotateMissense DuckDB database without changing it.

Usage:
    python -m pip install duckdb
    python Step7-InspectAnnotateMissenseDatabase.py variants.duckdb
    python Step7-InspectAnnotateMissenseDatabase.py variants.duckdb --count-rows
    python Step7-InspectAnnotateMissenseDatabase.py variants.duckdb --audit-alleles

CAUTION: count/duplicate audits can be expensive for ~90 million variants.
"""
import argparse
import json
import re
from pathlib import Path


def quoted(identifier):
    return '"' + identifier.replace('"', '""') + '"'


def pick(columns, candidates):
    # Avoid accidentally selecting protein/transcript coordinates as genome coordinates.
    def normal(name):
        return re.sub(r"[^a-z0-9]", "", name.lower())
    lookup = {normal(c): c for c in columns}
    for candidate in candidates:
        if normal(candidate) in lookup:
            return lookup[normal(candidate)]
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("database", type=Path, help="Path to decompressed variants.duckdb")
    parser.add_argument("--output", type=Path, default=Path("AnnotateMissense_database_schema.json"))
    parser.add_argument("--count-rows", action="store_true", help="May scan large tables")
    parser.add_argument("--audit-alleles", action="store_true", help="Expensive grouped uniqueness query")
    args = parser.parse_args()

    if not args.database.is_file():
        parser.error("Database does not exist: " + str(args.database))

    try:
        import duckdb
    except ImportError:
        parser.error("Install the lightweight lookup dependency: python -m pip install duckdb")

    con = duckdb.connect(str(args.database), read_only=True)
    try:
        table_rows = con.execute("""
            SELECT table_schema, table_name
            FROM information_schema.tables
            WHERE table_type = 'BASE TABLE'
              AND table_schema NOT IN ('information_schema', 'pg_catalog')
            ORDER BY 1, 2
        """).fetchall()
        if not table_rows:
            raise SystemExit("No base tables were found in the DuckDB file")

        report = {
            "database_file": args.database.name,
            "database_size_bytes": args.database.stat().st_size,
            "tables": []
        }
        for schema, table in table_rows:
            cols = con.execute("""
                SELECT column_name, data_type
                FROM information_schema.columns
                WHERE table_schema = ? AND table_name = ?
                ORDER BY ordinal_position
            """, [schema, table]).fetchall()
            item = {
                "schema": schema,
                "table": table,
                "columns": [{"name": n, "type": dtype} for n, dtype in cols]
            }
            names = [n for n, dtype in cols]
            candidates = {
                "chr": pick(names, ["#chr", "chr", "chrom", "chromosome", "chromosome_name"]),
                "pos": pick(names, ["pos(1-based)", "pos", "position", "start", "pos_1based"]),
                "ref": pick(names, ["ref", "reference"]),
                "alt": pick(names, ["alt", "alternate"])
            }
            item["candidate_genomic_allele_columns"] = candidates
            quoted_table = quoted(schema) + "." + quoted(table)

            if args.count_rows:
                item["row_count"] = con.execute(
                    "SELECT COUNT(*) FROM " + quoted_table
                ).fetchone()[0]

            if args.audit_alleles and all(candidates.values()):
                grouped = ", ".join(quoted(candidates[key]) for key in ("chr", "pos", "ref", "alt"))
                query = (
                    "SELECT " + grouped + ", COUNT(*) AS n "
                    "FROM " + quoted_table + " GROUP BY " + grouped
                    + " HAVING COUNT(*) > 1 LIMIT 10"
                )
                item["example_duplicate_alleles"] = [
                    list(row) for row in con.execute(query).fetchall()
                ]
            report["tables"].append(item)
            print(f"{schema}.{table}: {len(cols)} columns")
            for name, dtype in cols:
                print("  " + name + " : " + dtype)
            print("Allele key candidates:", candidates)
            if "row_count" in item:
                print("Rows:", item["row_count"])
            if "example_duplicate_alleles" in item:
                print("Example duplicated alleles:", item["example_duplicate_alleles"])

        args.output.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        print("\nSaved schema report to", args.output)
    finally:
        con.close()


if __name__ == "__main__":
    main()
