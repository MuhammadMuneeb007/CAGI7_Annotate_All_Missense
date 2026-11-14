# REST API

Base URL: `<server>/api/v1`. Responses are JSON with a `status` field (`success`, `ok` or `error`). CORS is enabled, no API key is needed and no cookies are set. Coordinates are GRCh38 and 1-based; chromosomes may be `22` or `chr22`.

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/variant/{chr}:{pos}:{ref}:{alt}` | one allele: score, annotations, transcript context |
| POST | `/variant` | the same, with `chrom`, `pos`, `ref`, `alt` as form or JSON |
| GET | `/variant/full?chrom=&pos=&ref=&alt=` | all 337 database columns (legacy names) |
| POST | `/batch` | up to 1,000 alleles: compact records plus summary |
| POST | `/annotate?format=json\|vcf\|tsv` | annotate an uploaded VCF or variant list |
| GET | `/gene/{symbol}?limit=100` | alleles annotated to a gene (1–500) |
| POST | `/compare` | 2–4 alleles side by side |
| GET | `/schema` | column names, types and sources |
| GET | `/stats` | row and prediction counts per chromosome |
| GET | `/health` | version, loaded resources, coverage, limits |

## Single variant

```bash
curl "$SERVER/api/v1/variant/22:36305985:G:A"
curl -X POST "$SERVER/api/v1/variant" -H 'Content-Type: application/json' \
     -d '{"chrom": "22", "pos": 36305985, "ref": "G", "alt": "A"}'
```

Response sections: `variant` (with rsID, protein change, hg19 position), `annotatemissense` (`score`, `prediction` D/T, `label`, `confidence`, `threshold` = 0.5), `gene`, `clinical`, `esm1v`, `pathogenicity_scores`, `conservation`, `population`, `transcripts` and `provenance` (`annotation_rows`).

Unknown alleles return HTTP 404:

```json
{"status": "error", "message": "Variant not found",
 "detail": "Allele is not among the precomputed GRCh38 missense SNVs served by this instance. Absence is not evidence of benignity.",
 "query": {"chr": "chr22", "pos": 10000000, "ref": "A", "alt": "G"}}
```

## Batch

```bash
curl -X POST "$SERVER/api/v1/batch" -F variants_text=$'22 36305985 G A\n22 19963748 G A\n22 21549460 C A,G'
curl -X POST "$SERVER/api/v1/batch" -F variants_file=@variants.tsv
curl -X POST "$SERVER/api/v1/batch" -H 'Content-Type: application/json' \
     -d '{"variants": ["22:36305985:G:A", {"chr": "22", "pos": 19963748, "ref": "G", "alt": "A"}]}'
```

Each result has `chr`, `pos`, `ref`, `alt`, `status` (`found`, `no_score`, `not_in_release`), `annotatemissense_score`, `annotatemissense_confidence`, `annotatemissense_pred`, `gene`, `aa_change`, `CLNSIG`, `AlphaMissense`, `CADD_phred`, `REVEL_score`, `MetaRNN_score`, `gnomad30_AF` and `annotation_rows`. Indels and malformed lines are listed under `skipped`.

## Annotate a VCF

```bash
curl -X POST "$SERVER/api/v1/annotate?format=vcf"  -F file=@sample.vcf.gz -o sample.annotatemissense.vcf
curl -X POST "$SERVER/api/v1/annotate?format=tsv"  -F file=@sample.vcf    -o sample.annotatemissense.tsv
curl -X POST "$SERVER/api/v1/annotate?format=json" -F file=@sample.vcf
```

See [Annotating a VCF](vcf.md) for the output fields.

## Python

```python
import requests

SERVER = "https://<annotatemissense-server>"
r = requests.get(f"{SERVER}/api/v1/variant/22:36305985:G:A", timeout=60)
r.raise_for_status()
am = r.json()["annotatemissense"]
print(am["score"], am["prediction"])          # 0.999574 D

with open("sample.vcf", "rb") as fh:
    out = requests.post(f"{SERVER}/api/v1/annotate", params={"format": "tsv"}, files={"file": fh}, timeout=300)
open("sample.annotatemissense.tsv", "wb").write(out.content)
```

## R

```r
library(httr2)
server <- "https://<annotatemissense-server>"
resp <- request(server) |> req_url_path_append("api/v1/variant/22:36305985:G:A") |> req_perform()
res  <- resp_body_json(resp)
res$annotatemissense$score       # 0.999574

batch <- request(server) |> req_url_path_append("api/v1/batch") |>
  req_body_json(list(variants = list("22:36305985:G:A", "22:19963748:G:A"))) |> req_perform() |>
  resp_body_json()
do.call(rbind, lapply(batch$results, \(x) data.frame(variant = x$variant, score = x$annotatemissense_score)))
```

## Limits and errors

| Code | Meaning |
|---|---|
| 400 | invalid input (unparseable chromosome or position, too many variants) |
| 404 | allele not in the release (not evidence of benignity) |
| 413 | upload above `MAX_UPLOAD_MB` (default 50 MB) |
| 429 | rate limit exceeded (default 300 requests/hour/IP; the public demo uses 600) |
| 501 | endpoint needs the annotation database (e.g. gene view on a score-only server) |

For more than a few thousand variants, use the [offline CLI](cli.md) or your own [server](server.md#self-hosting) with higher limits.
