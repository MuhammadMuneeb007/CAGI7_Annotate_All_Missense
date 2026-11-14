# Naming and releases

**AnnotateMissense** is the name of the tool, the resource and the score. Earlier development names survive in some file and column names that cannot be changed in the published v1.0.0 deposit. This table maps all of them:

| Where you see it | Historical name | Means | Name used by the server, API and CLI |
|---|---|---|---|
| GitHub (pipeline) | `CAGI7_Annotate_All_Missense` | the pipeline repository, now renamed (old links redirect) | [`AnnotateMissense`](https://github.com/MuhammadMuneeb007/AnnotateMissense) |
| GitHub (server) | `uq_biosig_variant`, "VarPath" | the web server | [`AnnotateMissense-Server`](https://github.com/MuhammadMuneeb007/AnnotateMissense-Server) |
| Zenodo file | `UQ_BioSig_model_Final.tsv.gz` | AnnotateMissense genome-wide prediction table | AnnotateMissense prediction table |
| Zenodo file | `variants.duckdb.gz` | AnnotateMissense annotation database | — |
| Database column | `UQ_score` | AnnotateMissense score (0–1) | `annotatemissense_score` |
| Database column | `UQ_pred` | AnnotateMissense prediction (`D` ≥ 0.5 > `T`) | `annotatemissense_pred` |
| Database column / TSV | `UQ_sd` / `sd` | probability of the predicted class, max(score, 1 − score) | `annotatemissense_confidence` |
| TSV | `pred = U` | no prediction available | NULL score, status `no_score` |
| VCF INFO | — | — | `AnnotateMissense_score`, `AnnotateMissense_pred`, `AnnotateMissense_status` |
| Older text | "AnnotateMissenseModel" | the AnnotateMissense model | AnnotateMissense |

`UQ_BioSig` refers to the authors' laboratory (Biosig, The University of Queensland). It is not a separate product.

## Releases

| Version | Date | Content |
|---|---|---|
| Data v1.0.0 | 2026-05-02 | [doi:10.5281/zenodo.19981867](https://doi.org/10.5281/zenodo.19981867): `variants.duckdb.gz`, `UQ_BioSig_model_Final.tsv.gz` |
| Server 1.1.0 | 2026-11 | web UI, REST API, VCF annotation, CLI ([AnnotateMissense-Server](https://github.com/MuhammadMuneeb007/AnnotateMissense-Server)) |

Recommended for the next data version (a new Zenodo version DOI, leaving v1.0.0 unchanged):

- rename the prediction table to `AnnotateMissense_predictions_GRCh38_v1.tsv.gz`, with columns `annotatemissense_score`, `annotatemissense_confidence`, `annotatemissense_pred`;
- point the Zenodo "related software" link to `github.com/MuhammadMuneeb007/AnnotateMissense`;
- add a schema file and per-file checksums, and state the licence terms for the embedded third-party columns ([Licensing](licensing.md)).

Cite the concept DOI to refer to all versions, or the version DOI for exact reproducibility.
