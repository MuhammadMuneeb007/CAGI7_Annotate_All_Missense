# Requirements and runtime

What you need depends on what you do. Using the precomputed scores needs **no** ANNOVAR, dbNSFP, GPU or HPC; those matter only if you rebuild the resource.

| Use | Software | Storage | Hardware | Typical runtime |
|---|---|---|---|---|
| **Web server / REST API** | browser or `curl` | none | any | ~0.1 s per lookup server-side |
| **Local lookup / offline VCF annotation** | Python ≥ 3.9, `duckdb ≥ 1.4.3` | 20.8 GB download; ≈50–55 GB extracted (estimate) | laptop or desktop; 8–16 GB RAM suggested | 100,000 VCF alleles in ~11 s (measured on chr22) |
| **Light score lookup** | as above | 1.4 GB TSV + score index | laptop | not benchmarked |
| **Self-hosted server** | above + Flask, flask-cors, flask-limiter, waitress/gunicorn (or Docker) | as local lookup | 2+ cores, 8–16 GB RAM suggested | as above |
| **Full genome-wide rebuild** | see below | ~500 GB raw + ~1 TB intermediate | HPC/SLURM, 50–128 GB RAM per job | days (see below) |

## Installing the lookup tools

```bash
python -m pip install "git+https://github.com/MuhammadMuneeb007/AnnotateMissense-Server"   # CLI + core
# or, to run the web server:
git clone https://github.com/MuhammadMuneeb007/AnnotateMissense-Server && cd AnnotateMissense-Server
pip install -r requirements.txt && pip install -e .
```

`requirements.txt` pins `duckdb>=1.4.3,<2`, `flask>=3.0,<4`, `flask-cors>=4.0`, `flask-limiter>=3.7`, `waitress>=3.0` and `gunicorn>=22` (Linux). The test suite (`pytest`) runs on a 146-row extract in a few seconds, and CI runs it on Python 3.9 and 3.12.

Docker alternative: `docker compose up -d --build` (Python 3.11-slim image, nginx in front).

## Measured performance (lookup and server)

The measurements were taken on a Windows 11 laptop with Python 3.13, DuckDB 1.5.6, and the chromosome-22 extract of the release (1,681,296 rows, 1.09 GB):

| Operation | Time |
|---|---|
| Server start-up (open database, compute coverage) | ~2 s |
| `GET /api/v1/variant/...` single allele, full report | 0.11 s |
| Batch of 4 alleles (`/api/v1/batch`) | < 0.1 s |
| 8-record VCF via `/api/v1/annotate` | 0.07 s |
| 99,949-allele VCF via `annotatemissense annotate-vcf` | 10.8 s |
| Full allele audit (`annotatemissense audit`) | 1.3 s |

These have not yet been repeated on the full genome-wide file. Lookups use DuckDB block skipping on the chromosome/position-sorted data. Large inputs are resolved with a single join, so time grows roughly with input size plus one pass over the database.

## Full genome-wide rebuild (research pipeline)

Rebuilding the annotations and re-running inference follows the `Step*` scripts in the [pipeline repository](pipeline.md). It needs:

| Component | Requirement |
|---|---|
| Annotation | ANNOVAR (Perl; free academic registration with the Wang lab), hg38 ANNOVAR databases (`refGene`, `ensGene`, `knownGene`, `ccdsGene`, `cytoBand`, `genomicSuperDups`, `rmsk`, `simpleRepeat`, `dbnsfp42a`, `dbnsfp42c`, `phastCons100way`, `clinvar_20220320`, `avsnp150`, `gnomad211_exome`, `gnomad211_genome`, `gnomad30_genome`, `exac03`, 1000 Genomes `*.sites.2015_08`, `cosmic70`, `dbscsnv11`, `wgEncodeGencodeBasicV44`, plus custom AlphaMissense and `gnomad_constraint` tables; full list in `Step2-AnnovarExecution.py`), dbNSFP v5.1 chromosome files, AlphaMissense hg38 predictions, ESM1v predictions |
| Data processing | Python 3.8+, pandas, NumPy, Polars, DuckDB, Bash |
| Machine learning | scikit-learn, XGBoost, imbalanced-learn, joblib; optional FLAML, PyCaret, auto-sklearn, LazyPredict |
| Deep learning (benchmark only) | PyTorch, TensorFlow, pytorch-tabnet, pytorch-tabular, fastai |
| Scheduling | SLURM (`Step3-Data.sh`: array of 76 annotation jobs, 50 GB RAM each, 240 h limit) |

Storage and runtime as recorded by the authors for their HPC run (wall-clock depends on the cluster; not an independent benchmark):

| Phase | Storage | Peak memory | Wall-clock |
|---|---|---|---|
| Download databases | ~200 GB ANNOVAR DBs + ~300 GB dbNSFP/AlphaMissense/ESM | — | 6–12 h |
| Parallel ANNOVAR annotation (25 chromosomes) | ~1 TB intermediate | ~50–64 GB per job | 24–48 h |
| Merge and feature integration | ~200 GB | ~50 GB | 8–12 h |
| ClinVar benchmark training (CPU; GPU optional for DL) | ~100 GB | ~32 GB | 4–8 h |
| Genome-wide inference (90.6 M variants) | ~50 GB outputs | up to 128 GB | 12–24 h |

The authors recommend a minimum of 64 GB RAM and 256 GB for comfortable headroom. GPUs are used only by the deep-learning benchmark models; XGBoost training and inference run on CPU. Package versions for the research pipeline are not fully pinned in the repository. Record the versions you use, following the reproducibility checklist below.

## Reproducibility checklist

When reporting results produced with AnnotateMissense, record:

1. Data release (Zenodo DOI and version, MD5 of the file used).
2. Software version (`annotatemissense --version`, git commit of the server/pipeline).
3. Genome build of your input (GRCh38) and how it was normalised or lifted over.
4. How missing alleles (`not_in_release`, `no_score`) were treated.
5. For rebuilds: upstream database versions, feature configuration, random seeds, and the train/test allele keys.
