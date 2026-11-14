# Installation, dependencies, hardware, runtime and storage

## End-user lookup versus full research rebuild

These are different tasks and should **not** be described as having identical requirements.

### A. Lookup an already scored hg38 missense variant

- **Install:** Python and `duckdb` package, or DuckDB CLI.
- **Download:** `variants.duckdb.gz` from [Zenodo](https://zenodo.org/records/19981867).
- **Storage:** 20,819,921,303 **compressed** bytes, *plus* room for the extracted database. Exact extracted size and typical per-query latency have not been benchmarked.
- **GPU:** not required.
- **ANNOVAR, dbNSFP, ML frameworks:** not required.
- **Input:** genomic allele already present in the database; no supported de-novo VCF scoring service.

See [QUICKSTART.md](QUICKSTART.md) and [DATABASE.md](DATABASE.md).

### B. Reproduce the full genome-wide research annotation and inference

This uses separate software and data sources. The GitHub README describes the HPC workflow (Steps 1–6) and the final chromosome-level inference.

| Component | Relevant dependencies / requirements |
|---|---|
| Genome annotation | ANNOVAR (Perl; independently registered/licensed), hg38 annotation databases, dbNSFP v5.1, AlphaMissense and ESM-derived source annotations |
| Preparation and integration | Python, Bash, Polars, pandas, NumPy, and relevant parser libraries; SLURM for the published HPC-oriented scheduling scripts |
| Machine learning | scikit-learn, XGBoost; FLAML optional; other model scripts use additional packages |
| Deep learning | PyTorch/TensorFlow and task-specific packages when using the DL scripts |
| Database inspection | DuckDB (separate lightweight lookup dependency) |

The exact versions for all components were **not fully pinned in the public repository** at the time of this documentation update. A single one-command fully reproducible environment has **not been validated**. Researchers must obtain third-party datasets and software separately under their own applicable terms.

### Resource estimates from the existing research README (not measured benchmarks)

- Approximate raw annotation source storage: **~500 GB**.
- Approximate intermediate annotation storage: **~1 TB**.
- Approximate total project working storage stated elsewhere in the README: **~1.9 TB**.
- Suggested full-run peak memory: **64–128 GB**; **256 GB** recommended for comfortable headroom.
- Multi-core CPU and chromosome-level SLURM parallelisation.
- Some DL stages can use CUDA GPUs; not all computational stages use GPU.
- Approximate stage timings in the old README include 24–48 hours for parallel annotation and 12–24 hours for final inference, **not a reproducible end-to-end runtime benchmark**. Actual time depends on cluster hardware, parallelism, input format and I/O throughput.

Do not cite these as independently measured per-sample runtime, and do not confuse genome-wide creation with lookup on a precomputed file.

## Reproducibility checklist

To reproduce a scientific result, record (1) genome assembly, (2) upstream resource release and licence, (3) preprocessing/selection scripts and commit, (4) train/test allele keys, (5) exact feature set, (6) model and random seed, (7) score thresholds and metrics, (8) storage/CPU/GPU usage, and (9) output checksums. The newer Dataset 9/10 ablation work is **not** automatically part of the previously published Zenodo v1.0.0 archive.
