# Research pipeline

The [AnnotateMissense repository](https://github.com/MuhammadMuneeb007/AnnotateMissense) contains the scripts that produced the release. They are intended for reproduction and method development. Lookup users do not need them; see [Requirements](requirements.md#full-genome-wide-rebuild-research-pipeline) for what a rebuild involves.

| Stage | Scripts | What happens |
|---|---|---|
| 1. Input | `Step1-UnzipFile.py` | unpack the dbNSFP v5.1 nonsynonymous-SNV chromosome files (CAGI7 Annotate-All-Missense input, 90,643,830 variants) and convert them to ANNOVAR input |
| 2. Annotation | `Step2-AnnovarExecution.py`, `Step3-Data.sh`, `Step3-commands.txt` | ANNOVAR gene-, region- and filter-based annotation per chromosome and database (SLURM array) |
| 3. Merge | `Step4-MergeFiles.py`, `Step4-MergeFiles.sh` | merge per-database outputs per chromosome (Polars); clashing column names get the database name as prefix |
| 4. Integration | `Step5.0-AppendAAchanges.py`, `Step5.1-AppendAlphaMissense.py`, `Step5.3-AppendESMPredictions.py` | add dbNSFP amino-acid context, AlphaMissense and ESM1v |
| 5. ClinVar benchmark | `Step6.1` – `Step6.8` | build the 132,714-variant ClinVar set, folds, feature configurations (Datasets 2–8), ML/DL benchmarks |
| 6. Feature analysis and final model | `Step6.10` – `Step6.19` | feature inventory, processing, engineering, model selection, final XGBoost (1,044 inputs) |
| 7. Genome-wide inference | `Step6.18.1`, `Step6.19.1`, `Step6.20` – `Step6.22` | score all chromosomes, merge predictions, format the CAGI submission / prediction table |
| 8. Inspection | `Step7-InspectAnnotateMissenseDatabase.py` | read-only schema and allele-multiplicity report for a released DuckDB file |

The repository README documents each step's commands, inputs and outputs in detail.

## Key results (manuscript)

| Analysis | MCC |
|---|---:|
| Dataset 2, full 303-feature benchmark, XGBoost, stratified 5-fold CV (ROC-AUC 0.9950) | 0.9411 |
| Same, genomic-allele-grouped 5-fold CV | 0.9380 |
| Dataset 3, naive features | 0.4989 |
| Dataset 4, location-oriented features | 0.5113 |
| Dataset 5, no prior predictors or population frequency | 0.7582 |
| Dataset 6, Dataset 5 plus no prediction-derived features | 0.7326 |
| Dataset 7, no AlphaMissense or ESM | 0.9409 |
| Dataset 8, engineered biological features only | 0.5120 |
| Temporal ClinVar validation (298,353 new variants) | 0.7613 |
