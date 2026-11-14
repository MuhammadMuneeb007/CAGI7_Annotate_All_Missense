# Licensing and data use

!!! note
    This page describes the terms that apply to the resource and its components. It is not legal advice; check the original terms of each upstream resource for your use case.

## Current status

| Item | Status |
|---|---|
| **Data deposit** (`variants.duckdb.gz`, `UQ_BioSig_model_Final.tsv.gz`) | Zenodo metadata: **CC BY 4.0** ([record 19981867](https://doi.org/10.5281/zenodo.19981867)) |
| **AnnotateMissense scores** (`UQ_score`/`UQ_pred`, the prediction table) | original work of the authors, released with the deposit; derived from features that include third-party values (see below) |
| **Embedded third-party annotations** | remain under their **original terms**; CC BY 4.0 on the deposit does not relicense them |
| **Code** (pipeline and server repositories) | no `LICENSE` file yet; until one is added, reuse of the code requires the authors' permission |
| **ANNOVAR** | not redistributed; needed only to rebuild the annotations |

## Third-party content inside the database

The release database carries values computed by other groups. Their terms travel with those columns:

| Component | Columns | Terms to check |
|---|---|---|
| **dbNSFP v4.2a, academic version** | 107 columns without a prefix (`SIFT_score` … `GTEx_V8_tissue`) | [dbNSFP licence](https://www.dbnsfp.org/license). **17 of these columns, from 6 tools, are absent from dbNSFP's commercial-use v4.2c:** PolyPhen-2 (`Polyphen2_HDIV/HVAR_*`), VEST4, REVEL, CADD (`CADD_raw`, `CADD_phred`), GenoCanyon and LINSIGHT. dbNSFP leaves these out of its commercial version because their own licences restrict commercial use. |
| **dbNSFP v4.2c, commercial-use version** | 90 `dbnsfp42c_*` columns | [dbNSFP licence](https://www.dbnsfp.org/license) (attribution; component terms) |
| **dbNSFP v5.1** (challenge input: transcript and protein context) | `aaref`, `aapos`, `Ensembl_*`, `genename`, … | [dbNSFP licence](https://www.dbnsfp.org/license) |
| **AlphaMissense** | `am_pathogenicity`, `am_class`, `AlphaMissense`, … | Google DeepMind released the hg38 predictions under **CC BY-NC-SA 4.0** (non-commercial, share-alike); see [AlphaMissense](https://github.com/google-deepmind/alphamissense) |
| **ESM1v** predictions | `esm1v_*`, `combined_score` | ESM models are MIT-licensed (Meta); check the terms of the precomputed score set used |
| **COSMIC v70** | `cosmic70` | COSMIC data are free for academic use; commercial use needs a licence ([COSMIC](https://cancer.sanger.ac.uk/cosmic/license)) |
| **gnomAD** v2.1.1 / v3.0, ExAC | `AF*`, `gnomad211_*`, `gnomad30_*`, `ExAC_*`, `gnomad_constraint` | [gnomAD terms](https://gnomad.broadinstitute.org/terms) |
| **ClinVar, dbSNP** (NCBI) | `CLN*`, `avsnp150` | freely available from NCBI ([ClinVar](https://www.ncbi.nlm.nih.gov/clinvar/docs/maintenance_use/)) |
| **1000 Genomes, UCSC/NCBI gene and region tracks** (via ANNOVAR) | `*.sites.2015_08`, `*.refGene`, `cytoBand`, `rmsk`, … | public data; cite the original resources |
| **ANNOVAR software** | none (only its output) | free for personal, academic and non-profit use after [registration](https://www.openbioinformatics.org/annovar/annovar_download_form.php); commercial use needs a licence |

### What this means in practice

- **Academic and non-commercial research:** you can use the database, the scores, the web server and the CLI, citing AnnotateMissense and the upstream resources.
- **Commercial use, or redistribution in a commercial product:** the database contains columns whose owners restrict commercial use (dbNSFP v4.2a academic-only fields such as CADD, PolyPhen-2, REVEL and VEST4; AlphaMissense under CC BY-NC-SA; COSMIC). Obtain the relevant licences, or drop those columns. The AnnotateMissense model used features from these sources. Whether the scores themselves carry any restriction is a question for your institution's legal advisers.
- **Redistribution:** if you redistribute subsets of the database, keep the upstream attributions and terms with the columns you include. Share-alike terms (AlphaMissense) require derivative datasets containing those values to use the same licence.
- **Rebuilding the resource:** you need your own ANNOVAR registration and must download each database under its own terms.

## Clinical use

AnnotateMissense scores are for research prioritisation, benchmarking and variant triage. They are not validated clinical classifications and must not be the sole basis for a diagnostic or treatment decision.
