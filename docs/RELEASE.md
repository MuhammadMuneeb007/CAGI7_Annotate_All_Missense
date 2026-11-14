# Naming, repository migration and Zenodo release checklist

## Canonical branding

**Product:** AnnotateMissense

**Study:** *AnnotateMissense: a genome-wide annotation and benchmarking framework for missense pathogenicity prediction*

**Existing GitHub repository (unrenamed as of this documentation change):**
https://github.com/MuhammadMuneeb007/CAGI7_Annotate_All_Missense

**Recommended repository slug:** `AnnotateMissense`

**Existing deposited Zenodo release:** https://doi.org/10.5281/zenodo.19981867, v1.0.0.

**Deposited filenames (must not be misreported as renamed):** `variants.duckdb.gz` and `UQ_BioSig_model_Final.tsv.gz`. The legacy `UQ_BioSig` wording is a historical output filename; the scientific resource is called AnnotateMissense.

## Rename GitHub (repository administrator)

The connected GitHub contents integration can update files but does **not** expose repository rename settings. To rename:

1. Open the current repository in GitHub.
2. Go to **Settings → General → Repository name**.
3. Enter `AnnotateMissense` and select **Rename**.
4. In active local clones, update the remote:

```bash
git remote set-url origin https://github.com/MuhammadMuneeb007/AnnotateMissense.git
git remote -v
```

GitHub generally redirects old repository links, but update the new URL throughout the manuscript, Zenodo, README, scripts and any pinned workflow links. Avoid reusing the old slug because it can break redirect behaviour.

## Align Zenodo metadata

The 2026-05-02 published record currently displays:

- Title: **Genome-wide AnnotateMissense predictions for 90.6 million hg38 missense variants**.
- Version: `1.0.0`.
- Dataset licence metadata: **CC BY 4.0** (upstream-rights review pending).
- Related software URL: old GitHub repository.
- A *different* historical manuscript subtitle in its description.
- Two filenames shown above, compressed total ~22.2 GB.

**Manual actions for the Zenodo owner:**

1. Update the related-software GitHub URL after a successful repository rename.
2. Align the description with the exact current manuscript title.
3. Add documentation links for lookup, schema, transcript caveats, software environment and third-party provenance.
4. Verify the licence and third-party usage rights before asserting that the entire derived database is unrestricted under CC BY 4.0.
5. If distributing renamed files (e.g., `AnnotateMissense_predictions_GRCh38.tsv.gz`), prepare a **new version** with verified checksums; do not claim the already published v1.0.0 attachments were renamed. Consider version-pinned and concept DOIs for citation.

The GitHub integration alone **cannot modify Zenodo account metadata**. These actions require the depositor's authenticated Zenodo session or a separately authorised deposit workflow.

## Manuscript sentence (pending legal and transcript verification)

“AnnotateMissense provides a downloadable precomputed research lookup for hg38 missense variants, with predictions and annotations in a compressed DuckDB database and TSV available at DOI:10.5281/zenodo.19981867. The release does not currently offer an end-user service for predicting arbitrary new VCF variants. Runtime and storage requirements for reproducing the full research workflow, lookup examples, database schema-inspection commands, transcript-provenance limitations and third-party licensing considerations are documented in the accompanying GitHub repository.”
