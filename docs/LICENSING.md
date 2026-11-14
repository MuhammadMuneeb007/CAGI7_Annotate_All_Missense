# Licensing, third-party resources and usage constraints

**This page distinguishes metadata that is verified from legal permissions that still require review.** It is not legal advice.

## Verified as published

- [Zenodo record 19981867](https://zenodo.org/records/19981867) currently lists **Creative Commons Attribution 4.0 International (CC BY 4.0)** for the **dataset deposit** (`variants.duckdb.gz` and `UQ_BioSig_model_Final.tsv.gz`).
- The repository's historical README claimed **MIT** licensing for the project, but **no corresponding root `LICENSE` file** was present at the time of review. The published code's intended licence therefore remains **unverified**.
- **The CC BY 4.0 metadata does not independently grant rights to redistribute third-party annotations or predictor outputs whose rights belong to other authors.** It should not be described as a clearance of upstream licensing obligations.

## Restrictions and upstream sources

- **ANNOVAR:** registration/acceptance of its terms is required; its official download site describes free personal/academic/non-profit use and separate commercial permissions. ANNOVAR itself is **not redistributed** here. https://www.openbioinformatics.org/annovar/annovar_download_form.php
- **dbNSFP v5.x:** academic/commercial usage rules apply and some components have additional conditions (e.g., CADD, VEST, M-CAP, PolyPhen-2, PrimateAI). Users and redistributors are responsible for checking the particular versions and embedded source fields. https://www.dbnsfp.org/license/
- **Other upstream sources:** AlphaMissense, ESM-derived scores, population annotations and other source resources may have independent terms. The conditions for the **specific released contents**, rather than only the code used to compute them, must be checked.

## Recommended author actions before revising the manuscript's legal claim

1. Inventory which third-party score fields and raw annotations **actually appear in the released database**, not just which appear in the model-building workflow.
2. Seek permission/confirmation where required from the upstream rightsholders or institutional research/legal support.
3. Decide and document a **separate, explicit code licence**, then add a correctly attributed root `LICENSE` file. Do **not** assume MIT is already legally established just because an old badge said so.
4. If the original Zenodo CC BY 4.0 claim is too broad for the released data, contact Zenodo support or create an appropriately corrected version/metadata after rights review; preserve the DOI/version provenance.
5. Add an upstream provenance/attribution table covering source, version, URL, licence and whether redistributed or only used in producing derived scores.

## Use and interpretation

AnnotateMissense scores are provided for **research prioritisation, variant triage and reproducible benchmarking**; they are **not standalone clinical classifications** and should not be used as the sole basis for medical decisions. Commercial/clinical use may require separate permissions.

Current documented metadata: **dataset deposit lists CC BY 4.0; upstream redistribution rights not yet verified; code licence not confirmed**.
