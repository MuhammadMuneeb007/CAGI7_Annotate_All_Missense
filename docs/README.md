# AnnotateMissense documentation

**AnnotateMissense** is a precomputed lookup resource for human **GRCh38/hg38 missense SNVs**. It provides pathogenicity scores and associated annotations for the variants already present in the released dataset. The public release **does not provide a supported arbitrary-VCF scoring pipeline or hosted inference API**.

- [Quick start: obtain data and look up variants](QUICKSTART.md)
- [Database files, sizes, schema inspection and SQL](DATABASE.md)
- [Transcript representation and allele uniqueness](TRANSCRIPTS.md)
- [Installation, computational resources and reproducibility](REPRODUCIBILITY.md)
- [Licensing, third-party restrictions and clinical-use limits](LICENSING.md)
- [Naming and Zenodo release checklist](RELEASE.md)
- [Security and credential handling](SECURITY.md)

**Publication:** *AnnotateMissense: a genome-wide annotation and benchmarking framework for missense pathogenicity prediction*.

**Data DOI:** https://doi.org/10.5281/zenodo.19981867 (Zenodo v1.0.0, published 2026-05-02).

**Scope:** 90,643,830 hg38 missense variants in the study. This is not a claim that every possible input allele or transcript is covered. Missing matches must be reported as **not found**, not benign.

The repository's `Step*.py` files document the *research-generation workflow*. Ordinary lookup users do not need ANNOVAR, dbNSFP, training dependencies, SLURM or a GPU.

Data rights are discussed in [LICENSING.md](LICENSING.md). Do not assume that a licence displayed for this deposit clears the separate permissions of upstream tools and embedded data.
