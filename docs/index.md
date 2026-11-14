---
title: AnnotateMissense Documentation
layout: default
---

# AnnotateMissense

**Precomputed missense-pathogenicity lookup for human GRCh38/hg38 variants.**

AnnotateMissense provides downloadable precomputed scores and annotations for the variants included in its published genome-wide resource. The released product is **not** a hosted pathogenicity-prediction API, and there is **no supported general-purpose new-VCF scoring command**.

## Start here

1. [Quick start: download, unpack and query](QUICKSTART.md)
2. [DuckDB files, sizes and schema](DATABASE.md)
3. [Transcript and genomic-allele interpretation](TRANSCRIPTS.md)
4. [Dependencies, runtime, hardware and reproducibility](REPRODUCIBILITY.md)
5. [Licensing and third-party usage constraints](LICENSING.md)
6. [Naming, repository and Zenodo release management](RELEASE.md)
7. [Security and credentials](SECURITY.md)

**Dataset:** [Zenodo record 19981867](https://zenodo.org/records/19981867), version 1.0.0.

**Research publication:** *AnnotateMissense: a genome-wide annotation and benchmarking framework for missense pathogenicity prediction*.

**Important limitations:** Scores are for research prioritisation, not standalone clinical classification. The current GitHub documentation does not yet assert an independently verified MANE/canonical transcript-selection rule or verified DuckDB row uniqueness, and licensing of some embedded upstream annotations remains under review.
