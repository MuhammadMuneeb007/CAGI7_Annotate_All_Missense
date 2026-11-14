| # | Column | Type | Source |
|---:|---|---|---|
| 1 | `Chr` | VARCHAR | Genomic allele key (ANNOVAR input, GRCh38) |
| 2 | `Start` | BIGINT | Genomic allele key (ANNOVAR input, GRCh38) |
| 3 | `End` | VARCHAR | Genomic allele key (ANNOVAR input, GRCh38) |
| 4 | `Ref` | VARCHAR | Genomic allele key (ANNOVAR input, GRCh38) |
| 5 | `Alt` | VARCHAR | Genomic allele key (ANNOVAR input, GRCh38) |
| 6 | `AFR.sites.2015_08` | VARCHAR | 1000 Genomes (2015-08) |
| 7 | `ALL.sites.2015_08` | VARCHAR | 1000 Genomes (2015-08) |
| 8 | `AMR.sites.2015_08` | VARCHAR | 1000 Genomes (2015-08) |
| 9 | `AlphaMissense` | VARCHAR | AlphaMissense |
| 10 | `EAS.sites.2015_08` | VARCHAR | 1000 Genomes (2015-08) |
| 11 | `EUR.sites.2015_08` | VARCHAR | 1000 Genomes (2015-08) |
| 12 | `SAS.sites.2015_08` | VARCHAR | 1000 Genomes (2015-08) |
| 13 | `avsnp150` | VARCHAR | dbSNP 150 |
| 14 | `Func.ccdsGene` | VARCHAR | ANNOVAR gene-based (ccdsGene) |
| 15 | `Gene.ccdsGene` | VARCHAR | ANNOVAR gene-based (ccdsGene) |
| 16 | `GeneDetail.ccdsGene` | VARCHAR | ANNOVAR gene-based (ccdsGene) |
| 17 | `ExonicFunc.ccdsGene` | VARCHAR | ANNOVAR gene-based (ccdsGene) |
| 18 | `AAChange.ccdsGene` | VARCHAR | ANNOVAR gene-based (ccdsGene) |
| 19 | `CLNALLELEID` | VARCHAR | ClinVar (ANNOVAR filter-based) |
| 20 | `CLNDN` | VARCHAR | ClinVar (ANNOVAR filter-based) |
| 21 | `CLNDISDB` | VARCHAR | ClinVar (ANNOVAR filter-based) |
| 22 | `CLNREVSTAT` | VARCHAR | ClinVar (ANNOVAR filter-based) |
| 23 | `CLNSIG` | VARCHAR | ClinVar (ANNOVAR filter-based) |
| 24 | `cosmic70` | VARCHAR | COSMIC v70 |
| 25 | `cytoBand` | VARCHAR | ANNOVAR region-based |
| 26 | `SIFT_score` | VARCHAR | dbNSFP v4.2a (ANNOVAR filter-based) |
| 27 | `SIFT_converted_rankscore` | VARCHAR | dbNSFP v4.2a (ANNOVAR filter-based) |
| 28 | `SIFT_pred` | VARCHAR | dbNSFP v4.2a (ANNOVAR filter-based) |
| 29 | `SIFT4G_score` | DOUBLE | dbNSFP v4.2a (ANNOVAR filter-based) |
| 30 | `SIFT4G_converted_rankscore` | DOUBLE | dbNSFP v4.2a (ANNOVAR filter-based) |
| 31 | `SIFT4G_pred` | VARCHAR | dbNSFP v4.2a (ANNOVAR filter-based) |
| 32 | `Polyphen2_HDIV_score` | VARCHAR | dbNSFP v4.2a (ANNOVAR filter-based) |
| 33 | `Polyphen2_HDIV_rankscore` | VARCHAR | dbNSFP v4.2a (ANNOVAR filter-based) |
| 34 | `Polyphen2_HDIV_pred` | VARCHAR | dbNSFP v4.2a (ANNOVAR filter-based) |
| 35 | `Polyphen2_HVAR_score` | VARCHAR | dbNSFP v4.2a (ANNOVAR filter-based) |
| 36 | `Polyphen2_HVAR_rankscore` | VARCHAR | dbNSFP v4.2a (ANNOVAR filter-based) |
| 37 | `Polyphen2_HVAR_pred` | VARCHAR | dbNSFP v4.2a (ANNOVAR filter-based) |
| 38 | `LRT_score` | VARCHAR | dbNSFP v4.2a (ANNOVAR filter-based) |
| 39 | `LRT_converted_rankscore` | VARCHAR | dbNSFP v4.2a (ANNOVAR filter-based) |
| 40 | `LRT_pred` | VARCHAR | dbNSFP v4.2a (ANNOVAR filter-based) |
| 41 | `MutationTaster_score` | DOUBLE | dbNSFP v4.2a (ANNOVAR filter-based) |
| 42 | `MutationTaster_converted_rankscore` | DOUBLE | dbNSFP v4.2a (ANNOVAR filter-based) |
| 43 | `MutationTaster_pred` | VARCHAR | dbNSFP v4.2a (ANNOVAR filter-based) |
| 44 | `MutationAssessor_score` | VARCHAR | dbNSFP v4.2a (ANNOVAR filter-based) |
| 45 | `MutationAssessor_rankscore` | VARCHAR | dbNSFP v4.2a (ANNOVAR filter-based) |
| 46 | `MutationAssessor_pred` | VARCHAR | dbNSFP v4.2a (ANNOVAR filter-based) |
| 47 | `FATHMM_score` | VARCHAR | dbNSFP v4.2a (ANNOVAR filter-based) |
| 48 | `FATHMM_converted_rankscore` | VARCHAR | dbNSFP v4.2a (ANNOVAR filter-based) |
| 49 | `FATHMM_pred` | VARCHAR | dbNSFP v4.2a (ANNOVAR filter-based) |
| 50 | `PROVEAN_score` | VARCHAR | dbNSFP v4.2a (ANNOVAR filter-based) |
| 51 | `PROVEAN_converted_rankscore` | VARCHAR | dbNSFP v4.2a (ANNOVAR filter-based) |
| 52 | `PROVEAN_pred` | VARCHAR | dbNSFP v4.2a (ANNOVAR filter-based) |
| 53 | `VEST4_score` | VARCHAR | dbNSFP v4.2a (ANNOVAR filter-based) |
| 54 | `VEST4_rankscore` | VARCHAR | dbNSFP v4.2a (ANNOVAR filter-based) |
| 55 | `MetaSVM_score` | VARCHAR | dbNSFP v4.2a (ANNOVAR filter-based) |
| 56 | `MetaSVM_rankscore` | VARCHAR | dbNSFP v4.2a (ANNOVAR filter-based) |
| 57 | `MetaSVM_pred` | VARCHAR | dbNSFP v4.2a (ANNOVAR filter-based) |
| 58 | `MetaLR_score` | VARCHAR | dbNSFP v4.2a (ANNOVAR filter-based) |
| 59 | `MetaLR_rankscore` | VARCHAR | dbNSFP v4.2a (ANNOVAR filter-based) |
| 60 | `MetaLR_pred` | VARCHAR | dbNSFP v4.2a (ANNOVAR filter-based) |
| 61 | `MetaRNN_score` | VARCHAR | dbNSFP v4.2a (ANNOVAR filter-based) |
| 62 | `MetaRNN_rankscore` | VARCHAR | dbNSFP v4.2a (ANNOVAR filter-based) |
| 63 | `MetaRNN_pred` | VARCHAR | dbNSFP v4.2a (ANNOVAR filter-based) |
| 64 | `M-CAP_score` | VARCHAR | dbNSFP v4.2a (ANNOVAR filter-based) |
| 65 | `M-CAP_rankscore` | VARCHAR | dbNSFP v4.2a (ANNOVAR filter-based) |
| 66 | `M-CAP_pred` | VARCHAR | dbNSFP v4.2a (ANNOVAR filter-based) |
| 67 | `REVEL_score` | VARCHAR | dbNSFP v4.2a (ANNOVAR filter-based) |
| 68 | `REVEL_rankscore` | VARCHAR | dbNSFP v4.2a (ANNOVAR filter-based) |
| 69 | `MutPred_score` | VARCHAR | dbNSFP v4.2a (ANNOVAR filter-based) |
| 70 | `MutPred_rankscore` | VARCHAR | dbNSFP v4.2a (ANNOVAR filter-based) |
| 71 | `MVP_score` | VARCHAR | dbNSFP v4.2a (ANNOVAR filter-based) |
| 72 | `MVP_rankscore` | VARCHAR | dbNSFP v4.2a (ANNOVAR filter-based) |
| 73 | `MPC_score` | VARCHAR | dbNSFP v4.2a (ANNOVAR filter-based) |
| 74 | `MPC_rankscore` | VARCHAR | dbNSFP v4.2a (ANNOVAR filter-based) |
| 75 | `PrimateAI_score` | VARCHAR | dbNSFP v4.2a (ANNOVAR filter-based) |
| 76 | `PrimateAI_rankscore` | VARCHAR | dbNSFP v4.2a (ANNOVAR filter-based) |
| 77 | `PrimateAI_pred` | VARCHAR | dbNSFP v4.2a (ANNOVAR filter-based) |
| 78 | `DEOGEN2_score` | DOUBLE | dbNSFP v4.2a (ANNOVAR filter-based) |
| 79 | `DEOGEN2_rankscore` | DOUBLE | dbNSFP v4.2a (ANNOVAR filter-based) |
| 80 | `DEOGEN2_pred` | VARCHAR | dbNSFP v4.2a (ANNOVAR filter-based) |
| 81 | `BayesDel_addAF_score` | DOUBLE | dbNSFP v4.2a (ANNOVAR filter-based) |
| 82 | `BayesDel_addAF_rankscore` | DOUBLE | dbNSFP v4.2a (ANNOVAR filter-based) |
| 83 | `BayesDel_addAF_pred` | VARCHAR | dbNSFP v4.2a (ANNOVAR filter-based) |
| 84 | `BayesDel_noAF_score` | DOUBLE | dbNSFP v4.2a (ANNOVAR filter-based) |
| 85 | `BayesDel_noAF_rankscore` | DOUBLE | dbNSFP v4.2a (ANNOVAR filter-based) |
| 86 | `BayesDel_noAF_pred` | VARCHAR | dbNSFP v4.2a (ANNOVAR filter-based) |
| 87 | `ClinPred_score` | VARCHAR | dbNSFP v4.2a (ANNOVAR filter-based) |
| 88 | `ClinPred_rankscore` | VARCHAR | dbNSFP v4.2a (ANNOVAR filter-based) |
| 89 | `ClinPred_pred` | VARCHAR | dbNSFP v4.2a (ANNOVAR filter-based) |
| 90 | `LIST-S2_score` | DOUBLE | dbNSFP v4.2a (ANNOVAR filter-based) |
| 91 | `LIST-S2_rankscore` | DOUBLE | dbNSFP v4.2a (ANNOVAR filter-based) |
| 92 | `LIST-S2_pred` | VARCHAR | dbNSFP v4.2a (ANNOVAR filter-based) |
| 93 | `Aloft_pred` | VARCHAR | dbNSFP v4.2a (ANNOVAR filter-based) |
| 94 | `Aloft_Confidence` | VARCHAR | dbNSFP v4.2a (ANNOVAR filter-based) |
| 95 | `CADD_raw` | VARCHAR | dbNSFP v4.2a (ANNOVAR filter-based) |
| 96 | `CADD_raw_rankscore` | VARCHAR | dbNSFP v4.2a (ANNOVAR filter-based) |
| 97 | `CADD_phred` | VARCHAR | dbNSFP v4.2a (ANNOVAR filter-based) |
| 98 | `DANN_score` | VARCHAR | dbNSFP v4.2a (ANNOVAR filter-based) |
| 99 | `DANN_rankscore` | VARCHAR | dbNSFP v4.2a (ANNOVAR filter-based) |
| 100 | `fathmm-MKL_coding_score` | VARCHAR | dbNSFP v4.2a (ANNOVAR filter-based) |
| 101 | `fathmm-MKL_coding_rankscore` | VARCHAR | dbNSFP v4.2a (ANNOVAR filter-based) |
| 102 | `fathmm-MKL_coding_pred` | VARCHAR | dbNSFP v4.2a (ANNOVAR filter-based) |
| 103 | `fathmm-XF_coding_score` | VARCHAR | dbNSFP v4.2a (ANNOVAR filter-based) |
| 104 | `fathmm-XF_coding_rankscore` | VARCHAR | dbNSFP v4.2a (ANNOVAR filter-based) |
| 105 | `fathmm-XF_coding_pred` | VARCHAR | dbNSFP v4.2a (ANNOVAR filter-based) |
| 106 | `Eigen-raw_coding` | VARCHAR | dbNSFP v4.2a (ANNOVAR filter-based) |
| 107 | `Eigen-raw_coding_rankscore` | VARCHAR | dbNSFP v4.2a (ANNOVAR filter-based) |
| 108 | `Eigen-PC-raw_coding` | VARCHAR | dbNSFP v4.2a (ANNOVAR filter-based) |
| 109 | `Eigen-PC-raw_coding_rankscore` | VARCHAR | dbNSFP v4.2a (ANNOVAR filter-based) |
| 110 | `GenoCanyon_score` | VARCHAR | dbNSFP v4.2a (ANNOVAR filter-based) |
| 111 | `GenoCanyon_rankscore` | VARCHAR | dbNSFP v4.2a (ANNOVAR filter-based) |
| 112 | `integrated_fitCons_score` | VARCHAR | dbNSFP v4.2a (ANNOVAR filter-based) |
| 113 | `integrated_fitCons_rankscore` | VARCHAR | dbNSFP v4.2a (ANNOVAR filter-based) |
| 114 | `integrated_confidence_value` | VARCHAR | dbNSFP v4.2a (ANNOVAR filter-based) |
| 115 | `LINSIGHT` | VARCHAR | dbNSFP v4.2a (ANNOVAR filter-based) |
| 116 | `LINSIGHT_rankscore` | VARCHAR | dbNSFP v4.2a (ANNOVAR filter-based) |
| 117 | `GERP++_NR` | DOUBLE | dbNSFP v4.2a (ANNOVAR filter-based) |
| 118 | `GERP++_RS` | DOUBLE | dbNSFP v4.2a (ANNOVAR filter-based) |
| 119 | `GERP++_RS_rankscore` | DOUBLE | dbNSFP v4.2a (ANNOVAR filter-based) |
| 120 | `phyloP100way_vertebrate` | DOUBLE | dbNSFP v4.2a (ANNOVAR filter-based) |
| 121 | `phyloP100way_vertebrate_rankscore` | DOUBLE | dbNSFP v4.2a (ANNOVAR filter-based) |
| 122 | `phyloP30way_mammalian` | DOUBLE | dbNSFP v4.2a (ANNOVAR filter-based) |
| 123 | `phyloP30way_mammalian_rankscore` | DOUBLE | dbNSFP v4.2a (ANNOVAR filter-based) |
| 124 | `phastCons100way_vertebrate` | DOUBLE | dbNSFP v4.2a (ANNOVAR filter-based) |
| 125 | `phastCons100way_vertebrate_rankscore` | DOUBLE | dbNSFP v4.2a (ANNOVAR filter-based) |
| 126 | `phastCons30way_mammalian` | DOUBLE | dbNSFP v4.2a (ANNOVAR filter-based) |
| 127 | `phastCons30way_mammalian_rankscore` | DOUBLE | dbNSFP v4.2a (ANNOVAR filter-based) |
| 128 | `SiPhy_29way_logOdds` | VARCHAR | dbNSFP v4.2a (ANNOVAR filter-based) |
| 129 | `SiPhy_29way_logOdds_rankscore` | VARCHAR | dbNSFP v4.2a (ANNOVAR filter-based) |
| 130 | `Interpro_domain` | VARCHAR | dbNSFP v4.2a (ANNOVAR filter-based) |
| 131 | `GTEx_V8_gene` | VARCHAR | dbNSFP v4.2a (ANNOVAR filter-based) |
| 132 | `GTEx_V8_tissue` | VARCHAR | dbNSFP v4.2a (ANNOVAR filter-based) |
| 133 | `dbnsfp42c_SIFT_score` | VARCHAR | dbNSFP v4.2c (ANNOVAR filter-based) |
| 134 | `dbnsfp42c_SIFT_converted_rankscore` | VARCHAR | dbNSFP v4.2c (ANNOVAR filter-based) |
| 135 | `dbnsfp42c_SIFT_pred` | VARCHAR | dbNSFP v4.2c (ANNOVAR filter-based) |
| 136 | `dbnsfp42c_SIFT4G_score` | DOUBLE | dbNSFP v4.2c (ANNOVAR filter-based) |
| 137 | `dbnsfp42c_SIFT4G_converted_rankscore` | DOUBLE | dbNSFP v4.2c (ANNOVAR filter-based) |
| 138 | `dbnsfp42c_SIFT4G_pred` | VARCHAR | dbNSFP v4.2c (ANNOVAR filter-based) |
| 139 | `dbnsfp42c_LRT_score` | VARCHAR | dbNSFP v4.2c (ANNOVAR filter-based) |
| 140 | `dbnsfp42c_LRT_converted_rankscore` | VARCHAR | dbNSFP v4.2c (ANNOVAR filter-based) |
| 141 | `dbnsfp42c_LRT_pred` | VARCHAR | dbNSFP v4.2c (ANNOVAR filter-based) |
| 142 | `dbnsfp42c_MutationTaster_score` | DOUBLE | dbNSFP v4.2c (ANNOVAR filter-based) |
| 143 | `dbnsfp42c_MutationTaster_converted_rankscore` | DOUBLE | dbNSFP v4.2c (ANNOVAR filter-based) |
| 144 | `dbnsfp42c_MutationTaster_pred` | VARCHAR | dbNSFP v4.2c (ANNOVAR filter-based) |
| 145 | `dbnsfp42c_MutationAssessor_score` | VARCHAR | dbNSFP v4.2c (ANNOVAR filter-based) |
| 146 | `dbnsfp42c_MutationAssessor_rankscore` | VARCHAR | dbNSFP v4.2c (ANNOVAR filter-based) |
| 147 | `dbnsfp42c_MutationAssessor_pred` | VARCHAR | dbNSFP v4.2c (ANNOVAR filter-based) |
| 148 | `dbnsfp42c_FATHMM_score` | VARCHAR | dbNSFP v4.2c (ANNOVAR filter-based) |
| 149 | `dbnsfp42c_FATHMM_converted_rankscore` | VARCHAR | dbNSFP v4.2c (ANNOVAR filter-based) |
| 150 | `dbnsfp42c_FATHMM_pred` | VARCHAR | dbNSFP v4.2c (ANNOVAR filter-based) |
| 151 | `dbnsfp42c_PROVEAN_score` | VARCHAR | dbNSFP v4.2c (ANNOVAR filter-based) |
| 152 | `dbnsfp42c_PROVEAN_converted_rankscore` | VARCHAR | dbNSFP v4.2c (ANNOVAR filter-based) |
| 153 | `dbnsfp42c_PROVEAN_pred` | VARCHAR | dbNSFP v4.2c (ANNOVAR filter-based) |
| 154 | `dbnsfp42c_MetaSVM_score` | VARCHAR | dbNSFP v4.2c (ANNOVAR filter-based) |
| 155 | `dbnsfp42c_MetaSVM_rankscore` | VARCHAR | dbNSFP v4.2c (ANNOVAR filter-based) |
| 156 | `dbnsfp42c_MetaSVM_pred` | VARCHAR | dbNSFP v4.2c (ANNOVAR filter-based) |
| 157 | `dbnsfp42c_MetaLR_score` | VARCHAR | dbNSFP v4.2c (ANNOVAR filter-based) |
| 158 | `dbnsfp42c_MetaLR_rankscore` | VARCHAR | dbNSFP v4.2c (ANNOVAR filter-based) |
| 159 | `dbnsfp42c_MetaLR_pred` | VARCHAR | dbNSFP v4.2c (ANNOVAR filter-based) |
| 160 | `dbnsfp42c_MetaRNN_score` | VARCHAR | dbNSFP v4.2c (ANNOVAR filter-based) |
| 161 | `dbnsfp42c_MetaRNN_rankscore` | VARCHAR | dbNSFP v4.2c (ANNOVAR filter-based) |
| 162 | `dbnsfp42c_MetaRNN_pred` | VARCHAR | dbNSFP v4.2c (ANNOVAR filter-based) |
| 163 | `dbnsfp42c_M-CAP_score` | VARCHAR | dbNSFP v4.2c (ANNOVAR filter-based) |
| 164 | `dbnsfp42c_M-CAP_rankscore` | VARCHAR | dbNSFP v4.2c (ANNOVAR filter-based) |
| 165 | `dbnsfp42c_M-CAP_pred` | VARCHAR | dbNSFP v4.2c (ANNOVAR filter-based) |
| 166 | `dbnsfp42c_MutPred_score` | VARCHAR | dbNSFP v4.2c (ANNOVAR filter-based) |
| 167 | `dbnsfp42c_MutPred_rankscore` | VARCHAR | dbNSFP v4.2c (ANNOVAR filter-based) |
| 168 | `dbnsfp42c_MVP_score` | VARCHAR | dbNSFP v4.2c (ANNOVAR filter-based) |
| 169 | `dbnsfp42c_MVP_rankscore` | VARCHAR | dbNSFP v4.2c (ANNOVAR filter-based) |
| 170 | `dbnsfp42c_MPC_score` | VARCHAR | dbNSFP v4.2c (ANNOVAR filter-based) |
| 171 | `dbnsfp42c_MPC_rankscore` | VARCHAR | dbNSFP v4.2c (ANNOVAR filter-based) |
| 172 | `dbnsfp42c_PrimateAI_score` | VARCHAR | dbNSFP v4.2c (ANNOVAR filter-based) |
| 173 | `dbnsfp42c_PrimateAI_rankscore` | VARCHAR | dbNSFP v4.2c (ANNOVAR filter-based) |
| 174 | `dbnsfp42c_PrimateAI_pred` | VARCHAR | dbNSFP v4.2c (ANNOVAR filter-based) |
| 175 | `dbnsfp42c_DEOGEN2_score` | DOUBLE | dbNSFP v4.2c (ANNOVAR filter-based) |
| 176 | `dbnsfp42c_DEOGEN2_rankscore` | DOUBLE | dbNSFP v4.2c (ANNOVAR filter-based) |
| 177 | `dbnsfp42c_DEOGEN2_pred` | VARCHAR | dbNSFP v4.2c (ANNOVAR filter-based) |
| 178 | `dbnsfp42c_BayesDel_addAF_score` | DOUBLE | dbNSFP v4.2c (ANNOVAR filter-based) |
| 179 | `dbnsfp42c_BayesDel_addAF_rankscore` | DOUBLE | dbNSFP v4.2c (ANNOVAR filter-based) |
| 180 | `dbnsfp42c_BayesDel_addAF_pred` | VARCHAR | dbNSFP v4.2c (ANNOVAR filter-based) |
| 181 | `dbnsfp42c_BayesDel_noAF_score` | DOUBLE | dbNSFP v4.2c (ANNOVAR filter-based) |
| 182 | `dbnsfp42c_BayesDel_noAF_rankscore` | DOUBLE | dbNSFP v4.2c (ANNOVAR filter-based) |
| 183 | `dbnsfp42c_BayesDel_noAF_pred` | VARCHAR | dbNSFP v4.2c (ANNOVAR filter-based) |
| 184 | `dbnsfp42c_ClinPred_score` | VARCHAR | dbNSFP v4.2c (ANNOVAR filter-based) |
| 185 | `dbnsfp42c_ClinPred_rankscore` | VARCHAR | dbNSFP v4.2c (ANNOVAR filter-based) |
| 186 | `dbnsfp42c_ClinPred_pred` | VARCHAR | dbNSFP v4.2c (ANNOVAR filter-based) |
| 187 | `dbnsfp42c_LIST-S2_score` | DOUBLE | dbNSFP v4.2c (ANNOVAR filter-based) |
| 188 | `dbnsfp42c_LIST-S2_rankscore` | DOUBLE | dbNSFP v4.2c (ANNOVAR filter-based) |
| 189 | `dbnsfp42c_LIST-S2_pred` | VARCHAR | dbNSFP v4.2c (ANNOVAR filter-based) |
| 190 | `dbnsfp42c_Aloft_pred` | VARCHAR | dbNSFP v4.2c (ANNOVAR filter-based) |
| 191 | `dbnsfp42c_Aloft_Confidence` | VARCHAR | dbNSFP v4.2c (ANNOVAR filter-based) |
| 192 | `dbnsfp42c_DANN_score` | VARCHAR | dbNSFP v4.2c (ANNOVAR filter-based) |
| 193 | `dbnsfp42c_DANN_rankscore` | VARCHAR | dbNSFP v4.2c (ANNOVAR filter-based) |
| 194 | `dbnsfp42c_fathmm-MKL_coding_score` | VARCHAR | dbNSFP v4.2c (ANNOVAR filter-based) |
| 195 | `dbnsfp42c_fathmm-MKL_coding_rankscore` | VARCHAR | dbNSFP v4.2c (ANNOVAR filter-based) |
| 196 | `dbnsfp42c_fathmm-MKL_coding_pred` | VARCHAR | dbNSFP v4.2c (ANNOVAR filter-based) |
| 197 | `dbnsfp42c_fathmm-XF_coding_score` | VARCHAR | dbNSFP v4.2c (ANNOVAR filter-based) |
| 198 | `dbnsfp42c_fathmm-XF_coding_rankscore` | VARCHAR | dbNSFP v4.2c (ANNOVAR filter-based) |
| 199 | `dbnsfp42c_fathmm-XF_coding_pred` | VARCHAR | dbNSFP v4.2c (ANNOVAR filter-based) |
| 200 | `dbnsfp42c_Eigen-raw_coding` | VARCHAR | dbNSFP v4.2c (ANNOVAR filter-based) |
| 201 | `dbnsfp42c_Eigen-raw_coding_rankscore` | VARCHAR | dbNSFP v4.2c (ANNOVAR filter-based) |
| 202 | `dbnsfp42c_Eigen-PC-raw_coding` | VARCHAR | dbNSFP v4.2c (ANNOVAR filter-based) |
| 203 | `dbnsfp42c_Eigen-PC-raw_coding_rankscore` | VARCHAR | dbNSFP v4.2c (ANNOVAR filter-based) |
| 204 | `dbnsfp42c_integrated_fitCons_score` | VARCHAR | dbNSFP v4.2c (ANNOVAR filter-based) |
| 205 | `dbnsfp42c_integrated_fitCons_rankscore` | VARCHAR | dbNSFP v4.2c (ANNOVAR filter-based) |
| 206 | `dbnsfp42c_integrated_confidence_value` | VARCHAR | dbNSFP v4.2c (ANNOVAR filter-based) |
| 207 | `dbnsfp42c_GERP++_NR` | DOUBLE | dbNSFP v4.2c (ANNOVAR filter-based) |
| 208 | `dbnsfp42c_GERP++_RS` | DOUBLE | dbNSFP v4.2c (ANNOVAR filter-based) |
| 209 | `dbnsfp42c_GERP++_RS_rankscore` | DOUBLE | dbNSFP v4.2c (ANNOVAR filter-based) |
| 210 | `dbnsfp42c_phyloP100way_vertebrate` | DOUBLE | dbNSFP v4.2c (ANNOVAR filter-based) |
| 211 | `dbnsfp42c_phyloP100way_vertebrate_rankscore` | DOUBLE | dbNSFP v4.2c (ANNOVAR filter-based) |
| 212 | `dbnsfp42c_phyloP30way_mammalian` | DOUBLE | dbNSFP v4.2c (ANNOVAR filter-based) |
| 213 | `dbnsfp42c_phyloP30way_mammalian_rankscore` | DOUBLE | dbNSFP v4.2c (ANNOVAR filter-based) |
| 214 | `dbnsfp42c_phastCons100way_vertebrate` | DOUBLE | dbNSFP v4.2c (ANNOVAR filter-based) |
| 215 | `dbnsfp42c_phastCons100way_vertebrate_rankscore` | DOUBLE | dbNSFP v4.2c (ANNOVAR filter-based) |
| 216 | `dbnsfp42c_phastCons30way_mammalian` | DOUBLE | dbNSFP v4.2c (ANNOVAR filter-based) |
| 217 | `dbnsfp42c_phastCons30way_mammalian_rankscore` | DOUBLE | dbNSFP v4.2c (ANNOVAR filter-based) |
| 218 | `dbnsfp42c_SiPhy_29way_logOdds` | VARCHAR | dbNSFP v4.2c (ANNOVAR filter-based) |
| 219 | `dbnsfp42c_SiPhy_29way_logOdds_rankscore` | VARCHAR | dbNSFP v4.2c (ANNOVAR filter-based) |
| 220 | `dbnsfp42c_Interpro_domain` | VARCHAR | dbNSFP v4.2c (ANNOVAR filter-based) |
| 221 | `dbnsfp42c_GTEx_V8_gene` | VARCHAR | dbNSFP v4.2c (ANNOVAR filter-based) |
| 222 | `dbnsfp42c_GTEx_V8_tissue` | VARCHAR | dbNSFP v4.2c (ANNOVAR filter-based) |
| 223 | `dbscSNV_ADA_SCORE` | VARCHAR | dbscSNV splicing |
| 224 | `dbscSNV_RF_SCORE` | VARCHAR | dbscSNV splicing |
| 225 | `Func.ensGene` | VARCHAR | ANNOVAR gene-based (ensGene) |
| 226 | `Gene.ensGene` | VARCHAR | ANNOVAR gene-based (ensGene) |
| 227 | `GeneDetail.ensGene` | VARCHAR | ANNOVAR gene-based (ensGene) |
| 228 | `ExonicFunc.ensGene` | VARCHAR | ANNOVAR gene-based (ensGene) |
| 229 | `AAChange.ensGene` | VARCHAR | ANNOVAR gene-based (ensGene) |
| 230 | `ExAC_ALL` | VARCHAR | ExAC |
| 231 | `ExAC_AFR` | VARCHAR | ExAC |
| 232 | `ExAC_AMR` | VARCHAR | ExAC |
| 233 | `ExAC_EAS` | VARCHAR | ExAC |
| 234 | `ExAC_FIN` | VARCHAR | ExAC |
| 235 | `ExAC_NFE` | VARCHAR | ExAC |
| 236 | `ExAC_OTH` | VARCHAR | ExAC |
| 237 | `ExAC_SAS` | VARCHAR | ExAC |
| 238 | `genomicSuperDups` | VARCHAR | ANNOVAR region-based |
| 239 | `gnomad_constraint` | VARCHAR | gnomAD gene constraint |
| 240 | `AF` | VARCHAR | gnomAD v2.1.1 genome (ANNOVAR gnomad211_genome) |
| 241 | `AF_popmax` | VARCHAR | gnomAD v2.1.1 genome (ANNOVAR gnomad211_genome) |
| 242 | `AF_male` | VARCHAR | gnomAD v2.1.1 genome (ANNOVAR gnomad211_genome) |
| 243 | `AF_female` | VARCHAR | gnomAD v2.1.1 genome (ANNOVAR gnomad211_genome) |
| 244 | `AF_raw` | VARCHAR | gnomAD v2.1.1 genome (ANNOVAR gnomad211_genome) |
| 245 | `AF_afr` | VARCHAR | gnomAD v2.1.1 genome (ANNOVAR gnomad211_genome) |
| 246 | `AF_sas` | VARCHAR | gnomAD v2.1.1 genome (ANNOVAR gnomad211_genome) |
| 247 | `AF_amr` | VARCHAR | gnomAD v2.1.1 genome (ANNOVAR gnomad211_genome) |
| 248 | `AF_eas` | VARCHAR | gnomAD v2.1.1 genome (ANNOVAR gnomad211_genome) |
| 249 | `AF_nfe` | VARCHAR | gnomAD v2.1.1 genome (ANNOVAR gnomad211_genome) |
| 250 | `AF_fin` | VARCHAR | gnomAD v2.1.1 genome (ANNOVAR gnomad211_genome) |
| 251 | `AF_asj` | VARCHAR | gnomAD v2.1.1 genome (ANNOVAR gnomad211_genome) |
| 252 | `AF_oth` | VARCHAR | gnomAD v2.1.1 genome (ANNOVAR gnomad211_genome) |
| 253 | `non_topmed_AF_popmax` | VARCHAR | gnomAD v2.1.1 genome (ANNOVAR gnomad211_genome) |
| 254 | `non_neuro_AF_popmax` | VARCHAR | gnomAD v2.1.1 genome (ANNOVAR gnomad211_genome) |
| 255 | `non_cancer_AF_popmax` | VARCHAR | gnomAD v2.1.1 genome (ANNOVAR gnomad211_genome) |
| 256 | `controls_AF_popmax` | VARCHAR | gnomAD v2.1.1 genome (ANNOVAR gnomad211_genome) |
| 257 | `gnomad211_AF` | VARCHAR | gnomAD v2.1.1 exome (ANNOVAR gnomad211_exome) |
| 258 | `gnomad211_AF_popmax` | VARCHAR | gnomAD v2.1.1 exome (ANNOVAR gnomad211_exome) |
| 259 | `gnomad211_AF_male` | VARCHAR | gnomAD v2.1.1 exome (ANNOVAR gnomad211_exome) |
| 260 | `gnomad211_AF_female` | VARCHAR | gnomAD v2.1.1 exome (ANNOVAR gnomad211_exome) |
| 261 | `gnomad211_AF_raw` | VARCHAR | gnomAD v2.1.1 exome (ANNOVAR gnomad211_exome) |
| 262 | `gnomad211_AF_afr` | VARCHAR | gnomAD v2.1.1 exome (ANNOVAR gnomad211_exome) |
| 263 | `gnomad211_AF_sas` | VARCHAR | gnomAD v2.1.1 exome (ANNOVAR gnomad211_exome) |
| 264 | `gnomad211_AF_amr` | VARCHAR | gnomAD v2.1.1 exome (ANNOVAR gnomad211_exome) |
| 265 | `gnomad211_AF_eas` | VARCHAR | gnomAD v2.1.1 exome (ANNOVAR gnomad211_exome) |
| 266 | `gnomad211_AF_nfe` | VARCHAR | gnomAD v2.1.1 exome (ANNOVAR gnomad211_exome) |
| 267 | `gnomad211_AF_fin` | VARCHAR | gnomAD v2.1.1 exome (ANNOVAR gnomad211_exome) |
| 268 | `gnomad211_AF_asj` | VARCHAR | gnomAD v2.1.1 exome (ANNOVAR gnomad211_exome) |
| 269 | `gnomad211_AF_oth` | VARCHAR | gnomAD v2.1.1 exome (ANNOVAR gnomad211_exome) |
| 270 | `gnomad211_non_topmed_AF_popmax` | VARCHAR | gnomAD v2.1.1 exome (ANNOVAR gnomad211_exome) |
| 271 | `gnomad211_non_neuro_AF_popmax` | VARCHAR | gnomAD v2.1.1 exome (ANNOVAR gnomad211_exome) |
| 272 | `gnomad211_non_cancer_AF_popmax` | VARCHAR | gnomAD v2.1.1 exome (ANNOVAR gnomad211_exome) |
| 273 | `gnomad211_controls_AF_popmax` | VARCHAR | gnomAD v2.1.1 exome (ANNOVAR gnomad211_exome) |
| 274 | `gnomad30_AF` | VARCHAR | gnomAD v3.0 genome (ANNOVAR gnomad30_genome) |
| 275 | `gnomad30_AF_raw` | VARCHAR | gnomAD v3.0 genome (ANNOVAR gnomad30_genome) |
| 276 | `gnomad30_AF_male` | VARCHAR | gnomAD v3.0 genome (ANNOVAR gnomad30_genome) |
| 277 | `gnomad30_AF_female` | VARCHAR | gnomAD v3.0 genome (ANNOVAR gnomad30_genome) |
| 278 | `gnomad30_AF_afr` | VARCHAR | gnomAD v3.0 genome (ANNOVAR gnomad30_genome) |
| 279 | `AF_ami` | VARCHAR | gnomAD v3.0 genome (ANNOVAR gnomad30_genome) |
| 280 | `gnomad30_AF_amr` | VARCHAR | gnomAD v3.0 genome (ANNOVAR gnomad30_genome) |
| 281 | `gnomad30_AF_asj` | VARCHAR | gnomAD v3.0 genome (ANNOVAR gnomad30_genome) |
| 282 | `gnomad30_AF_eas` | VARCHAR | gnomAD v3.0 genome (ANNOVAR gnomad30_genome) |
| 283 | `gnomad30_AF_fin` | VARCHAR | gnomAD v3.0 genome (ANNOVAR gnomad30_genome) |
| 284 | `gnomad30_AF_nfe` | VARCHAR | gnomAD v3.0 genome (ANNOVAR gnomad30_genome) |
| 285 | `gnomad30_AF_oth` | VARCHAR | gnomAD v3.0 genome (ANNOVAR gnomad30_genome) |
| 286 | `gnomad30_AF_sas` | VARCHAR | gnomAD v3.0 genome (ANNOVAR gnomad30_genome) |
| 287 | `Func.knownGene` | VARCHAR | ANNOVAR gene-based (knownGene) |
| 288 | `Gene.knownGene` | VARCHAR | ANNOVAR gene-based (knownGene) |
| 289 | `GeneDetail.knownGene` | VARCHAR | ANNOVAR gene-based (knownGene) |
| 290 | `ExonicFunc.knownGene` | VARCHAR | ANNOVAR gene-based (knownGene) |
| 291 | `AAChange.knownGene` | VARCHAR | ANNOVAR gene-based (knownGene) |
| 292 | `phastCons100way` | VARCHAR | ANNOVAR region-based |
| 293 | `Func.refGene` | VARCHAR | ANNOVAR gene-based (refGene) |
| 294 | `Gene.refGene` | VARCHAR | ANNOVAR gene-based (refGene) |
| 295 | `GeneDetail.refGene` | VARCHAR | ANNOVAR gene-based (refGene) |
| 296 | `ExonicFunc.refGene` | VARCHAR | ANNOVAR gene-based (refGene) |
| 297 | `AAChange.refGene` | VARCHAR | ANNOVAR gene-based (refGene) |
| 298 | `rmsk` | VARCHAR | ANNOVAR region-based |
| 299 | `simpleRepeat` | VARCHAR | ANNOVAR region-based |
| 300 | `aaref` | VARCHAR | dbNSFP v5.1 transcript/protein context (challenge input) |
| 301 | `aaalt` | VARCHAR | dbNSFP v5.1 transcript/protein context (challenge input) |
| 302 | `aapos` | VARCHAR | dbNSFP v5.1 transcript/protein context (challenge input) |
| 303 | `genename` | VARCHAR | dbNSFP v5.1 transcript/protein context (challenge input) |
| 304 | `cds_strand` | VARCHAR | dbNSFP v5.1 transcript/protein context (challenge input) |
| 305 | `refcodon` | VARCHAR | dbNSFP v5.1 transcript/protein context (challenge input) |
| 306 | `codonpos` | VARCHAR | dbNSFP v5.1 transcript/protein context (challenge input) |
| 307 | `Ensembl_geneid` | VARCHAR | dbNSFP v5.1 transcript/protein context (challenge input) |
| 308 | `Ensembl_transcriptid` | VARCHAR | dbNSFP v5.1 transcript/protein context (challenge input) |
| 309 | `Ensembl_proteinid` | VARCHAR | dbNSFP v5.1 transcript/protein context (challenge input) |
| 310 | `hg19_chr` | VARCHAR | dbNSFP v5.1 transcript/protein context (challenge input) |
| 311 | `hg19_pos` | VARCHAR | dbNSFP v5.1 transcript/protein context (challenge input) |
| 312 | `hg18_chr` | VARCHAR | dbNSFP v5.1 transcript/protein context (challenge input) |
| 313 | `hg18_pos` | VARCHAR | dbNSFP v5.1 transcript/protein context (challenge input) |
| 314 | `genome` | VARCHAR | dbNSFP v5.1 transcript/protein context (challenge input) |
| 315 | `uniprot_id` | VARCHAR | AlphaMissense |
| 316 | `transcript_id` | VARCHAR | AlphaMissense |
| 317 | `protein_variant` | VARCHAR | AlphaMissense |
| 318 | `am_pathogenicity` | VARCHAR | AlphaMissense |
| 319 | `am_class` | VARCHAR | AlphaMissense |
| 320 | `GENCODE.v43.ENSP` | VARCHAR | ESM1v variant-effect predictions |
| 321 | `esm1v_t33_650M_UR90S_5_next` | VARCHAR | ESM1v variant-effect predictions |
| 322 | `esm1v_t33_650M_UR90S_3` | VARCHAR | ESM1v variant-effect predictions |
| 323 | `HGVS.p` | VARCHAR | ESM1v variant-effect predictions |
| 324 | `esm1v_t33_650M_UR90S_1` | VARCHAR | ESM1v variant-effect predictions |
| 325 | `esm1v_t33_650M_UR90S_1_next` | VARCHAR | ESM1v variant-effect predictions |
| 326 | `esm1v_t33_650M_UR90S_3_next` | VARCHAR | ESM1v variant-effect predictions |
| 327 | `GENCODE.v43.ENST` | VARCHAR | ESM1v variant-effect predictions |
| 328 | `esm1v_t33_650M_UR90S_4` | VARCHAR | ESM1v variant-effect predictions |
| 329 | `combined_score` | DOUBLE | ESM1v variant-effect predictions |
| 330 | `esm1v_t33_650M_UR90S_2` | VARCHAR | ESM1v variant-effect predictions |
| 331 | `esm1v_t33_650M_UR90S_2_next` | VARCHAR | ESM1v variant-effect predictions |
| 332 | `esm1v_t33_650M_UR90S_5` | VARCHAR | ESM1v variant-effect predictions |
| 333 | `esm1v_t33_650M_UR90S_4_next` | VARCHAR | ESM1v variant-effect predictions |
| 334 | `GENCODE.v43.ENSG` | VARCHAR | ESM1v variant-effect predictions |
| 335 | `UQ_score` | DOUBLE | AnnotateMissense prediction |
| 336 | `UQ_sd` | DOUBLE | AnnotateMissense prediction |
| 337 | `UQ_pred` | VARCHAR | AnnotateMissense prediction |
