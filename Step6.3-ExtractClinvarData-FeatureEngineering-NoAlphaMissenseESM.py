#!/usr/bin/env python3
"""
Complete ClinVar Biological Feature Engineering Pipeline with Unified Prediction Mapping
Preserves all original detailed biological features + adds unified prediction mapping
Usage: python complete_clinvar_pipeline.py [dataset_dir] [--config config.json] [--test-mode]
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
import json
import sys
import logging
import argparse
from typing import Dict, List, Tuple, Optional
from datetime import datetime

# Configure logging for better debugging and tracking
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('feature_engineering.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')

class RowIndependentFeatureEngineer:
    def __init__(self, dataset_dir: str = "Clinvar_Dataset7", config_path: Optional[str] = None):
        """
        Initialize the row-independent feature engineering pipeline.
        
        Args:
            dataset_dir (str): Directory containing fold data (default: Clinvar_Dataset)
            config_path (Optional[str]): Path to configuration file
        """
        self.dataset_dir = Path(dataset_dir)
        logger.info(f"Initializing feature engineer with dataset directory: {self.dataset_dir}")
        
        # Setup unified prediction mapping FIRST
        self.setup_unified_prediction_mapping()
        
        # Setup curated feature lists
        self.setup_curated_features()
        
        # Setup biological parameters and fixed mappings
        self.setup_biological_parameters()
        
        # Load or create feature configuration
        self.feature_config = self.get_default_feature_config()
        if config_path and Path(config_path).exists():
            self.load_config(config_path)
            logger.info(f"Loaded configuration from {config_path}")
        else:
            logger.info("Using default feature configuration")

    def setup_unified_prediction_mapping(self):
        """Setup unified mapping for all prediction values - NEW ADDITION"""
        logger.info("Setting up unified prediction mapping")
        
        self.unified_mapping = {
            # PATHOGENIC CATEGORY (mapped to 1)
            # Standard prediction tool outputs
            'D': 1, 'DAMAGING': 1, 'DELETERIOUS': 1,
            'DISEASE_CAUSING': 1, 'DISEASE_CAUSING_AUTOMATIC': 1, 'A': 1,
            'PROBABLY_DAMAGING': 1, 'HIGH': 1, 'H': 1,
            
            # ClinVar pathogenic classifications
            'PATHOGENIC': 1, 'Pathogenic': 1, 'pathogenic': 1,
            'LIKELY_PATHOGENIC': 1, 'Likely_pathogenic': 1, 'likely_pathogenic': 1,
            'PATHOGENIC/LIKELY_PATHOGENIC': 1, 'Pathogenic/Likely_pathogenic': 1,
            
            # Pathogenic with additional annotations
            'Pathogenic|_other': 1, 'Pathogenic|_risk_factor': 1,
            'Pathogenic|_drug_response': 1, 'Pathogenic|_Affects': 1,
            'Pathogenic|_association': 1, 'Pathogenic|_confers_sensitivity': 1,
            'Pathogenic|_protective': 1, 'Pathogenic/Likely_pathogenic|_other': 1,
            'Pathogenic/Likely_pathogenic|_risk_factor': 1, 'Pathogenic/Likely_pathogenic|_drug_response': 1,
            'Likely_pathogenic|_risk_factor': 1, 'Likely_pathogenic|_drug_response': 1,
            'Likely_pathogenic|_other': 1, 'Likely_pathogenic|_Affects': 1,
            
            # BENIGN CATEGORY (mapped to 0)
            # Standard prediction tool outputs  
            'B': 0, 'BENIGN': 0, 'T': 0, 'TOLERATED': 0,
            'N': 0, 'NEUTRAL': 0, 'POLYMORPHISM': 0, 'P': 0,
            'POLYMORPHISM_AUTOMATIC': 0, 'LOW': 0, 'L': 0,
            
            # ClinVar benign classifications
            'BENIGN': 0, 'Benign': 0, 'benign': 0,
            'LIKELY_BENIGN': 0, 'Likely_benign': 0, 'likely_benign': 0,
            'BENIGN/LIKELY_BENIGN': 0, 'Benign/Likely_benign': 0,
            
            # Benign with additional annotations
            'Benign|_other': 0, 'Benign|_risk_factor': 0, 'Benign|_drug_response': 0,
            'Benign|_confers_sensitivity': 0, 'Benign|_association|_confers_sensitivity': 0,
            'Benign/Likely_benign|_other': 0, 'Benign/Likely_benign|_risk_factor': 0,
            'Benign/Likely_benign|_drug_response': 0, 'Benign/Likely_benign|_drug_response|_other': 0,
            'Benign/Likely_benign|_other|_risk_factor': 0, 'Likely_benign|_other': 0,
            'Likely_benign|_drug_response|_other': 0,
            
            # INTERMEDIATE/UNCERTAIN (mapped to 0.5 for PolyPhen)
            'POSSIBLY_DAMAGING': 0.5, 'MEDIUM': 0.5, 'M': 0.5,
            
            # UNKNOWN/MISSING VALUES (mapped to -1)
            'UNKNOWN': -1, 'Unknown': -1, 'unknown': -1,
            'UNCERTAIN_SIGNIFICANCE': -1, 'uncertain_significance': -1,
            'AMBIGUOUS': -1, 'ambiguous': -1, 'VUS': -1, 'VOUS': -1,
            'U': -1,  # Added: U typically means "Unknown" or "Uncertain"
            '.': -1, '': -1, 'nan': -1, 'NaN': -1, 'NULL': -1, 'null': -1
        }
        
        logger.info(f"Created unified mapping with {len(self.unified_mapping)} entries")

    def setup_curated_features(self) -> None:
        """Setup curated recommended features from high-performance pipeline"""
        logger.info("Setting up curated feature lists")
        
        # COMPUTATIONAL PREDICTION SCORES & PREDICTIONS
        self.computational_prediction_features = [
            'CADD_phred', 'CADD_raw', 'CADD_raw_rankscore',
            'fathmm-MKL_coding_score', 'fathmm-MKL_coding_pred', 'fathmm-MKL_coding_rankscore',
            'fathmm-MKL_coding_score_dbnsfp42c', 'fathmm-MKL_coding_pred_dbnsfp42c', 'fathmm-MKL_coding_rankscore_dbnsfp42c',
            'VEST4_score', 'VEST4_rankscore',
            'SIFT4G_score', 'SIFT4G_pred', 'SIFT4G_converted_rankscore',
            'SIFT4G_score_dbnsfp42c', 'SIFT4G_pred_dbnsfp42c', 'SIFT4G_converted_rankscore_dbnsfp42c',
            'PROVEAN_score', 'PROVEAN_pred', 'PROVEAN_converted_rankscore',
            'PROVEAN_score_dbnsfp42c', 'PROVEAN_pred_dbnsfp42c', 'PROVEAN_converted_rankscore_dbnsfp42c',
            'SIFT_score', 'SIFT_pred', 'SIFT_converted_rankscore',
            'SIFT_score_dbnsfp42c', 'SIFT_pred_dbnsfp42c', 'SIFT_converted_rankscore_dbnsfp42c',
            'fathmm-XF_coding_score', 'fathmm-XF_coding_pred', 'fathmm-XF_coding_rankscore',
            'fathmm-XF_coding_score_dbnsfp42c', 'fathmm-XF_coding_pred_dbnsfp42c', 'fathmm-XF_coding_rankscore_dbnsfp42c',
            'FATHMM_score', 'FATHMM_pred', 'FATHMM_converted_rankscore',
            'FATHMM_score_dbnsfp42c', 'FATHMM_pred_dbnsfp42c', 'FATHMM_converted_rankscore_dbnsfp42c',
            'MetaLR_score', 'MetaLR_pred', 'MetaLR_rankscore',
            'MetaLR_score_dbnsfp42c', 'MetaLR_pred_dbnsfp42c', 'MetaLR_rankscore_dbnsfp42c',
            'MetaSVM_score', 'MetaSVM_pred', 'MetaSVM_rankscore',
            'MetaSVM_score_dbnsfp42c', 'MetaSVM_pred_dbnsfp42c', 'MetaSVM_rankscore_dbnsfp42c',
            'REVEL_score', 'REVEL_rankscore',
            'Polyphen2_HDIV_score', 'Polyphen2_HDIV_pred', 'Polyphen2_HDIV_rankscore',
            'Polyphen2_HVAR_score', 'Polyphen2_HVAR_pred', 'Polyphen2_HVAR_rankscore'
        ]
        
        # EVOLUTIONARY CONSERVATION SCORES
        self.conservation_features = [
            'phastCons100way_vertebrate', 'phastCons100way_vertebrate_rankscore',
            'phastCons100way_vertebrate_dbnsfp42c', 'phastCons100way_vertebrate_rankscore_dbnsfp42c',
            'phyloP100way_vertebrate', 'phyloP100way_vertebrate_rankscore',
            'phyloP100way_vertebrate_dbnsfp42c', 'phyloP100way_vertebrate_rankscore_dbnsfp42c',
            'phastCons30way_mammalian', 'phastCons30way_mammalian_rankscore',
            'phastCons30way_mammalian_dbnsfp42c', 'phastCons30way_mammalian_rankscore_dbnsfp42c',
            'phyloP30way_mammalian', 'phyloP30way_mammalian_rankscore',
            'phyloP30way_mammalian_dbnsfp42c', 'phyloP30way_mammalian_rankscore_dbnsfp42c',
            'GERP++_NR', 'GERP++_NR_dbnsfp42c', 'GERP++_RS', 'GERP++_RS_dbnsfp42c',
            'GERP++_RS_rankscore', 'GERP++_RS_rankscore_dbnsfp42c',
            'SiPhy_29way_logOdds', 'SiPhy_29way_logOdds_rankscore',
            'SiPhy_29way_logOdds_dbnsfp42c', 'SiPhy_29way_logOdds_rankscore_dbnsfp42c'
        ]
        
        # FUNCTIONAL ANNOTATIONS
        self.functional_annotation_features = [
            'Func.ensGene', 'Func.knownGene', 'Func.ccdsGene', 'Func.refGene',
            'ExonicFunc.knownGene', 'ExonicFunc.ensGene', 'ExonicFunc.ccdsGene', 'ExonicFunc.refGene'
        ]
        
        # POPULATION ALLELE FREQUENCIES
        self.population_frequency_features = [
            'BayesDel_addAF_score', 'BayesDel_addAF_pred', 'BayesDel_addAF_rankscore',
            'BayesDel_addAF_score_dbnsfp42c', 'BayesDel_addAF_pred_dbnsfp42c', 'BayesDel_addAF_rankscore_dbnsfp42c',
            'BayesDel_noAF_score', 'BayesDel_noAF_pred', 'BayesDel_noAF_rankscore',
            'BayesDel_noAF_score_dbnsfp42c', 'BayesDel_noAF_pred_dbnsfp42c', 'BayesDel_noAF_rankscore_dbnsfp42c'
        ]
        
        # OTHER IMPORTANT SCORES
        self.other_prediction_features = [
        
            
            'DANN_score', 'DANN_rankscore',  'DANN_score_dbnsfp42c', 'DANN_rankscore_dbnsfp42c',
            'GenoCanyon_score', 'GenoCanyon_rankscore',
            'MutationTaster_score', 'MutationTaster_pred', 'MutationTaster_converted_rankscore',
            'MutationTaster_score_dbnsfp42c', 'MutationTaster_pred_dbnsfp42c', 'MutationTaster_converted_rankscore_dbnsfp42c',
            'MetaRNN_score', 'MetaRNN_pred', 'MetaRNN_rankscore',
            'MetaRNN_score_dbnsfp42c', 'MetaRNN_pred_dbnsfp42c', 'MetaRNN_rankscore_dbnsfp42c',
            'LIST-S2_score', 'LIST-S2_pred', 'LIST-S2_rankscore',
            'LIST-S2_score_dbnsfp42c', 'LIST-S2_pred_dbnsfp42c', 'LIST-S2_rankscore_dbnsfp42c',
            'MVP_score', 'MVP_rankscore', 'MVP_score_dbnsfp42c', 'MVP_rankscore_dbnsfp42c',
            'integrated_fitCons_score', 'integrated_fitCons_rankscore', 'integrated_confidence_value',
            'integrated_fitCons_score_dbnsfp42c', 'integrated_fitCons_rankscore_dbnsfp42c', 'integrated_confidence_value_dbnsfp42c',
            'DEOGEN2_score', 'DEOGEN2_pred', 'DEOGEN2_rankscore',
            'DEOGEN2_score_dbnsfp42c', 'DEOGEN2_pred_dbnsfp42c', 'DEOGEN2_rankscore_dbnsfp42c',
 
            'M-CAP_score', 'M-CAP_pred', 'M-CAP_rankscore',
            'M-CAP_score_dbnsfp42c', 'M-CAP_pred_dbnsfp42c', 'M-CAP_rankscore_dbnsfp42c',
            'ClinPred_score', 'ClinPred_pred', 'ClinPred_rankscore',
            'ClinPred_score_dbnsfp42c', 'ClinPred_pred_dbnsfp42c', 'ClinPred_rankscore_dbnsfp42c',
            'Eigen-raw_coding', 'Eigen-raw_coding_rankscore', 'Eigen-raw_coding_dbnsfp42c', 'Eigen-raw_coding_rankscore_dbnsfp42c',
            'Eigen-PC-raw_coding', 'Eigen-PC-raw_coding_rankscore', 'Eigen-PC-raw_coding_dbnsfp42c', 'Eigen-PC-raw_coding_rankscore_dbnsfp42c',
            'PrimateAI_score', 'PrimateAI_pred', 'PrimateAI_rankscore',
            'PrimateAI_score_dbnsfp42c', 'PrimateAI_pred_dbnsfp42c', 'PrimateAI_rankscore_dbnsfp42c',
            'MutPred_score', 'MutPred_rankscore', 'MutPred_score_dbnsfp42c', 'MutPred_rankscore_dbnsfp42c',
            'LRT_score', 'LRT_pred', 'LRT_converted_rankscore',
            'LRT_score_dbnsfp42c', 'LRT_pred_dbnsfp42c', 'LRT_converted_rankscore_dbnsfp42c',
            'MutationAssessor_score', 'MutationAssessor_pred', 'MutationAssessor_rankscore',
            'MutationAssessor_score_dbnsfp42c', 'MutationAssessor_pred_dbnsfp42c', 'MutationAssessor_rankscore_dbnsfp42c',
            'MPC_score', 'MPC_rankscore', 'MPC_score_dbnsfp42c', 'MPC_rankscore_dbnsfp42c'
        ]
        
        # CATEGORICAL FEATURES
        self.categorical_features = [
            'cytoBand', 'Aloft_pred', 'Aloft_Confidence', 'Aloft_pred_dbnsfp42c', 'Aloft_Confidence_dbnsfp42c',
            'Func.ensGene', 'Func.knownGene', 'Func.ccdsGene', 'Func.refGene',
            'ExonicFunc.knownGene', 'ExonicFunc.ensGene', 'ExonicFunc.ccdsGene', 'ExonicFunc.refGene'
        ]
        
        # BIOLOGICAL FEATURES FOR ENGINEERING
        self.biological_raw_features = [
            'aaref_merged', 'aaalt_merged', 'Ref', 'Alt', 'refcodon_merged'
        ]

    def get_default_feature_config(self) -> Dict:
        """Return default feature configuration with row-independent toggles"""
        logger.info("Generating default feature configuration")
        return {
            # Base feature categories
            "use_computational_predictions": True,
            "use_conservation_scores": True,
            "use_functional_annotations": True,
            "use_population_frequencies": True,
            "use_other_predictions": True,
            "use_categorical_features": True,
            "use_biological_raw": True,
            
            # NEW: Unified prediction mapping
            "use_unified_mapping": True,
            
            # Row-independent engineered features
            "amino_acid_features": {
                "enabled": True,
                "blosum62_score": True,
                "blosum62_categories": True,
                "aa_property_changes": True,
                "aa_chemical_classes": True,
                "aa_size_changes": True,
                "aa_structural_impact": True,
                "aa_polarity_change": True,
                "aa_charge_change": True
            },
            
            "codon_features": {
                "enabled": True,
                "codon_gc_content": True,
                "codon_cpg_content": True,
                "codon_degeneracy": True,
                "wobble_position": True,
                "codon_optimality": True,
                "start_stop_codons": True,
                "codon_bias_score": True,
                "codon_position_impact": True
            },
            
            "nucleotide_features": {
                "enabled": True,
                "transition_transversion": True,
                "purine_pyrimidine": True,
                "gc_content_change": True,
                "cpg_context": True,
                "base_composition": True,
                "nucleotide_complexity": True,
                "dinucleotide_change": True
            },
            
            "prediction_features": {
                "enabled": True,
                "severity_encoding": True,
                "consensus_voting": True,
                "tool_agreement": True,
                "prediction_confidence": True,
                "prediction_spread": True
            },
            
            "interaction_features": {
                "enabled": True,
                "score_ratios": True,
                "score_products": True,
                "conservation_prediction_products": True,
                "pairwise_interactions": True,
                "score_differences": True,
                "combined_scores": True
            },
            
            "categorical_encoding": {
                "enabled": True,
                "fixed_mapping": True
            },
            
            "missing_value_handling": {
                "enabled": True,
                "use_fixed_defaults": True,
                "numerical_default": 0,
                "categorical_default": "Unknown"
            }
        }

    def setup_biological_parameters(self) -> None:
        """Setup biological parameters and fixed mappings for deterministic feature engineering"""
        logger.info("Setting up biological parameters and fixed mappings")
        
        # FULL BLOSUM62 substitution matrix (all original entries preserved)
        self.blosum62 = {
            'AA': 4, 'AR': -1, 'AN': -2, 'AD': -2, 'AC': 0, 'AQ': -1, 'AE': -1, 'AG': 0, 'AH': -2, 'AI': -1, 'AL': -1, 'AK': -1, 'AM': -1, 'AF': -2, 'AP': -1, 'AS': 1, 'AT': 0, 'AW': -3, 'AY': -2, 'AV': 0,
            'RA': -1, 'RR': 5, 'RN': 0, 'RD': -2, 'RC': -3, 'RQ': 1, 'RE': 0, 'RG': -2, 'RH': 0, 'RI': -3, 'RL': -2, 'RK': 2, 'RM': -1, 'RF': -3, 'RP': -2, 'RS': -1, 'RT': -1, 'RW': -3, 'RY': -2, 'RV': -3,
            'NA': -2, 'NR': 0, 'NN': 6, 'ND': 1, 'NC': -3, 'NQ': 0, 'NE': 0, 'NG': 0, 'NH': 1, 'NI': -3, 'NL': -3, 'NK': 0, 'NM': -2, 'NF': -3, 'NP': -2, 'NS': 1, 'NT': 0, 'NW': -4, 'NY': -2, 'NV': -3,
            'DA': -2, 'DR': -2, 'DN': 1, 'DD': 6, 'DC': -3, 'DQ': 0, 'DE': 2, 'DG': -1, 'DH': -1, 'DI': -3, 'DL': -4, 'DK': -1, 'DM': -3, 'DF': -3, 'DP': -1, 'DS': 0, 'DT': -1, 'DW': -4, 'DY': -3, 'DV': -3,
            'CA': 0, 'CR': -3, 'CN': -3, 'CD': -3, 'CC': 9, 'CQ': -3, 'CE': -4, 'CG': -3, 'CH': -3, 'CI': -1, 'CL': -1, 'CK': -3, 'CM': -1, 'CF': -2, 'CP': -3, 'CS': -1, 'CT': -1, 'CW': -2, 'CY': -2, 'CV': -1,
            'QA': -1, 'QR': 1, 'QN': 0, 'QD': 0, 'QC': -3, 'QQ': 5, 'QE': 2, 'QG': -2, 'QH': 0, 'QI': -3, 'QL': -2, 'QK': 1, 'QM': 0, 'QF': -3, 'QP': -1, 'QS': 0, 'QT': -1, 'QW': -2, 'QY': -1, 'QV': -2,
            'EA': -1, 'ER': 0, 'EN': 0, 'ED': 2, 'EC': -4, 'EQ': 2, 'EE': 5, 'EG': -2, 'EH': 0, 'EI': -3, 'EL': -3, 'EK': 1, 'EM': -2, 'EF': -3, 'EP': -1, 'ES': 0, 'ET': -1, 'EW': -3, 'EY': -2, 'EV': -2,
            'GA': 0, 'GR': -2, 'GN': 0, 'GD': -1, 'GC': -3, 'GQ': -2, 'GE': -2, 'GG': 6, 'GH': -2, 'GI': -4, 'GL': -4, 'GK': -2, 'GM': -3, 'GF': -3, 'GP': -2, 'GS': 0, 'GT': -2, 'GW': -2, 'GY': -3, 'GV': -3,
            'HA': -2, 'HR': 0, 'HN': 1, 'HD': -1, 'HC': -3, 'HQ': 0, 'HE': 0, 'HG': -2, 'HH': 8, 'HI': -3, 'HL': -3, 'HK': -1, 'HM': -2, 'HF': -1, 'HP': -2, 'HS': -1, 'HT': -2, 'HW': -2, 'HY': 2, 'HV': -3,
            'IA': -1, 'IR': -3, 'IN': -3, 'ID': -3, 'IC': -1, 'IQ': -3, 'IE': -3, 'IG': -4, 'IH': -3, 'II': 4, 'IL': 2, 'IK': -3, 'IM': 1, 'IF': 0, 'IP': -3, 'IS': -2, 'IT': -1, 'IW': -3, 'IY': -1, 'IV': 3,
            'LA': -1, 'LR': -2, 'LN': -3, 'LD': -4, 'LC': -1, 'LQ': -2, 'LE': -3, 'LG': -4, 'LH': -3, 'LI': 2, 'LL': 4, 'LK': -2, 'LM': 2, 'LF': 0, 'LP': -3, 'LS': -2, 'LT': -1, 'LW': -2, 'LY': -1, 'LV': 1,
            'KA': -1, 'KR': 2, 'KN': 0, 'KD': -1, 'KC': -3, 'KQ': 1, 'KE': 1, 'KG': -2, 'KH': -1, 'KI': -3, 'KL': -2, 'KK': 5, 'KM': -1, 'KF': -3, 'KP': -1, 'KS': 0, 'KT': -1, 'KW': -3, 'KY': -2, 'KV': -2,
            'MA': -1, 'MR': -1, 'MN': -2, 'MD': -3, 'MC': -1, 'MQ': 0, 'ME': -2, 'MG': -3, 'MH': -2, 'MI': 1, 'ML': 2, 'MK': -1, 'MM': 5, 'MF': 0, 'MP': -2, 'MS': -1, 'MT': -1, 'MW': -1, 'MY': -1, 'MV': 1,
            'FA': -2, 'FR': -3, 'FN': -3, 'FD': -3, 'FC': -2, 'FQ': -3, 'FE': -3, 'FG': -3, 'FH': -1, 'FI': 0, 'FL': 0, 'FK': -3, 'FM': 0, 'FF': 6, 'FP': -4, 'FS': -2, 'FT': -2, 'FW': 1, 'FY': 3, 'FV': -1,
            'PA': -1, 'PR': -2, 'PN': -2, 'PD': -1, 'PC': -3, 'PQ': -1, 'PE': -1, 'PG': -2, 'PH': -2, 'PI': -3, 'PL': -3, 'PK': -1, 'PM': -2, 'PF': -4, 'PP': 7, 'PS': -1, 'PT': -1, 'PW': -4, 'PY': -3, 'PV': -2,
            'SA': 1, 'SR': -1, 'SN': 1, 'SD': 0, 'SC': -1, 'SQ': 0, 'SE': 0, 'SG': 0, 'SH': -1, 'SI': -2, 'SL': -2, 'SK': 0, 'SM': -1, 'SF': -2, 'SP': -1, 'SS': 4, 'ST': 1, 'SW': -3, 'SY': -2, 'SV': -2,
            'TA': 0, 'TR': -1, 'TN': 0, 'TD': -1, 'TC': -1, 'TQ': -1, 'TE': -1, 'TG': -2, 'TH': -2, 'TI': -1, 'TL': -1, 'TK': -1, 'TM': -1, 'TF': -2, 'TP': -1, 'TS': 1, 'TT': 5, 'TW': -2, 'TY': -2, 'TV': 0,
            'WA': -3, 'WR': -3, 'WN': -4, 'WD': -4, 'WC': -2, 'WQ': -2, 'WE': -3, 'WG': -2, 'WH': -2, 'WI': -3, 'WL': -2, 'WK': -3, 'WM': -1, 'WF': 1, 'WP': -4, 'WS': -3, 'WT': -2, 'WW': 11, 'WY': 2, 'WV': -3,
            'YA': -2, 'YR': -2, 'YN': -2, 'YD': -3, 'YC': -2, 'YQ': -1, 'YE': -2, 'YG': -3, 'YH': 2, 'YI': -1, 'YL': -1, 'YK': -2, 'YM': -1, 'YF': 3, 'YP': -3, 'YS': -2, 'YT': -2, 'YW': 2, 'YY': 7, 'YV': -1,
            'VA': 0, 'VR': -3, 'VN': -3, 'VD': -3, 'VC': -1, 'VQ': -2, 'VE': -2, 'VG': -3, 'VH': -3, 'VI': 3, 'VL': 1, 'VK': -2, 'VM': 1, 'VF': -1, 'VP': -2, 'VS': -2, 'VT': 0, 'VW': -3, 'VY': -1, 'VV': 4
        }
        
        # Amino acid properties (full original set preserved)
        self.aa_properties = {
            'A': {'hydrophobic': True, 'polar': False, 'charged': False, 'aromatic': False, 'size': 1, 'polarity': 'nonpolar', 'charge': 'neutral'},
            'R': {'hydrophobic': False, 'polar': True, 'charged': True, 'aromatic': False, 'size': 4, 'polarity': 'polar', 'charge': 'positive'},
            'N': {'hydrophobic': False, 'polar': True, 'charged': False, 'aromatic': False, 'size': 2, 'polarity': 'polar', 'charge': 'neutral'},
            'D': {'hydrophobic': False, 'polar': True, 'charged': True, 'aromatic': False, 'size': 2, 'polarity': 'polar', 'charge': 'negative'},
            'C': {'hydrophobic': True, 'polar': False, 'charged': False, 'aromatic': False, 'size': 1, 'polarity': 'nonpolar', 'charge': 'neutral'},
            'Q': {'hydrophobic': False, 'polar': True, 'charged': False, 'aromatic': False, 'size': 3, 'polarity': 'polar', 'charge': 'neutral'},
            'E': {'hydrophobic': False, 'polar': True, 'charged': True, 'aromatic': False, 'size': 3, 'polarity': 'polar', 'charge': 'negative'},
            'G': {'hydrophobic': False, 'polar': False, 'charged': False, 'aromatic': False, 'size': 1, 'polarity': 'nonpolar', 'charge': 'neutral'},
            'H': {'hydrophobic': False, 'polar': True, 'charged': True, 'aromatic': True, 'size': 3, 'polarity': 'polar', 'charge': 'positive'},
            'I': {'hydrophobic': True, 'polar': False, 'charged': False, 'aromatic': False, 'size': 3, 'polarity': 'nonpolar', 'charge': 'neutral'},
            'L': {'hydrophobic': True, 'polar': False, 'charged': False, 'aromatic': False, 'size': 3, 'polarity': 'nonpolar', 'charge': 'neutral'},
            'K': {'hydrophobic': False, 'polar': True, 'charged': True, 'aromatic': False, 'size': 3, 'polarity': 'polar', 'charge': 'positive'},
            'M': {'hydrophobic': True, 'polar': False, 'charged': False, 'aromatic': False, 'size': 3, 'polarity': 'nonpolar', 'charge': 'neutral'},
            'F': {'hydrophobic': True, 'polar': False, 'charged': False, 'aromatic': True, 'size': 3, 'polarity': 'nonpolar', 'charge': 'neutral'},
            'P': {'hydrophobic': True, 'polar': False, 'charged': False, 'aromatic': False, 'size': 2, 'polarity': 'nonpolar', 'charge': 'neutral'},
            'S': {'hydrophobic': False, 'polar': True, 'charged': False, 'aromatic': False, 'size': 1, 'polarity': 'polar', 'charge': 'neutral'},
            'T': {'hydrophobic': False, 'polar': True, 'charged': False, 'aromatic': False, 'size': 2, 'polarity': 'polar', 'charge': 'neutral'},
            'W': {'hydrophobic': True, 'polar': False, 'charged': False, 'aromatic': True, 'size': 4, 'polarity': 'nonpolar', 'charge': 'neutral'},
            'Y': {'hydrophobic': True, 'polar': True, 'charged': False, 'aromatic': True, 'size': 3, 'polarity': 'polar', 'charge': 'neutral'},
            'V': {'hydrophobic': True, 'polar': False, 'charged': False, 'aromatic': False, 'size': 2, 'polarity': 'nonpolar', 'charge': 'neutral'}
        }
        
        # Codon properties (all original entries preserved)
        self.optimal_codons = {'TTT', 'TTC', 'TTA', 'TTG', 'TCT', 'TCC', 'TCA', 'TCG', 'TAT', 'TAC'}
        self.start_codons = {'ATG'}
        self.stop_codons = {'TAA', 'TAG', 'TGA'}
        self.degenerate_codons = {
            'TTT': 2, 'TTC': 2, 'TTA': 4, 'TTG': 4, 'TCT': 4, 'TCC': 4, 'TCA': 4, 'TCG': 4,
            'TAT': 2, 'TAC': 2, 'TAA': 3, 'TAG': 3, 'TGA': 3, 'TGT': 2, 'TGC': 2, 'TGG': 1,
            'CTT': 4, 'CTC': 4, 'CTA': 4, 'CTG': 4, 'CCT': 4, 'CCC': 4, 'CCA': 4, 'CCG': 4,
            'CAT': 2, 'CAC': 2, 'CAA': 2, 'CAG': 2, 'CGT': 4, 'CGC': 4, 'CGA': 4, 'CGG': 4,
            'ATT': 3, 'ATC': 3, 'ATA': 3, 'ATG': 1, 'ACT': 4, 'ACC': 4, 'ACA': 4, 'ACG': 4,
            'AAT': 2, 'AAC': 2, 'AAA': 2, 'AAG': 2, 'AGT': 2, 'AGC': 2, 'AGA': 4, 'AGG': 4,
            'GTT': 4, 'GTC': 4, 'GTA': 4, 'GTG': 4, 'GCT': 4, 'GCC': 4, 'GCA': 4, 'GCG': 4,
            'GAT': 2, 'GAC': 2, 'GAA': 2, 'GAG': 2, 'GGT': 4, 'GGC': 4, 'GGA': 4, 'GGG': 4
        }
        
        # Nucleotide classifications (all original entries preserved)
        self.transitions = {('A', 'G'), ('G', 'A'), ('C', 'T'), ('T', 'C')}
        self.purines = {'A', 'G'}
        self.pyrimidines = {'C', 'T'}
        
        # Prediction tool severity mapping (keeping original mapping values)
        self.severity_mapping = {
            'D': 2, 'DAMAGING': 2, 'DELETERIOUS': 2,
            'P': 1, 'PROBABLY_DAMAGING': 1, 'POSSIBLY_DAMAGING': 1,
            'T': 0, 'TOLERATED': 0,
            'B': 0, 'BENIGN': 0, 'NEUTRAL': 0,
            'N': 0, 'UNKNOWN': 0, '.': 0, '': 0, 'nan': 0,
            'H': 3, 'HIGH': 3, 'M': 2, 'MEDIUM': 2, 'L': 1, 'LOW': 1
        }
        
        # Fixed categorical mappings (keeping all original mappings)
        self.categorical_mappings = {
            'cytoBand': {
                'p': 0, 'q': 1, 'Unknown': 0, '.': 0, '': 0, 'nan': 0
            },
            'Aloft_pred': {
                'D': 1, 'T': 0, 'N': 0, 'B': 0, '.': 0, 'Unknown': 0, '': 0, 'nan': 0
            },
            'Aloft_Confidence': {
                'High': 2, 'Medium': 1, 'Low': 0, 'Unknown': 0, '.': 0, '': 0, 'nan': 0
            },
            'Aloft_pred_dbnsfp42c': {
                'D': 1, 'T': 0, 'N': 0, 'B': 0, '.': 0, 'Unknown': 0, '': 0, 'nan': 0
            },
            'Aloft_Confidence_dbnsfp42c': {
                'High': 2, 'Medium': 1, 'Low': 0, 'Unknown': 0, '.': 0, '': 0, 'nan': 0
            },
            'Func.ensGene': {
                'exonic': 2, 'intronic': 0, 'intergenic': 0, 'UTR5': 1, 'UTR3': 1, 
                'Unknown': 0, '.': 0, '': 0, 'nan': 0, 'splicing': 1
            },
            'Func.knownGene': {
                'exonic': 2, 'intronic': 0, 'intergenic': 0, 'UTR5': 1, 'UTR3': 1, 
                'Unknown': 0, '.': 0, '': 0, 'nan': 0, 'splicing': 1
            },
            'Func.ccdsGene': {
                'exonic': 2, 'intronic': 0, 'intergenic': 0, 'UTR5': 1, 'UTR3': 1, 
                'Unknown': 0, '.': 0, '': 0, 'nan': 0, 'splicing': 1
            },
            'Func.refGene': {
                'exonic': 2, 'intronic': 0, 'intergenic': 0, 'UTR5': 1, 'UTR3': 1, 
                'Unknown': 0, '.': 0, '': 0, 'nan': 0, 'splicing': 1
            },
            'ExonicFunc.ensGene': {
                'nonsynonymous_SNV': 2, 'synonymous_SNV': 1, 'stopgain': 3, 'stoploss': 3, 
                'frameshift_deletion': 3, 'frameshift_insertion': 3, 'nonframeshift_deletion': 2, 
                'nonframeshift_insertion': 2, 'Unknown': 0, '.': 0, '': 0, 'nan': 0
            },
            'ExonicFunc.knownGene': {
                'nonsynonymous_SNV': 2, 'synonymous_SNV': 1, 'stopgain': 3, 'stoploss': 3, 
                'frameshift_deletion': 3, 'frameshift_insertion': 3, 'nonframeshift_deletion': 2, 
                'nonframeshift_insertion': 2, 'Unknown': 0, '.': 0, '': 0, 'nan': 0
            },
            'ExonicFunc.ccdsGene': {
                'nonsynonymous_SNV': 2, 'synonymous_SNV': 1, 'stopgain': 3, 'stoploss': 3, 
                'frameshift_deletion': 3, 'frameshift_insertion': 3, 'nonframeshift_deletion': 2, 
                'nonframeshift_insertion': 2, 'Unknown': 0, '.': 0, '': 0, 'nan': 0
            },
            'ExonicFunc.refGene': {
                'nonsynonymous_SNV': 2, 'synonymous_SNV': 1, 'stopgain': 3, 'stoploss': 3, 
                'frameshift_deletion': 3, 'frameshift_insertion': 3, 'nonframeshift_deletion': 2, 
                'nonframeshift_insertion': 2, 'Unknown': 0, '.': 0, '': 0, 'nan': 0
            }
        }

    # NEW METHOD: Get prediction columns for unified mapping
    def get_prediction_columns(self, df):
        """Identify CATEGORICAL prediction columns for unified mapping (not numeric scores)"""
        prediction_suffixes = ['_pred', '_class', '_prediction']
        prediction_keywords = ['pred', 'class', 'pathogenicity']
        
        # Keywords that indicate numeric scores (should NOT be mapped)
        score_keywords = ['score', 'rankscore', 'phred', 'raw']
        
        pred_cols = []
        for col in df.columns:
            col_lower = col.lower()
            
            # Skip if it's clearly a numeric score column
            if any(score_kw in col_lower for score_kw in score_keywords):
                continue
                
            # Check if column contains categorical predictions
            if any(col_lower.endswith(suffix) for suffix in prediction_suffixes):
                # Additional check: is this actually categorical data?
                if self._is_categorical_prediction_column(df[col]):
                    pred_cols.append(col)
            elif any(keyword in col_lower for keyword in prediction_keywords):
                if self._is_categorical_prediction_column(df[col]):
                    pred_cols.append(col)
        
        logger.info(f"Found {len(pred_cols)} categorical prediction columns for unified mapping: {pred_cols}")
        return pred_cols

    def _is_categorical_prediction_column(self, series):
        """Check if a column contains categorical predictions rather than numeric scores"""
        # Get non-null unique values
        unique_vals = series.dropna().unique()
        
        if len(unique_vals) == 0:
            return False
            
        # Check if values look like categorical predictions
        categorical_indicators = [
            'D', 'T', 'B', 'N', 'P', 'H', 'M', 'L',  # Single letter codes
            'DAMAGING', 'TOLERATED', 'BENIGN', 'NEUTRAL', 'PATHOGENIC',  # Full words
            'DELETERIOUS', 'PROBABLY_DAMAGING', 'POSSIBLY_DAMAGING',
            'DISEASE_CAUSING', 'POLYMORPHISM', 'HIGH', 'MEDIUM', 'LOW',
            'Pathogenic', 'Benign', 'Likely_pathogenic', 'Likely_benign',
            'pathogenic', 'benign', 'likely_pathogenic', 'likely_benign'
        ]
        
        # Convert to strings and check
        str_vals = [str(val).strip() for val in unique_vals[:20]]  # Check first 20 unique values
        
        # If any values match categorical indicators, it's likely categorical
        matches = sum(1 for val in str_vals if val in categorical_indicators)
        
        # If more than 20% of unique values are categorical indicators, treat as categorical
        is_categorical = (matches / len(str_vals)) > 0.2
        
        logger.debug(f"Column categorical check: {matches}/{len(str_vals)} matches, is_categorical: {is_categorical}")
        return is_categorical

    # UPDATED METHOD: Apply unified mapping with proper type checking
    def apply_unified_mapping(self, value):
        """Apply unified mapping to a single prediction value"""
        if pd.isna(value):
            return -1
        
        # Convert to string for mapping
        value_str = str(value).strip()
        
        # Direct mapping
        if value_str in self.unified_mapping:
            return self.unified_mapping[value_str]
        
        # Case-insensitive fallback
        value_upper = value_str.upper()
        if value_upper in self.unified_mapping:
            return self.unified_mapping[value_upper]
        
        # Check if it's a numeric value that shouldn't be mapped
        try:
            float_val = float(value_str)
            # If it's a number between 0 and 1, it's likely a score, not a prediction
            if 0 <= float_val <= 1:
                logger.warning(f"Numeric value '{value_str}' found in prediction column - this may be a score column")
                return -1  # Treat as unknown since it's likely misclassified
        except ValueError:
            pass  # Not a number, continue with string mapping
        
        # Default to unknown
        logger.warning(f"No unified mapping found for '{value_str}' - assigning -1 (unknown)")
        return -1

    # UPDATED METHOD: Transform predictions using unified mapping with better filtering
    def transform_predictions_unified(self, X, test_mode=False):
        """Apply unified prediction mapping to categorical prediction columns only"""
        logger.info("Applying unified prediction mapping")
        X_work = X.copy()
        
        pred_cols = self.get_prediction_columns(X_work)
        
        if not pred_cols:
            logger.warning("No categorical prediction columns found for unified mapping")
            return X_work
        
        for col in pred_cols:
            logger.info(f"Applying unified mapping to column: {col}")
            
            if test_mode:
                # Show before/after mapping for test mode
                unique_before = X_work[col].value_counts().head(10)
                logger.info(f"Values in {col} before mapping:\n{unique_before}")
            
            # Double-check this is actually categorical before mapping
            if self._is_categorical_prediction_column(X_work[col]):
                # Apply unified mapping
                X_work[col] = X_work[col].apply(self.apply_unified_mapping)
                
                if test_mode:
                    unique_after = X_work[col].value_counts()
                    logger.info(f"Values in {col} after mapping:\n{unique_after}")
            else:
                logger.info(f"Skipping {col} - appears to contain numeric scores, not categorical predictions")
        
        return X_work

    def get_enabled_base_features(self, available_columns: List[str]) -> List[str]:
        """Get list of enabled base features that are available"""
        logger.info("Filtering enabled base features")
        enabled_features = []
        
        if self.feature_config["use_computational_predictions"]:
            enabled_features.extend([f for f in self.computational_prediction_features if f in available_columns])
        
        if self.feature_config["use_conservation_scores"]:
            enabled_features.extend([f for f in self.conservation_features if f in available_columns])
        
        if self.feature_config["use_functional_annotations"]:
            enabled_features.extend([f for f in self.functional_annotation_features if f in available_columns])
        
        if self.feature_config["use_population_frequencies"]:
            enabled_features.extend([f for f in self.population_frequency_features if f in available_columns])
        
        if self.feature_config["use_other_predictions"]:
            enabled_features.extend([f for f in self.other_prediction_features if f in available_columns])
        
        if self.feature_config["use_categorical_features"]:
            enabled_features.extend([f for f in self.categorical_features if f in available_columns])
        
        if self.feature_config["use_biological_raw"]:
            enabled_features.extend([f for f in self.biological_raw_features if f in available_columns])
        
        logger.info(f"Selected {len(enabled_features)} base features")
        return enabled_features

    def validate_input_data(self, X: pd.DataFrame) -> bool:
        """Validate input DataFrame for required columns and data types"""
        logger.info("Validating input data")
        required_cols = ['aaref_merged', 'aaalt_merged', 'Ref', 'Alt', 'refcodon_merged']
        missing_cols = [col for col in required_cols if col not in X.columns]
        
        if missing_cols:
            logger.warning(f"Missing required columns: {missing_cols}")
            return False
        
        logger.info("Input data validation passed")
        return True

    def create_amino_acid_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create deterministic amino acid features - ROW INDEPENDENT"""
        if not self.feature_config["amino_acid_features"]["enabled"]:
            logger.info("Amino acid features disabled")
            return X
        
        logger.info("Creating amino acid features")
        X_aa = X.copy()
        
        if 'aaref_merged' in X.columns and 'aaalt_merged' in X.columns:
            # BLOSUM62 score
            if self.feature_config["amino_acid_features"]["blosum62_score"]:
                def get_blosum_score(ref, alt):
                    if pd.isna(ref) or pd.isna(alt) or ref == '.' or alt == '':
                        return 0
                    key = f"{ref}{alt}"
                    return self.blosum62.get(key, -4)
                
                X_aa['blosum62_score'] = X_aa.apply(lambda row: get_blosum_score(row['aaref_merged'], row['aaalt_merged']), axis=1)
            
            # BLOSUM62 categories
            if self.feature_config["amino_acid_features"]["blosum62_categories"]:
                if 'blosum62_score' in X_aa.columns:
                    X_aa['blosum_conservative'] = (X_aa['blosum62_score'] > 0).astype(int)
                    X_aa['blosum_semi_conservative'] = (X_aa['blosum62_score'] == 0).astype(int)
                    X_aa['blosum_non_conservative'] = (X_aa['blosum62_score'] < 0).astype(int)
            
            # Amino acid property changes
            if self.feature_config["amino_acid_features"]["aa_property_changes"]:
                for prop in ['hydrophobic', 'polar', 'charged', 'aromatic']:
                    def get_property_change(ref, alt, property_name):
                        if (pd.isna(ref) or pd.isna(alt) or ref == '.' or alt == '' or
                            ref not in self.aa_properties or alt not in self.aa_properties):
                            return 0
                        return int(self.aa_properties[ref][property_name] != self.aa_properties[alt][property_name])
                    
                    X_aa[f'{prop}_change'] = X_aa.apply(lambda row: get_property_change(row['aaref_merged'], row['aaalt_merged'], prop), axis=1)
            
            # Size changes
            if self.feature_config["amino_acid_features"]["aa_size_changes"]:
                def get_size_change(ref, alt):
                    if (pd.isna(ref) or pd.isna(alt) or ref == '.' or alt == '' or
                        ref not in self.aa_properties or alt not in self.aa_properties):
                        return 0
                    return self.aa_properties[alt]['size'] - self.aa_properties[ref]['size']
                
                X_aa['aa_size_change'] = X_aa.apply(lambda row: get_size_change(row['aaref_merged'], row['aaalt_merged']), axis=1)
            
            # Chemical class indicators
            if self.feature_config["amino_acid_features"]["aa_chemical_classes"]:
                hydrophobic_aa = {'A', 'I', 'L', 'V', 'M', 'F', 'Y', 'W', 'C'}
                charged_aa = {'R', 'D', 'E', 'H', 'K'}
                aromatic_aa = {'F', 'Y', 'W'}
                
                def safe_isin(aa, aa_set):
                    return 1 if (pd.notna(aa) and aa != '.' and aa in aa_set) else 0
                
                X_aa['ref_hydrophobic'] = X_aa['aaref_merged'].apply(lambda x: safe_isin(x, hydrophobic_aa))
                X_aa['alt_hydrophobic'] = X_aa['aaalt_merged'].apply(lambda x: safe_isin(x, hydrophobic_aa))
                X_aa['ref_charged'] = X_aa['aaref_merged'].apply(lambda x: safe_isin(x, charged_aa))
                X_aa['alt_charged'] = X_aa['aaalt_merged'].apply(lambda x: safe_isin(x, charged_aa))
                X_aa['ref_aromatic'] = X_aa['aaref_merged'].apply(lambda x: safe_isin(x, aromatic_aa))
                X_aa['alt_aromatic'] = X_aa['aaalt_merged'].apply(lambda x: safe_isin(x, aromatic_aa))
            
            # Structural impact
            if self.feature_config["amino_acid_features"]["aa_structural_impact"]:
                def get_structural_impact(ref, alt):
                    if (pd.isna(ref) or pd.isna(alt) or ref == '.' or alt == '' or
                        ref not in self.aa_properties or alt not in self.aa_properties):
                        return 0
                    return 1 if self.aa_properties[ref]['size'] != self.aa_properties[alt]['size'] or \
                                self.aa_properties[ref]['hydrophobic'] != self.aa_properties[alt]['hydrophobic'] else 0
                
                X_aa['aa_structural_impact'] = X_aa.apply(lambda row: get_structural_impact(row['aaref_merged'], row['aaalt_merged']), axis=1)
            
            # Polarity change
            if self.feature_config["amino_acid_features"]["aa_polarity_change"]:
                def get_polarity_change(ref, alt):
                    if (pd.isna(ref) or pd.isna(alt) or ref == '.' or alt == '' or
                        ref not in self.aa_properties or alt not in self.aa_properties):
                        return 0
                    return 1 if self.aa_properties[ref]['polarity'] != self.aa_properties[alt]['polarity'] else 0
                
                X_aa['aa_polarity_change'] = X_aa.apply(lambda row: get_polarity_change(row['aaref_merged'], row['aaalt_merged']), axis=1)
            
            # Charge change
            if self.feature_config["amino_acid_features"]["aa_charge_change"]:
                def get_charge_change(ref, alt):
                    if (pd.isna(ref) or pd.isna(alt) or ref == '.' or alt == '' or
                        ref not in self.aa_properties or alt not in self.aa_properties):
                        return 0
                    return 1 if self.aa_properties[ref]['charge'] != self.aa_properties[alt]['charge'] else 0
                
                X_aa['aa_charge_change'] = X_aa.apply(lambda row: get_charge_change(row['aaref_merged'], row['aaalt_merged']), axis=1)
        
        logger.info(f"Created amino acid features: {list(X_aa.columns.difference(X.columns))}")
        return X_aa

    def create_codon_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create deterministic codon features - ROW INDEPENDENT"""
        if not self.feature_config["codon_features"]["enabled"]:
            logger.info("Codon features disabled")
            return X
        
        logger.info("Creating codon features")
        X_codon = X.copy()
        
        if 'refcodon_merged' in X.columns:
            # GC content
            if self.feature_config["codon_features"]["codon_gc_content"]:
                def get_gc_content(codon):
                    if pd.isna(codon) or codon == '.' or not isinstance(codon, str) or len(codon) == 0:
                        return 0
                    codon = str(codon).upper()
                    return (codon.count('G') + codon.count('C')) / max(len(codon), 1)
                
                X_codon['codon_gc_content'] = X_codon['refcodon_merged'].apply(get_gc_content)
            
            # CpG content
            if self.feature_config["codon_features"]["codon_cpg_content"]:
                def get_cpg_count(codon):
                    if pd.isna(codon) or codon == '.' or not isinstance(codon, str):
                        return 0
                    return str(codon).upper().count('CG')
                
                X_codon['codon_cpg_count'] = X_codon['refcodon_merged'].apply(get_cpg_count)
            
            # Codon optimality
            if self.feature_config["codon_features"]["codon_optimality"]:
                def is_optimal_codon(codon):
                    if pd.isna(codon) or codon == '.' or not isinstance(codon, str):
                        return 0
                    return 1 if str(codon).upper() in self.optimal_codons else 0
                
                X_codon['codon_optimal'] = X_codon['refcodon_merged'].apply(is_optimal_codon)
            
            # Wobble position
            if self.feature_config["codon_features"]["wobble_position"]:
                def wobble_gc(codon):
                    if pd.isna(codon) or codon == '.' or not isinstance(codon, str) or len(str(codon)) < 3:
                        return 0
                    return 1 if str(codon)[2].upper() in 'GC' else 0
                
                X_codon['wobble_gc'] = X_codon['refcodon_merged'].apply(wobble_gc)
            
            # Start/Stop codons
            if self.feature_config["codon_features"]["start_stop_codons"]:
                def is_start_codon(codon):
                    if pd.isna(codon) or codon == '.' or not isinstance(codon, str):
                        return 0
                    return 1 if str(codon).upper() in self.start_codons else 0
                
                def is_stop_codon(codon):
                    if pd.isna(codon) or codon == '.' or not isinstance(codon, str):
                        return 0
                    return 1 if str(codon).upper() in self.stop_codons else 0
                
                X_codon['is_start_codon'] = X_codon['refcodon_merged'].apply(is_start_codon)
                X_codon['is_stop_codon'] = X_codon['refcodon_merged'].apply(is_stop_codon)
            
            # Codon degeneracy
            if self.feature_config["codon_features"]["codon_degeneracy"]:
                def get_codon_degeneracy(codon):
                    if pd.isna(codon) or codon == '.' or not isinstance(codon, str):
                        return 0
                    return self.degenerate_codons.get(str(codon).upper(), 0)
                
                X_codon['codon_degeneracy'] = X_codon['refcodon_merged'].apply(get_codon_degeneracy)
            
            # Codon bias score
            if self.feature_config["codon_features"]["codon_bias_score"]:
                def get_codon_bias_score(codon):
                    if pd.isna(codon) or codon == '.' or not isinstance(codon, str):
                        return 0
                    codon = str(codon).upper()
                    return 1 if codon in self.optimal_codons else -1 if codon in self.stop_codons else 0
                
                X_codon['codon_bias_score'] = X_codon['refcodon_merged'].apply(get_codon_bias_score)
            
            # Codon position impact
            if self.feature_config["codon_features"]["codon_position_impact"]:
                def get_position_impact(codon):
                    if pd.isna(codon) or codon == '.' or not isinstance(codon, str) or len(str(codon)) < 3:
                        return 0
                    codon = str(codon).upper()
                    return 1 if codon[2] in 'GC' else 0 if codon[0] in 'GC' else -1
                
                X_codon['codon_position_impact'] = X_codon['refcodon_merged'].apply(get_position_impact)
        
        logger.info(f"Created codon features: {list(X_codon.columns.difference(X.columns))}")
        return X_codon

    def create_nucleotide_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create deterministic nucleotide features - ROW INDEPENDENT"""
        if not self.feature_config["nucleotide_features"]["enabled"]:
            logger.info("Nucleotide features disabled")
            return X
        
        logger.info("Creating nucleotide features")
        X_nuc = X.copy()
        
        if 'Ref' in X.columns and 'Alt' in X.columns:
            # Transition/Transversion
            if self.feature_config["nucleotide_features"]["transition_transversion"]:
                def is_transition(ref, alt):
                    if pd.isna(ref) or pd.isna(alt) or ref == '.' or alt == '':
                        return 0
                    return 1 if (ref, alt) in self.transitions else 0
                
                X_nuc['is_transition'] = X_nuc.apply(lambda row: is_transition(row['Ref'], row['Alt']), axis=1)
                X_nuc['is_transversion'] = 1 - X_nuc['is_transition']
            
            # Purine/Pyrimidine
            if self.feature_config["nucleotide_features"]["purine_pyrimidine"]:
                def purine_to_pyrimidine(ref, alt):
                    if pd.isna(ref) or pd.isna(alt) or ref == '.' or alt == '':
                        return 0
                    return 1 if (ref in self.purines and alt in self.pyrimidines) else 0
                
                def pyrimidine_to_purine(ref, alt):
                    if pd.isna(ref) or pd.isna(alt) or ref == '.' or alt == '':
                        return 0
                    return 1 if (ref in self.pyrimidines and alt in self.purines) else 0
                
                X_nuc['purine_to_pyrimidine'] = X_nuc.apply(lambda row: purine_to_pyrimidine(row['Ref'], row['Alt']), axis=1)
                X_nuc['pyrimidine_to_purine'] = X_nuc.apply(lambda row: pyrimidine_to_purine(row['Ref'], row['Alt']), axis=1)
            
            # GC content change
            if self.feature_config["nucleotide_features"]["gc_content_change"]:
                def get_gc_change(ref, alt):
                    if pd.isna(ref) or pd.isna(alt) or ref == '.' or alt == '':
                        return 0
                    ref_gc = 1 if ref in 'GC' else 0
                    alt_gc = 1 if alt in 'GC' else 0
                    return alt_gc - ref_gc
                
                X_nuc['gc_content_change'] = X_nuc.apply(lambda row: get_gc_change(row['Ref'], row['Alt']), axis=1)
            
            # CpG context
            if self.feature_config["nucleotide_features"]["cpg_context"]:
                def affects_cpg(ref, alt):
                    if pd.isna(ref) or pd.isna(alt) or ref == '.' or alt == '':
                        return 0
                    return 1 if ((ref == 'C' and alt != 'C') or (ref != 'C' and alt == 'C')) else 0
                
                X_nuc['affects_cpg'] = X_nuc.apply(lambda row: affects_cpg(row['Ref'], row['Alt']), axis=1)
            
            # Base composition
            if self.feature_config["nucleotide_features"]["base_composition"]:
                def get_base_composition(ref, alt):
                    if pd.isna(ref) or pd.isna(alt) or ref == '.' or alt == '':
                        return 0
                    return 1 if ref == alt else -1
                
                X_nuc['base_composition_match'] = X_nuc.apply(lambda row: get_base_composition(row['Ref'], row['Alt']), axis=1)
            
            # Nucleotide complexity
            if self.feature_config["nucleotide_features"]["nucleotide_complexity"]:
                def get_nucleotide_complexity(ref, alt):
                    if pd.isna(ref) or pd.isna(alt) or ref == '.' or alt == '':
                        return 0
                    return 1 if (ref in self.purines and alt in self.purines) or (ref in self.pyrimidines and alt in self.pyrimidines) else -1
                
                X_nuc['nucleotide_complexity'] = X_nuc.apply(lambda row: get_nucleotide_complexity(row['Ref'], row['Alt']), axis=1)
            
            # Dinucleotide change
            if self.feature_config["nucleotide_features"]["dinucleotide_change"]:
                def get_dinucleotide_change(ref, alt):
                    if pd.isna(ref) or pd.isna(alt) or ref == '.' or alt == '':
                        return 0
                    return 1 if (ref in 'CG' and alt not in 'CG') or (ref not in 'CG' and alt in 'CG') else 0
                
                X_nuc['dinucleotide_change'] = X_nuc.apply(lambda row: get_dinucleotide_change(row['Ref'], row['Alt']), axis=1)
        
        logger.info(f"Created nucleotide features: {list(X_nuc.columns.difference(X.columns))}")
        return X_nuc

    def create_prediction_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create deterministic prediction consensus features - ROW INDEPENDENT"""
        if not self.feature_config["prediction_features"]["enabled"]:
            logger.info("Prediction features disabled")
            return X
        
        logger.info("Creating prediction features")
        X_pred = X.copy()
        
        pred_cols = [col for col in X.columns if '_pred' in col.lower()]
        
        if len(pred_cols) >= 3:
            # Severity encoding (using original severity mapping)
            if self.feature_config["prediction_features"]["severity_encoding"]:
                for col in pred_cols[:15]:
                    severity_col = f'{col}_severity'
                    def map_severity(val):
                        if pd.isna(val):
                            return 0
                        val_str = str(val).upper()
                        return self.severity_mapping.get(val_str, 0)
                    
                    X_pred[severity_col] = X_pred[col].apply(map_severity)
            
            # Consensus voting
            if self.feature_config["prediction_features"]["consensus_voting"]:
                severity_cols = [f'{col}_severity' for col in pred_cols[:15] if f'{col}_severity' in X_pred.columns]
                
                if len(severity_cols) >= 3:
                    def count_damaging(row):
                        valid_preds = [row[col] for col in severity_cols if pd.notna(row[col]) and row[col] >= 0]
                        if len(valid_preds) == 0:
                            return 0
                        return sum(1 for pred in valid_preds if pred >= 1)
                    
                    def count_total_valid(row):
                        return sum(1 for col in severity_cols if pd.notna(row[col]) and row[col] >= 0)
                    
                    X_pred['damaging_votes'] = X_pred.apply(count_damaging, axis=1)
                    X_pred['total_valid_predictions'] = X_pred.apply(count_total_valid, axis=1)
                    X_pred['damaging_fraction'] = X_pred['damaging_votes'] / (X_pred['total_valid_predictions'] + 1e-8)
                    
                    X_pred['high_confidence_damaging'] = (X_pred['damaging_fraction'] > 0.75).astype(int)
                    X_pred['high_confidence_benign'] = (X_pred['damaging_fraction'] < 0.25).astype(int)
                    X_pred['conflicting_predictions'] = (
                        X_pred['damaging_fraction'].between(0.25, 0.75) & 
                        (X_pred['total_valid_predictions'] >= 3)
                    ).astype(int)
            
            # Prediction confidence
            if self.feature_config["prediction_features"]["prediction_confidence"]:
                def get_confidence_score(row):
                    valid_preds = [row[f'{col}_severity'] for col in pred_cols[:15] 
                                 if f'{col}_severity' in X_pred.columns and pd.notna(row[f'{col}_severity']) and row[f'{col}_severity'] >= 0]
                    return max(valid_preds, default=0) if valid_preds else 0
                
                X_pred['max_severity_score'] = X_pred.apply(get_confidence_score, axis=1)
            
            # Prediction spread
            if self.feature_config["prediction_features"]["prediction_spread"]:
                def get_prediction_spread(row):
                    valid_preds = [row[f'{col}_severity'] for col in pred_cols[:15] 
                                 if f'{col}_severity' in X_pred.columns and pd.notna(row[f'{col}_severity']) and row[f'{col}_severity'] >= 0]
                    return max(valid_preds, default=0) - min(valid_preds, default=0) if valid_preds else 0
                
                X_pred['prediction_spread'] = X_pred.apply(get_prediction_spread, axis=1)
        
        logger.info(f"Created prediction features: {list(X_pred.columns.difference(X.columns))}")
        return X_pred

    def create_interaction_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create deterministic interaction features - ROW INDEPENDENT"""
        if not self.feature_config["interaction_features"]["enabled"]:
            logger.info("Interaction features disabled")
            return X
        
        logger.info("Creating interaction features")
        X_int = X.copy()
        
        important_pairs = [
            ('CADD_phred', 'REVEL_score'),
            ('SIFT4G_score', 'Polyphen2_HDIV_score'),
            ('MetaSVM_score', 'MetaLR_score'),
            ('phastCons100way_vertebrate', 'phyloP100way_vertebrate'),
            ('GERP++_RS', 'SiPhy_29way_logOdds'),
            ('CADD_phred', 'VEST4_score'),
            ('REVEL_score', 'ClinPred_score'),
            ('phastCons30way_mammalian', 'phyloP30way_mammalian')
        ]
        
        for col1_pattern, col2_pattern in important_pairs:
            col1_matches = [col for col in X.columns if col1_pattern.lower() in col.lower()]
            col2_matches = [col for col in X.columns if col2_pattern.lower() in col.lower()]
            
            if col1_matches and col2_matches:
                col1, col2 = col1_matches[0], col2_matches[0]
                
                col1_numeric = pd.api.types.is_numeric_dtype(X[col1])
                col2_numeric = pd.api.types.is_numeric_dtype(X[col2])
                
                if col1_numeric and col2_numeric:
                    # Ratios
                    if self.feature_config["interaction_features"]["score_ratios"]:
                        def safe_ratio(val1, val2):
                            if pd.isna(val1) or pd.isna(val2) or val2 == 0:
                                return 0
                            return val1 / (val2 + 1e-8)
                        
                        X_int[f'{col1}_{col2}_ratio'] = X_int.apply(lambda row: safe_ratio(row[col1], row[col2]), axis=1)
                    
                    # Products
                    if self.feature_config["interaction_features"]["score_products"]:
                        def safe_product(val1, val2):
                            if pd.isna(val1) or pd.isna(val2):
                                return 0
                            return val1 * val2
                        
                        X_int[f'{col1}_{col2}_product'] = X_int.apply(lambda row: safe_product(row[col1], row[col2]), axis=1)
                    
                    # Differences
                    if self.feature_config["interaction_features"]["score_differences"]:
                        def safe_difference(val1, val2):
                            if pd.isna(val1) or pd.isna(val2):
                                return 0
                            return val1 - val2
                        
                        X_int[f'{col1}_{col2}_difference'] = X_int.apply(lambda row: safe_difference(row[col1], row[col2]), axis=1)
                    
                    # Combined scores
                    if self.feature_config["interaction_features"]["combined_scores"]:
                        def combined_score(val1, val2):
                            if pd.isna(val1) or pd.isna(val2):
                                return 0
                            return (val1 + val2) / 2
                        
                        X_int[f'{col1}_{col2}_combined'] = X_int.apply(lambda row: combined_score(row[col1], row[col2]), axis=1)
        
        logger.info(f"Created interaction features: {list(X_int.columns.difference(X.columns))}")
        return X_int

    def handle_missing_values(self, X: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values with fixed defaults - ROW INDEPENDENT"""
        if not self.feature_config["missing_value_handling"]["enabled"]:
            logger.info("Missing value handling disabled")
            return X
        
        logger.info("Handling missing values")
        X_clean = X.copy()
        
        if self.feature_config["missing_value_handling"]["use_fixed_defaults"]:
            numerical_default = self.feature_config["missing_value_handling"]["numerical_default"]
            categorical_default = self.feature_config["missing_value_handling"]["categorical_default"]
            
            for col in X_clean.columns:
                if pd.api.types.is_object_dtype(X_clean[col]):
                    X_clean[col] = X_clean[col].fillna(categorical_default)
                    X_clean[col] = X_clean[col].replace(['.', '', 'nan', 'NaN', 'NULL'], categorical_default)
                else:
                    X_clean[col] = X_clean[col].fillna(numerical_default)
                    # Only check for infinity in numeric columns
                    if pd.api.types.is_numeric_dtype(X_clean[col]):
                        inf_mask = np.isinf(X_clean[col])
                        if inf_mask.any():
                            X_clean.loc[inf_mask, col] = numerical_default
        
        logger.info("Completed missing value handling")
        return X_clean

    def encode_categorical_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical features with fixed mappings - ROW INDEPENDENT"""
        if not self.feature_config["categorical_encoding"]["enabled"]:
            logger.info("Categorical encoding disabled")
            return X
        
        logger.info("Encoding categorical features")
        X_encoded = X.copy()
        
        try:
            if self.feature_config["categorical_encoding"]["fixed_mapping"]:
                categorical_cols = X_encoded.select_dtypes(include=['object']).columns
                logger.info(f"Processing {len(categorical_cols)} categorical columns: {list(categorical_cols)}")
                
                for col in categorical_cols:
                    try:
                        logger.debug(f"Processing categorical column: {col}")
                        
                        if col in self.categorical_mappings:
                            # Apply mapping with proper handling of missing values
                            def safe_map(val):
                                try:
                                    if pd.isna(val):
                                        return 0
                                    val_str = str(val).upper()
                                    return self.categorical_mappings[col].get(val_str, self.categorical_mappings[col].get('.', 0))
                                except Exception as e:
                                    logger.warning(f"Error mapping value '{val}' in column '{col}': {e}")
                                    return 0
                            
                            X_encoded[col] = X_encoded[col].apply(safe_map)
                            logger.debug(f"Successfully applied predefined mapping to column: {col}")
                            
                        else:
                            # For unmapped categorical columns, create a simple numerical encoding
                            try:
                                unique_vals = X_encoded[col].dropna().unique()
                                logger.debug(f"Column {col} has {len(unique_vals)} unique values")
                                
                                if len(unique_vals) > 0:
                                    # Create a simple mapping for unmapped columns
                                    sorted_vals = sorted([str(v) for v in unique_vals])
                                    mapping = {val: i for i, val in enumerate(sorted_vals)}
                                    mapping.update({'.': 0, '': 0, 'nan': 0, 'NaN': 0, 'NULL': 0})
                                    
                                    def safe_map_unknown(val):
                                        try:
                                            if pd.isna(val):
                                                return 0
                                            return mapping.get(str(val), 0)
                                        except Exception as e:
                                            logger.warning(f"Error mapping unknown value '{val}' in column '{col}': {e}")
                                            return 0
                                    
                                    X_encoded[col] = X_encoded[col].apply(safe_map_unknown)
                                    logger.debug(f"Successfully applied simple encoding to column: {col}")
                                    
                                else:
                                    # If no unique values, fill with 0
                                    X_encoded[col] = 0
                                    logger.debug(f"No unique values in column {col}, filled with 0")
                                
                                logger.info(f"Applied simple encoding for unmapped categorical column: {col}")
                                
                            except Exception as e:
                                logger.error(f"Error processing unmapped column {col}: {e}")
                                # Fallback: convert to string and then to simple numeric
                                try:
                                    X_encoded[col] = pd.Categorical(X_encoded[col].astype(str)).codes
                                except Exception as e2:
                                    logger.error(f"Fallback failed for column {col}: {e2}")
                                    X_encoded[col] = 0
                        
                        # Verify the column is now numeric
                        if not pd.api.types.is_numeric_dtype(X_encoded[col]):
                            logger.warning(f"Column {col} is still not numeric after encoding, forcing conversion")
                            X_encoded[col] = pd.to_numeric(X_encoded[col], errors='coerce').fillna(0)
                        
                    except Exception as e:
                        logger.error(f"Error processing column {col}: {e}")
                        # Ultimate fallback
                        X_encoded[col] = 0
            
            # Final check: ensure all remaining object columns are converted
            remaining_object_cols = X_encoded.select_dtypes(include=['object']).columns
            if len(remaining_object_cols) > 0:
                logger.warning(f"Still have object columns after encoding: {list(remaining_object_cols)}")
                for col in remaining_object_cols:
                    try:
                        X_encoded[col] = pd.to_numeric(X_encoded[col], errors='coerce').fillna(0)
                    except:
                        X_encoded[col] = 0
            
            logger.info("Completed categorical encoding successfully")
            return X_encoded
            
        except Exception as e:
            logger.error(f"Major error in categorical encoding: {e}")
            # Emergency fallback: convert all object columns to 0
            object_cols = X_encoded.select_dtypes(include=['object']).columns
            for col in object_cols:
                X_encoded[col] = 0
            logger.warning("Applied emergency fallback: converted all object columns to 0")
            return X_encoded

    def engineer_features(self, X: pd.DataFrame, test_mode: bool = False) -> pd.DataFrame:
        """Main row-independent feature engineering pipeline"""
        logger.info(f"Starting feature engineering (test_mode={test_mode})")
        
        if test_mode:
            logger.info("TEST MODE: Processing only first 100 rows")
            X_work = X.head(100).copy()
        else:
            X_work = X.copy()
        
        if not self.validate_input_data(X_work):
            logger.error("Input data validation failed")
            raise ValueError("Invalid input data")
        
        # STEP 1: Apply unified prediction mapping FIRST
        if self.feature_config["use_unified_mapping"]:
            X_work = self.transform_predictions_unified(X_work, test_mode)
        
        # STEP 2: Filter to enabled base features
        available_cols = list(X_work.columns)
        enabled_features = self.get_enabled_base_features(available_cols)
        X_work = X_work[enabled_features].copy()
        
        # STEP 3: Apply row-independent feature engineering steps
        X_work = self.create_amino_acid_features(X_work)
        X_work = self.create_codon_features(X_work)
        X_work = self.create_nucleotide_features(X_work)
        X_work = self.create_prediction_features(X_work)
        X_work = self.create_interaction_features(X_work)
        
        # STEP 4: Handle missing values with fixed defaults
        X_work = self.handle_missing_values(X_work)
        
        # STEP 5: Encode categorical features deterministically
        X_work = self.encode_categorical_features(X_work)
        
        logger.info(f"Feature engineering completed. Total features: {len(X_work.columns)}")
        return X_work

    def process_fold(self, fold_num: int, test_mode: bool = False) -> bool:
        """Process a single fold with row-independent feature engineering"""
        logger.info(f"Processing fold {fold_num} (test_mode={test_mode})")
        fold_dir = self.dataset_dir / f"Fold_{fold_num}"
        
        if not fold_dir.exists():
            logger.error(f"Fold directory {fold_dir} does not exist")
            return False
        
        try:
            # Load data
            X_train = pd.read_csv(fold_dir / "X_train.csv", low_memory=False)
            X_test = pd.read_csv(fold_dir / "X_test.csv", low_memory=False)
            logger.info(f"Loaded X_train ({X_train.shape}) and X_test ({X_test.shape})")
            
            # Apply same row-independent transformation to both train and test
            X_train_featured = self.engineer_features(X_train, test_mode)
            X_test_featured = self.engineer_features(X_test, test_mode)
            
            # Ensure same columns
            common_cols = list(set(X_train_featured.columns) & set(X_test_featured.columns))
            X_train_featured = X_train_featured[common_cols]
            X_test_featured = X_test_featured[common_cols]
            logger.info(f"Common columns after feature engineering: {len(common_cols)}")
            
            # Save enhanced files
            suffix = "_test" if test_mode else "_processed"
            X_train_featured.to_csv(fold_dir / f"X_train{suffix}.csv", index=False)
            X_test_featured.to_csv(fold_dir / f"X_test{suffix}.csv", index=False)
            logger.info(f"Saved featured data for fold {fold_num}")
            
            # Save configuration
            config_file = fold_dir / f"feature_config{suffix}.json"
            with open(config_file, 'w') as f:
                json.dump(self.feature_config, f, indent=2)
            logger.info(f"Saved configuration to {config_file}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error processing fold {fold_num}: {str(e)}")
            return False

    def process_all_folds(self, test_mode: bool = False) -> int:
        """Process all available folds"""
        logger.info(f"Processing all folds (test_mode={test_mode})")
        fold_dirs = [d for d in self.dataset_dir.iterdir() 
                    if d.is_dir() and d.name.startswith('Fold_')]
        fold_dirs.sort()
        
        if not fold_dirs:
            logger.error("No fold directories found")
            return 0
        
        processed = 0
        for fold_dir in fold_dirs:
            fold_num = int(fold_dir.name.split('_')[1])
            if self.process_fold(fold_num, test_mode):
                processed += 1
        
        logger.info(f"Processed {processed} folds successfully")
        return processed

    def save_config(self, config_path: str) -> None:
        """Save current configuration"""
        logger.info(f"Saving configuration to {config_path}")
        with open(config_path, 'w') as f:
            json.dump(self.feature_config, f, indent=2)

    def load_config(self, config_path: str) -> None:
        """Load configuration from file"""
        logger.info(f"Loading configuration from {config_path}")
        with open(config_path, 'r') as f:
            self.feature_config.update(json.load(f))

def main():
    parser = argparse.ArgumentParser(description="Complete ClinVar feature engineering with unified prediction mapping")
    parser.add_argument('dataset_dir', nargs='?', default='Clinvar_Dataset7',
                       help='Directory containing fold data')
    parser.add_argument('--config', help='Configuration file path')
    parser.add_argument('--save-config', help='Save current config to file')
    parser.add_argument('--test-mode', action='store_true', help='Run in test mode (100 rows only)')
    
    args = parser.parse_args()
    
    logger.info("="*80)
    logger.info("COMPLETE CLINVAR FEATURE ENGINEERING PIPELINE")
    logger.info("Unified Prediction Mapping + All Original Biological Features")
    logger.info("="*80)
    logger.info(f"Dataset directory: {args.dataset_dir}")
    logger.info(f"Test mode: {args.test_mode}")
    
    engineer = RowIndependentFeatureEngineer(args.dataset_dir, args.config)
    
    if args.save_config:
        engineer.save_config(args.save_config)
        logger.info(f"Configuration saved to {args.save_config}")
        print(f"Configuration saved to {args.save_config}")
        return
    
    processed_folds = engineer.process_all_folds(test_mode=args.test_mode)
    
    if processed_folds > 0:
        suffix = "test" if args.test_mode else "processed"
        print("="*80)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*80)
        print(f"Successfully processed {processed_folds} folds")
        print(f"Created files: X_train_{suffix}.csv, X_test_{suffix}.csv")
        print("\nFeatures created:")
        print("? Unified prediction mapping (Pathogenic=1, Benign=0, Unknown=-1)")
        print("? Full BLOSUM62 amino acid substitution scores")
        print("? Complete amino acid property changes")
        print("? Comprehensive codon features (GC, CpG, optimality, degeneracy)")
        print("? Nucleotide transition/transversion analysis")
        print("? Prediction consensus and voting features")
        print("? Interaction features between important scores")
        print("? Row-independent processing (no data leakage)")
        print("? Deterministic feature engineering")
        
        if args.test_mode:
            print("\n" + "="*60)
            print("TEST MODE COMPLETED - Review results before full processing")
            print("Run without --test-mode for full dataset processing")
            print("="*60)
        
        logger.info("Pipeline execution completed successfully")
    else:
        print("? No folds were processed successfully")
        print("Check the logs for error details")
        logger.error("No folds processed successfully")

if __name__ == "__main__":
    main()