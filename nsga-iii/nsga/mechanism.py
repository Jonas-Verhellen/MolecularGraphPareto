import sys
import hydra
import joblib
import numpy as np
import pandas as pd
from typing import List, Tuple
from scipy.stats import gmean

from rdkit import Chem
from rdkit import rdBase
from rdkit import RDConfig
rdBase.DisableLog('rdApp.error')

from rdkit.Chem import AllChem
from rdkit.Chem import Crippen
from rdkit.Chem import Lipinski
from rdkit.Chem import Descriptors
from rdkit.Chem import rdMolDescriptors
from rdkit.DataStructs.cDataStructs import TanimotoSimilarity

from rdkit import Chem
from rdkit.Chem import rdDistGeom
from rdkit.Chem.rdMolDescriptors import GetUSRScore, GetUSRCAT

from guacamol.score_modifier import MinGaussianModifier, MaxGaussianModifier, ClippedScoreModifier, GaussianModifier, AbsoluteScoreModifier
from guacamol.common_scoring_functions import TanimotoScoringFunction, RdkitScoringFunction, CNS_MPO_ScoringFunction, IsomerScoringFunction, SMARTSScoringFunction
from guacamol.utils.descriptors import num_rotatable_bonds, num_aromatic_rings, logP, qed, tpsa, bertz, mol_weight, AtomCounter, num_rings
from guacamol.scoring_function import MoleculewiseScoringFunction, BatchScoringFunction

class Fitness:
    """
    A strategy class for calculating the fitness values of a molecule.
    """
    def __init__(self, config_fitness) -> None:
        self.calls = 0
        self.task = getattr(self, config_fitness.task)
        if config_fitness.task == "mpo_dap_kinases":
            self.clf_DAPk1 = joblib.load(hydra.utils.to_absolute_path("data/models/clf_DAPK1.joblib"))
            self.clf_DRP1 = joblib.load(hydra.utils.to_absolute_path("data/models/clf_DRP1.joblib"))
            self.clf_ZIPk = joblib.load(hydra.utils.to_absolute_path("data/models/clf_ZIPK.joblib"))
            self.clf_HERG = joblib.load(hydra.utils.to_absolute_path("data/models/clf_HERG.joblib"))
            self.clf_SCN2A = joblib.load(hydra.utils.to_absolute_path("data/models/clf_SCN2A.joblib"))
        if config_fitness.task == "mpo_antipsychotics":
            self.clf_HT2A = joblib.load(hydra.utils.to_absolute_path("data/models/clf_HT2A.joblib"))
            self.clf_HT2B = joblib.load(hydra.utils.to_absolute_path("data/models/clf_HT2B.joblib"))
            self.clf_D2 = joblib.load(hydra.utils.to_absolute_path("data/models/clf_D2.joblib"))
            self.clf_HERG = joblib.load(hydra.utils.to_absolute_path("data/models/clf_HERG.joblib"))
        self.dimensionality = len(self.task())
        return None

    def __call__(self, molecule) -> None:
        """
        Updates the fitness values of a molecule. 
        """
        fitnesses = []
        for objective in self.task():
            fitnesses.append(objective.modify_score(objective.raw_score(molecule.smiles)))
        molecule.fitnesses = fitnesses
        return molecule

    def batch_score(self, molecules):
        fitnesses_list = []
        for objective in self.task():
            fitnesses_list.append(objective.score_list([molecule.smiles for molecule in molecules]))
        for molecule, fitnesses in zip(molecules, list(map(list, zip(*fitnesses_list)))):
            molecule.fitnesses = fitnesses
        return molecules

    def mpo_cobimetinib(self, max_logP=5.0):
        smiles = 'OC1(CN(C1)C(=O)C1=C(NC2=C(F)C=C(I)C=C2)C(F)=C(F)C=C1)C1CCCCN1'
        similarity = TanimotoScoringFunction(smiles, fp_type='FCFP4', score_modifier=ClippedScoreModifier(upper_x=0.7))
        deviation = TanimotoScoringFunction(smiles, fp_type='ECFP6', score_modifier=MinGaussianModifier(mu=0.75, sigma=0.1))
        rot_b = RdkitScoringFunction(descriptor=num_rotatable_bonds, score_modifier= MinGaussianModifier(mu=3, sigma=1))
        rings = RdkitScoringFunction(descriptor=num_aromatic_rings, score_modifier= MaxGaussianModifier(mu=3, sigma=1)) 
        return [similarity, deviation, rot_b, rings, CNS_MPO_ScoringFunction(max_logP=max_logP)]

    def mpo_osimertinib(self):
        smiles = 'COc1cc(N(C)CCN(C)C)c(NC(=O)C=C)cc1Nc2nccc(n2)c3cn(C)c4ccccc34'
        similarity = TanimotoScoringFunction(smiles, fp_type='FCFP4', score_modifier=ClippedScoreModifier(upper_x=0.8))
        deviation = TanimotoScoringFunction(smiles, fp_type='ECFP6', score_modifier=MinGaussianModifier(mu=0.85, sigma=0.1))
        tpsa_over_100 = RdkitScoringFunction(descriptor=tpsa, score_modifier=MaxGaussianModifier(mu=100, sigma=10))
        logP_scoring = RdkitScoringFunction(descriptor=logP, score_modifier=MinGaussianModifier(mu=1, sigma=1))
        return [similarity, deviation, tpsa_over_100, logP_scoring]

    def mpo_fexofenadine(self):
        smiles = 'CC(C)(C(=O)O)c1ccc(cc1)C(O)CCCN2CCC(CC2)C(O)(c3ccccc3)c4ccccc4'
        similarity = TanimotoScoringFunction(smiles, fp_type='AP', score_modifier=ClippedScoreModifier(upper_x=0.8))
        tpsa_over_90 = RdkitScoringFunction(descriptor=tpsa,score_modifier=MaxGaussianModifier(mu=90, sigma=10))
        logP_under_4 = RdkitScoringFunction(descriptor=logP, score_modifier=MinGaussianModifier(mu=4, sigma=1))
        return [similarity, tpsa_over_90, logP_under_4]

    def mpo_pioglitazone(self):
        smiles = 'O=C1NC(=O)SC1Cc3ccc(OCCc2ncc(cc2)CC)cc3'
        target_molw = mol_weight(Chem.MolFromSmiles(smiles))
        disimilarity = TanimotoScoringFunction(smiles, fp_type='ECFP4', score_modifier=GaussianModifier(mu=0, sigma=0.1))
        mw = RdkitScoringFunction(descriptor=mol_weight, score_modifier=GaussianModifier(mu=target_molw, sigma=10))
        rb = RdkitScoringFunction(descriptor=num_rotatable_bonds, score_modifier=GaussianModifier(mu=2, sigma=0.5))
        return [disimilarity, mw, rb]

    def mpo_ranolazine(self):
        smiles = 'COc1ccccc1OCC(O)CN2CCN(CC(=O)Nc3c(C)cccc3C)CC2'
        similarity = TanimotoScoringFunction(smiles, fp_type='AP', score_modifier=ClippedScoreModifier(upper_x=0.7))
        logP_under_4 = RdkitScoringFunction(descriptor=logP, score_modifier=MaxGaussianModifier(mu=7, sigma=1))
        tpsa_f = RdkitScoringFunction(descriptor=tpsa, score_modifier=MaxGaussianModifier(mu=95, sigma=20))
        fluorine = RdkitScoringFunction(descriptor=AtomCounter('F'), score_modifier=GaussianModifier(mu=1, sigma=1.0))
        return [similarity, logP_under_4, fluorine, tpsa_f]

    def mpo_dap_kinases(self):
        DAPk1 = CLFScoringFunction(self.clf_DAPk1, score_modifier=ClippedScoreModifier(upper_x=0.9))
        DRP1 = CLFScoringFunction(self.clf_DRP1, score_modifier=ClippedScoreModifier(upper_x=0.9))
        ZIPk = CLFScoringFunction(self.clf_ZIPk, score_modifier=ClippedScoreModifier(upper_x=0.9))
        not_hERG = CLFScoringFunction(self.clf_HERG, score_modifier=GaussianModifier(mu=0, sigma=0.1))
        not_SCN2A = CLFScoringFunction(self.clf_SCN2A, score_modifier=GaussianModifier(mu=0, sigma=0.1))
        return [DAPk1, DRP1, ZIPk, not_hERG, not_SCN2A]

    def mpo_antipsychotics(self, max_logP=5.0):
        HT2A = CLFScoringFunction(self.clf_HT2A, score_modifier=ClippedScoreModifier(upper_x=0.8))
        HT2B = CLFScoringFunction(self.clf_HT2B, score_modifier=ClippedScoreModifier(upper_x=0.8))
        D2 = CLFScoringFunction(self.clf_D2, score_modifier=ClippedScoreModifier(upper_x=0.8))
        not_hERG = CLFScoringFunction(self.clf_HERG, score_modifier=GaussianModifier(mu=0, sigma=0.1))
        return [HT2A, HT2B, D2, not_hERG, CNS_MPO_ScoringFunction(max_logP=max_logP)]


class CLFScoringFunction(BatchScoringFunction):
    def __init__(self, clf, score_modifier):
        super().__init__(score_modifier=score_modifier)
        self.clf = clf

    def raw_score_list(self, smiles_list):
        fingerprints = [np.array(AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smiles), 2, nBits=1024)) for smiles in smiles_list]
        raw_scores = self.clf.predict_proba(np.array(fingerprints))[:, 1] 
        return raw_scores

class CNS_MPO_BatchScoringFunction(BatchScoringFunction):
    def __init__(self, max_logP=5.0, maxMW=360, min_tpsa=40, max_tpsa=90, max_hbd=0) -> None:
        super().__init__()

        self.logP_gauss = MinGaussianModifier(max_logP, 1)
        self.molW_gauss = MinGaussianModifier(maxMW, 60)
        self.tpsa_maxgauss = MaxGaussianModifier(min_tpsa, 20)
        self.tpsa_mingauss = MinGaussianModifier(max_tpsa, 30)
        self.hbd_gauss = MinGaussianModifier(max_hbd, 2.0)

    def score_smiles(self, smiles) -> float:
        mol = Chem.MolFromSmiles(smiles)
        mw = mol_weight(mol)
        lp = logP(mol)
        hbd = num_H_donors(mol)
        mol_tpsa = tpsa(mol)

        o1 = self.tpsa_mingauss(mol_tpsa)
        o2 = self.tpsa_maxgauss(mol_tpsa)
        o3 = self.hbd_gauss(hbd)
        o4 = self.logP_gauss(lp)
        o5 = self.molW_gauss(mw)

        return 0.2 * (o1 + o2 + o3 + o4 + o5)

    def raw_score_list(self, smiles_list):
        return [self.score_smiles(smiles) for smiles in smiles_list]