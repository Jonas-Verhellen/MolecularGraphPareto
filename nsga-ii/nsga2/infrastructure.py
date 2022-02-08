import os
import csv
import hydra
import random
import itertools

import numpy as np
import pandas as pd
from typing import List, Tuple
from dataclasses import dataclass

from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.neighbors import KDTree

from rdkit import Chem
from rdkit import rdBase
from rdkit.Chem import AllChem
rdBase.DisableLog('rdApp.error')

from rdkit.Chem import Crippen
from rdkit.Chem import Lipinski
from rdkit.Chem import Descriptors
from rdkit.Chem import rdMolDescriptors

from nsga2.base import Molecule

class Arbiter:
    """
    A catalog class containing different druglike filters for small molecules.
    Includes the option to run the structural filters from ChEMBL.
    """
    def __init__(self, arbiter_config) -> None:
        self.cache_smiles = []
        self.rules_dict = pd.read_csv(hydra.utils.to_absolute_path("data/smarts/alert_collection.csv"))
        self.rules_dict= self.rules_dict[self.rules_dict.rule_set_name.isin(arbiter_config.rules)]
        self.rules_list = self.rules_dict["smarts"].values.tolist()
        self.tolerance_list = pd.to_numeric(self.rules_dict["max"]).values.tolist()
        self.pattern_list = [Chem.MolFromSmarts(smarts) for smarts in self.rules_list]

    def __call__(self, molecules):
        """
        Applies the chosen filters (hologenicity, veber_infractions,
        ChEMBL structural alerts, ...) to a list of molecules and removes duplicates.
        """
        filtered_molecules = []
        molecules = self.neutralize(molecules)
        molecules = self.unique_molecules(molecules)
        for molecule in molecules:
            molecular_graph = Chem.MolFromSmiles(molecule.smiles)
            if self.molecule_filter(molecular_graph):
                filtered_molecules.append(molecule)
        return filtered_molecules

    def unique_molecules(self, molecules: List[Molecule]) -> List[Molecule]:
        """
        Checks if a molecule in a lost of molcules is duplicated, either in this batch or before.
        """
        unique_molecules = []
        for molecule in molecules:
            if molecule.smiles not in self.cache_smiles:
                unique_molecules.append(molecule)
                self.cache_smiles.append(molecule.smiles)
        return unique_molecules

    def molecule_filter(self, molecular_graph: Chem.Mol) -> bool:
        """
        Checks if a given molecular structure passes through the chosen filters (hologenicity,
        veber_infractions, ChEMBL structural alerts, ...).
        """
        toxicity = self.toxicity(molecular_graph)
        hologenicity = self.hologenicity(molecular_graph)
        veber_infraction = self.veber_infraction(molecular_graph)
        validity = not (toxicity or hologenicity or veber_infraction)
        if molecular_graph.HasSubstructMatch(Chem.MolFromSmarts('[R]')):
            ring_infraction = self.ring_infraction(molecular_graph)
            validity = validity and not (ring_infraction)
        validity = validity and self.constraints(molecular_graph)
        return validity

    def constraints(self, molecular_graph: Chem.Mol) -> bool:
        """
        Checks if a given molecule fails the structural filters.
        """
        mw = 140.0 <= Descriptors.ExactMolWt(molecular_graph) <= 555.0
        logp = 0.0 <= Descriptors.MolLogP(molecular_graph) <= 7.0
        tpsa = 0.0 <= Descriptors.TPSA(molecular_graph) <= 140.0
        mr = 40.0 <= Crippen.MolMR(molecular_graph) <= 130.0
        return mw and logp and tpsa and mr

    def toxicity(self, molecular_graph: Chem.Mol) -> bool:
        """
        Checks if a given molecule fails the structural filters.
        """
        for (pattern, tolerance) in zip(self.pattern_list, self.tolerance_list):
            if len(molecular_graph.GetSubstructMatches(pattern)) > tolerance:
                return True
        return False

    @staticmethod
    def hologenicity(molecular_graph: Chem.Mol) -> bool:
        """
        Checks if a given molecule fails the hologenicity filters.
        """
        fluorine_saturation = len(molecular_graph.GetSubstructMatches(Chem.MolFromSmarts('[F]'))) > 6
        bromide_saturation = len(molecular_graph.GetSubstructMatches(Chem.MolFromSmarts('[Br]'))) > 3
        chlorine_saturation = len(molecular_graph.GetSubstructMatches(Chem.MolFromSmarts('[Cl]'))) > 3
        return chlorine_saturation or bromide_saturation or fluorine_saturation

    @staticmethod
    def ring_infraction(molecular_graph: Chem.Mol) -> bool:
        """
        Checks if a given molecule fails the ring infraction filters.
        """
        ring_allene = molecular_graph.HasSubstructMatch(Chem.MolFromSmarts('[R]=[R]=[R]'))
        macro_cycle = max([len(j) for j in molecular_graph.GetRingInfo().AtomRings()]) > 6
        double_bond_in_small_ring = molecular_graph.HasSubstructMatch(Chem.MolFromSmarts('[r3,r4]=[r3,r4]'))
        return ring_allene or macro_cycle or double_bond_in_small_ring

    @staticmethod
    def veber_infraction(molecular_graph: Chem.Mol) -> bool:
        """
        Checks if a given molecule fails the veber infraction filters.
        """
        rotatable_bond_saturation = Lipinski.NumRotatableBonds(molecular_graph) > 10
        hydrogen_bond_saturation = Lipinski.NumHAcceptors(molecular_graph) + Lipinski.NumHDonors(molecular_graph) > 10
        return rotatable_bond_saturation or hydrogen_bond_saturation

    @staticmethod
    def neutralize(molecules: List[Molecule]) -> List[Molecule]:
        pattern = Chem.MolFromSmarts("[+1!h0!$([*]~[-1,-2,-3,-4]),-1!$([*]~[+1,+2,+3,+4])]")
        for molecule in molecules: 
            molecular_graph = Chem.MolFromSmiles(molecule.smiles)
            atomic_matches = molecular_graph.GetSubstructMatches(pattern)
            atomic_matches_list = [y[0] for y in atomic_matches]
            if len(atomic_matches_list) > 0:
                for atomic_idx in atomic_matches_list:
                    atom = molecular_graph.GetAtomWithIdx(atomic_idx)
                    number_of_hydrogens = atom.GetTotalNumHs()
                    formal_charge = atom.GetFormalCharge()
                    atom.SetFormalCharge(0)
                    atom.SetNumExplicitHs(number_of_hydrogens - formal_charge)
                    atom.UpdatePropertyCache()
                molecule.smiles = Chem.MolToSmiles(molecular_graph)
        return molecules    


    def unique_molecules(self, molecules: List[Molecule]) -> List[Molecule]:
        """
        Checks if a molecule in a list of molcules is duplicated, either in this batch or before.
        """
        unique_molecules = []
        for molecule in molecules:
            if molecule.smiles not in self.cache_smiles:
                unique_molecules.append(molecule)
                self.cache_smiles.append(molecule.smiles)
        return unique_molecules
