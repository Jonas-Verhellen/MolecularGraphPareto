import os
import csv
import hydra
import warnings
import random
import numpy as np
import pandas as pd

from scipy.stats import gmean
from scipy.spatial import distance_matrix
from indices.faith import Faith
from typing import List, Tuple, Type

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import PandasTools as pdtl

from dask import bag
from dask.distributed import Client

from nsga2.base import Molecule
from nsga2.infrastructure import Arbiter
from nsga2.operations import Mutator, Crossover
from nsga2.mechanism import Fitness

from pymoo.factory import get_reference_directions
from pymoo.util.ref_dirs.energy import RieszEnergyReferenceDirectionFactory
import pygmo as pg 
from omegaconf import DictConfig, OmegaConf

class NSGA2:
    def __init__(self, config) -> None:
        self.data_file = config.data_file
        self.generations = config.generations
        self.batch_size = config.batch_size
        self.initial_size = config.initial_size
        self.population_size = config.population_size
        self.arbiter = Arbiter(config.arbiter)
        self.fitness = Fitness(config.fitness)
        self.mutator = Mutator(config.mutator)
        self.crossover = Crossover()
        self.client = Client(n_workers=config.workers, threads_per_worker=config.threads)
        return None

    def __call__(self) -> None:
        molecules = self.initial_population()
        for generation in range(1, self.generations):
            offspring = self.generate_molecules(molecules)
            offspring = self.calculate_fitnesses(offspring)
            molecules = self.selection(molecules + offspring)
            molecules = self.fingerprints(molecules)
            max_geometric_mean =  np.max([gmean(molecule.fitnesses) for molecule in molecules])
            dominated_hypervolume = pg.hypervolume(np.array([-1.0*np.array(molecule.fitnesses) for molecule in molecules if np.linalg.norm(molecule.fitnesses) > 0.00])).compute(np.zeros(len(molecules[0].fitnesses)))
            internal_similarity = self.internal_similarity(molecules)
            fitness_calls = self.fitness.calls
            print(f'Generation: {generation}, Max Geometric Mean: {max_geometric_mean}, Hypervolume: {dominated_hypervolume}, Internal Similarity: {internal_similarity}, Fitness Calls: {fitness_calls}')
            self.store_data(generation, molecules, max_geometric_mean, dominated_hypervolume, internal_similarity, fitness_calls)
        return None

    def initial_population(self) -> None:
        molecules = self.arbiter(self.load_from_database())
        molecules = self.calculate_fitnesses(molecules)
        molecules = self.selection(molecules)
        molecules = self.fingerprints(molecules)
        return molecules

    def load_from_database(self) -> List[Molecule]:
        dataframe = pd.read_csv(hydra.utils.to_absolute_path(self.data_file))
        smiles_list = dataframe['smiles'].sample(n=self.initial_size).tolist()
        molecules = [Molecule(Chem.CanonSmiles(smiles)) for smiles in smiles_list]
        return molecules

    def store_data(self, generation, molecules, max_geometric_mean, dominated_hypervolume, internal_similarity, fitness_calls) -> None:
        if os.path.isfile('statistics.csv'):
            with open('statistics.csv', 'a') as file:
                csv.writer(file).writerow([generation, max_geometric_mean, dominated_hypervolume, internal_similarity, fitness_calls])
                file.close()
        else:
            with open('statistics.csv', 'w') as file:
                csv.writer(file).writerow(["generation", "max geometric mean", "dominated hypervolume", "internal similarity", "fitness calls (cumulative)"])
                csv.writer(file).writerow([generation, max_geometric_mean, dominated_hypervolume, internal_similarity, fitness_calls])
                file.close()
        archive_data = {'smiles': [molecule.smiles for molecule in molecules], 'fitnesses': [molecule.fitnesses for molecule in molecules], "rank": [molecule.rank for molecule in molecules], "crowding distance": [molecule.crowding_distance for molecule in molecules]}
        pd.DataFrame(data=archive_data).to_csv("archive_{}.csv".format(generation), index=False)
        return None

    def generate_molecules(self, molecules: List[Molecule]) -> List[Molecule]:
        generated_molecules = []
        molecule_samples = self.sample(molecules)
        molecule_sample_pairs = self.sample_pairs(molecules)
        for molecule in molecule_samples:
            generated_molecules.extend(self.mutator(molecule)) 
        for molecule_pair in molecule_sample_pairs:
            generated_molecules.extend(self.crossover(molecule_pair)) 
        return self.arbiter(generated_molecules)

    def calculate_fitnesses(self, molecules: List[Molecule]) -> List[Molecule]:
        molecules = bag.map(self.fitness, bag.from_sequence(molecules)).compute() 
        self.fitness.calls += len(molecules)
        return molecules

    def sample(self, molecules: List[Molecule]) -> List[Chem.Mol]:
        pairs = [(molecule, np.mean(molecule.fitnesses)) for molecule in molecules]
        molecules, weights = map(list, zip(*pairs))
        return random.choices(molecules, k=self.batch_size, weights=weights)

    def sample_pairs(self, molecules: List[Molecule]) -> List[Tuple[Chem.Mol, Chem.Mol]]:
        pairs = [(molecule, gmean(molecule.fitnesses)) for molecule in molecules]
        molecules, weights = map(list, zip(*pairs))
        sample_molecules = random.choices(molecules, k=self.batch_size, weights=weights)
        sample_pairs = np.random.choice(list(filter(None, sample_molecules)), size=(self.batch_size, 2), replace=True)
        sample_pairs = [tuple(sample_pair) for sample_pair in sample_pairs]       
        return sample_pairs

    def selection(self, molecules): 
        selected = []
        fronts = self.calculate_fronts(molecules)   
        minimal_values, maximal_values = self.extrema(molecules)
        molecules = self.crowding_distance(fronts, minimal_values, maximal_values)     
        for front in fronts: 
            if len(selected) + len(front) > self.population_size:
                front.sort(key=lambda x: x.crowding_distance, reverse=True)
                selected.extend(front[:self.population_size - len(selected)])
            else:
                selected.extend(front)
        return selected

    def internal_similarity(self, molecules, c_threshold=None, w_factor="fraction") -> float:
        """
        Returns the extended Faith similarity for the list of molecules 
        """
        fingerprints = np.array([molecule.fingerprint for molecule in molecules])
        multiple_comparisons_index = Faith(fingerprints=fingerprints, c_threshold=c_threshold, w_factor=w_factor)
        return multiple_comparisons_index.__dict__["Fai_1sim_wdis"] 

    def fingerprints(self, molecules) -> List[Chem.Mol]:
        for molecule in molecules:
            molecule.fingerprint = np.array(AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(molecule.smiles), 4, nBits=1024))
        return molecules

    def extrema(self, molecules): 
        stacked_fitnesses = np.stack([molecule.fitnesses for molecule in molecules], axis = 0)
        minimal_values = stacked_fitnesses.min(axis = 0)
        maximal_values = stacked_fitnesses.max(axis = 0)
        return minimal_values, maximal_values

    def crowding_distance(self, fronts, minimal_values, maximal_values):
        for front in fronts:
            for dimension in range(len(fronts[0][0].fitnesses)):
                value_range = maximal_values[dimension] - minimal_values[dimension]
                if value_range == 0.0:
                    value_range = 1.0
                normalized_fitnesses = ([molecule.fitnesses[dimension] for molecule in front] - minimal_values[dimension])/value_range
                normalized_and_ordered_front = [(normalized_fitnesses, molecule) for normalized_fitnesses, molecule in sorted(zip(normalized_fitnesses, front), key = lambda x: x[0])]
                for index, pair in enumerate(normalized_and_ordered_front[1:-1]):
                    pair[1].crowding_distance += normalized_and_ordered_front[index+2][0] - normalized_and_ordered_front[index][0] 
                normalized_and_ordered_front[0][1].crowding_distance = np.inf
                normalized_and_ordered_front[-1][1].crowding_distance = np.inf
        return None

    def calculate_fronts(self, molecules: List[Molecule]) -> List[Molecule]: 
        fitnesses = np.array([molecule.fitnesses for molecule in molecules])
        domination_sets = []
        domination_counts = []
        for fitnesses_a in fitnesses:
            current_dimination_set = set()
            domination_counts.append(0)
            for index, fitnesses_b in enumerate(fitnesses):
                if np.all(fitnesses_a >= fitnesses_b) and np.any(fitnesses_a > fitnesses_b):
                    current_dimination_set.add(index)
                elif np.all(fitnesses_b >= fitnesses_a) and np.any(fitnesses_b > fitnesses_a):
                    domination_counts[-1] += 1
            domination_sets.append(current_dimination_set)
        domination_counts = np.array(domination_counts)
        fronts = []
        while True:
            current_front = np.where(domination_counts==0)[0]
            if len(current_front) == 0:
                break
            fronts.append(current_front)
            for individual in current_front:
                domination_counts[individual] = -1 
                dominated_by_current_set = domination_sets[individual]
                for dominated_by_current in dominated_by_current_set:
                    domination_counts[dominated_by_current] -= 1
        molecular_fronts = [list(map(molecules.__getitem__, front)) for front in fronts]
        for index, molecular_front in enumerate(molecular_fronts):
            for molecule in molecular_front:
                molecule.rank = index
        return molecular_fronts

@hydra.main(config_path="configuration", config_name="config.yaml")
def launch(config) -> None:
    print(OmegaConf.to_yaml(config))
    current_instance = NSGA2(config)
    current_instance()
    current_instance.client.close()

if __name__ == "__main__":
    launch()
