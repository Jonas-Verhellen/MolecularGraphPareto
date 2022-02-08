import os
import csv
import hydra
import warnings
import random
import numpy as np
import pandas as pd
from scipy.stats import gmean
from typing import List, Tuple, Type

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import PandasTools as pdtl

from dask import bag
from dask.distributed import Client

from nsga.base import Molecule
from nsga.infrastructure import Arbiter
from nsga.operations import Mutator, Crossover
from nsga.mechanism import Fitness

from pymoo.factory import  get_reference_directions
from pymoo.util.ref_dirs.energy import RieszEnergyReferenceDirectionFactory
import pygmo as pg 

from indices.faith import Faith
from omegaconf import DictConfig, OmegaConf

class NSGA3:
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
        self.dimensionality = self.fitness.dimensionality
        self.reference_directions = get_reference_directions("energy", self.dimensionality, config.direction_size)
        self.client = Client(n_workers=config.workers, threads_per_worker=config.threads)
        return None

    def __call__(self) -> None:
        molecules = self.initial_population()
        for generation in range(1, self.generations):
            offspring = self.generate_molecules(molecules)
            offspring = self.calculate_fitnesses(offspring)
            offspring = self.fingerprints(offspring)
            candidate_molecules = molecules + offspring
            molecules = self.directional_selection(candidate_molecules) 
            fraction = len(molecules)/len(self.reference_directions)
            max_geometric_mean = np.max(np.array([gmean(molecule.fitnesses) for molecule in molecules])) 
            dominated_hypervolume = pg.hypervolume(np.array([-1.0*np.array(molecule.fitnesses) for molecule in molecules if np.linalg.norm(molecule.fitnesses) > 0.00])).compute(np.zeros(len(molecules[0].fitnesses)))
            internal_similarity = self.internal_similarity(molecules)
            fitness_calls = self.fitness.calls
            print(f'Generation: {generation}, Max geometric mean: {max_geometric_mean}, Hypervolume: {dominated_hypervolume}, Fraction: {fraction}, Internal Similarity: {internal_similarity}, Fitness Calls: {fitness_calls}')
            self.store_data(generation, molecules, fraction, max_geometric_mean, dominated_hypervolume, internal_similarity, fitness_calls)
        return None

    def initial_population(self) -> None:
        molecules = self.arbiter(self.load_from_database())
        molecules = self.calculate_fitnesses(molecules)
        molecules = self.directional_selection(molecules)
        molecules = self.fingerprints(molecules)
        return molecules

    def load_from_database(self) -> List[Molecule]:
        dataframe = pd.read_csv(hydra.utils.to_absolute_path(self.data_file))
        smiles_list = dataframe['smiles'].sample(n=self.initial_size).tolist()
        molecules = [Molecule(Chem.CanonSmiles(smiles)) for smiles in smiles_list]
        return molecules
    
    def store_data(self, generation, molecules, fraction, max_geometric_mean, dominated_hypervolume, internal_similarity, fitness_calls) -> None:
        if os.path.isfile('statistics.csv'):
            with open('statistics.csv', 'a') as file:
                csv.writer(file).writerow([generation, fraction, max_geometric_mean, dominated_hypervolume, internal_similarity, fitness_calls])
                file.close()
        else:
            with open('statistics.csv', 'w') as file:
                csv.writer(file).writerow(["generation", "fraction", "max geometric mean", "dominated hypervolume", "internal similarity", "fitness calls (cumulative)"])
                csv.writer(file).writerow([generation, fraction, max_geometric_mean, dominated_hypervolume, internal_similarity, fitness_calls])
                file.close()
        archive_data = {'smiles': [molecule.smiles for molecule in molecules], 'fitnesses': [molecule.fitnesses for molecule in molecules], 'rank': [molecule.rank for molecule in molecules], 'nearest_direction': [molecule.nearest_direction for molecule in molecules] , 'orthogonal_distance': [molecule.orthogonal_distance for molecule in molecules]}
        pd.DataFrame(data=archive_data).to_csv("archive_{}.csv".format(generation), index=False)
        return None

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

    def directional_selection(self, molecules: List[Molecule]) -> List[Molecule]:
        selected, remaining = [], []
        possible_directions, underrepresented_directions = [], []
        fronts = self.calculate_fronts(molecules)        
        for front in fronts: 
            if len(selected) + len(front) > self.population_size:
                splitting_front = front
                possible_directions = [self.nearest_direction(molecule) for molecule in splitting_front]
                break
            else:
                selected.extend(front)
        if self.population_size - len(selected) <= 0:
            return selected
        current_directions = [self.nearest_direction(molecule) for molecule in selected]
        for direction in possible_directions:
            if direction in set(current_directions):
                remaining.extend([molecule for molecule in splitting_front if molecule.nearest_direction == direction])
            else: 
                underrepresented_directions.append(direction)
        if len(underrepresented_directions) > 0: 
            for underrepresented_direction in set(underrepresented_directions):
                underrepresented_molecules = [molecule for molecule in splitting_front if molecule.nearest_direction == underrepresented_direction] 
                underrepresented_molecules.sort(key = lambda molecule : molecule.orthogonal_distance)
                selected.append(underrepresented_molecules[0])
                if len(underrepresented_molecules) > 1: 
                    remaining.extend(underrepresented_molecules[1:])
                if (self.population_size - len(selected) <= 0):
                    break
        leftover_size = min(self.population_size - len(selected), len(remaining))
        geometric_means = [gmean(molecule.fitnesses) for molecule in remaining]      
        if len(remaining) > 0:
        	selected.extend(random.choices(remaining, k = leftover_size, weights = geometric_means))
        return selected

    def nearest_direction(self, molecule: Molecule) -> float: 
        fitness_vector = np.array(molecule.fitnesses)
        orthogonal_distances = [self.orthogonal_distance(fitness_vector, reference_direction) for reference_direction in self.reference_directions]
        orthogonal_distance = min(orthogonal_distances)
        nearest_direction = orthogonal_distances.index(orthogonal_distance) 
        molecule.orthogonal_distance = orthogonal_distance
        molecule.nearest_direction = nearest_direction
        return nearest_direction

    def orthogonal_distance(self, fitness_vector, reference_direction) -> float: 
        normalised_fitness_vector = fitness_vector/np.linalg.norm(fitness_vector)
        return np.linalg.norm(normalised_fitness_vector - np.dot(normalised_fitness_vector, reference_direction)*reference_direction)

    def pareto_dominates(self, molecule_a: Molecule, molecule_b: Molecule) -> int:
        larger_or_equal = molecule_a.fitnesses >= molecule_b.fitnesses
        larger = molecule_a.fitnesses > molecule_b.fitnesses
        if np.all(larger_or_equal) and np.any(larger):
            return True
        return False

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
    current_instance = NSGA3(config)
    current_instance()
    current_instance.client.close()

if __name__ == "__main__":
    launch()
