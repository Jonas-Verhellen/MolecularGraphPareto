# Pareto Optimisation for Molecular Design

</div>

[![GitHub issues](https://img.shields.io/github/issues/Jonas-Verhellen/MolecularGraphPareto)](https://github.com/Jonas-Verhellen/MolecularGraphPareto/issues)
![GitHub](https://img.shields.io/github/license/Jonas-Verhellen/MolecularGraphPareto)
[![DOI](https://img.shields.io/badge/DOI-10.1039/D2SC00821A-blue)](https://pubs.rsc.org/en/content/articlehtml/2022/sc/d2sc00821a)

</div>

![Logo](/figures/logo.png "Logo")

## Description

This repo contains open-source implementations of NSGA-II, NSGA-III and GB-EPI for optimization of small organic molecules.

![Logo](/figures/overview.png "Overview")



## Instuctions on Running the Code

To manage configuration files and outputs we make use of [Hydra](), which allows us to dynamically override configuration files through the command line. Example bash files (called run.sh) of how to apply these capabilities to the different algorithms to create multiruns can be found in the each algorithm folder. To use the algorithms for a single run, go to the configuration file, set your preferences, and run the algorithm from its homefolder. For instance for NSGA-III, the configuration file looks like 

```
---
data_file: data/smiles/guacamol_filtered.smi
batch_size: 20
initial_size: 100
population_size: 35
direction_size: 25
workers: 1
threads: 2
generations: 150
fitness:
  task: "mpo_pioglitazone"
arbiter:
  rules:
  - Glaxo
mutator:
  data_file: data/smarts/mutation_collection.tsv
```

and the algorithms can be run by the following commands 


```
python illuminate.py 
```


```
python nsga2.py 
```

```
python nsga3.py 
```


## Dependencies

Important dependencies of the Argenomic software environment and where to find the source.

* [Python](https://www.python.org/) - Python is a widely used scientific and numeric programming language.
* [RDKit](https://github.com/rdkit/rdkit) - Cheminformatics and machine-learning software toolkit, used for storing and manipulating molecules.
* [Pymoo](https://pymoo.org/index.html) - Multi-objective optimization toolkit in Python, providing state-of-the-art reference directions.
* [Pygmo](https://github.com/esa/pygmo2) - Massively parallel optimization library in python, providing implementations of fast dominatefd hypervolume methods. 
* [MultipleComparisons](https://github.com/ramirandaq/MultipleComparisons) - Reference implementations of the extended similarity metrics used to calculate the internal similarity. 
* [Omegaconf](https://github.com/omry/omegaconf) - Configuration system for multiple sources, providing a consistent API.

## Authors
Based on the paper *[Graph-based molecular Pareto optimisation.](https://pubs.rsc.org/en/content/articlehtml/2020/sc/d0sc03544k) Chemical Science 13.25 (2022): 7526-7535.*
* **Jonas Verhellen**

## Acknowledgments

* Pat Walters for his scripts indicating how to run structural alerts using the RDKit and ChEMBL, and for his many enlightening medicinal chemistry blog posts.

## Copyright License

This project is licensed under the GNU Affero General Public License v3.0 (AGPLv3).
