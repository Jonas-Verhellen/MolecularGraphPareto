![Logo](/figures/logo.png "Logo")

## Description

This repo contains open-source implementations of NSGA-II, NSGA-III and GB-EPI for optimization of small organic molecules.

![Logo](/figures/overview.png "Overview")

## Authors

* **Jonas Verhellen**

## Dependencies

Important dependencies of the Argenomic software environment and where to find the source.

* [Python](https://www.python.org/) - Python is a widely used scientific and numeric programming language.
* [RDKit](https://github.com/rdkit/rdkit) - Cheminformatics and machine-learning software toolkit, used for storing and manipulating molecules.
* [Pymoo](https://pymoo.org/index.html) - Multi-objective optimization toolkit in Python, providing state-of-the-art reference directions.
* [Pymoo](https://github.com/esa/pygmo2) - Massively parallel optimization library in python, providing implementations of fast dominatefd hypervolume methods. 
* [MultipleComparisons] (https://github.com/ramirandaq/MultipleComparisons) - Reference implementations of the extended similarity metrics used to calculate the internal similarity. 
* [Omegaconf](https://github.com/omry/omegaconf) - Configuration system for multiple sources, providing a consistent API.

## Instuctions on Running the Code

To manage configuration files and outputs we make use of [Hydra](), which allows us to dynamically override configuration files through the command line. Examples bash files (called run.sh) of how to apply these capabilities to the different algorithms to create multiruns can be found in the each algorithm folder. To use the algorithms for a single run, go to the configuration file, set your preferences, and run the algorithm from its homefolder. For instance for NSGA-III, the configuration file looks like 

```
---
data_file: data/smiles/guacamol_filtered.smi
batch_size: 20
initial_size: 100
workers: 1
threads: 2
generations: 150
archive:
  name: mpo_cobimetinib
  size: 150
  accuracy: 25000
descriptor:
  properties:
  - Descriptors.ExactMolWt
  - Descriptors.MolLogP
  - Descriptors.TPSA
  - Crippen.MolMR
  ranges:
  - - 140
    - 555
  - - 0.0
    - 7.0
  - - 0
    - 140
  - - 40
    - 130
fitness:
  task: mpo_cobimetinib
arbiter:
  rules:
  - Glaxo
mutator:
  data_file: data/smarts/mutation_collection.tsv

```

and the algorithm can be run by 

```
python nsga3.py 
```


## Acknowledgments

* Pat Walters for his scripts indicating how to run structural alerts using the RDKit and ChEMBL, and for his many enlightening medicinal chemistry blog posts.

## Copyright License

This project is licensed under the GNU Affero General Public License v3.0 (AGPLv3).
