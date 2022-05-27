#!/bin/bash
source /home/jove/miniconda3/etc/profile.d/conda.sh
conda activate komorebi-develop
for i in {0..19}
do
  echo "Run: $i"
  python nsga2.py -m fitness.task=mpo_cobimetinib,mpo_fexofenadine,mpo_ranolazine,mpo_pioglitazone,mpo_osimertinib,mpo_dap_kinases,mpo_antipsychotics
done


