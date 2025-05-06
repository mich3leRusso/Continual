#!/bin/bash

#PBS -N plot_core50_CSIx3_ruotato
#PBS -o Odin_002-.txt
#PBS -q gpu
#PBS -e Odin_002-.txt
#PBS -k oe
#PBS -m e
#PBS -M davide.mor@leonardo.com
#PBS -l select=1:ngpus=1:ncpus=4,walltime=150:00:00

# Add conda to source
source /archive/apps/miniconda/miniconda3/py312_2/etc/profile.d/conda.sh
# Conda activate
conda activate env_9


python /davinci-1/home/dmor/PycharmProjects/MIND/num_pert_2.py --run_name "tiny_CSIx3" \
        --dataset "CORE50_CI" \
        --cuda 0 \
        --seed 1 \
        --n_experiences 10 \
        --model "gresnet32" \
        --temperature 3 \
        --mode 4 \
        --extra_classes 10 \
        --aug_inf 1 \
        --num_aug 100 \
        --dropout 0.0
