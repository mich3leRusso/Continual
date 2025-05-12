#!/bin/bash

#PBS -N synbols_CSIx4o
#PBS -o exp.txt
#PBS -q gpu
#PBS -e exp.txt
#PBS -k oe
#PBS -m e
#PBS -M davide.mor@leonardo.com
#PBS -l select=1:ngpus=1:ncpus=4,walltime=72:00:00

# Add conda to source
source /archive/apps/miniconda/miniconda3/py312_2/etc/profile.d/conda.sh
# Conda activate
conda activate env_9

# Experiments Table 1 (A) - Synbols
for seed in 0 #1 2 3 4 5 6 7 8 9;
do
    python /davinci-1/home/dmor/PycharmProjects/Refactoring_MIND/main.py --run_name "synbols_CSIx4" \
            --dataset "Synbols" \
            --cuda 0 \
            --seed $seed \
            --n_experiences 10 \
            --model "gresnet32" \
            --epochs 5 \
            --lr 0.005 \
            --scheduler 10 20 \
            --epochs_distillation 5 \
            --lr_distillation 0.035 \
            --scheduler_distillation 15 \
            --temperature 4 \
            --class_augmentation 1 \

done