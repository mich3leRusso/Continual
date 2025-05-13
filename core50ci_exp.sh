#!/bin/bash

#PBS -N Core50_CSIx4
#PBS -o continual_1.txt
#PBS -q gpu
#PBS -e continual_1.txt
#PBS -k oe
#PBS -m e
#PBS -M davide.mor@leonardo.com
#PBS -l select=1:ngpus=1:ncpus=12,walltime=72:00:00

# Add conda to source
source /archive/apps/miniconda/miniconda3/py312_2/etc/profile.d/conda.sh
# Conda activate
conda activate env_9

# Experiments Table 1 (A) - CORE50-CI
for seed in 0 #1 2 3 4 5 6 7 8 9;
do
    python /davinci-1/home/dmor/PycharmProjects/Refactoring_MIND/main.py --run_name "core50_CSIx4" \
            --dataset "CORE50_CI" \
            --cuda 0 \
            --seed $seed \
            --n_experiences 10 \
            --model "gresnet32" \
            --epochs 20 \
            --lr 0.005 \
            --scheduler 15 \
            --epochs_distillation 20 \
            --lr_distillation 0.035 \
            --scheduler_distillation 15 \
            --temperature 3 \
            --class_augmentation 2

done