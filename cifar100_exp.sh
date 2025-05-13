#!/bin/bash

#PBS -N cifar100_CSIx2
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

for seed in 0 #1 2 3 4 5 6 7 8 9;
#mode = 0 baseline, 3 controllo, 4 CSI
do
    python /davinci-1/home/dmor/PycharmProjects/Refactoring_MIND/main.py --run_name "Refactoring" \
            --dataset "CIFAR100" \
            --cuda 0 \
            --seed $seed \
            --n_experiences 10 \
            --model "gresnet32" \
            --epochs 50 \
            --lr 0.005 \
            --scheduler 35 \
            --epochs_distillation 50 \
            --lr_distillation 0.035 \
            --scheduler_distillation 40 \
            --temperature 6.5 \
            --class_augmentation 2
done