#!/bin/bash

#PBS -N tiny_CSIx4
#PBS -o continual_2.txt
#PBS -q gpu
#PBS -e continual_2.txt
#PBS -k oe
#PBS -m e
#PBS -M davide.mor@leonardo.com
#PBS -l select=1:ngpus=1:ncpus=12,walltime=240:00:00

# Add conda to source
source /archive/apps/miniconda/miniconda3/py312_2/etc/profile.d/conda.sh
# Conda activate
conda activate env_9

# Experiments Table 1 (A) - TinyImageNet
for seed in 0 #1 2 3 4 5 6 7 8 9;
do
    python /davinci-1/home/dmor/PycharmProjects/Refactoring_MIND/main.py --run_name "tiny_CSIx4" \
            --dataset "TinyImageNet" \
            --cuda 0 \
            --seed $seed \
            --n_experiences 10 \
            --model "gresnet32" \
            --epochs 10 \
            --lr 0.005 \
            --scheduler 70 90 \
            --epochs_distillation 12 \
            --lr_distillation 0.035 \
            --scheduler_distillation 80 110 \
            --temperature 12 \
            --class_augmentation 2

done