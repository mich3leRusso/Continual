#!/bin/bash

# Experiments Table 1 (A) - tinyimagenet
#PBS -N one_ring_test_tiny
#PBS -o one_ring_test_tiny.txt
#PBS -q gpu
#PBS -e one_ring_test_tiny.txt
#PBS -k oe
#PBS -m e
#PBS -M michele.russo03@leonardo.comg
#PBS -l select=1:ngpus=1:ncpus=5,walltime=90:00:00

cd /davinci-1/home/micherusso/PycharmProjects/MIND_real
source /archive/apps/miniconda/miniconda3/py312_2/etc/profile.d/conda.sh

conda activate MIND



for seed in 3 4 5;
do
    python /davinci-1/home/micherusso/PycharmProjects/MIND_real/main.py --run_name "tinyimgnet_experiment" \
            --dataset "TinyImageNet" \
            --cuda 0 \
            --seed $seed \
            --n_experiences 10 \
            --model "gresnet32" \
            --epochs 100 \
            --lr 0.005 \
            --scheduler 70 90 \
            --epochs_distillation 120 \
            --lr_distillation 0.035 \
            --scheduler_distillation 80 110 \
            --temperature 5 7 10 12 \
            --sweep 90 \
            --load_model_from_run "tinyimgnet_experiment"
done