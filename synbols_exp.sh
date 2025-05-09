#!/bin/bash


# Experiments Table 1 (A) - tinyimagenet
#PBS -N one_ring_test_synbols
#PBS -o one_ring_test_synbols.txt
#PBS -q gpu
#PBS -e one_ring_test_synbols.txt
#PBS -k oe
#PBS -m e
#PBS -M michele.russo03@leonardo.comg
#PBS -l select=1:ngpus=1:ncpus=5,walltime=90:00:00

cd /davinci-1/home/micherusso/PycharmProjects/MIND_real
source /archive/apps/miniconda/miniconda3/py312_2/etc/profile.d/conda.sh

conda activate MIND


# Experiments Table 1 (A) - Synbols
for seed in 0 1 2 3 4 5 6 7 8 9;
do

    python main.py --run_name "synbols_experiment" \
            --dataset "Synbols" \
            --cuda 0 \
            --seed $seed \
            --n_experiences 10 \
            --model "gresnet32" \
            --epochs 25 \
            --lr 0.005 \
            --scheduler 10 20 \
            --epochs_distillation 25 \
            --lr_distillation 0.035 \
            --scheduler_distillation 15 \
            --temperature  3 \
            --sweep 15 \
            --load_model_from_run "synbols_experiment" \
            --number_perturbations 10 20 \


done