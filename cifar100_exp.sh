#!/bin/bash

# Experiments Table 1 (A) - cifar100
#PBS -N one_ring_test
#PBS -o one_ring_test.txt
#PBS -q gpu
#PBS -e one_ring_test.txt
#PBS -k oe
#PBS -m e
#PBS -M michele.russo03@leonardo.comg
#PBS -l select=1:ngpus=1:ncpus=5,walltime=20:00:00

cd /davinci-1/home/micherusso/PycharmProjects/MIND_real
source /archive/apps/miniconda/miniconda3/py312_2/etc/profile.d/conda.sh

conda activate MIND

#for seed in 8 9;
#do
  python  /davinci-1/home/micherusso/PycharmProjects/MIND_real/main.py --run_name "cifar100_experiment" \
              --dataset "CIFAR100" \
              --cuda 0 \
              --seed 0 \
              --n_experiences 10 \
              --model "gresnet32" \
              --epochs 50 \
              --lr 0.005 \
              --scheduler 35 \
              --epochs_distillation 60 \
              --lr_distillation 0.035 \
              --scheduler_distillation 40 \
              --temperature 3.5 6.5 8.5  \
              --load_model_from_run "cifar100_experiment"

#done
