#!/bin/bash
#!/bin/bash

# Experiments Table 1 (A) - tinyimagenet
#PBS -N one_ring_test_core50
#PBS -o one_ring_test_core50.txt
#PBS -q gpu
#PBS -e one_ring_test_50.txt
#PBS -k oe
#PBS -m e
#PBS -M michele.russo03@leonardo.comg
#PBS -l select=1:ngpus=1:ncpus=7,walltime=90:00:00

cd /davinci-1/home/micherusso/PycharmProjects/MIND_real
source /archive/apps/miniconda/miniconda3/py312_2/etc/profile.d/conda.sh

conda activate MIND



for seed in   8 9 ;
do
    python /davinci-1/home/micherusso/PycharmProjects/MIND_real/main2.py --run_name "core50_experiment" \
            --dataset "CORE50_CI" \
            --cuda 0 \
            --seed $seed \
            --n_experiences 10 \
            --model "gresnet32" \
            --epochs 20 \
            --lr 0.005 \
            --scheduler 15\
            --epochs_distillation 25 \
            --lr_distillation 0.035 \
            --scheduler_distillation  15 \
            --temperature 6 \
            --sweep 15 \
            --load_model_from_run "core50_experiment" \
            --number_perturbations 10 20




done

