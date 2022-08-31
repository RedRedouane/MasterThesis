#!/bin/bash

#SBATCH -p gpu_shared
#SBATCH -N 1
#SBATCH --gpus=2
#SBATCH --time=120:00:00

#SBATCH --job-name=BART_10
#SBATCH --output=/home/dahmanir/lisa/Jobberts/Final/Out_files/BART_10.out

# Load modules
module purge
module load 2021
module load Python/3.9.5-GCCcore-10.3.0
module load CUDA/11.3.1
module load cuDNN/8.2.1.32-CUDA-11.3.1
module load NCCL/2.10.3-GCCcore-10.3.0-CUDA-11.3.1
module load Anaconda3/2021.05

# Activate your environment
source activate test2

# Your job starts in the directory where you call sbatch
cd $HOME/

# Run your code
python -O fairseq/train.py "/home/dahmanir/lisa/Datasets/10_percent" --restore-file "/home/dahmanir/lisa/Models/bart.pt"  --fp16 --mask 0.3 --tokens-per-sample 512 --total-num-update 500000 --max-update 500000 --warmup-updates 500 --task denoising  --arch bart_base --optimizer adam --lr-scheduler polynomial_decay --lr 0.0004  --dropout 0.1 --criterion cross_entropy --max-tokens 3200 --weight-decay 0.01 --attention-dropout 0.1 --clip-norm 0.1 --skip-invalid-size-inputs-valid-test --log-format json --log-interval 200 --update-freq 32 --seed 4 --mask-length span-poisson --replace-length 1 --rotate 0.0 --mask-random 0.1 --permute-sentences 1.0 --insert 0.0 --poisson-lambda 3.5 --wandb-project BART