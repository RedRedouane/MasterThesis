#!/bin/bash

#SBATCH -p gpu_shared
#SBATCH -N 1
#SBATCH --gpus=1
#SBATCH --time=24:00:00

#SBATCH --job-name=BART_25_summ
#SBATCH --output=/home/dahmanir/lisa/Jobberts/Final/Out_files/BART_25_summ.out

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
python -O fairseq/train.py "/home/dahmanir/lisa/Datasets/wiki_binarized" --restore-file "/home/dahmanir/lisa/Models/BART_25_finetune.pt" --reset-dataloader --reset-meters --reset-optimizer --fp16 --total-num-update 500000 --max-update 500000 --max-epoch 50 --warmup-updates 500 --task translation  --arch bart_base --optimizer adam --lr-scheduler polynomial_decay --lr 0.00001  --dropout 0.1 --criterion label_smoothed_cross_entropy --max-tokens 3200 --weight-decay 0.01 --attention-dropout 0.1 --clip-norm 0.1 --skip-invalid-size-inputs-valid-test --log-format json --log-interval 200 --update-freq 32 --seed 4 --wandb-project BART_summ --save-dir checkpoints_BART_25_summ --no-epoch-checkpoints
