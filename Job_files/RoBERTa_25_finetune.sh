#!/bin/bash

#SBATCH -p gpu_shared
#SBATCH -N 1
#SBATCH --gpus=2
#SBATCH --time=120:00:00

#SBATCH --job-name=RoBERTa_25
#SBATCH --output=/home/dahmanir/lisa/Jobberts/Final/Out_files/RoBERTa_25_finetune.out

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

# Copy input data and model to scratch disk
#cp -r $HOME/lisa/Datasets/base_test "$TMPDIR"
#cp $HOME/lisa/Models/model_adjusted.pt "$TMPDIR"

# Run your code
fairseq-hydra-train -m --config-dir lisa/Configs/Final/ --config-name RoBERTa_25_finetune

# Copy output files back to home folder
