#!/bin/bash

#SBATCH -p gpu_shared
#SBATCH -N 1
#SBATCH --gpus=1
#SBATCH --time=24:00:00

#SBATCH --job-name=RoBERTa_25_encdec
#SBATCH --output=/home/dahmanir/lisa/Jobberts/Final/Out_files/RoBERTa_25_encdec.out

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
fairseq-hydra-train -m  --config-dir lisa/Configs/Final/ --config-name RoBERTa_25_encdec_dec_only
