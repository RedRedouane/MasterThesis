#!/bin/bash

#SBATCH -p gpu_shared
#SBATCH -N 1
#SBATCH --gpus=1
#SBATCH --time=01:30:00

#SBATCH --job-name=europarl_test_RoBERTa_10
#SBATCH --output=/home/dahmanir/lisa/Jobberts/Final/Out_files/europarl_test_RoBERTa_10.out

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
python -m fairseq_cli.validate lisa/Datasets/europarl_nl_bin --path lisa/Models/RoBERTa_10_finetune.pt --task masked_lm --batch-size 8 --skip-invalid-size-inputs-valid-test
