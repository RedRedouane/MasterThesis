#!/bin/bash

#SBATCH -p gpu_shared
#SBATCH -N 1
#SBATCH --gpus=1
#SBATCH --time=00:10:00

#SBATCH --job-name=RTL_test
#SBATCH --output=/home/dahmanir/lisa/Jobberts/Final/Out_files/RTL_test.out

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
fairseq-generate lisa/Datasets/RTL_totalset_bin \
    --path lisa/Models/BART_10_summ.pt \
    --max-len-a 0 \
    --max-len-b 10 \
    --batch-size 1 \
    --skip-invalid-size-inputs-valid-test > RTL_outputs/BART_10.txt

fairseq-generate lisa/Datasets/RTL_totalset_bin \
    --path lisa/Models/BART_25_summ.pt \
    --max-len-a 0 \
    --max-len-b 10 \
    --batch-size 1 \
    --skip-invalid-size-inputs-valid-test > RTL_outputs/BART_25.txt

fairseq-generate lisa/Datasets/RTL_totalset_bin \
    --path lisa/Models/BART_50_summ.pt \
    --max-len-a 0 \
    --max-len-b 10 \
    --batch-size 1 \
    --skip-invalid-size-inputs-valid-test > RTL_outputs/BART_50.txt

fairseq-generate lisa/Datasets/RTL_totalset_bin \
    --path lisa/Models/RoBERTa_10_encdec_finetune.pt \
    --max-len-a 0 \
    --max-len-b 10 \
    --batch-size 1 \
    --skip-invalid-size-inputs-valid-test > RTL_outputs/RoBERTa_10.txt

fairseq-generate lisa/Datasets/RTL_totalset_bin \
    --path lisa/Models/RoBERTa_25_encdec_finetune.pt \
    --max-len-a 0 \
    --max-len-b 10 \
    --batch-size 1 \
    --skip-invalid-size-inputs-valid-test > RTL_outputs/RoBERTa_25.txt

fairseq-generate lisa/Datasets/RTL_totalset_bin \
    --path lisa/Models/RoBERTa_50_encdec_finetune.pt \
    --max-len-a 0 \
    --max-len-b 10 \
    --batch-size 1 \
    --skip-invalid-size-inputs-valid-test > RTL_outputs/RoBERTa_50.txt

fairseq-generate lisa/Datasets/RTL_totalset_bin \
    --path lisa/Models/RoBERTa_100_encdec_finetune.pt \
    --max-len-a 0 \
    --max-len-b 10 \
    --batch-size 1 \
    --skip-invalid-size-inputs-valid-test > RTL_outputs/RoBERTa_100.txt

fairseq-generate lisa/Datasets/RTL_totalset_bin \
    --path lisa/Models/RobBERT_encdec_finetune.pt \
    --max-len-a 0 \
    --max-len-b 10 \
    --batch-size 1 \
    --skip-invalid-size-inputs-valid-test > RTL_outputs/RobBERT.txt
