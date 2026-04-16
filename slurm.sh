#!/bin/bash
#SBATCH --job-name=MyFirstTestRun
#SBATCH --output=/scratch/ssci-adityag/%x-%j.out
#SBATCH --error=/scratch/ssci-adityag/%x-%j.err
#SBATCH --partition=b200
#SBATCH --nodes=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:4

#SBATCH --time 7-00:00:00
# module load cuda/13.0.2
sleep infinity  