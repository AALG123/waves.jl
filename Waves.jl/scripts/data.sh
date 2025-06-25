#!/bin/bash
#SBATCH -J data
#SBATCH -p gpuqs
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=2G
#SBATCH -t 2-00:00:00
#SBATCH -w cs004
#SBATCH -o logs/log-%j.out
#SBATCH --mail-user=assem.alghathi@sjsu.edu
module load julia
module load nvhpc-hpcx-cuda11/.24.11.bak
module load nvhpc-hpcx-cuda12/24.11
module load hpcx
srun julia --project scripts/data.jl