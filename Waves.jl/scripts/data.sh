#!/bin/bash
#SBATCH -J data
#SBATCH -p gpuqs
#SBATCH -N 1
# OPTIMIZATION: Increased from 1 to 8 cores for parallel episode generation
# This allows Julia to use multiple threads simultaneously
#SBATCH -n 8
#SBATCH -c 1
# OPTIMIZATION: Increased memory from 2G to 16G
# More memory allows for larger batches and prevents out-of-memory errors
# when generating episodes in parallel
#SBATCH --mem=16G
# OPTIMIZATION: Extended runtime from 2 days to 4 days
# Longer runtime allows generation of more episodes (5000 vs 500)
#SBATCH -t 4-00:00:00
#SBATCH -w cs004
#SBATCH -o logs/log-%j.out
#SBATCH --mail-user=example@sjsu.edu

# Load required modules
module load julia
module load nvhpc-hpcx-cuda11/.24.11.bak
module load nvhpc-hpcx-cuda12/24.11
module load hpcx

# OPTIMIZATION: Set Julia to use 8 threads for parallel processing
# This environment variable tells Julia how many threads to use
export JULIA_NUM_THREADS=8

# OPTIMIZATION: Run Julia with threading enabled (-t 8)
# This matches the number of SLURM cores and JULIA_NUM_THREADS
# for maximum parallel efficiency
srun julia --project -t 8 scripts/data.jl
