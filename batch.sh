#!/bin/bash

# Author: Rachel Kane
#SBATCH --job-name=scramble
#SBATCH --partition=teach_cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=28
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:0
#SBATCH --mem-per-cpu=1000M
#SBATCH --account=phys026162




## Direct output to the following files.

#SBATCH -e scramble.txt
#SBATCH -o scramble.txt


# load in Julia
module purge
module load languages/julia/1.8.5


# Change to working directory, where the job was submitted from.
cd "${SLURM_SUBMIT_DIR}"

# Record some potentially useful details about the job: 
echo "Running on host $(hostname)"
echo "Started on $(date)"
echo "Directory is $(pwd)"
echo "Slurm job ID is ${SLURM_JOBID}"
echo "This jobs runs on the following machines:"
echo "${SLURM_JOB_NODELIST}" 
printf "\n\n"

julia tdvp.jl 20.0 21

# Output the end time
printf "\n\n"
echo "Ended on: $(date)"
