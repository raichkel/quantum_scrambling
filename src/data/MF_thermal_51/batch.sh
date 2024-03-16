#!/bin/bash

# Author: Rachel Kane
#SBATCH --job-name=MF_thermal
#SBATCH --nodes=1
#SBATCH --partition=QuamNESS
#SBATCH --ntasks-per-node=28
#SBATCH --cpus-per-task=1
#SBATCH --time=72:00:00
#SBATCH --mem-per-cpu=1000M
#SBATCH --account=phys025062




## Direct output to the following files.

#SBATCH -e scramble_err.txt
#SBATCH -o scramble_out.txt


# load in Julia
module purge
module load lang/julia/1.8.5


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

julia old_tdvp_thermal_state.jl 55.0 51 "MF" 0.1 1.0 1.0 1.0

#T, N, H, ß, Jx, Jy, Jz

# Output the end time
printf "\n\n"
echo "Ended on: $(date)"
