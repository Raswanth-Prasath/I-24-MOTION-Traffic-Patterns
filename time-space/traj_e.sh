#!/bin/bash
#SBATCH -N 1                    # number of nodes
#SBATCH --mem=256G              # amount of memory for the job
#SBATCH -c 1                   # number of CPU cores (10 cores)
#SBATCH -t 10:00:00       # time limit (24 hours, adjust as needed)
#SBATCH -p general              # partition
#SBATCH -q public               # QOS
#SBATCH -o traj-e.out      # file to save job's STDOUT (%j = JobId)
#SBATCH -e traj-e.err      # file to save job's STDERR (%j = JobId)
#SBATCH --mail-type=ALL         # Send an e-mail when a job starts, stops, or fails
#SBATCH --mail-user=raswanth@asu.edu  # Replace with your ASU email

# Load required modules
module load mamba/latest

source activate tds
source deactivate

# Activate your conda environment
conda activate tds # Replace with your actual environment name

# Change to the directory of your script
#cd ~/path/to/your/project/directory   # Replace with your actual directory path

# Run the Python script
python Trajectory_E_v1.py 