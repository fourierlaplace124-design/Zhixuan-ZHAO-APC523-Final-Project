#!/bin/bash
#SBATCH --job-name=apc523-fdm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%x-%j.out

module purge
module load anaconda3/2025.6
conda activate apc523-fdm

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

bash run_adroit.sh "${RUN_MODE:-verify}"
