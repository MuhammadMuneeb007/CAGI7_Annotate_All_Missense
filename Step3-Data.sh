#!/bin/bash
#SBATCH --job-name=GeneExpressionPrediction
#SBATCH --nodes=1
#SBATCH --partition=ascher
#SBATCH --time=240:00:00
#SBATCH --output=ttemp.%A_%a.out
#SBATCH --error=ttemp.%A_%a.err
#SBATCH --array=1-76
#SBATCH --mem=50G

# Path to the file containing all ANNOVAR commands (one per line)
COMMANDS_FILE="commands.txt"

# Get the command for this array task
CMD=$(sed -n "${SLURM_ARRAY_TASK_ID}p" $COMMANDS_FILE)

# Print and execute the command
echo "Running command: $CMD"
eval $CMD
