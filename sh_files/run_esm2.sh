#!/bin/bash
#SBATCH --mem=30gb
#SBATCH -c8
#SBATCH --gres=gpu:1,vmem:20g
#SBATCH --time=05:00:00
#SBATCH --error=/cs/labs/dina/sapir_amittai/code/spring_24/TCRep/info_sc_computer/error%A.txt
#SBATCH --output=/cs/labs/dina/sapir_amittai/code/spring_24/TCRep/info_sc_computer/output%A.txt
#SBATCH --job-name=RunEsm2ForEmbedding
#SBATCH --killable

module load nvidia
nvidia-smi

cd /cs/labs/dina/sapir_amittai/code/spring_24/TCRep/
ls -l
source /cs/labs/dina/sapir_amittai/code/spring_24/gal_proj_env/bin/activate

echo z============================

python /cs/labs/dina/sapir_amittai/code/spring_24/TCRep/main.py
