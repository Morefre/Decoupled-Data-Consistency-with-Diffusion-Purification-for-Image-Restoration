#!/bin/bash
#SBATCH --account=qingqu1
#SBATCH --job-name=motion_deblur
#SBATCH --nodes=1
#SBATCH --time=48:00:00
#SBATCH --mem=47GB
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --output=motion_deblur.log
#SBATCH --mail-user=forkobe@umich.edu
#SBATCH --mail-type=END

module purge
# module load cuda/10.2.89
module load python3.9-anaconda/2021.11
eval "$(conda shell.bash hook)"
conda activate stable_diffusion

python dcdp.py --task_config=./task_configurations/motion_deblur_config.yaml --purification_config=./purification_configurations/purification_config_motion_deblur.yaml \
                 --model_config=./model_configurations/model_config_ffhq.yaml 