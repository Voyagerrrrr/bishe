#!/bin/bash
#SBATCH --time=02:00:00
#SBATCH --mem=128gb
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:1

source /home/demiao/miniconda3/etc/profile.d/conda.sh 
conda activate bishe 

echo "Trained On: $1";

_ROOT='/home/demiao/bishe' 
# _WEIGHTS="${_ROOT}/weights/pfe_stun_dropout/dropout_minkloc_$1.pth" 
_WEIGHTS="${_ROOT}/weights/batch_jobs/dropout_minkloc_$1/checkpoint_final.pth" 
_RESULTS="${_ROOT}/save/dropout_minkloc.csv" 

export PYTHONPATH=$PYTHONPATH:$_ROOT

python $_ROOT/eval/eval_wrapper.py \
    --config $_ROOT/config/eval_datasets/minkloc3d_mcdropout.yaml \
    --uncertainty_method dropout \
    --dropout_passes 5 \
    --weights $_WEIGHTS \
    save_results $_RESULTS \
