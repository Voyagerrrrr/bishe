#!/bin/bash
#SBATCH --time=01:59:00
#SBATCH --mem=128gb
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:1

source /home/demiao/miniconda3/etc/profile.d/conda.sh
conda activate bishe

echo "Architecture: $1, Trained On: $2, Model no.: $3";

_ROOT='/home/demiao/bishe'
# _WEIGHTS="${_ROOT}/weights/$1_$2.pth" 
_WEIGHTS="${_ROOT}/weights/batch_jobs/$1_$2_$3/checkpoint_final.pth"
_RESULTS="${_ROOT}/results/results_standard/$1_$2_$3.csv" 

export PYTHONPATH=$PYTHONPATH:$_ROOT

python $_ROOT/eval/eval_wrapper.py \
    --config $_ROOT/config/eval_datasets/$1.yaml \
    --weights $_WEIGHTS \
    --save_results $_RESULTS \
