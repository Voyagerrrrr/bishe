#!/bin/bash
#SBATCH --time=08:00:00
#SBATCH --mem=128gb
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:1

source /home/demiao/miniconda3/etc/profile.d/conda.sh 
conda activate bishe

echo "Trained On: $1";

_ROOT='/home/demiao/bishe' 
_SAVEDIR="${_ROOT}/weights/batch_jobs/dropout_minkloc_$1"

if [ "$1" == "oxford" ]; then
    trainfile="oxford/training_queries_oxford"
fi

export PYTHONPATH=$PYTHONPATH:$_ROOT

python $_ROOT/training/train.py  \
    --config $_ROOT/config/eval_datasets/minkloc3d_mcdropout.yaml \
    --uncertainty_method dropout \
    --teacher_net "${_ROOT}/weights/minkloc_$1.pth" \
    data.train_file "${_ROOT}/pickles/${trainfile}.pickle" \
    save_path $_SAVEDIR \

