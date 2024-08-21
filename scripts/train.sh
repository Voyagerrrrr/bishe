#!/bin/bash
#SBATCH --time=08:00:00
#SBATCH --mem=40gb
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:1

source /home/demiao/miniconda3/etc/profile.d/conda.sh 
conda activate bishe 
#export OMP_NUM_THREADS=1 # set maximum threads
echo "Architecture: $1, Trained On: $2, Model no.: $3";

_ROOT='/home/demiao/bishe'
_SAVEDIR="${_ROOT}/weights/batch_jobs/$1_$2_$3" 
_RESULTS="${_ROOT}/results/results_standard/$1_$2_$3.csv" 

if [ "$2" == "oxford" ]; then
    trainfile="oxford/training_queries_oxford"
fi

export PYTHONPATH=$PYTHONPATH:$_ROOT

python $_ROOT/training/train.py  \
    --config $_ROOT/config/eval_datasets/$1.yaml \
    data.train_file "${_ROOT}/pickles/${trainfile}.pickle" \
    save_path $_SAVEDIR \
    save_results $_RESULTS \
    

