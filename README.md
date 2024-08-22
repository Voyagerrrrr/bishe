# Uncertainty Qualification in Lidar Place Recognition

This repository contains the code implementation used in the paper 


## Environment

### Dependencies

Code was tested using Python 3.8 with PyTorch 1.9.0 and MinkowskiEngine 0.5.4 on Ubuntu 20.04 with CUDA 11.1.

The following Python packages are required:
* PyTorch (version 1.9.0)
* MinkowskiEngine (version 0.5.4)
* pytorch_metric_learning (version 1.0 or above)
* torchpack
* tensorboard
* pandas

### Datasets

The **Oxford RobotCar** and **NUS Inhouse** datasets were introduced in [PointNetVLAD: Deep Point Cloud Based Retrieval for Large-Scale Place Recognition](https://arxiv.org/pdf/1804.03492). 

You can download training and evaluation datasets from [here](https://drive.google.com/open?id=1rflmyfZ1v9cGGH0RL4qXRrKhg-8A-U9q)


Pickles were created following the split of the original and are available under `pickles/`

After downloading datasets, please save folders `oxford`, `inhouse_datasets` directly under `datasets/`.

## Getting Started

### Preprocessing

To run any bash file, please making the change to `config/default.yaml`:

1.  Replace with `datasets/` directory containing all subfolders of datasets.
2.  Change path to your conda installation
3.  Replace environment name with your conda environment
4.  Replace root directory
5.  The `batch_size` and `batch_size_limit` may need to be changed to account for available GPU memory


### Training

To train a standard network across a variety of configurations:

```
# bash train.sh minkloc oxford <model number>
bash train.sh minkloc oxford 1
```

To train the dropout method:

```
# bash /train_dropout.sh oxford
bash train_dropout.sh oxford
```
### Evaluation

To evaluate a standard model:

```
# bash eval.sh minkloc oxford <model number>
bash eval.sh minkloc oxford 1
```

To evaluate the dropout model:

```
# basheval_dropout.sh oxford
bash eval_dropout.sh oxford

```
