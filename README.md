# EpoD
To facilitate a thorough understanding of our paper, we provide the source code of our EpoD. When reproducing the results, please adhere to the following steps.


## 0. Installation
Our code runs on `Python 3.9` with `PyTorch 1.10.0`, `PyG 2.0.3` and `CUDA 11.3`. Please follow the following steps to create a virtual environment and install the required packages.

1. Create a conda environment:

```
conda create -n EpoD python=3.9
conda activate EpoD
```
2. Install dependencies:

```
pip install torch==1.11 --extra-index-url https://download.pytorch.org/whl/cu113
pip install torch-scatter==2.0.9 torch-sparse==0.6.13 torch-cluster==1.6.0 torch-spline-conv==1.2.1 torch-geometric -f https://data.pyg.org/whl/torch-1.11.0+cu113.html
pip install matplotlib networkx scikit-learn scipy jupyter jupyterlab tensorboard gensim pandas
```

## 1. Dataset 

### 1.1 Download traffic flow Dataset

Download dataset at ./traffic/data from following links

```
https://drive.google.com/file/d/19SOqzYEKvkna6DKd74gcJ50Wd4phOHr3/view?usp=share_link
```


### 1.2 Download the social link Dataset

Download dataset at ./social/data from following links

```
https://drive.google.com/file/d/19SOqzYEKvkna6DKd74gcJ50Wd4phOHr3/view?usp=share_link
```

## 2. Experiments Running

### 2.1 Running traffic flow prediction task

To run on one dataset, please execute following commands in the directory ./traffic

```
python experiments/agcrn/main.py --device cuda:0 --dataset PEMS08 --years 2019 --model_name agcrn --seed 2024 --bs 32 --seq_len 12 --horizon 24
python experiments/agcrn/main.py --device cuda:0 --dataset PEMS04 --years 2019 --model_name agcrn --seed 2024 --bs 32 --seq_len 12 --horizon 24
python experiments/agcrn/main.py --device cuda:0 --dataset SD --years 2019_2020 --model_name agcrn --seed 2024 --bs 32 --seq_len 12 --horizon 24
python experiments/agcrn/main.py --device cuda:0 --dataset GBA --years 2019_2020 --model_name agcrn --seed 2024 --bs 32 --seq_len 12 --horizon 24
```

### 2.2 Running social link prediction task

To run on one dataset, please execute following commands in the directory ./social

```
python scripts/main.py --dataset collab --log_dir ../logs/collab --device_id 0 --seed 2024
python scripts/main.py --dataset yelp --log_dir ../logs/yelp --device_id 0 --seed 2024
python scripts/main.py --dataset act --log_dir ../logs/act --device_id 0 --seed 2024
```
