# TwoComplementaryPerspectivesCL
This repository contains the codebase that was used to produce the results for the paper: "Two Complementary Perspectives to Continual Learning:Ask Not Only What to Optimize, But Also How"


# Two Complementary Perspectives to Continual Learning 
This repository contains the codebase that was used to produce the results for the paper: *"Two Complementary Perspectives to Continual Learning:Ask Not Only What to Optimize, But Also How"*

## Setup
This code uses:
* Python 3.8
* Pytorch 2.1
* Avalanche 0.4.0a (The library code is included in this repository, https://github.com/ContinualAI/avalanche)

To setup your python environmet we provide a [conda_environment.yml](conda_environment.yml) to mirrors our Anaconda packages.
Please use the following bash command:

    conda env create -n TwoCompCLEnv --file conda_environment.yml
    conda activate TwoCompCLEnv

## Reproducing results
To reporduce results of individual experiments, run the python code as below. 
1. Select the benchmark configuration from `reproduce/benchmark/`. Optionally you can provide the `--dset_rootpath` argument to specify the locaiton of dataset. Otherwise the dataset are stored in `./data`.
2. Select a strategy configuration from `reproduce/strategy`. Set the learning rate (`lr`) and for strategies involving experience replay, set the total size of the auxilliary buffer (`mem_size`). For online experiments you also need to set the preferred batch_size (`bs`).
3. Select a seed and a location to store the results (`save_path`). 

#### Generally:

    python train.py --benchmark_config {path_to_config} --strategy_config {path_to_config} --mem_size {total_buffer_size} --lr {0.1, 0.01, 0.001} --bs {10, 64, 128} --seed {142, 152, 162, 172, 182} --save_path {path_to_storage}

* **mem_size**
    * Rotated MNIST - Offline: `3000` | Online: `60000`
    * CIFAR-100 / MiniImageNet - Offline: `10000` | Online: `50000`

#### Example for offline RotatedMNIST: 

    python train.py --benchmark_config ./reproduce/benchmark/rotmnist_offline.yml --strategy_config ./reproduce/strategy/er_gem.yml --mem_size 3000 --lr 0.1 --bs 128 --seed 142 --save_path ./results/

    

#### Note: 
We didn't run with deterministic CUDNN backbone for computational efficiency, which might result in small deviations in results. 
We average all results over 5 initialization seeds {142, 152, 162, 172, 182}.

## Access Results
The results are logged with Tensorboard. To view the results use:

    tensorboard --logdir={path_to_storage}

## Credit 
We particularly thank the repositories containing the Avalanche continual learning library (https://github.com/ContinualAI/avalanche), and the codebase for continual evaluation (https://github.com/Mattdl/ContinualPrototypeEvolution)

## Lisence
Code is available under MIT license: A short and simple permissive license with conditions only requiring preservation of copyright and license notices.