# RELATION: REceptor-LigAnd interacTION
# A Deep Generative Model for Structure-based De Novo Drug Design

![overview of the architecture of RELATION](/images/figure.png)

## Overview
This repository contains the source of RELATION, a software for DL-based de novo drug design.


## Requirements
- Python >= 3.7
- pytorch >= 1.1.0
- openbabel == 2.4.1
- RDKit
- pyscreener [README](https://github.com/coleygroup/pyscreener)

if utilizing GPU accelerated model training 
- CUDA & cudnn 




## Running MolPAL

### Prepare molecular dataset
To train the RELATION network, the source dataset (`./data/zinc/zinc.csv`) and target dataset (`./data/akt1`, `./data/cdk2`) must by converted to a 4D-tensor-(19,16,16,16), which means the 3D gird with 19 channels(np array in `./data/zinc/zinc.npz`,`./data/akt1/akt_pkis.npz`,`./data/cdk2/akt_cdk2.npz`).
 
 `python model/data_prepare.py`


### Training RELATION

`python model/training.py`




### Sampling

Load the `./akt1_relation.pth` or `./cdk2_relation.pth` generative model, and typing the following codes on any python interpreter   in `/model` directory:

`import sample`  


`sample.greedy_search(n_sampled=1000, path='./gen_smi.csv')`

or you can also use the bayesian optimization in sampling process:

`main_bo(n_sampled=500,iterations=5,random_seed=1, path='./')`


