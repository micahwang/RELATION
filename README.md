# RELATION: REceptor-LigAnd interacTION
# A Deep Generative Model for Structure-based De Novo Drug Design

![overview of the architecture of RELATION](/images/figure.png)

## Overview
This repository contains the source of RELATION, a software for DL-based de novo drug design.


## Requirements
- Python == 3.7
- pytorch >= 1.1.0
- openbabel == 2.4.1
- RDKit == 2020.09.5
- theano == 1.0.5
- pyscreener [README](https://github.com/coleygroup/pyscreener)

if utilizing GPU accelerated model training 
- CUDA==10.2 & cudnn==7.5 

### Creat a new environment in conda 

 `conda env create -f env.yml `



## Running RELATION

### Prepare molecular dataset
To train the RELATION network, the source dataset and target dataset (akt1 and cdk2) must by converted to a 4D-tensor-(19,16,16,16), which means the 3D gird with 19 channels.
#### Source dataset
 `python model/data_prepare.py --input ./data/zinc/zinc.csv 
                               --output ./data/zinc/zinc.npz 
                               --mode 0 `
                            
####  Target dataset
 `python model/data_prepare.py --input ./data/akt1 
                               --output ./data/akt1/akt_pkis.npz
                               --pkidir ./data/akt1.csv
                               --mode 1`

### Training RELATION
Load sourch dataset (`./data/zinc/zinc.npz`) and target dataset (`./data/akt1/akt_pkis.npz`or `./data/cdk2/cdk2_pkis.npz`).

`python model/training.py`




### Sampling

Load the `./akt1_relation.pth` or `./cdk2_relation.pth` generative model, and typing the following codes:


`python sample.py --num 500
                  --output ./gen_smi.csv`

or you can also use the bayesian optimization in sampling process:

`python sample.py --num 500
                  --output ./gen_smi.csv
                  --method bo
                  --iter 5`




