# CUFAR
This repository provides a reference implementation of paper: Overcoming Forgetting in Fine-Grained Urban Flow Inference via Adaptive Knowledge Replay. The codes include the implementation of CUFAR and other FUFI methods (UrbanFM, DeepLGR, FODE, UrbanODE, UrbanPy).

## Requirements
We implement CUFAR and other FUFI methods with following dependencies:
* python 3.7.12
* pytorch 1.8.0
* numpy
* einops
* sciki-learn

For Linux, install the enviroment via Anaconda:
```shell
bash install_env.sh
```
For Windows, follow the step in the [install_env.sh](install_env.sh).


## Datasets
TaxiBJ datasets can be obtained in baseline [UrbanFM's repo](https://github.com/yoshall/UrbanFM/tree/master/data).


## Usage
Before you run the code, you may need to ensure the package structure of CUFAR is as follows:
```
.
├── buffers
├── datasets
│   └── TaxiBJ
│       ├── P1
│       ├── P2
│       ├── P3
│       └── P4
├── model
├── src
└── README.md
```

We also provide the training approches of all protocols, they are `train_single_task.py`, `train_finetune.py`, `train_continual.py` and `train_joint.py`. You can change the backbone through `model` argument.

```
# Run single-task protocol of CUFAR
python train_single_task.py --model=CUFAR

# Run joint protocol of CUFAR
python train_joint.py --model=CUFAR

# Run fine-tune protocol of CUFAR
python train_finetune.py --model=CUFAR --initial_train

# Run continual protocol to evaluate our AKR
python train_continual.py --model=CUFAR --initial_train
```

## Citing
If you find CUFAR useful in your research, please cite the following paper:
```bibtex
@inproceedings{yu2023Overcoming,
  title={Overcoming Forgetting in Fine-Grained Urban Flow Inference via Adaptive Knowledge Replay},
  author={Yu, Haoyang and Xu, Xovee and Zhong, Ting and Zhou, Fan},
  booktitle={AAAI},
  year={2023}
} 
```