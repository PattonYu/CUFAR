# CUFAR

The codes include the implementation of CUFAR (continual Urban Flow inference with Adaptive knowledge Replay) and other FUFI methods (UrbanFM, DeePLGR, FODE, UrbanODE, UrbanPy).

## Requirements
We implement CUFAR and other FUFI methods with following dependencies:
* python 3.7.12
* pytorch 1.8.0
* numpy
* einops
* sciki-learn

For Linux, install the enviroments via Anaconda:
```shell
bash install_env.sh
```
For Windows, follow the step in the [install_env.sh](install_env.sh).


## Datasets
In our code & data file, we only upload a subset dataset (P1) due to the size limit of CMT, ohter three datasets (P2, P3, P4) can be obtained in baseline [UrbanFM's repo](https://github.com/yoshall/UrbanFM/tree/master/data). For $\textit{continual}$ protocol, we suggest you download the complete datasets to conduct experiments.



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
