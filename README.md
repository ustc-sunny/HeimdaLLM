# HeimdaLLM: Efficient Cloud-assisted Federated Fine-tuning with Zeroth-Order Rectification for LLMs

## Installation
<!-- http://doc.fedml.ai/#/installation -->
After `git clone`-ing this repository, please run the following command to install our dependencies.

```bash
conda create --name HeimdaLLM python=3.7.15
conda activate HeimdaLLM
pip3 install cpython
pip install -r requirements.txt
conda install mpi4py=3.0.3=py37hf046da1_1
conda install six==1.15.0

cd FedML; git submodule init; git submodule update; cd ../; 
```

## Code Structure of FwdLLM

- `FedML`: a soft repository link generated using `git submodule add https://github.com/FedML-AI/FedML`. We modified it and added the cloud-assisted fine-tuning functions of HeimdaLLM

- `data`: provide data downloading scripts and raw data loader to process original data and generate h5py files. Besides, `data/advanced_partition` offers some practical partition functions to split data for each client.

- `data_preprocessing`: preprocessors, examples and utility functions for each task formulation.

- `data_manager`: data manager is responsible for loading dataset and partition data from h5py files and driving preprocessor to transform data to features.

- `model`: advanced NLP models. You can define your own models in this folder.

- `trainer`: please define your own `trainer.py` by inheriting the base class in `FedML/fedml-core/trainer/fedavg_trainer.py`.
Some tasks can share the same trainer.

- `experiments`: 
    1. `experiments/distributed/transformer_exps` is the entry point for federated training. It contains experiments for different tasks. We start from `experiments/distributed/transformer_exps/run_tc_exps`.

## Data Preparation
We have pre-processed four datasets including AGNEWS, SST-2, 20News.(Need network access to drive.google.com)
```bash
gdown https://drive.google.com/uc?id=10S3Zg9HFmBuDkOusycefkugOCu27s0JT
tar -zxvf fednlp_data.tar
```

## Demo Experiments: AGNEWS for DistilBERT (Discriminative)
<!-- ## HeimdaLLM测试实验，模型: DistilBERT, 数据集: AGNEWS。 -->
```python
conda activate HeimdaLLM
cd experiments/distributed/transformer_exps/run_tc_exps
sh run_text_classification.sh 100 0.01 FedFwd
```
You can modify the run_text_classification.sh and gpu_mapping.yaml files to configure the experiment settings

## Results
The training log will be saved in `FwdFL/experiments/distributed/transformer_exps/run_tc_exps/log/new/`
You can find the the accuracy changes of the model by searching for `acc`.

Alternatively, you can run the following command to print the model's acc:
```bash
grep "'acc':" log/new/test_fedFwd_distilbert_agnews_lr0.01_client_num_100_numerical.log
```
## Citation
Please cite our HeimdaLLM paper if it helps your research (will update the arxiv version).
```bib
@misc{sun2026cooperllm,
      title={CooperLLM: Cloud-Edge-End Cooperative Federated Fine-tuning for LLMs via ZOO-based Gradient Correction}, 
      author={He Sun and Jinrui Zhou and Li Li and Mingjun Xiao},
      year={2026},
      eprint={2601.12917},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2601.12917}, 
}
```

## Acknowledge
We thank anonymous SIGKDD Artifacts reviewers to make this artifact better.
