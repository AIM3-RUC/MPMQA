# Multimodal Product Manual Question Answering
This is the official repository of the paper *MPMQA: Multimodal Question Answering on Product Manuals (AAAI 2023)*. It provides the PM209 dataset and code of the URA model.\
The metadata (products and brands) of PM209 can be found in `meta_data/`
## Setup
Create virtual environment with conda
```
conda create -n mpmqa python==3.7.6
conda activate mpmqa
```
Install environments
```
# Install pytorch
pip install torch torchvision torchaudio 
# Please make sure that the versions of torch and CUDA is compatible.

# Install detectron2-0.2
# Clone detectron2 from github
cd detectron2
git checkout be792b959bca9a
python python -m pip install -e detectron2
cd ..

# Install apex 
# Clone apex from github
cd apex
python setup.py build develop

# Install bua detector
cd ./detector
python setup.py build develop
cd ..

# Install java>=8
sudo apt-get install openjdk-8-jdk

# Install other requirements
pip install -r requirements.txt
```
## PM209 dataset
```
mkdir -p data && cd data

# Download PM209.zip from: https://drive.google.com/file/d/1K6BPBYdTwKgA1OkNt_BUqVJAn3RDy59B/view?usp=sharing
unzip PM209.zip
cd ..
```
## URA config file and checkpoint
```
# Download expr.zip from: https://drive.google.com/file/d/1CgY5pg2Z1DtfZVcvLtSIsxYjMiBSBQyx/view?usp=sharing
unzip expr.zip
```
## Pre-trained T5 and BUA models
```
# Download pretrained.zip from: https://drive.google.com/file/d/1-o1LMbZKCZQOBJtX1C1T_MkZzOG8tsxu/view?usp=share_link
unzip pretrained.zip

# Make symbolic link to the ./detector dir 
cd detector
ln -s ../pretrained .
cd ..
```
## Config file
Hyper-parameters can be found in `config.json`, and their default values are in `parser.py`. The config file of the `URA` model can be found at `expr/URA/config.json`. 
## Train, validate, and evaluate
```
config_json_path="expr/URA/config.json"
gpu_id=0
bash ds_train.sh ${config_json_path} ${gpu_id}
```
## Load checkpoint and evaluate
```
# Change the 'checkpoint' field to the model path.

config_json_path="expr/URA/eval_opt/config.json"
gpu_id=0
bash ds_eval.sh ${config_json_path} ${gpu_id}
```
## Acknowledge & License
The code is partly based on [transformers](https://github.com/huggingface/transformers) and [detectron2](https://github.com/facebookresearch/detectron2), and is released under the Apache 2.0 license.
