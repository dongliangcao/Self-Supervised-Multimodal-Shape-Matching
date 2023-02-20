# Self-Supervised Learning for Multimodal Non-Rigid 3D Shape Matching

## Installation
```bash 
conda create -n fmnet python=3.8 # create new viertual environment
conda activate fmnet
conda install pytorch cudatoolkit -c pytorch # install pytorch
pip install -r requirements.txt # install other necessary libraries via pip
```

## Train
To train a model for 3D shape matching. You only need to write or use a YAML config file. 
In the YAML config file, you can specify everything around training. 
Here is an example to train.
```python
python train.py --opt options/train.yaml 
```
You can visualize the training process via TensorBoard.
```bash
tensorboard --logdir experiments/
```

## Test
After finishing training. You can evaluate the model performance using a YAML config file similar to training.
Here is an example to evaluate.
```python
python test.py --opt options/test.yaml 
```

## Pretrained models
You can find the pre-trained models on SURREAL-5k dataset in [checkpoints](checkpoints) for reproducibility.