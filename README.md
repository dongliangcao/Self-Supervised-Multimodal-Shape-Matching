# Self-Supervised Learning for Multimodal Non-Rigid 3D Shape Matching
![img](figures/framework.png)

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

## Acknowledgement
The implementation of DiffusionNet is modified from [the official implementation](https://github.com/nmwsharp/diffusion-net), 
the computation of the FM unsupervised loss and FM solvers are modified from [SURFMNet-pytorch](https://github.com/pvnieo/SURFMNet-pytorch)
and [DPFM](https://github.com/pvnieo/DPFM), the Sinkhorn normalization is adapted from [tensorflow implementation](https://github.com/google/gumbel_sinkhorn).  
We thank the authors for making their codes publicly available.

## Citation
If you find the code is useful, please cite the following paper
```bibtex
@inproceedings{cao2023,
title = {Self-Supervised Learning for Multimodal Non-Rigid 3D Shape Matching},
author = {D. Cao and F. Bernard},
year  = {2023},
booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)}
}