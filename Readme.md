 <h1 align="center"> Neural Mesh Refinement </h1>



Neural Mesh Refinement utilized a learned geometric prior on fine shapes to adaptively refine coarse meshes through subdivision, demonstrating robust generalization.

![Teaser of Neural Mesh Refinement](fig/teaser.jpg)

This is a prototype implementation in Python 3.8  with PyTorch 1.12.1.

# Getting Started

## Set up environment

```bash
conda create -n nmr python==3.8
conda activate nmr
pip install -r requirments.txt
```

Depending on your setup, please change the dependency version of pytorch/cudatoolkit in `requirments.txt` by following [this document](https://pytorch.org/get-started/previous-versions/).

We have tested successfully in the following environment:
- Ubuntu 20.04 with NVIDIA GeForce RTX 3090: `pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3`
- Windows 10 without GPU: `pytorch==1.12.1`

# Test

For a quick demo, please use the pre-trained model and test on new shapes. To test the pre-trained model please run
```bash
python test.py -p ckpt/thingi10k_netparams.dat -t data_meshes/coarse/sphere.obj -ns 3
```

Then, you will get refined meshes in  `data_meshes/refined/thingi10k_netparams/`

You can also try other pre-trained models and coarse meshes.

## ckpt

- `ckpt/thingi10k_netparams.dat`: We trained this net using the [Thingi10k](https://ten-thousand-models.appspot.com/) dataset, which can adaptively refine the coarse mesh.
- `ckpt/bunny_netparams.dat`: We trained this net using the `data_meshes/original/bunny.obj`, which tends to smoothly refine the coarse mesh.
- `ckpt/gear_netparams.dat`:  We trained this net using the `data_meshes/original/gear.obj`, which tends to sharply refine the coarse mesh.

## coarse meshes

- `data_meshes/coarse/sphere.obj`

- `data_meshes/coarse/cube.obj`

You can also try other coarse manifold meshes, otherwise using [fTetWild](https://github.com/wildmeshing/fTetWild) to preprocess them.

# Train

Coming soon.

