# Towards Faster and Stabilized GAN Training for High Fidelity Few-shot Image Synthesis

Implementation of FastGAN from https://arxiv.org/abs/2101.04775.

### Dataset

Few-shot images dataset from default configuration is automatically downloaded from http://silentz.ml/few-shot-images.zip. In case domain has expired, you can download archive manually and unpack it to `data/` directory inside project root. _NOTE: you can download few-shot-images.zip from github releases of this repository_. Final `data/` directory layout should look like this:

```
data/
├── few-shot-images
│   ├── anime
│   ├── art
│   ├── cat_faces
│   ├── dog_faces
│   ├── grumpy_cat
│   ├── moongate
│   ├── obama
│   ├── panda
│   ├── pokemon
│   ├── shells
│   └── skulls
└── few-shot-images.zip
```
