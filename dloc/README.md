# dloc - the deep learning image matching toolbox

This repository provides accessible interfaces for several existing SotA methods to match image feature correspondences between image pairs. We provide scripts to evaluate their predicted correspondences on common benchmarks for the tasks of image matching, homography estimation, and visual localization.


## Support Methods
The code supports three processes: co-view area estimation, feature point extraction, and feature point matching, and supports detector-base and detector-free methods, mainly including:
- [d2net](https://arxiv.org/abs/1905.03561): extract keypoint from 1/8 feature map
- [superpoint](https://arxiv.org/abs/1712.07629): could extract in corner points, pretrained with magicpoint
- [superglue](https://arxiv.org/abs/1911.11763): excellent matching algorithm, but pretrained model only support superpoint, we have implementation superglue with sift/superpoint in megadepth datasets.
- [disk](https://arxiv.org/abs/2006.13566): add reinforcement for keypoints extraction
- [aslfeat](https://arxiv.org/abs/2003.10071): build multiscale extraction network
- [cotr](https://arxiv.org/abs/2103.14167): build transformer network for points matching
- [loftr](https://arxiv.org/abs/2104.00680): dense extraction and matching with end-to-end network
- [r2d2](https://arxiv.org/abs/1906.06195): add repeatability and reliability for keypoints extraction
- [contextdesc](https://arxiv.org/abs/1904.04084): keypoints use sift, use full image context to enhance descriptor. expensive calculation.
- [OETR](https://arxiv.org/abs/2202.09050): image pairs co-visible area estimation.


## Installation
This repository support different SOTA methods. If you want use this code, you could reference these steps:
1. Download this repository and initialize the submodules to the third_party folder
```
git clone

# Install submodules non-recursively
cd OETR/
git submodule update --init
```
2. install requirements for different submodules
3. Download model weights and place them in the weights folder.


## Inference and evaluation
Download the image pairs and relative pose groundtruth of [IMC]() and [megadepth]() to assets/. You could also chose dataset and methods, please reference to `evaluate_imc.sh` and `evaluate_megadepth.sh`, after the results process, you could run evaluation pipeline for imc or megadepth:
```
python3 dloc/evaluate/eval_imc.py --input_pairs ./dataset/ImageMatching/imc_0.1.txt --results_path outputs/imc_all/ --methods_file assets/methods.txt
or
python3 dloc/evaluate/eval_imc.py --input_pairs ./dataset/megadepth/imc_0.1.txt --results_path outputs/imc_all/ --methods_file assets/methods.txt
```

1. Benchmark on IMC dataset
```sh evaluate_imc.sh```


2. Benchmark on Megadepth dataset
```sh evaluate_megadepth.sh```


## Debugging and Visualization
Download https://drive.weixin.qq.com/s?k=AJEAIQdfAAob00fTSEAKMA9AatACk datasets to assets.

Weights could be downloaded from https://drive.weixin.qq.com/s?k=AJEAIQdfAAo97Nnovq.
