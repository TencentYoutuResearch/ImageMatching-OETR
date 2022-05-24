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
3. Download model weights and place them in the weights folder, and weights could be downloaded from https://drive.google.com/drive/folders/1UedCycHJph4PDoStAAyxtdRxUX9PwLsJ?usp=sharing.


## Inference and evaluation
Download the image pairs and relative pose groundtruth of [imc](https://drive.google.com/drive/folders/1-kAESEYPXe3Byzgu51XWDwlTaDx0Jldo?usp=sharing) and [megadepth](https://drive.google.com/drive/folders/1D0u64-SaMufpTiBVQQAg7C1NpOtQSBNs?usp=sharing) to `assets/` folder. You could also chose dataset and methods, please reference to `evaluate_imc.sh` and `evaluate_megadepth.sh`:
1. Benchmark on IMC dataset
```sh evaluate_imc.sh```


2. Benchmark on Megadepth dataset
```sh evaluate_megadepth.sh```

You can choose only to execute part of the algorithm inside. After the results process, you could run an evaluation pipeline for imc or megadepth:
```
python3 dloc/evaluate/eval_imc.py --input_pairs ./assets/imc/imc_0.1.txt --results_path outputs/imc_2011/ --methods_file assets/methods.txt
or
python3 dloc/evaluate/eval_megadepth.py --input_pairs ./assets/megadepth/megadepth_scale_34.txt --results_path outputs/megadepth_34/ --methods_file assets/methods.txt
```



