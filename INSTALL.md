# Installation

This repository is built in PyTorch 2.0.1 and tested on Ubuntu 20.04 environment (Python3.9, CUDA12.0, cuDNN8.9.7).
Follow these intructions

1. Clone our repository
```
git clone https://github.com/sunshangquan/TransMamba.git
cd TransMamba
```

2. Make conda environment
```
conda create -n TransMamba python=3.9
conda activate TransMamba
```

3. Install dependencies
```
conda install pytorch=2.0.1 torchvision cudatoolkit=12.0 -c pytorch
pip install matplotlib scikit-learn scikit-image opencv-python yacs joblib natsort h5py tqdm mamba-ssm
pip install einops gdown addict future lmdb numpy pyyaml requests scipy tb-nightly yapf lpips
```

4. Install basicsr
```
python setup.py develop --no_cuda_ext
```