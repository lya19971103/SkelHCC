# SkelHCC: A Hyperbolic CLIP-Driven Cache Adaptation Framework for Skeleton-based One-Shot Action Recognition

<div align="center">

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![ICML](https://img.shields.io/badge/ICML-2026-brightgreen.svg)]()
[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)]()
[![PyTorch](https://img.shields.io/badge/pytorch-1.9%2B-red.svg)]()

</div>

## Overview

**SkelHCC** is an official implementation of the paper *"SkelHCC: A Hyperbolic CLIP-Driven Cache Adaptation Framework for Skeleton-based One-Shot Action Recognition"* accepted by **ICML 2026**.

This framework introduces a novel approach to skeleton-based action recognition in the one-shot learning scenario by leveraging hyperbolic geometry and CLIP-based knowledge distillation. Our method achieves superior performance through intelligent cache adaptation mechanisms.

## Key Features

- 🎯 **One-Shot Action Recognition**: Learn to recognize new action classes from just a single example
- 📐 **Hyperbolic Geometry**: Utilize hyperbolic space for better hierarchical representation learning
- 🎨 **CLIP Integration**: Leverage pre-trained CLIP models for semantic understanding
- ⚡ **Cache Adaptation**: Efficient adaptation mechanism for quick model generalization
- 🦴 **Skeleton-Based**: Works directly with skeleton sequences without RGB information
- 🚀 **High Performance**: State-of-the-art results on multiple benchmarks

## Method Highlights

1. **Hyperbolic Space Representation**: Exploits the properties of hyperbolic geometry to capture hierarchical structure in action classes
2. **CLIP-Driven Learning**: Integrates vision-language models to provide robust semantic priors
3. **Cache Adaptation Framework**: Dynamically adapts cached knowledge for new action classes with minimal computational overhead
4. **Efficient Few-Shot Learning**: Achieves strong performance with minimal training data

## Requirements

- Python >= 3.8
- PyTorch >= 1.9.0
- NumPy
- CUDA >= 11.0 (optional, for GPU acceleration)

### Installation

```bash
# Clone the repository
git clone https://github.com/lya19971103/SkelHCC.git
cd SkelHCC

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Training

```bash
python train.py --config configs/config.yaml
```

### Evaluation

```bash
python eval.py --model_path checkpoint.pth --config configs/config.yaml
```

### One-Shot Adaptation

```bash
python one_shot_adapt.py --query_skeleton query_skeleton.npy --support_skeleton support_skeleton.npy
```

## Dataset

The framework supports skeleton-based action recognition datasets:

- **NTU RGB+D 60**: Large-scale skeleton action recognition dataset
- **NTU RGB+D 120**: Extended version of NTU RGB+D
- **PKU-MMD**

### Data Format

Skeleton sequences should be in the format:
- **Input**: NumPy arrays of shape `(T, J, 3)` where:
  - `T`: Number of frames in the sequence
  - `J`: Number of joints (e.g., 25 for NTU dataset)
  - `3`: (x, y, z) coordinates



## Citation

If you use SkelHCC in your research, please cite:

```bibtex
@inproceedings{SkelHCC2026,
  title={SkelHCC: A Hyperbolic CLIP-Driven Cache Adaptation Framework for Skeleton-based One-Shot Action Recognition},
  author={Liu, Yanan and Zhu, Anqi and Zhu, Jingmin and Liu, Jun and Rahmani, Hossein and Bennamoun, Mohammed and Boussaid, Farid and Xu, Dan and Ke, Qiuhong},
  booktitle={Proceedings of the International Conference on Machine Learning (ICML)},
  year={2026}
}
```


## Project Structure

```
SkelHCC/
├── configs/
│   └── config.yaml
├── data/
│   ├── loaders.py
│   └── datasets.py
├── models/
│   ├── clip_models.py
│   ├── hyperbolic_encoder.py
│   └── cache_adapter.py
├── utils/
│   ├── losses.py
│   ├── metrics.py
│   └── helpers.py
├── train.py
├── eval.py
├── one_shot_adapt.py
├── requirements.txt
└── README.md
```


## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Contact

For questions, issues, or collaborations, please reach out:

- **Email**: [liuyanan@mail.ynu.edu.cn](mailto:liuyanan@mail.ynu.edu.cn)

## Acknowledgments

We thank the authors of [CLIP](https://github.com/openai/CLIP) and the skeleton action recognition community for their contributions and inspiration. This work builds upon advances in:

- Vision-Language Models (CLIP)
- Hyperbolic Neural Networks
- Few-Shot Learning
- Skeleton-based Action Recognition


## Disclaimer

This code is provided for research and educational purposes. The project is currently under development. More details and pre-trained weights will be released upon official publication.

---

**Status**: 🚧 Under Development | Expected Release: 2026

For updates and more information, please watch this repository or follow the author on GitHub.
