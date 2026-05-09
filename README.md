<div align="center">
<h1>FAConformer</h1>
<h3>Frequency-Aware Convolutional Transformer for Auditory Attention Detection</h3>

[Ziwei Wang](https://scholar.google.com/citations?user=fjlXqvQAAAAJ&hl=en)<sup>1</sup>, [Xingyi He](https://github.com/BAY040210)<sup>1</sup>, [Tianwang Jia](https://github.com/TianwangJia)<sup>1</sup>, [Hongbin Wang](https://github.com/WangHongbinary)<sup>1</sup>, and [Dongrui Wu](https://scholar.google.com/citations?user=UYGzCPEAAAAJ&hl=en)<sup>1 :email:</sup>

<sup>1</sup> School of Artificial Intelligence and Automation, Huazhong University of Science and Technology

(<sup>:email:</sup>) Corresponding Author

</div>

> This repository contains the implementation of our paper: **"FAConformer: Frequency-Aware Convolutional Transformer for Auditory Attention Detection"**, serving as a **benchmark codebase** for auditory attention detection models. We implemented and fairly evaluated 13 state-of-the-art auditory attention detection models, including CNN-based, AAD-speciﬁc, and CNN-Transformer hybrid auditory attention detection models.

## Overview

**FAConformer**, a **frequency-aware convolutional Transformer** network tailored for auditory attention detection:

- **Within-Band Encoding**: Learns band-speciﬁc representations
- **Cross-Band Hierarchical Fusion**: Models adaptive cross-band dependencies
- **Band-Wise Auxiliary Supervision**: Ensures effective optimizations of each frequency branch

<div align="center">
<img width="1809" height="903" alt="Image" src="https://github.com/user-attachments/assets/af20aaf0-5a1f-4d10-abd4-6bc26b9adb40" />
</div>

## Features

- 🔀 **Frequency-aware CNN-Transformer design** for hierarchical EEG frequency modeling in AAD
- 🧩 **Band-specific encoding strategy** for discriminative within-band spatio-temporal representation learning
- 📡 **Adaptive cross-band fusion** for modeling inter-band dependency relationships
- ⚖️ **Band-wise auxiliary supervision** for balanced multi-branch training and enhanced branch reliability
- 📈 **State-of-the-art decoding performance** with gains over related approaches
- 💡 **Strong robustness and interpretability** validated through ablation, band importance, and sensitivity analyses

Most related approaches have not yet fully explored hierarchical modeling of band-specific EEG representations or adaptive integration of unequal frequency-band contributions for final decision-making. FAConformer forms a complete frequency-aware hierarchical decoding pipeline, rather than simply combining multi-band decomposition and feature fusion.

<img width="831" height="1155" alt="Image" src="https://github.com/user-attachments/assets/337a4ea9-1801-4e24-8249-659e6af587cf" />

## Code Structure
```
FAConformer/
│
├── main.py                   # Main script
│
├── models/                   # Model architectures (FAConformer and baselines)
│   ├── FAConformer.py        # Frequency-Aware Convolutional Transformer (Ours)
│   ├── DARNet.py             # AAD-Speciﬁc model
│   ├── DBPNet.py             # AAD-Speciﬁc model
│   └── DHGCN.py              # AAD-Speciﬁc model
│
├── utils/                    # Helper functions and common utilities
│   ├── data_loader.py        # Chronological data splitting
│   ├── functions.py          # EEG frequency spatial processing
│   ├── hypergraph_utils.py   # Hypergraph construction
│   └── utils.py              # General utilities
│
├── locs_orig.mat             # Channel electrode coordinates, used for DBPNet
│
└── README.md
```

## Baselines
Twelve EEG decoding models were reproduced and compared with the proposed FAConformer in this paper. FAConformer achieves the **state-of-the-art performance**.

- CNNs: EEGNet, SCNN, IFNet
- AAD-Speciﬁc Models: DBPNet, DARNet, DHGCN
- CNN-Transformers: CTNet, TMSA-Net, EEGConformer, MSCFormer, MSVTNet, DBConformer

<div align="center">
<img width="1302" height="528" alt="Image" src="https://github.com/user-attachments/assets/3369c8b4-5516-4d61-8fa3-15e8503472c2" />
</div>

## Datasets
FAConformer conducted experiments on two representative public AAD datasets. AAD datasets can be downloaded from [DTU](https://zenodo.org/records/1199011), and [KUL](https://zenodo.org/records/4004271).

- Auditory Attention Detection:
  - DTU: 18 subjects, 64 channels, 512 Hz sampling rate, Danish speech stimuli presented from ±60° directions.
  - KUL: 16 subjects, 64 channels, 8192 Hz sampling rate, Dutch speech stimuli presented from ±90° directions.

## Visualizations
### Subject-wise Classification Performance
To further verify the cross-subject robustness and generalization ability of the proposed FAConformer, we illustrated the subject-wise classification results of all compared models on both DTU and KUL datasets.

<div align="center">
<img width="1302" height="1547" alt="Image" src="https://github.com/user-attachments/assets/07de62c9-8aa7-459e-9ff6-850c351809bb" />
</div>

### Cross-Band Attention for Band Importance Analysis
To analyze how the FAA module characterizes band importance during cross-band fusion, we visualized the subject-wise self-attention maps learned by FAA on both DTU and KUL datasets.

<div align="center">
<img width="1341" height="1458" alt="Image" src="https://github.com/user-attachments/assets/4890a1ca-260e-4c2f-8a49-c6ad31276e4c" />
</div>

### Effect of Frequency-Aware Modeling
To further examine the effect of frequency-aware modeling on representation learning, we visualized the learned feature distributions using t-SNE for FAConformer and baseline models on both DTU and KUL datasets.

<div align="center">
<img width="1323" height="1635" alt="Image" src="https://github.com/user-attachments/assets/0c2f10aa-2b74-427c-be63-421de42398cd" />
</div>

### Parameter Sensitivity Analysis
To evaluate the robustness of FAConformer to hyperparameter selection, we analyzed the sensitivity of three key model parameters on both DTU and KUL datasets.

<div align="center">
<img width="648" height="747" alt="Image" src="https://github.com/user-attachments/assets/14e1e1b3-e672-47f6-9ea0-136021690a87" />
</div>

---

## 🙌 Acknowledgments

Special thanks to the source code of EEG decoding models: [DBConformer](https://github.com/wzwvv/DBConformer), [EEGNet](https://github.com/vlawhern/arl-eegmodels), [IFNet](https://github.com/Jiaheng-Wang/IFNet), [DBPNet](https://github.com/fchest/DBPNet), [DARNet](https://github.com/fchest/darnet), [DHGCN](https://github.com/nobody1219/DHGCN), [CTNet](https://github.com/snailpt/CTNet), [TMSA-Net](https://www.sciencedirect.com/science/article/pii/S1746809424012473), [EEGConformer](https://github.com/eeyhsong/EEG-Conformer), [MSCFormer](https://www.nature.com/articles/s41598-025-96611-5), and [MSVTNet](https://ieeexplore.ieee.org/abstract/document/10652246).

We appreciate your interest and patience. Feel free to raise issues or pull requests for questions or improvements.
