# MedFinder

This repository contains the official implementation of the paper [BIMCV-R: A Landmark Dataset for 3D CT Text-Image Retrieval](https://arxiv.org/pdf/2403.15992).  
MedFinder is a multimodal retrieval system for medical imaging and text, specifically designed for CT scans and radiology reports. It leverages contrastive learning and cross-attention mechanisms to build a joint embedding space for effective cross-modal retrieval.

## Dataset Availability

Our dataset, **BIMCV-R**, is publicly accessible and can be directly downloaded from **Hugging Face**:  
🔗 [BIMCV-R on Hugging Face](https://huggingface.co/datasets/cyd0806/BIMCV-R)  

The dataset includes:  
- **8069** 3D CT volumes  
- **Over 2 million** 2D CT slices  
- **Medical reports** paired with each scan  

To access the dataset, you will need to agree to the terms and conditions on the Hugging Face page.

## Features
- **Multimodal Retrieval**: Supports **text-to-image**, **image-to-text**, and **keyword-based** retrieval.  
- **Advanced AI Integration**: Utilizes **BiomedCLIP** and **3D vision transformers** for enhanced retrieval accuracy.  
- **Medical Expert Review**: Data has been carefully curated, translated, and reviewed by medical professionals.  

For more details, check out our paper and dataset page. 🚀

## Requirements

- torch>=1.12.0
- monai>=0.9.0
- transformers>=4.20.0
- nibabel>=3.2.0
- pandas>=1.3.0
- scikit-learn>=1.0.0
- wandb>=0.13.0
- tqdm

## Data Structure

The model is designed to work with the BIMCV-R dataset format, but can be adapted for other medical imaging datasets. Expected directory structure:
data_root/
```plaintext
data_root/
├── CT*/
│   └── ct/
│       └── *.nii.gz (CT volume files)
└── meta/
    └── curated_ct_report_path_En.csv (Metadata with reports and file paths)
```


## Usage

### Training

```bash
# Single-node multi-GPU training
torchrun --nproc_per_node=8 --master_port=12345 main_ddp.py \
    --data_root /path/to/dataset \
    --output_dir /path/to/save/model \
    --backbone resnet50 \
    --batch_size 4 \
    --epochs 20 \
    --lr 1e-4 \
    --alpha 1.0 \
    --use_amp \
    --use_wandb \
    --seed 42
```

## Evaluation
```bash
torchrun --nproc_per_node=8 --master_port=12346 main_ddp.py \
    --data_root /path/to/dataset \
    --output_dir /path/to/evaluation/results \
    --backbone resnet50 \
    --batch_size 2 \
    --eval_only \
    --model_path /path/to/best_model.pt \
    --use_amp \
    --use_wandb \
    --seed 42
```

## Command Line Arguments

| Argument         | Type   | Default                                | Description                                       |
|-----------------|--------|----------------------------------------|---------------------------------------------------|
| `--data_root`   | str    | `/h3cstore_ns/CT_data/CT_retrieval`    | Root directory of dataset                        |
| `--output_dir`  | str    | `output`                               | Directory to save models and results             |
| `--backbone`    | str    | `resnet50`                             | Visual encoder backbone (`resnet50` or `vit`)    |
| `--batch_size`  | int    | `2`                                    | Batch size per GPU                               |
| `--epochs`      | int    | `30`                                   | Number of training epochs                        |
| `--lr`          | float  | `1e-4`                                 | Learning rate                                    |
| `--alpha`       | float  | `1.0`                                  | Weight for similarity loss                       |
| `--seed`        | int    | `42`                                   | Random seed for reproducibility                 |
| `--eval_only`   | flag   | `-`                                    | Run only evaluation (no training)               |
| `--model_path`  | str    | `None`                                 | Path to pretrained model for evaluation         |
| `--use_amp`     | flag   | `False`                                | Use automatic mixed precision                   |
| `--use_wandb`   | flag   | `True`                                 | Use Weights & Biases for logging                |
| `--wandb_project` | str  | `MedFinder`                            | Weights & Biases project name                   |
| `--wandb_name`  | str    | `run_name`                             | Weights & Biases run name                       |

## Model Architecture

MedFinder consists of three main components:

- **Visual Encoder**: Processes 3D medical images using either **ResNet50** or **Vision Transformer**  
- **Text Encoder**: Processes medical reports using **BioBERT**  
- **Cross-Attention Module**: Enhances feature representations by combining information from multiple augmented views  

The model is trained using a combination of **MSE loss** for view consistency and **contrastive loss** for cross-modal alignment.

## Evaluation Metrics

The model is evaluated using standard retrieval metrics:

- **R@K**: Recall at K (K=1, 5, 10)  
- **MdR**: Median Rank  
- **MnR**: Mean Rank  

Metrics are computed in both directions: **text-to-image** and **image-to-text**.

## Citation
If you use this code in your research, please cite:

### BibTeX
```bibtex
@inproceedings{chen2024bimcv,
  title={Bimcv-r: A landmark dataset for 3d ct text-image retrieval},
  author={Chen, Yinda and Liu, Che and Liu, Xiaoyu and Arcucci, Rossella and Xiong, Zhiwei},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={124--134},
  year={2024},
  organization={Springer}
}
```
## Contact
For any questions or issues, please contact:  

📧 **[`cyd0806@mail.ustc.edu.cn`](mailto:cyd0806@mail.ustc.edu.cn)**
