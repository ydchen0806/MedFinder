import os
import random
import monai.networks
import monai.networks.nets
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from torchvision import transforms
from monai.networks.nets import vit, resnet
import nibabel as nib
import monai
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    ScaleIntensityd,
    RandRotated,
    RandZoomd,
    RandGaussianNoised,
    SpatialPadd,
    CropForegroundd,
    Resized
)
from sklearn.metrics import precision_score
from tqdm import tqdm
import logging
import argparse
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import pandas as pd
from glob import glob 

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

# Dataset class for BIMCV-R
class BIMCVRDataset(Dataset):
    def __init__(self, data_root, split='train', max_text_length=100, transform=None, split_rate = 0.8):
        """
        Args:
            data_root: Root directory of the BIMCV-R dataset
            split: 'train', 'val', or 'test'
            max_text_length: Maximum length of text to sample
            transform: Transforms to apply to the 3D images
        """
        if data_root is None:
            data_root = '/h3cstore_ns/CT_data/CT_retrieval'
        self.data_root = data_root
        self.split = split
        self.max_text_length = max_text_length
        self.transform = transform
        self.data = pd.read_csv('/h3cstore_ns/CT_data/CT_retrieval/meta/curated_ct_report_path_En.csv')
        self.set = split
        self.split_rate = split_rate
        # self.image_resolution = image_resolution
        if self.set == 'train':
            self.data = self.data[:int(len(self.data)*self.split_rate)]
        else:
            self.data = self.data[int(len(self.data)*self.split_rate):]
        self.father_path = sorted(glob(os.path.join(self.data_root,'CT*','ct','*.nii.gz')))

        self.total_report = self.data['Report_en'].values.tolist()
        self.total_path = self.data['path'].values.tolist()
        path_map = {os.path.basename(path): path for path in self.father_path}
        full_path = [path_map[os.path.basename(path)] for path in self.total_path]
        self.data['full_path'] = full_path
        # Load dataset information (assuming a metadata file exists)
        self.data_info = self._load_data_info()
        
        # Initialize tokenizer for medical text
        self.tokenizer = AutoTokenizer.from_pretrained("/h3cstore_ns/ydchen/code/MedicalFinder/models_HF/BioBert")
    
    def _load_data_info(self):
        """Load data information from metadata file"""
        # This should be adapted based on how your dataset is structured
        # For now, let's assume a simple CSV-like structure
        data_info = []
        for idx, row in self.data.iterrows():
            item_info = {
                "image_path": row["full_path"],
                "report_path": row["Report_en"],
                # "diagnosis": row["diagnosis"]
            }
            data_info.append(item_info)
        return data_info
    
    def __len__(self):
        return len(self.data_info)
    
    def _sample_text(self, text):
        """Sample a continuous segment of text of length max_text_length"""
        words = text.split()
        if len(words) <= self.max_text_length:
            return text
        
        start_idx = random.randint(0, len(words) - self.max_text_length)
        sampled_text = ' '.join(words[start_idx:start_idx + self.max_text_length])
        return sampled_text
    
    def __getitem__(self, idx):
        item_info = self.data_info[idx]
        
        # Load 3D CT volume
        volume = self._load_volume(item_info["image_path"])
        
        report_text = item_info["report_path"]
        
        # Sample text
        sampled_text = self._sample_text(report_text)
        
        # Tokenize text
        tokenized_text = self.tokenizer(
            sampled_text, 
            padding='max_length', 
            truncation=True, 
            max_length=self.max_text_length, 
            return_tensors='pt'
        )
        
        # Apply image transformations
        if self.transform:
            volume = self.transform({"volume": volume})["volume"]
        
        return {
            "volume": volume,
            "text": {
                "input_ids": tokenized_text["input_ids"].squeeze(0),
                "attention_mask": tokenized_text["attention_mask"].squeeze(0)
            },
            "original_text": report_text,
            # "diagnosis": item_info["diagnosis"]
        }
    
    def _load_volume(self, path):
        """Load 3D CT volume from file"""
        if path.endswith('.nii.gz'):
            nii_img = nib.load(path)
            volume = nii_img.get_fdata()  # 获取数组数据
            volume = np.asarray(volume)  # 确保是 NumPy 数组
        
        elif path.endswith('.npy') or path.endswith('.npz'):
            volume = np.load(path)
        
        else:
            raise ValueError(f"Unsupported file format: {path}")
        
        # 转换为张量并确保是 float 类型
        if not isinstance(volume, torch.Tensor):
            volume = torch.from_numpy(volume).float()
        
        # 确保有通道维度
        if volume.ndim == 3:
            volume = volume.unsqueeze(0)
        
        return volume

# Data transformations
def get_transforms(mode='train'):
    if mode == 'train':
        return Compose([
            # LoadImaged(keys=["volume"]),
            EnsureChannelFirstd(keys=["volume"], channel_dim=0, strict_check=False),
            ScaleIntensityd(keys=["volume"]),
            CropForegroundd(keys=["volume"], source_key="volume"),
            SpatialPadd(keys=["volume"], spatial_size=(224, 224, 96)),
            Resized(keys=["volume"], spatial_size=(224, 224, 96)),
            RandRotated(keys=["volume"], range_x=0.2, range_y=0.2, range_z=0.2, prob=0.5),
            RandZoomd(keys=["volume"], min_zoom=0.9, max_zoom=1.1, prob=0.5),
            RandGaussianNoised(keys=["volume"], prob=0.5, mean=0.0, std=0.1)
        ])
    else:
        return Compose([
            # LoadImaged(keys=["volume"]),
            EnsureChannelFirstd(keys=["volume"], channel_dim=0, strict_check=False),
            ScaleIntensityd(keys=["volume"]),
            CropForegroundd(keys=["volume"], source_key="volume"),
            SpatialPadd(keys=["volume"], spatial_size=(224, 224, 96)),
            Resized(keys=["volume"], spatial_size=(224, 224, 96))
        ])

# Visual Encoder Model
class VisualEncoder(nn.Module):
    def __init__(self, backbone='resnet50', feature_dim=512):
        super(VisualEncoder, self).__init__()
        if backbone == 'resnet50':
            # Use ResNet3D from MONAI
            self.encoder = monai.networks.nets.ResNet(
                block="bottleneck",
                layers=[3, 4, 6, 3],
                block_inplanes=[64, 128, 256, 512],
                spatial_dims=3,
                n_input_channels=1
            )
            # delete the last layer
            self.encoder.fc = nn.Identity()
            self.feature_dim = 2048
        elif backbone == 'vit':
            # Use Vision Transformer from MONAI
            self.encoder = monai.networks.nets.ViT(
                in_channels=1,
                img_size=(224, 224, 96),
                patch_size=(16, 16, 16),
                hidden_size=768,
                mlp_dim=3072,
                num_layers=12,
                num_heads=12,
                pos_embed="conv",
                classification=True,
                spatial_dims=3,
            )
            self.feature_dim = 768
        
        # Projection layer to match text embedding dimension
        self.projection = nn.Linear(self.feature_dim, feature_dim)
        
    def forward(self, x):
        features = self.encoder(x)
        projected = self.projection(features)
        return projected

# Cross-Attention Module
class CrossAttention(nn.Module):
    def __init__(self, dim):
        super(CrossAttention, self).__init__()
        self.scale = dim ** -0.5
        
    def forward(self, x1, x2):
        # Normalize feature vectors
        x1_norm = F.normalize(x1, dim=-1)
        x2_norm = F.normalize(x2, dim=-1)
        
        # Compute attention
        attn = torch.matmul(x1_norm, x2_norm.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        
        # Apply attention to x1
        output = x1 * attn
        return output

# MedFinder Model
class MedFinder(nn.Module):
    def __init__(self, backbone='resnet50', text_encoder='/h3cstore_ns/ydchen/code/MedicalFinder/models_HF/BioBert', feature_dim=512):
        super(MedFinder, self).__init__()
        
        # Visual encoder
        self.visual_encoder = VisualEncoder(backbone, feature_dim)
        
        # Text encoder (frozen)
        self.text_encoder = AutoModel.from_pretrained(text_encoder)
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        
        # Text projection
        self.text_projection = nn.Linear(self.text_encoder.config.hidden_size, feature_dim)
        
        # Cross-attention module
        self.cross_attention = CrossAttention(feature_dim)
        
        # Final pooling layer
        self.pooling = nn.AdaptiveAvgPool1d(1)
    
    def encode_text(self, input_ids, attention_mask):
        text_outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        # Use CLS token embedding
        text_features = text_outputs.last_hidden_state[:, 0, :]
        text_features = self.text_projection(text_features)
        return text_features
    
    def encode_image(self, x):
        return self.visual_encoder(x)
    
    def forward(self, volume, input_ids, attention_mask):
        # 提取原始体积特征
        vis_features = self.encode_image(volume)
        
        # 在验证或推理过程中直接复用特征
        if self.training or not self.training:
            vis_features_fusion = vis_features
            text_features = self.encode_text(input_ids, attention_mask)
            return vis_features, vis_features, vis_features, vis_features_fusion, text_features
        
        # 在训练时创建增强版本
        with torch.no_grad():  # 分离增强创建过程
            volume_aug1 = torch.stack([augment_volume(v) for v in volume])
            volume_aug2 = torch.stack([augment_volume(v) for v in volume])
        
        # 编码增强版本
        vis_features_aug1 = self.encode_image(volume_aug1)
        vis_features_aug2 = self.encode_image(volume_aug2)
        
        # 应用交叉注意力
        vis_features_fusion = self.cross_attention(vis_features_aug1, vis_features_aug2)
        
        # 编码文本特征
        text_features = self.encode_text(input_ids, attention_mask)
        
        return vis_features, vis_features_aug1, vis_features_aug2, vis_features_fusion, text_features

# Augmentation function for 3D volumes
def augment_volume(volume):
    # Apply random augmentations
    transforms_aug = Compose([
        RandRotated(keys=["volume"], range_x=0.2, range_y=0.2, range_z=0.2, prob=0.3),
        RandZoomd(keys=["volume"], min_zoom=0.9, max_zoom=1.1, prob=0.3),
        RandGaussianNoised(keys=["volume"], prob=0.3, mean=0.0, std=0.1)
    ])
    
    return transforms_aug({"volume": volume})["volume"]

# Loss functions
class MedFinderLoss(nn.Module):
    def __init__(self, alpha=1.0, temperature=0.07):
        super(MedFinderLoss, self).__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.mse_loss = nn.MSELoss()
        self.cross_entropy = nn.CrossEntropyLoss()
    
    def forward(self, vis_features_aug1, vis_features_aug2, vis_features_fusion, text_features, batch_size):
        # MSE loss for view consistency
        mse_loss = self.mse_loss(vis_features_aug1, vis_features_aug2)
        
        # Similarity loss
        logits = torch.matmul(text_features, vis_features_fusion.t()) / self.temperature
        targets = torch.arange(batch_size, device=logits.device)
        sim_loss = self.cross_entropy(logits, targets)
        
        # Total loss
        total_loss = mse_loss + self.alpha * sim_loss
        
        return total_loss, mse_loss, sim_loss

# Training function
def train(model, train_loader, optimizer, scheduler, criterion, device, epoch):
    model.train()
    running_loss = 0.0
    running_mse_loss = 0.0
    running_sim_loss = 0.0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
    for batch in pbar:
        # Get data
        volumes = batch["volume"].to(device)
        input_ids = batch["text"]["input_ids"].to(device)
        attention_mask = batch["text"]["attention_mask"].to(device)
        
        # Apply augmentations
        # volumes_aug1 = torch.stack([augment_volume(volume) for volume in volumes])
        # volumes_aug2 = torch.stack([augment_volume(volume) for volume in volumes])
        
        # Forward pass
        vis_features, vis_features_aug1, vis_features_aug2, vis_features_fusion, text_features = model(
            # volumes, volumes_aug1, volumes_aug2, input_ids, attention_mask
            volumes, input_ids, attention_mask
        )
        
        # Compute loss
        batch_size = volumes.size(0)
        loss, mse_loss, sim_loss = criterion(
            vis_features_aug1, vis_features_aug2, vis_features_fusion, text_features, batch_size
        )
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update metrics
        running_loss += loss.item()
        running_mse_loss += mse_loss.item()
        running_sim_loss += sim_loss.item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': running_loss / (pbar.n + 1),
            'mse_loss': running_mse_loss / (pbar.n + 1),
            'sim_loss': running_sim_loss / (pbar.n + 1)
        })
    
    # Update learning rate
    scheduler.step()
    
    # Return average losses
    return {
        'loss': running_loss / len(train_loader),
        'mse_loss': running_mse_loss / len(train_loader),
        'sim_loss': running_sim_loss / len(train_loader)
    }

# Function to compute retrieval metrics
def compute_metrics(similarities, batch_size):
    # Compute R@K, MdR, MnR
    metrics = {}
    
    # For each query, sort the similarities
    sorted_indices = torch.argsort(similarities, dim=1, descending=True)
    
    # Ground truth indices (diagonal)
    gt_indices = torch.arange(batch_size, device=similarities.device)
    
    # Calculate R@K
    for k in [1, 5, 10]:
        # Check if the ground truth is in the top-k predictions
        top_k_indices = sorted_indices[:, :k]
        is_match = (top_k_indices == gt_indices.unsqueeze(1)).any(dim=1)
        recall_at_k = is_match.float().mean().item()
        metrics[f'R@{k}'] = recall_at_k * 100
    
    # Calculate MdR (Median Rank)
    ranks = []
    for i in range(batch_size):
        rank = (sorted_indices[i] == i).nonzero(as_tuple=True)[0].item() + 1
        ranks.append(rank)
    
    median_rank = np.median(ranks)
    mean_rank = np.mean(ranks)
    
    metrics['MdR'] = median_rank
    metrics['MnR'] = mean_rank
    
    return metrics

# Validation function
def validate(model, val_loader, device):
    model.eval()
    text_embeddings = []
    image_embeddings = []
    all_text = []
    # all_diagnoses = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            # Get data
            volumes = batch["volume"].to(device)
            input_ids = batch["text"]["input_ids"].to(device)
            attention_mask = batch["text"]["attention_mask"].to(device)
            all_text.extend(batch["original_text"])
            # all_diagnoses.extend(batch["diagnosis"])
            
            # Forward pass
            vis_features, _, _, vis_features_fusion, text_features = model(
                volumes, volumes, volumes, input_ids, attention_mask
            )
            
            # Store embeddings
            text_embeddings.append(text_features)
            image_embeddings.append(vis_features_fusion)
    
    # Concatenate embeddings
    text_embeddings = torch.cat(text_embeddings, dim=0)
    image_embeddings = torch.cat(image_embeddings, dim=0)
    
    # Normalize embeddings
    text_embeddings = F.normalize(text_embeddings, dim=1)
    image_embeddings = F.normalize(image_embeddings, dim=1)
    
    # Compute similarities
    similarities_t2i = torch.matmul(text_embeddings, image_embeddings.t())
    similarities_i2t = similarities_t2i.t()
    
    # Compute metrics
    batch_size = text_embeddings.size(0)
    t2i_metrics = compute_metrics(similarities_t2i, batch_size)
    i2t_metrics = compute_metrics(similarities_i2t, batch_size)
    
    results = {
        'text_to_image': t2i_metrics,
        'image_to_text': i2t_metrics
    }
    
    return results, text_embeddings, image_embeddings, all_text

# Keyword-based retrieval evaluation
def evaluate_keyword_retrieval(text_embeddings, image_embeddings, all_text, all_diagnoses, keywords, device):
    results = {}
    
    for keyword in keywords:
        # Find ground truth samples with this keyword
        gt_indices = [i for i, diag in enumerate(all_diagnoses) if keyword.lower() in diag.lower()]
        
        if not gt_indices:
            logger.warning(f"No samples found with keyword: {keyword}")
            continue
        
        # Create a keyword embedding using a sample text
        keyword_texts = [all_text[i] for i in gt_indices]
        keyword_text = keyword_texts[0]  # Use the first one
        
        # Encode the keyword
        tokenizer = AutoTokenizer.from_pretrained("/h3cstore_ns/ydchen/code/MedicalFinder/models_HF/BioBert")
        text_encoder = AutoModel.from_pretrained("/h3cstore_ns/ydchen/code/MedicalFinder/models_HF/BioBert").to(device)
        
        with torch.no_grad():
            inputs = tokenizer(keyword, return_tensors="pt").to(device)
            keyword_embedding = text_encoder(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                return_dict=True
            ).last_hidden_state[:, 0, :]  # CLS token
            
            # Normalize the embedding
            keyword_embedding = F.normalize(keyword_embedding, dim=1)
        
        # Compute similarities
        similarities = torch.matmul(keyword_embedding, image_embeddings.t()).squeeze(0)
        
        # Sort by similarity
        sorted_indices = torch.argsort(similarities, descending=True)
        
        # Calculate precision@K
        for k in [20, 50, 100]:
            if k > len(sorted_indices):
                continue
                
            top_k_indices = sorted_indices[:k].cpu().numpy()
            top_k_relevant = sum(1 for idx in top_k_indices if idx in gt_indices)
            precision_at_k = top_k_relevant / k
            
            if keyword not in results:
                results[keyword] = {}
            results[keyword][f'P@{k}'] = precision_at_k * 100
    
    return results

# Main function
def main():
    parser = argparse.ArgumentParser(description="Train and evaluate MedFinder")
    parser.add_argument("--data_root", type=str, default="/h3cstore_ns/CT_data/CT_retrieval", help="Root directory of BIMCV-R dataset")
    parser.add_argument("--output_dir", type=str, default="/h3cstore_ns/ydchen/code/MedicalFinder/output", help="Output directory")
    parser.add_argument("--backbone", type=str, default="resnet50", choices=["resnet50", "vit"], help="Visual backbone")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--alpha", type=float, default=1.0, help="Weight for similarity loss")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--eval_only", action="store_true", help="Only run evaluation")
    parser.add_argument("--model_path", type=str, default=None, help="Path to pretrained model for evaluation")
    
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Create datasets and data loaders
    train_transform = get_transforms(mode='train')
    val_transform = get_transforms(mode='val')
    
    train_dataset = BIMCVRDataset(
        data_root=args.data_root,
        split='train',
        max_text_length=100,
        transform=train_transform
    )
    
    val_dataset = BIMCVRDataset(
        data_root=args.data_root,
        split='val',
        max_text_length=100,
        transform=val_transform
    )
    
    test_dataset = BIMCVRDataset(
        data_root=args.data_root,
        split='test',
        max_text_length=100,
        transform=val_transform
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Create model
    model = MedFinder(
        backbone=args.backbone,
        text_encoder="/h3cstore_ns/ydchen/code/MedicalFinder/models_HF/BioBert",
        feature_dim=512
    )
    model = model.to(device)
    
    # If evaluation only, load pre-trained model
    if args.eval_only:
        assert args.model_path is not None, "Model path must be provided for evaluation"
        model.load_state_dict(torch.load(args.model_path))
        logger.info(f"Loaded model from {args.model_path}")
        
        # Run evaluation
        logger.info("Evaluating model on test set...")
        results, text_embeddings, image_embeddings, all_text, all_diagnoses = validate(
            model, test_loader, device
        )
        
        # Log retrieval results
        logger.info("Text-to-Image Retrieval Results:")
        for metric, value in results['text_to_image'].items():
            logger.info(f"  {metric}: {value:.2f}")
        
        logger.info("Image-to-Text Retrieval Results:")
        for metric, value in results['image_to_text'].items():
            logger.info(f"  {metric}: {value:.2f}")
        
        # Evaluate keyword-based retrieval
        # keywords = ["atelectasis", "consolidation", "adenopathy"]
        # keyword_results = evaluate_keyword_retrieval(
        #     text_embeddings, image_embeddings, all_text, all_diagnoses, keywords, device
        # )
        
        # logger.info("Keyword-based Retrieval Results:")
        # for keyword, metrics in keyword_results.items():
        #     logger.info(f"  {keyword}:")
        #     for metric, value in metrics.items():
        #         logger.info(f"    {metric}: {value:.2f}")
        
        return
    
    # Create optimizer, scheduler, and criterion
    optimizer = AdamW(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = MedFinderLoss(alpha=args.alpha)
    
    # Training loop
    best_recall = 0.0
    for epoch in range(args.epochs):
        # Train
        train_metrics = train(model, train_loader, optimizer, scheduler, criterion, device, epoch)
        logger.info(f"Epoch {epoch+1}/{args.epochs} - Train Loss: {train_metrics['loss']:.4f}")
        
        # Validate
        val_results, _, _, _, _ = validate(model, val_loader, device)
        
        # Log validation results
        logger.info("Validation Results:")
        logger.info("Text-to-Image Retrieval:")
        for metric, value in val_results['text_to_image'].items():
            logger.info(f"  {metric}: {value:.2f}")
        
        logger.info("Image-to-Text Retrieval:")
        for metric, value in val_results['image_to_text'].items():
            logger.info(f"  {metric}: {value:.2f}")
        
        # Save best model
        current_recall = val_results['text_to_image']['R@10']
        if current_recall > best_recall:
            best_recall = current_recall
            torch.save(model.state_dict(), os.path.join(args.output_dir, "best_model.pt"))
            logger.info(f"Saved best model with R@10: {best_recall:.2f}")
        
        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_recall': best_recall,
        }, os.path.join(args.output_dir, f"checkpoint_epoch_{epoch+1}.pt"))
    
    # Load best model for final evaluation
    model.load_state_dict(torch.load(os.path.join(args.output_dir, "best_model.pt")))
    
    # # Evaluate on test set
    # logger.info("Evaluating model on test set...")
    # test_results, text_embeddings, image_embeddings, all_text, all_diagnoses = validate(
    #     model, test_loader, device
    # )
    
    # # Log test results
    # logger.info("Test Results:")
    # logger.info("Text-to-Image Retrieval:")
    # for metric, value in test_results['text_to_image'].items():
    #     logger.info(f"  {metric}: {value:.2f}")
    
    # logger.info("Image-to-Text Retrieval:")
    # for metric, value in test_results['image_to_text'].items():
    #     logger.info(f"  {metric}: {value:.2f}")
    
    # Evaluate keyword-based retrieval
    # keywords = ["atelectasis", "consolidation", "adenopathy"]
    # keyword_results = evaluate_keyword_retrieval(
    #     text_embeddings, image_embeddings, all_text, all_diagnoses, keywords, device
    # )
    
    # logger.info("Keyword-based Retrieval Results:")
    # for keyword, metrics in keyword_results.items():
    #     logger.info(f"  {keyword}:")
    #     for metric, value in metrics.items():
    #         logger.info(f"    {metric}: {value:.2f}")

if __name__ == "__main__":
    main()