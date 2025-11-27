"""
================================================================================
USDNet 完整训练程序 - 集成所有修复
================================================================================

✅ 修复内容：
1. 官方Res16UNet架构（正确的Skip Connection特征维度）
2. 修复体素化后的标签对应（使用np.unique去重）
3. 两阶段训练：预训练（DINO蒸馏） + 微调（语义分割）
4. 类别权重平衡
5. Warmup学习率调度
6. 混合精度训练

文件名: train_usdnet_complete.py

================================================================================
"""

import os
import sys
import logging
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field, asdict
from collections import defaultdict, Counter
import argparse
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler

import MinkowskiEngine as ME
import zarr
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


# ============================================================================
# 配置
# ============================================================================

@dataclass
class TrainingConfig:
    """训练配置"""
    
    # 数据
    zarr_root: str = "./td5"
    voxel_size: float = 0.05
    num_workers: int = 0  # 数据加载进程数（0=单进程稳定，>0需Docker --shm-size=8g）
    batch_size: int = 2
    pin_memory: bool = True
    max_points_per_sample: int = 50000  # 增加采样点数以保留更多信息
    
    # 模型
    num_classes: Optional[int] = None
    feature_dim_3d: int = 256
    feature_dim_2d: int = 768
    dropout: float = 0.1  # Dropout (0.3太高会降低性能)
    bn_momentum: float = 0.1
    
    # 早停
    early_stop_patience: int = 100  # 验证集性能不提升100个epoch则停止
    
    # 训练模式
    pretrain_mode: bool = False
    weight_seg: float = 1.0
    weight_distill: float = 0.5
    
    # 优化器
    optimizer: str = "adamw"
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    max_epochs: int = 100
    decay_rate: float = 0.95
    decay_step: int = 15
    gradient_clip: float = 1.0
    
    # Warmup
    warmup_epochs: int = 5
    warmup_lr: float = 1e-6
    
    # 其他
    use_mixed_precision: bool = True
    pretrain_checkpoint: Optional[str] = None
    device: str = "cuda:0"
    save_dir: str = "./checkpoints"
    seed: int = 42


# ============================================================================
# 工具函数
# ============================================================================

class GlobalLabelReader:
    """从 global_label_mapping.json 读取全局标签信息"""
    
    @staticmethod
    def load_global_mapping(zarr_root: str) -> Tuple[int, List[str], Dict[str, int]]:
        zarr_root = Path(zarr_root)
        mapping_file = zarr_root / "global_label_mapping.json"
        
        if not mapping_file.exists():
            logger.error(f"❌ 未找到全局标签映射文件: {mapping_file}")
            return 0, [], {}
        
        logger.info(f"✅ 加载全局标签映射: {mapping_file}")
        
        with open(mapping_file, 'r') as f:
            data = json.load(f)
        
        num_classes = data['num_classes']
        label_map = data['mapping']
        class_names = data['class_names']
        
        logger.info(f"  - 全局类别数: {num_classes}")
        logger.info(f"  - 类别名称数: {len(class_names)}")
        
        return num_classes, class_names, label_map


def setup_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================================
# 评估指标
# ============================================================================

class SegmentationMetrics:
    """语义分割评估指标"""
    
    def __init__(self, num_classes: int, class_names: Optional[List[str]] = None, ignore_index: int = -1):
        self.num_classes = num_classes
        self.class_names = class_names or [f"Class_{i}" for i in range(num_classes)]
        self.ignore_index = ignore_index
        self.confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    
    def update(self, pred: np.ndarray, target: np.ndarray):
        valid_mask = target != self.ignore_index
        if valid_mask.sum() == 0:
            return
        
        pred = pred[valid_mask]
        target = target[valid_mask]
        
        for p, t in zip(pred, target):
            if 0 <= p < self.num_classes and 0 <= t < self.num_classes:
                self.confusion_matrix[p, t] += 1
    
    def reset(self):
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
    
    def get_overall_accuracy(self) -> float:
        if self.confusion_matrix.sum() == 0:
            return 0.0
        correct = np.diag(self.confusion_matrix).sum()
        total = self.confusion_matrix.sum()
        return correct / total if total > 0 else 0.0
    
    def get_mean_iou(self) -> float:
        if self.confusion_matrix.sum() == 0:
            return 0.0
        
        ious = []
        for i in range(self.num_classes):
            tp = self.confusion_matrix[i, i]
            fp = self.confusion_matrix[i, :].sum() - tp
            fn = self.confusion_matrix[:, i].sum() - tp
            if tp + fp + fn > 0:
                ious.append(tp / (tp + fp + fn))
        
        return np.mean(ious) if ious else 0.0
    
    def print_summary(self, top_k: int = 5):
        if self.confusion_matrix.sum() == 0:
            logger.info("  (无有效数据)")
            return
        
        logger.info(f"  Overall Accuracy: {self.get_overall_accuracy()*100:.2f}%")
        logger.info(f"  Mean IoU: {self.get_mean_iou()*100:.2f}%")
        
        per_class_iou = {}
        for i in range(self.num_classes):
            tp = self.confusion_matrix[i, i]
            fp = self.confusion_matrix[i, :].sum() - tp
            fn = self.confusion_matrix[:, i].sum() - tp
            if tp + fp + fn > 0:
                per_class_iou[self.class_names[i]] = tp / (tp + fp + fn)
        
        if per_class_iou:
            sorted_classes = sorted(per_class_iou.items(), key=lambda x: x[1], reverse=True)
            logger.info(f"  Top-{top_k} Classes:")
            for i, (class_name, iou) in enumerate(sorted_classes[:top_k], 1):
                logger.info(f"    {i}. {class_name}: IoU={iou*100:.2f}%")


# ============================================================================
# 数据加载
# ============================================================================

class ZarrDataset(Dataset):
    """Zarr数据集加载器"""
    
    def __init__(
        self,
        zarr_files: List[Path],
        voxel_size: float = 0.05,
        max_points_per_sample: int = 8192,
        pretrain_mode: bool = False,
        augment: bool = False,  # 是否启用数据增强
    ):
        self.zarr_files = zarr_files
        self.voxel_size = voxel_size
        self.max_points_per_sample = max_points_per_sample
        self.pretrain_mode = pretrain_mode
        self.augment = augment
        
        logger.info(f"✓ 数据集: {len(self.zarr_files)} 个场景")
        logger.info(f"✓ 模式: {'预训练' if pretrain_mode else '微调'}")
        logger.info(f"✓ 数据增强: {'启用' if augment else '禁用'}")
    
    def __len__(self) -> int:
        return len(self.zarr_files)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        zarr_path = self.zarr_files[idx]
        scene_id = zarr_path.stem.replace('_dino_patch_level', '')
        
        root = zarr.open(str(zarr_path), mode='r')
        
        try:
            # 🚀 优化：直接读取为numpy，避免重复转换
            points = root['points'][:].astype(np.float32)
            dino_features = root['dino_features'][:].astype(np.float32)
            valid_mask = root['valid_mask'][:]
            
            # 优化颜色读取
            if 'colors' in root:
                colors = root['colors'][:].astype(np.float32)
                if colors.max() > 1.0:
                    colors = colors * (1.0 / 255.0)  # 向量化除法
            else:
                colors = np.full((len(points), 3), 0.5, dtype=np.float32)
            
            # 优化法向量读取
            if 'normals' in root:
                normals = root['normals'][:].astype(np.float32)
            else:
                normals = np.zeros((len(points), 3), dtype=np.float32)
                normals[:, 2] = 1.0
            
            # 优化标签读取
            if not self.pretrain_mode and 'semantic_labels' in root:
                semantic_labels = root['semantic_labels'][:].astype(np.int32)
            else:
                semantic_labels = np.full(len(points), -1, dtype=np.int32)
            
            # 过滤有效点
            valid_idx = np.where(valid_mask)[0]
            if len(valid_idx) == 0:
                valid_idx = np.arange(len(points))
            
            points = points[valid_idx]
            dino_features = dino_features[valid_idx]
            semantic_labels = semantic_labels[valid_idx]
            colors = colors[valid_idx]
            normals = normals[valid_idx]
            
            # 智能降采样：优先保留小类别的点
            if len(points) > self.max_points_per_sample:
                if not self.pretrain_mode and len(np.unique(semantic_labels[semantic_labels >= 0])) > 1:
                    # 🚀 优化：向量化计算采样权重
                    unique_labels, inverse_indices, counts = np.unique(
                        semantic_labels, return_inverse=True, return_counts=True
                    )
                    
                    # 计算每个类别的权重（向量化）
                    class_weights = np.ones(len(unique_labels), dtype=np.float32)
                    valid_mask = unique_labels >= 0
                    class_weights[valid_mask] = np.sqrt(len(points) / counts[valid_mask])
                    
                    # 映射到每个点（向量化）
                    sampling_weights = class_weights[inverse_indices]
                    sampling_weights = sampling_weights / sampling_weights.sum()
                    
                    # 加权采样
                    idx = np.random.choice(len(points), self.max_points_per_sample, replace=False, p=sampling_weights)
                else:
                    # 预训练模式或单类别场景：均匀采样
                    idx = np.random.choice(len(points), self.max_points_per_sample, replace=False)
                
                points = points[idx]
                dino_features = dino_features[idx]
                semantic_labels = semantic_labels[idx]
                colors = colors[idx]
                normals = normals[idx]
            
            # 数据增强（仅训练集）
            # ⚠️ 注意：不能dropout点，因为会破坏DINO特征对应关系！
            if self.augment:
                # 随机旋转（绕Z轴）
                theta = np.random.uniform(0, 2 * np.pi)
                cos_t, sin_t = np.cos(theta), np.sin(theta)
                rotation_z = np.array([[cos_t, -sin_t, 0],
                                      [sin_t, cos_t, 0],
                                      [0, 0, 1]], dtype=np.float32)
                points = points @ rotation_z.T
                normals = normals @ rotation_z.T
                
                # 随机缩放（轻微）
                scale = np.random.uniform(0.95, 1.05)
                points = points * scale
                
                # 随机抖动（减小幅度）
                jitter = np.random.normal(0, 0.005, points.shape).astype(np.float32)
                points = points + jitter
            
            # 🚀 优化：L2归一化DINO特征（向量化+clip）
            dino_norms = np.linalg.norm(dino_features, axis=1, keepdims=True)
            dino_features = dino_features / np.clip(dino_norms, 1e-6, None)
            
            return {
                'scene_id': scene_id,
                'points': torch.from_numpy(points).float(),
                'dino_features': torch.from_numpy(dino_features).float(),
                'semantic_labels': torch.from_numpy(semantic_labels).long(),
                'colors': torch.from_numpy(colors).float(),
                'normals': torch.from_numpy(normals).float(),
            }
        
        finally:
            root.store.close() if hasattr(root.store, 'close') else None


def collate_fn_sparse(batch: List[Dict], voxel_size: float = 0.05) -> Dict[str, Any]:
    """
    ✅ 修复版collate函数（向后兼容接口）
    
    关键修复：在体素化前对坐标去重，确保labels和coords一一对应
    """
    all_coords = []
    all_features = []
    all_labels = []
    all_teacher_feats = []
    
    for batch_idx, data in enumerate(batch):
        points = data['points'].numpy()
        dino_features = data['dino_features'].numpy()
        labels = data['semantic_labels'].numpy()
        colors = data['colors'].numpy()
        normals = data['normals'].numpy()
        
        # 计算体素坐标
        voxel_coords = np.floor(points / voxel_size).astype(np.int32)
        
        # ✅ 关键修复：对体素坐标去重
        unique_coords, unique_indices = np.unique(voxel_coords, axis=0, return_index=True)
        
        # 只保留唯一体素对应的点
        points_unique = points[unique_indices]
        dino_features_unique = dino_features[unique_indices]
        labels_unique = labels[unique_indices]
        colors_unique = colors[unique_indices]
        normals_unique = normals[unique_indices]
        
        # 添加batch索引
        batch_indices = np.full((len(unique_indices), 1), batch_idx, dtype=np.int32)
        coords_with_batch = np.hstack([batch_indices, unique_coords])
        
        # 特征：RGB + 法向量 + 坐标 (9维)
        combined_features = np.hstack([colors_unique, normals_unique, points_unique])
        
        all_coords.append(coords_with_batch)
        all_features.append(combined_features)
        all_labels.append(labels_unique)
        all_teacher_feats.append(dino_features_unique)
    
    coords = np.vstack(all_coords)
    features = np.vstack(all_features)
    labels = np.concatenate(all_labels)
    teacher_feats = np.vstack(all_teacher_feats)
    
    return {
        'coords': torch.from_numpy(coords).int(),
        'features': torch.from_numpy(features).float(),
        'labels': torch.from_numpy(labels).long(),
        'teacher_features': torch.from_numpy(teacher_feats).float(),
    }


# ============================================================================
# 模型 - 官方Res16UNet架构
# ============================================================================

class MinkowskiBasicBlock(nn.Module):
    """BasicBlock - 官方实现"""
    expansion = 1
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, 
                 dilation: int = 1, bn_momentum: float = 0.1):
        super().__init__()
        
        self.conv1 = ME.MinkowskiConvolution(
            in_channels=in_channels, out_channels=out_channels,
            kernel_size=3, stride=stride, dilation=dilation, dimension=3,
        )
        self.bn1 = ME.MinkowskiBatchNorm(out_channels, momentum=bn_momentum)
        
        self.conv2 = ME.MinkowskiConvolution(
            in_channels=out_channels, out_channels=out_channels,
            kernel_size=3, stride=1, dilation=dilation, dimension=3,
        )
        self.bn2 = ME.MinkowskiBatchNorm(out_channels, momentum=bn_momentum)
        
        self.relu = ME.MinkowskiReLU(inplace=True)
        
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                ME.MinkowskiConvolution(
                    in_channels=in_channels, out_channels=out_channels,
                    kernel_size=1, stride=stride, dimension=3,
                ),
                ME.MinkowskiBatchNorm(out_channels, momentum=bn_momentum),
            )
    
    def forward(self, x: ME.SparseTensor) -> ME.SparseTensor:
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out = out + identity
        out = self.relu(out)
        
        return out


class Res16UNetBackbone(nn.Module):
    """
    ✅ 官方Res16UNet架构
    
    关键修复：
    1. Skip connection时正确更新inplanes
    2. 使用expansion计算特征维度
    """
    
    BLOCK = MinkowskiBasicBlock
    PLANES = (32, 64, 128, 256, 128, 128, 96, 96)
    LAYERS = (2, 2, 2, 2, 2, 2, 2, 2)
    INIT_DIM = 32
    
    def __init__(self, in_channels: int = 9, out_channels: int = 256, bn_momentum: float = 0.1):
        super().__init__()
        
        self.inplanes = self.INIT_DIM
        self.relu = ME.MinkowskiReLU(inplace=True)
        
        # 初始卷积
        self.conv0p1s1 = ME.MinkowskiConvolution(
            in_channels=in_channels, out_channels=self.INIT_DIM,
            kernel_size=3, stride=1, dimension=3,
        )
        self.bn0 = ME.MinkowskiBatchNorm(self.INIT_DIM, momentum=bn_momentum)
        
        # Encoder
        self.conv1p1s2 = ME.MinkowskiConvolution(
            in_channels=self.INIT_DIM, out_channels=self.INIT_DIM,
            kernel_size=2, stride=2, dimension=3,
        )
        self.bn1 = ME.MinkowskiBatchNorm(self.INIT_DIM, momentum=bn_momentum)
        self.block1 = self._make_layer(self.INIT_DIM, self.PLANES[0], self.LAYERS[0], bn_momentum)
        
        self.conv2p2s2 = ME.MinkowskiConvolution(
            in_channels=self.inplanes, out_channels=self.inplanes,
            kernel_size=2, stride=2, dimension=3,
        )
        self.bn2 = ME.MinkowskiBatchNorm(self.inplanes, momentum=bn_momentum)
        self.block2 = self._make_layer(self.inplanes, self.PLANES[1], self.LAYERS[1], bn_momentum)
        
        self.conv3p4s2 = ME.MinkowskiConvolution(
            in_channels=self.inplanes, out_channels=self.inplanes,
            kernel_size=2, stride=2, dimension=3,
        )
        self.bn3 = ME.MinkowskiBatchNorm(self.inplanes, momentum=bn_momentum)
        self.block3 = self._make_layer(self.inplanes, self.PLANES[2], self.LAYERS[2], bn_momentum)
        
        self.conv4p8s2 = ME.MinkowskiConvolution(
            in_channels=self.inplanes, out_channels=self.inplanes,
            kernel_size=2, stride=2, dimension=3,
        )
        self.bn4 = ME.MinkowskiBatchNorm(self.inplanes, momentum=bn_momentum)
        self.block4 = self._make_layer(self.inplanes, self.PLANES[3], self.LAYERS[3], bn_momentum)
        
        # Decoder
        self.convtr4p16s2 = ME.MinkowskiConvolutionTranspose(
            in_channels=self.inplanes, out_channels=self.PLANES[4],
            kernel_size=2, stride=2, dimension=3,
        )
        self.bntr4 = ME.MinkowskiBatchNorm(self.PLANES[4], momentum=bn_momentum)
        self.inplanes = self.PLANES[4] + self.PLANES[2] * self.BLOCK.expansion
        self.block5 = self._make_layer(self.inplanes, self.PLANES[4], self.LAYERS[4], bn_momentum)
        
        self.convtr5p8s2 = ME.MinkowskiConvolutionTranspose(
            in_channels=self.inplanes, out_channels=self.PLANES[5],
            kernel_size=2, stride=2, dimension=3,
        )
        self.bntr5 = ME.MinkowskiBatchNorm(self.PLANES[5], momentum=bn_momentum)
        self.inplanes = self.PLANES[5] + self.PLANES[1] * self.BLOCK.expansion
        self.block6 = self._make_layer(self.inplanes, self.PLANES[5], self.LAYERS[5], bn_momentum)
        
        self.convtr6p4s2 = ME.MinkowskiConvolutionTranspose(
            in_channels=self.inplanes, out_channels=self.PLANES[6],
            kernel_size=2, stride=2, dimension=3,
        )
        self.bntr6 = ME.MinkowskiBatchNorm(self.PLANES[6], momentum=bn_momentum)
        self.inplanes = self.PLANES[6] + self.PLANES[0] * self.BLOCK.expansion
        self.block7 = self._make_layer(self.inplanes, self.PLANES[6], self.LAYERS[6], bn_momentum)
        
        self.convtr7p2s2 = ME.MinkowskiConvolutionTranspose(
            in_channels=self.inplanes, out_channels=self.PLANES[7],
            kernel_size=2, stride=2, dimension=3,
        )
        self.bntr7 = ME.MinkowskiBatchNorm(self.PLANES[7], momentum=bn_momentum)
        self.inplanes = self.PLANES[7] + self.INIT_DIM
        self.block8 = self._make_layer(self.inplanes, self.PLANES[7], self.LAYERS[7], bn_momentum)
        
        # Final projection
        self.final = ME.MinkowskiConvolution(
            in_channels=self.inplanes, out_channels=out_channels,
            kernel_size=1, stride=1, dimension=3,
        )
    
    def _make_layer(self, in_channels, out_channels, blocks, bn_momentum):
        layers = []
        layers.append(MinkowskiBasicBlock(in_channels, out_channels, stride=1, bn_momentum=bn_momentum))
        self.inplanes = out_channels * self.BLOCK.expansion
        for _ in range(1, blocks):
            layers.append(MinkowskiBasicBlock(self.inplanes, out_channels, stride=1, bn_momentum=bn_momentum))
        return nn.Sequential(*layers)
    
    def forward(self, x: ME.SparseTensor) -> ME.SparseTensor:
        # 初始特征
        out = self.conv0p1s1(x)
        out = self.bn0(out)
        out_p1 = self.relu(out)
        
        # Encoder
        out = self.conv1p1s2(out_p1)
        out = self.bn1(out)
        out = self.relu(out)
        out_b1p2 = self.block1(out)
        
        out = self.conv2p2s2(out_b1p2)
        out = self.bn2(out)
        out = self.relu(out)
        out_b2p4 = self.block2(out)
        
        out = self.conv3p4s2(out_b2p4)
        out = self.bn3(out)
        out = self.relu(out)
        out_b3p8 = self.block3(out)
        
        out = self.conv4p8s2(out_b3p8)
        out = self.bn4(out)
        out = self.relu(out)
        out = self.block4(out)
        
        # Decoder with skip connections
        out = self.convtr4p16s2(out)
        out = self.bntr4(out)
        out = self.relu(out)
        out = ME.cat(out, out_b3p8)
        out = self.block5(out)
        
        out = self.convtr5p8s2(out)
        out = self.bntr5(out)
        out = self.relu(out)
        out = ME.cat(out, out_b2p4)
        out = self.block6(out)
        
        out = self.convtr6p4s2(out)
        out = self.bntr6(out)
        out = self.relu(out)
        out = ME.cat(out, out_b1p2)
        out = self.block7(out)
        
        out = self.convtr7p2s2(out)
        out = self.bntr7(out)
        out = self.relu(out)
        out = ME.cat(out, out_p1)
        out = self.block8(out)
        
        feat_3d = self.final(out)
        
        return feat_3d


class USDNetStudent(nn.Module):
    """USDNet学生网络"""
    
    def __init__(self, num_classes: int, feature_dim_3d: int = 256, 
                 feature_dim_2d: int = 768, bn_momentum: float = 0.1, dropout: float = 0.1):
        super().__init__()
        
        self.num_classes = num_classes
        self.feature_dim_3d = feature_dim_3d
        self.feature_dim_2d = feature_dim_2d
        
        self.backbone = Res16UNetBackbone(
            in_channels=9, out_channels=feature_dim_3d, bn_momentum=bn_momentum,
        )
        
        self.distill_head = nn.Sequential(
            ME.MinkowskiLinear(feature_dim_3d, 512),
            ME.MinkowskiBatchNorm(512),
            ME.MinkowskiReLU(inplace=True),
            ME.MinkowskiDropout(p=dropout),
            ME.MinkowskiLinear(512, feature_dim_2d),
        )
        
        self.seg_head = nn.Sequential(
            ME.MinkowskiLinear(feature_dim_3d, 128),
            ME.MinkowskiBatchNorm(128),
            ME.MinkowskiReLU(inplace=True),
            ME.MinkowskiDropout(p=dropout),
            ME.MinkowskiLinear(128, num_classes),
        )
    
    def forward(self, x: ME.SparseTensor) -> Tuple[ME.SparseTensor, ME.SparseTensor]:
        feat_3d = self.backbone(x)
        seg_logits = self.seg_head(feat_3d)
        distill_features = self.distill_head(feat_3d)
        return seg_logits, distill_features


# ============================================================================
# 损失函数
# ============================================================================

class TwoStageDistillationLoss(nn.Module):
    """两阶段蒸馏损失"""
    
    def __init__(self, num_classes: int, weight_seg: float = 1.0,
                 weight_distill: float = 0.5, pretrain_mode: bool = False,
                 class_weights: Optional[torch.Tensor] = None):
        super().__init__()
        
        self.num_classes = num_classes
        self.weight_seg = weight_seg
        self.weight_distill = weight_distill
        self.pretrain_mode = pretrain_mode
        
        self.seg_loss_fn = nn.CrossEntropyLoss(ignore_index=-1, weight=class_weights)
    
    def forward(self, seg_logits: ME.SparseTensor, distill_features: ME.SparseTensor,
                semantic_labels: torch.Tensor, teacher_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        
        logits = seg_logits.features
        distill_feats = distill_features.features
        
        # 语义分割损失
        if self.weight_seg > 0 and not self.pretrain_mode:
            loss_seg = self.seg_loss_fn(logits, semantic_labels)
        else:
            loss_seg = torch.tensor(0.0, device=logits.device)
        
        # 蒸馏损失
        if self.weight_distill > 0:
            loss_distill_l2 = F.mse_loss(distill_feats, teacher_features)
            cos_sim = F.cosine_similarity(distill_feats, teacher_features, dim=1)
            loss_distill_cos = (1 - cos_sim).mean()
            loss_distill = loss_distill_l2 + loss_distill_cos
        else:
            loss_distill = torch.tensor(0.0, device=logits.device)
        
        loss_total = self.weight_seg * loss_seg + self.weight_distill * loss_distill
        
        return {
            'loss_total': loss_total,
            'loss_seg': loss_seg.detach() if isinstance(loss_seg, torch.Tensor) else loss_seg,
            'loss_distill': loss_distill.detach() if isinstance(loss_distill, torch.Tensor) else loss_distill,
        }


# ============================================================================
# 训练器
# ============================================================================

class Trainer:
    def __init__(self, model, loss_fn, optimizer, scheduler, config, 
                 class_names=None, device='cuda:0', val_loader=None):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.device = device
        self.val_loader = val_loader
        
        Path(config.save_dir).mkdir(parents=True, exist_ok=True)
        
        self.global_step = 0
        self.train_metrics = SegmentationMetrics(config.num_classes, class_names=class_names, ignore_index=-1)
        self.val_metrics = SegmentationMetrics(config.num_classes, class_names=class_names, ignore_index=-1)
        
        self.use_mixed_precision = config.use_mixed_precision
        self.scaler = GradScaler() if self.use_mixed_precision else None
        # 预训练：损失越小越好(初始无穷大) | 微调：mIoU越大越好(初始0)
        self.best_metric = float('inf') if config.pretrain_mode else 0.0
        self.patience_counter = 0  # 早停计数器
    
    def get_lr(self, epoch):
        if not self.config.pretrain_mode and epoch < self.config.warmup_epochs:
            return self.config.warmup_lr + (self.config.learning_rate - self.config.warmup_lr) * epoch / self.config.warmup_epochs
        return self.optimizer.param_groups[0]['lr']
    
    def set_lr(self, lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
    
    def train_epoch(self, train_loader, epoch):
        self.model.train()
        self.train_metrics.reset()
        
        current_lr = self.get_lr(epoch)
        self.set_lr(current_lr)
        
        loss_meter = defaultdict(list)
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train] LR={current_lr:.2e}", leave=False)
        
        for batch in pbar:
            try:
                coords = batch['coords'].to(self.device)
                features = batch['features'].to(self.device)
                labels = batch['labels'].to(self.device)
                teacher_features = batch['teacher_features'].to(self.device)
                
                x = ME.SparseTensor(features=features, coordinates=coords, device=self.device)
                
                if self.use_mixed_precision:
                    with autocast():
                        seg_logits, distill_features = self.model(x)
                        loss_dict = self.loss_fn(seg_logits, distill_features, labels, teacher_features)
                        loss_total = loss_dict['loss_total']
                    
                    self.optimizer.zero_grad()
                    self.scaler.scale(loss_total).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    seg_logits, distill_features = self.model(x)
                    loss_dict = self.loss_fn(seg_logits, distill_features, labels, teacher_features)
                    loss_total = loss_dict['loss_total']
                    
                    self.optimizer.zero_grad()
                    loss_total.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
                    self.optimizer.step()
                
                if not self.config.pretrain_mode:
                    pred_labels = seg_logits.features.argmax(dim=-1).cpu().numpy()
                    valid_labels = labels.cpu().numpy()
                    self.train_metrics.update(pred_labels, valid_labels)
                
                for key, val in loss_dict.items():
                    if isinstance(val, torch.Tensor):
                        loss_meter[key].append(val.item())
                
                self.global_step += 1
                pbar.set_postfix({'loss': f"{loss_total.item():.4f}"})
            
            except RuntimeError as e:
                if 'out of memory' in str(e).lower():
                    logger.error("❌ OOM")
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise
        
        if self.scheduler and (self.config.pretrain_mode or epoch >= self.config.warmup_epochs):
            self.scheduler.step()
        
        avg_losses = {key: np.mean(vals) if vals else 0 for key, vals in loss_meter.items()}
        
        if not self.config.pretrain_mode:
            avg_losses['oa'] = self.train_metrics.get_overall_accuracy()
            avg_losses['miou'] = self.train_metrics.get_mean_iou()
        
        return avg_losses
    
    @torch.no_grad()
    def validate(self, val_loader, epoch):
        self.model.eval()
        self.val_metrics.reset()
        
        loss_meter = defaultdict(list)
        pbar = tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]", leave=False)
        
        for batch in pbar:
            try:
                coords = batch['coords'].to(self.device)
                features = batch['features'].to(self.device)
                labels = batch['labels'].to(self.device)
                teacher_features = batch['teacher_features'].to(self.device)
                
                x = ME.SparseTensor(features=features, coordinates=coords, device=self.device)
                seg_logits, distill_features = self.model(x)
                loss_dict = self.loss_fn(seg_logits, distill_features, labels, teacher_features)
                
                if not self.config.pretrain_mode:
                    pred_labels = seg_logits.features.argmax(dim=-1).cpu().numpy()
                    valid_labels = labels.cpu().numpy()
                    self.val_metrics.update(pred_labels, valid_labels)
                
                for key, val in loss_dict.items():
                    if isinstance(val, torch.Tensor):
                        loss_meter[key].append(val.item())
            except RuntimeError as e:
                if 'out of memory' in str(e).lower():
                    logger.warning("⚠️ 验证时OOM，跳过该批次")
                torch.cuda.empty_cache()
                continue
        
        avg_losses = {key: np.mean(vals) if vals else 0 for key, vals in loss_meter.items()}
        
        if not self.config.pretrain_mode:
            avg_losses['oa'] = self.val_metrics.get_overall_accuracy()
            avg_losses['miou'] = self.val_metrics.get_mean_iou()
        
        return avg_losses
    
    def save_checkpoint(self, epoch, is_best=False):
        ckpt = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'config': asdict(self.config),
            'best_metric': self.best_metric,
        }
        
        if self.scheduler:
            ckpt['scheduler'] = self.scheduler.state_dict()
        
        ckpt_path = Path(self.config.save_dir) / f"ckpt_epoch_{epoch:03d}.pt"
        torch.save(ckpt, ckpt_path)
        
        if is_best:
            best_path = Path(self.config.save_dir) / "ckpt_best.pt"
            torch.save(ckpt, best_path)
            logger.info(f"  ✓ 最佳模型: {best_path}")
    
    def train(self, train_loader, val_loader=None):
        logger.info("="*80)
        logger.info(f"🚀 开始训练 ({'预训练' if self.config.pretrain_mode else '微调'})")
        logger.info("="*80)
        
        for epoch in range(self.config.max_epochs):
            train_losses = self.train_epoch(train_loader, epoch)
            
            if val_loader:
                val_losses = self.validate(val_loader, epoch)
            else:
                val_losses = {}
            
            logger.info(f"\n[Epoch {epoch+1}/{self.config.max_epochs}]")
            
            if self.config.pretrain_mode:
                logger.info(f"Loss: {train_losses.get('loss_total', 0):.4f} | Distill: {train_losses.get('loss_distill', 0):.4f}")
            else:
                logger.info(f"Loss: {train_losses.get('loss_total', 0):.4f} | Seg: {train_losses.get('loss_seg', 0):.4f}")
                logger.info("\n📊 Train:")
                self.train_metrics.print_summary(top_k=5)
                
                if val_loader:
                    logger.info("\n📊 Val:")
                    self.val_metrics.print_summary(top_k=5)
            
            if self.config.pretrain_mode:
                # 预训练：蒸馏损失越小越好
                current_metric = train_losses.get('loss_distill', float('inf'))
                is_best = current_metric < self.best_metric
            else:
                # 微调：mIoU越大越好
                current_metric = val_losses.get('miou', 0) if val_losses else train_losses.get('miou', 0)
                is_best = current_metric > self.best_metric
            
            if is_best:
                self.best_metric = current_metric
                self.patience_counter = 0  # 重置计数器
                logger.info("✨ 新的最佳模型!")
            else:
                self.patience_counter += 1
                logger.info(f"⏳ 验证指标未提升 ({self.patience_counter}/{self.config.early_stop_patience})")
            
            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(epoch, is_best)
            
            # 早停检查
            if self.patience_counter >= self.config.early_stop_patience:
                logger.info(f"\n⚠️  早停触发：验证集性能连续 {self.config.early_stop_patience} 个epoch未提升")
                logger.info(f"✓ 最佳指标: {self.best_metric:.4f}")
                break
        
        logger.info("\n🎉 训练完成！")


# ============================================================================
# 主程序
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="USDNet完整训练程序")
    
    parser.add_argument('--stage', type=str, required=True, choices=['pretrain', 'finetune'])
    parser.add_argument('--zarr_root', type=str, default="./td6")
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument('--pretrain_checkpoint', type=str, default=None)
    parser.add_argument('--save_dir', type=str, default="./c6")
    parser.add_argument('--voxel_size', type=float, default=0.05)
    parser.add_argument('--num_classes', type=int, default=None)
    parser.add_argument('--num_workers', type=int, default=0, help='数据加载进程数（0=单进程稳定，>0需Docker --shm-size）')
    parser.add_argument('--use_val', action='store_true', help='使用验证集（否则过拟合测试）')
    parser.add_argument('--no_distill', action='store_true', help='禁用DINO蒸馏，只训练语义分割')
    
    args = parser.parse_args()
    
    setup_seed(42)
    
    config = TrainingConfig()
    config.zarr_root = args.zarr_root
    config.save_dir = args.save_dir
    config.device = args.device
    config.batch_size = args.batch_size
    config.max_epochs = args.max_epochs
    config.learning_rate = args.learning_rate
    config.voxel_size = args.voxel_size
    config.num_workers = args.num_workers
    
    # 设置训练模式
    if args.stage == 'pretrain':
        config.pretrain_mode = True
        config.weight_seg = 0.0
        config.weight_distill = 1.0
        config.save_dir = str(Path(config.save_dir) / "pretrain")
        logger.info("🔧 模式：【预训练】- 仅DINO蒸馏")
    else:
        config.pretrain_mode = False
        config.weight_seg = 1.0
        
        # 如果指定了 --no_distill，禁用蒸馏
        if args.no_distill:
            config.weight_distill = 0.0
            logger.info("🔧 模式：【微调】- 纯语义分割（禁用蒸馏）")
        else:
            config.weight_distill = 0.5  # 微调阶段继续蒸馏
            logger.info("🔧 模式：【微调】- 语义分割 + DINO蒸馏")
        
        config.pretrain_checkpoint = args.pretrain_checkpoint
        config.save_dir = str(Path(config.save_dir) / "finetune")
    
    logger.info("="*80)
    logger.info(f"🎓 USDNet训练 - {args.stage.upper()}")
    logger.info("="*80)
    
    # 加载全局类别映射
    num_classes, class_names, label_map = GlobalLabelReader.load_global_mapping(args.zarr_root)
    
    if num_classes == 0:
        logger.error("❌ 无法加载全局标签映射！")
        return
    
    # 自动使用标签映射文件中的类别数（推荐）
    if args.num_classes is not None:
        logger.warning(f"⚠️  手动覆盖类别数: {args.num_classes}（标签映射文件中为 {num_classes}）")
        logger.warning(f"   建议删除 --num_classes 参数，自动使用标签映射文件中的类别数")
        config.num_classes = args.num_classes
    else:
        config.num_classes = num_classes
        logger.info(f"✅ 自动使用标签映射文件中的类别数: {num_classes}")
        logger.info(f"   这是推荐做法，确保模型输出与标签一致")
    
    # 加载数据
    zarr_root = Path(args.zarr_root)
    all_zarr_files = sorted(zarr_root.glob("*_dino_patch_level.zarr"))
    
    logger.info(f"\n✓ 找到 {len(all_zarr_files)} 个场景")
    
    if len(all_zarr_files) == 0:
        logger.error("❌ 未找到任何 Zarr 文件！请检查 --zarr_root 路径")
        return
    
    # 判断是否真正分割数据集
    use_separate_val = args.use_val and len(all_zarr_files) >= 5
    
    if use_separate_val:
        # 只有足够多的场景时才分割训练/验证集
        split_idx = int(len(all_zarr_files) * 0.8)
        train_files = all_zarr_files[:split_idx]
        val_files = all_zarr_files[split_idx:]
        logger.info(f"📊 训练: {len(train_files)} | 验证: {len(val_files)}")
    else:
        # 场景较少时，训练集=验证集（过拟合测试）
        train_files = all_zarr_files
        val_files = all_zarr_files
        logger.info("🧪 过拟合测试模式：训练=验证（场景数较少）")
    
    # 数据增强策略：
    # 1. 预训练模式：禁用数据增强
    # 2. 微调模式 + 真正的验证集分割：启用数据增强
    # 3. 微调模式 + 训练=验证（过拟合测试）：禁用数据增强（避免训练/验证不一致）
    enable_augmentation = (not config.pretrain_mode) and use_separate_val
    
    train_dataset = ZarrDataset(
        zarr_files=train_files,
        voxel_size=config.voxel_size,
        max_points_per_sample=config.max_points_per_sample,
        pretrain_mode=config.pretrain_mode,
        augment=enable_augmentation,
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=partial(collate_fn_sparse, voxel_size=config.voxel_size),  # 使用partial固定参数（可pickle）
        pin_memory=config.pin_memory,
        persistent_workers=config.num_workers > 0,  # 保持worker存活，避免重启开销
        prefetch_factor=2 if config.num_workers > 0 else None,  # 预取2个batch
        multiprocessing_context='spawn' if config.num_workers > 0 else None,  # 修复worker崩溃
    )
    
    val_dataset = ZarrDataset(
        zarr_files=val_files,
        voxel_size=config.voxel_size,
        max_points_per_sample=config.max_points_per_sample,
        pretrain_mode=config.pretrain_mode,
        augment=False,  # 验证集禁用数据增强
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,  # 验证集不shuffle，便于调试和复现
        num_workers=config.num_workers,
        collate_fn=partial(collate_fn_sparse, voxel_size=config.voxel_size),  # 使用partial固定参数（可pickle）
        pin_memory=config.pin_memory,
        persistent_workers=config.num_workers > 0,
        prefetch_factor=2 if config.num_workers > 0 else None,
        multiprocessing_context='spawn' if config.num_workers > 0 else None,  # 修复worker崩溃
    )
    
    # 初始化模型
    logger.info("\n🧠 初始化模型...")
    logger.info(f"  - Dropout: {config.dropout}")
    logger.info(f"  - 数据增强: {'启用' if (not config.pretrain_mode) else '禁用'}")
    
    model = USDNetStudent(
        num_classes=config.num_classes,
        feature_dim_3d=config.feature_dim_3d,
        feature_dim_2d=config.feature_dim_2d,
        dropout=config.dropout,
    ).to(config.device)
    
    # 微调：加载预训练权重
    if args.stage == 'finetune' and args.pretrain_checkpoint:
        logger.info(f"📥 加载预训练权重: {args.pretrain_checkpoint}")
        ckpt = torch.load(args.pretrain_checkpoint, map_location=config.device)
        
        pretrained_dict = ckpt['model']
        model_dict = model.state_dict()
        
        # 只加载backbone和distill_head
        filtered_dict = {k: v for k, v in pretrained_dict.items() 
                        if k in model_dict and not k.startswith('seg_head')}
        
        model_dict.update(filtered_dict)
        model.load_state_dict(model_dict)
        
        logger.info(f"✓ 加载: {len(filtered_dict)}/{len(model_dict)} 个参数")
    
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"✓ 参数量: {total_params:,}")
    
    # 计算类别权重
    class_weights = None
    if args.stage == 'finetune':
        logger.info("\n计算类别权重...")
        all_labels = []
        for zarr_file in tqdm(all_zarr_files[:min(10, len(all_zarr_files))], desc="统计类别"):
            root = zarr.open(str(zarr_file), mode='r')
            if 'semantic_labels' in root:
                labels = np.array(root['semantic_labels'][:])
                valid_labels = labels[labels >= 0]
                all_labels.extend(valid_labels.tolist())
            root.store.close() if hasattr(root.store, 'close') else None
        
        if len(all_labels) > 0:
            label_counts = Counter(all_labels)
            weights = np.ones(config.num_classes, dtype=np.float32)
            total_count = sum(label_counts.values())
            
            # 使用平方根平滑，避免极端权重
            for label_id, count in label_counts.items():
                if 0 <= label_id < config.num_classes and count > 0:
                    weights[label_id] = np.sqrt(total_count / (count * config.num_classes))
            
            # 对完全未出现的类别赋予较高权重（推测为稀有类别）
            for label_id in range(config.num_classes):
                if label_id not in label_counts:
                    weights[label_id] = 5.0  # 稀有类别高权重
            
            # 归一化权重
            weights = weights / weights.mean()
            weights = np.clip(weights, 0.5, 3.0)  # 更温和的权重范围
            class_weights = torch.from_numpy(weights).float().to(config.device)
            
            logger.info(f"✓ 类别权重: min={weights.min():.3f}, max={weights.max():.3f}")
    
    loss_fn = TwoStageDistillationLoss(
        num_classes=config.num_classes,
        weight_seg=config.weight_seg,
        weight_distill=config.weight_distill,
        pretrain_mode=config.pretrain_mode,
        class_weights=class_weights,
    ).to(config.device)
    
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config.decay_step, gamma=config.decay_rate)
    
    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        class_names=class_names,
        device=config.device,
        val_loader=val_loader,
    )
    
    trainer.train(train_loader, val_loader)
    
    logger.info("\n✨ 完成！")


if __name__ == "__main__":
    main()
