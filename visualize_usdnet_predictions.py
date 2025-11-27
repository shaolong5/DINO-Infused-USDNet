"""
================================================================================
USDNet 可视化程序 - 适配Zarr数据和训练好的模型
================================================================================

功能：
1. 加载训练好的USDNet模型
2. 从Zarr文件读取点云数据
3. 推理预测语义标签
4. 可视化：Ground Truth vs Prediction
5. 生成HTML交互式可视化

使用方法：
python visualize_usdnet_predictions.py \
    --checkpoint ./c6/finetune/ckpt_best.pt \
    --zarr_root ./td5 \
    --scene_id <scene_id> \
    --output_dir ./visualizations

================================================================================
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import MinkowskiEngine as ME
import zarr
from tqdm import tqdm

try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False
    print("⚠️  Open3D未安装，无法使用Open3D可视化")

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    print("⚠️  Plotly未安装，无法生成HTML可视化")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# 配色方案
# ============================================================================

class ColorPalette:
    """类别颜色映射"""
    
    @staticmethod
    def generate_colors(num_classes: int) -> np.ndarray:
        """生成区分度高的颜色"""
        np.random.seed(42)
        colors = np.random.rand(num_classes, 3)
        colors = (colors * 255).astype(np.uint8)
        return colors
    
    @staticmethod
    def get_semantic_colors(num_classes: int) -> Dict[int, Tuple[int, int, int]]:
        """返回类别ID到RGB颜色的映射"""
        colors = ColorPalette.generate_colors(num_classes)
        return {i: tuple(colors[i]) for i in range(num_classes)}


# ============================================================================
# 模型定义（与train_usdnet_complete.py相同）
# ============================================================================

class MinkowskiBasicBlock(nn.Module):
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
    BLOCK = MinkowskiBasicBlock
    PLANES = (32, 64, 128, 256, 128, 128, 96, 96)
    LAYERS = (2, 2, 2, 2, 2, 2, 2, 2)
    INIT_DIM = 32
    
    def __init__(self, in_channels: int = 9, out_channels: int = 256, bn_momentum: float = 0.1):
        super().__init__()
        
        self.inplanes = self.INIT_DIM
        self.relu = ME.MinkowskiReLU(inplace=True)
        
        self.conv0p1s1 = ME.MinkowskiConvolution(
            in_channels=in_channels, out_channels=self.INIT_DIM,
            kernel_size=3, stride=1, dimension=3,
        )
        self.bn0 = ME.MinkowskiBatchNorm(self.INIT_DIM, momentum=bn_momentum)
        
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
        out = self.conv0p1s1(x)
        out = self.bn0(out)
        out_p1 = self.relu(out)
        
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
# Zarr数据加载
# ============================================================================

class ZarrSceneLoader:
    """从Zarr文件加载场景数据"""
    
    @staticmethod
    def load_scene(zarr_path: Path, voxel_size: float = 0.05) -> Dict:
        """
        加载Zarr场景数据
        
        Returns:
            dict with keys: points, colors, normals, semantic_labels, dino_features
        """
        logger.info(f"加载Zarr: {zarr_path.name}")
        
        root = zarr.open(str(zarr_path), mode='r')
        
        try:
            points = np.array(root['points'][:], dtype=np.float32)
            colors = np.array(root['colors'][:], dtype=np.float32) if 'colors' in root else None
            normals = np.array(root['normals'][:], dtype=np.float32) if 'normals' in root else None
            semantic_labels = np.array(root['semantic_labels'][:], dtype=np.int32) if 'semantic_labels' in root else None
            dino_features = np.array(root['dino_features'][:], dtype=np.float32) if 'dino_features' in root else None
            valid_mask = np.array(root['valid_mask'][:], dtype=bool) if 'valid_mask' in root else np.ones(len(points), dtype=bool)
            
            # 归一化颜色
            if colors is not None and colors.max() > 1.0:
                colors = colors / 255.0
            
            # 默认法向量
            if normals is None:
                normals = np.zeros_like(points, dtype=np.float32)
                normals[:, 2] = 1.0
            
            logger.info(f"  - 点数: {len(points):,}")
            logger.info(f"  - 有效点: {valid_mask.sum():,}")
            
            if semantic_labels is not None:
                valid_labels = semantic_labels[semantic_labels >= 0]
                logger.info(f"  - 有效标签: {len(valid_labels):,}")
                logger.info(f"  - 标签范围: [{valid_labels.min()}, {valid_labels.max()}]")
            
            return {
                'points': points,
                'colors': colors,
                'normals': normals,
                'semantic_labels': semantic_labels,
                'dino_features': dino_features,
                'valid_mask': valid_mask,
                'voxel_size': voxel_size,
            }
        
        finally:
            root.store.close() if hasattr(root.store, 'close') else None


# ============================================================================
# 推理引擎
# ============================================================================

class InferenceEngine:
    """USDNet推理引擎"""
    
    def __init__(self, checkpoint_path: str, device: str = 'cuda:0'):
        self.device = device
        
        logger.info(f"加载模型: {checkpoint_path}")
        
        # 加载checkpoint
        ckpt = torch.load(checkpoint_path, map_location=device)
        config = ckpt.get('config', {})
        
        self.num_classes = config.get('num_classes', 100)
        logger.info(f"  - 类别数: {self.num_classes}")
        
        # 初始化模型
        self.model = USDNetStudent(
            num_classes=self.num_classes,
            feature_dim_3d=config.get('feature_dim_3d', 256),
            feature_dim_2d=config.get('feature_dim_2d', 768),
            dropout=config.get('dropout', 0.1),
        ).to(device)
        
        # 加载权重
        self.model.load_state_dict(ckpt['model'])
        self.model.eval()
        
        logger.info(f"  ✓ 模型加载完成")
    
    @torch.no_grad()
    def predict(self, scene_data: Dict, chunk_size: int = 100000) -> np.ndarray:
        """
        流式分块预测场景的语义标签
        
        Args:
            scene_data: 从ZarrSceneLoader.load_scene()返回的字典
            chunk_size: 每次推理的最大点数（默认100k点）
        
        Returns:
            predictions: [N] 预测的类别ID数组
        """
        points = scene_data['points']
        colors = scene_data['colors']
        normals = scene_data['normals']
        voxel_size = scene_data['voxel_size']
        
        num_points = len(points)
        num_chunks = int(np.ceil(num_points / chunk_size))
        
        if num_chunks > 1:
            logger.info(f"🚀 大场景检测 ({num_points:,} 点)，使用分块推理")
            logger.info(f"  - 块大小: {chunk_size:,}")
            logger.info(f"  - 总块数: {num_chunks}")
        
        # 初始化预测结果
        predictions = np.full(num_points, -1, dtype=np.int32)
        
        # 流式分块推理
        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min(start_idx + chunk_size, num_points)
            
            if num_chunks > 1:
                logger.info(f"\n📦 推理块 {chunk_idx+1}/{num_chunks} (点 {start_idx:,}-{end_idx:,})...")
            
            # 提取当前块
            chunk_points = points[start_idx:end_idx]
            chunk_colors = colors[start_idx:end_idx] if colors is not None else None
            chunk_normals = normals[start_idx:end_idx]
            
            # 体素化当前块
            voxel_coords = np.floor(chunk_points / voxel_size).astype(np.int32)
            unique_coords, unique_indices = np.unique(voxel_coords, axis=0, return_index=True)
            
            # 只保留唯一体素
            points_unique = chunk_points[unique_indices]
            colors_unique = chunk_colors[unique_indices] if chunk_colors is not None else np.ones((len(unique_indices), 3), dtype=np.float32) * 0.5
            normals_unique = chunk_normals[unique_indices]
            
            # 构建特征 (RGB + 法向量 + 坐标)
            combined_features = np.hstack([colors_unique, normals_unique, points_unique])
            
            # 添加batch索引
            batch_indices = np.zeros((len(unique_indices), 1), dtype=np.int32)
            coords_with_batch = np.hstack([batch_indices, unique_coords])
            
            # 转换为Tensor
            coords = torch.from_numpy(coords_with_batch).int().to(self.device)
            features = torch.from_numpy(combined_features).float().to(self.device)
            
            # 创建SparseTensor
            x = ME.SparseTensor(features=features, coordinates=coords, device=self.device)
            
            # 推理当前块
            seg_logits, _ = self.model(x)
            pred_labels = seg_logits.features.argmax(dim=-1).cpu().numpy()
            
            # 映射回当前块的原始点云
            for i, coord in enumerate(voxel_coords):
                match_idx = np.where((unique_coords == coord).all(axis=1))[0]
                if len(match_idx) > 0:
                    predictions[start_idx + i] = pred_labels[match_idx[0]]
            
            if num_chunks > 1:
                logger.info(f"  ✅ 块 {chunk_idx+1} 完成 ({end_idx-start_idx:,} 点)")
            
            # 释放GPU内存
            del coords, features, x, seg_logits
            torch.cuda.empty_cache()
        
        logger.info(f"  ✓ 推理完成: {(predictions >= 0).sum():,}/{len(predictions):,} 个点")
        
        return predictions


# ============================================================================
# 可视化工具
# ============================================================================

class Visualizer:
    """点云可视化工具"""
    
    @staticmethod
    def visualize_open3d(points: np.ndarray, colors: np.ndarray, 
                         gt_labels: Optional[np.ndarray] = None,
                         pred_labels: Optional[np.ndarray] = None,
                         class_names: Optional[List[str]] = None):
        """Open3D交互式可视化"""
        
        if not HAS_OPEN3D:
            logger.error("Open3D未安装")
            return
        
        logger.info("启动Open3D可视化...")
        
        # 创建点云
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        if colors is not None:
            pcd.colors = o3d.utility.Vector3dVector(colors)
        
        # 可视化
        o3d.visualization.draw_geometries([pcd], window_name="Point Cloud")
    
    @staticmethod
    def visualize_plotly_by_objects(points: np.ndarray, 
                                     gt_labels: np.ndarray,
                                     pred_labels: np.ndarray,
                                     class_names: List[str],
                                     output_dir: Path,
                                     scene_id: str,
                                     max_points_per_object: int = 50000):
        """
        按物体/语义区域分组可视化（每个物体一个HTML）
        
        流式处理策略：
        1. 按GT标签聚类，找出每个语义类别的所有点
        2. 对每个类别，如果点数太多则空间分割（DBSCAN或网格）
        3. 每个物体/区域生成独立HTML
        4. 流式处理，逐个物体加载和释放内存
        
        Args:
            points: 点云坐标 [N, 3]
            gt_labels: Ground Truth标签 [N]
            pred_labels: 预测标签 [N]
            class_names: 类别名称列表
            output_dir: 输出目录
            scene_id: 场景ID
            max_points_per_object: 单个物体最大点数（超过则空间分割）
        """
        
        if not HAS_PLOTLY:
            logger.error("Plotly未安装")
            return
        
        num_points = len(points)
        num_classes = len(class_names)
        
        logger.info(f"🎯 按物体分组可视化...")
        logger.info(f"  - 总点数: {num_points:,}")
        logger.info(f"  - 类别数: {num_classes}")
        
        color_map = ColorPalette.get_semantic_colors(num_classes)
        
        # 颜色转换函数
        def labels_to_colors(labels):
            colors = np.zeros((len(labels), 3), dtype=np.float32)
            for i, label in enumerate(labels):
                if label >= 0 and label < num_classes:
                    colors[i] = np.array(color_map[label]) / 255.0
                else:
                    colors[i] = [0.5, 0.5, 0.5]
            return colors
        
        # 统计每个类别的点数
        unique_labels = np.unique(gt_labels[gt_labels >= 0])
        logger.info(f"  - 发现 {len(unique_labels)} 个语义类别")
        
        object_groups = []
        for label in unique_labels:
            mask = (gt_labels == label)
            num_pts = mask.sum()
            
            if num_pts == 0:
                continue
            
            class_name = class_names[label] if label < len(class_names) else f"class_{label}"
            
            # 如果单个类别点数太多，按空间分割
            if num_pts > max_points_per_object:
                # 使用简单的空间网格分割
                pts = points[mask]
                labels_gt = gt_labels[mask]
                labels_pred = pred_labels[mask]
                
                # 计算空间范围
                mins = pts.min(axis=0)
                maxs = pts.max(axis=0)
                ranges = maxs - mins
                
                # 估算需要多少个子区域
                num_subdivisions = int(np.ceil(num_pts / max_points_per_object))
                grid_size = int(np.ceil(num_subdivisions ** (1/3)))  # 3D网格
                
                logger.info(f"  - {class_name}: {num_pts:,} 点 → 空间分割为 {grid_size}³ 网格")
                
                # 网格划分
                grid_indices = np.floor((pts - mins) / (ranges / grid_size + 1e-6)).astype(int)
                grid_indices = np.clip(grid_indices, 0, grid_size - 1)
                grid_keys = [tuple(idx) for idx in grid_indices]
                
                # 按网格分组
                from collections import defaultdict
                grid_groups = defaultdict(list)
                for i, key in enumerate(grid_keys):
                    grid_groups[key].append(i)
                
                # 为每个非空网格创建object group
                for sub_idx, (grid_key, indices) in enumerate(grid_groups.items()):
                    if len(indices) < 100:  # 过滤掉太小的碎片
                        continue
                    
                    object_groups.append({
                        'name': f"{class_name}_part{sub_idx+1}",
                        'label': label,
                        'indices': np.where(mask)[0][indices],
                        'num_points': len(indices)
                    })
            else:
                # 单个物体
                object_groups.append({
                    'name': class_name,
                    'label': label,
                    'indices': np.where(mask)[0],
                    'num_points': num_pts
                })
                logger.info(f"  - {class_name}: {num_pts:,} 点")
        
        logger.info(f"\n📦 总共分为 {len(object_groups)} 个物体/区域")
        
        # 生成索引HTML
        index_html_path = output_dir / f"{scene_id}_objects_index.html"
        index_html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{scene_id} - 物体分组可视化</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1 {{ color: #333; }}
                .object-list {{ list-style: none; padding: 0; }}
                .object-item {{ 
                    margin: 10px 0; 
                    padding: 15px; 
                    background: #f0f0f0; 
                    border-radius: 5px;
                }}
                .object-item a {{ 
                    text-decoration: none; 
                    color: #007bff; 
                    font-size: 18px;
                    font-weight: bold;
                }}
                .object-item a:hover {{ text-decoration: underline; }}
                .info {{ color: #666; margin-top: 5px; }}
                .category {{ display: inline-block; padding: 3px 10px; 
                           background: #007bff; color: white; 
                           border-radius: 3px; font-size: 12px; }}
            </style>
        </head>
        <body>
            <h1>🎯 {scene_id} - 按物体分组可视化</h1>
            <p><strong>总点数:</strong> {num_points:,}</p>
            <p><strong>物体数:</strong> {len(object_groups)}</p>
            <hr>
            <ul class="object-list">
        """
        
        # 流式处理每个物体
        for obj_idx, obj_info in enumerate(object_groups):
            logger.info(f"\n🎨 处理物体 {obj_idx+1}/{len(object_groups)}: {obj_info['name']}...")
            
            # 提取物体数据
            indices = obj_info['indices']
            obj_points = points[indices].copy()
            obj_gt_labels = gt_labels[indices].copy()
            obj_pred_labels = pred_labels[indices].copy()
            
            # 生成颜色
            obj_gt_colors = labels_to_colors(obj_gt_labels)
            obj_pred_colors = labels_to_colors(obj_pred_labels)
            
            # 计算准确率
            valid_mask = (obj_gt_labels >= 0) & (obj_pred_labels >= 0)
            if valid_mask.sum() > 0:
                obj_accuracy = (obj_gt_labels[valid_mask] == obj_pred_labels[valid_mask]).mean()
            else:
                obj_accuracy = 0.0
            
            # 创建可视化
            fig = make_subplots(
                rows=1, cols=2,
                specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}]],
                subplot_titles=('Ground Truth', 'Prediction')
            )
            
            # GT点云
            fig.add_trace(
                go.Scatter3d(
                    x=obj_points[:, 0],
                    y=obj_points[:, 1],
                    z=obj_points[:, 2],
                    mode='markers',
                    marker=dict(
                        size=2,
                        color=['rgb({},{},{})'.format(int(c[0]*255), int(c[1]*255), int(c[2]*255)) 
                               for c in obj_gt_colors],
                    ),
                    text=[f"{class_names[l]}" if 0 <= l < num_classes else "unknown" for l in obj_gt_labels],
                    hovertemplate='<b>%{text}</b><br>X: %{x}<br>Y: %{y}<br>Z: %{z}<extra></extra>',
                ),
                row=1, col=1
            )
            
            # Prediction点云
            fig.add_trace(
                go.Scatter3d(
                    x=obj_points[:, 0],
                    y=obj_points[:, 1],
                    z=obj_points[:, 2],
                    mode='markers',
                    marker=dict(
                        size=2,
                        color=['rgb({},{},{})'.format(int(c[0]*255), int(c[1]*255), int(c[2]*255)) 
                               for c in obj_pred_colors],
                    ),
                    text=[f"{class_names[l]}" if 0 <= l < num_classes else "unknown" for l in obj_pred_labels],
                    hovertemplate='<b>%{text}</b><br>X: %{x}<br>Y: %{y}<br>Z: %{z}<extra></extra>',
                ),
                row=1, col=2
            )
            
            fig.update_layout(
                title=f"{obj_info['name']} - Accuracy: {obj_accuracy*100:.1f}%",
                height=800,
                showlegend=False,
            )
            
            # 保存HTML
            html_filename = f"{scene_id}_{obj_info['name'].replace(' ', '_')}.html"
            html_path = output_dir / html_filename
            fig.write_html(str(html_path))
            
            logger.info(f"  ✅ 已保存: {html_filename} ({obj_info['num_points']:,} 点, 准确率 {obj_accuracy*100:.1f}%)")
            
            # 添加到索引
            color_rgb = color_map[obj_info['label']]
            index_html_content += f"""
                <li class="object-item">
                    <a href="{html_filename}">🎯 {obj_info['name']}</a>
                    <div class="info">
                        <span class="category" style="background: rgb({color_rgb[0]},{color_rgb[1]},{color_rgb[2]});">
                            {class_names[obj_info['label']] if obj_info['label'] < len(class_names) else 'unknown'}
                        </span>
                        点数: {obj_info['num_points']:,} | 准确率: {obj_accuracy*100:.1f}%
                    </div>
                </li>
            """
            
            # 释放内存
            del obj_points, obj_gt_labels, obj_pred_labels
            del obj_gt_colors, obj_pred_colors, fig
        
        # 完成索引HTML
        index_html_content += """
            </ul>
            <hr>
            <p style="color: #666; font-size: 14px;">💡 提示: 点击上方链接查看每个物体的分割效果</p>
        </body>
        </html>
        """
        
        with open(index_html_path, 'w', encoding='utf-8') as f:
            f.write(index_html_content)
        
        logger.info(f"\n  ✅ 索引页面: {index_html_path}")
        logger.info(f"  ✅ 共生成 {len(object_groups)} 个物体HTML文件")
    
    @staticmethod
    def visualize_plotly(points: np.ndarray, 
                         gt_labels: np.ndarray,
                         pred_labels: np.ndarray,
                         class_names: List[str],
                         output_path: Path):
        """Plotly HTML可视化（GT vs Prediction）- 单文件版本"""
        
        if not HAS_PLOTLY:
            logger.error("Plotly未安装")
            return
        
        logger.info("生成Plotly HTML可视化（单文件）...")
        
        num_classes = len(class_names)
        color_map = ColorPalette.get_semantic_colors(num_classes)
        
        # 为GT和Pred生成颜色
        def labels_to_colors(labels):
            colors = np.zeros((len(labels), 3), dtype=np.float32)
            for i, label in enumerate(labels):
                if label >= 0 and label < num_classes:
                    colors[i] = np.array(color_map[label]) / 255.0
                else:
                    colors[i] = [0.5, 0.5, 0.5]  # 灰色表示无效
            return colors
        
        gt_colors = labels_to_colors(gt_labels)
        pred_colors = labels_to_colors(pred_labels)
        
        # 创建子图
        fig = make_subplots(
            rows=1, cols=2,
            specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}]],
            subplot_titles=('Ground Truth', 'Prediction')
        )
        
        # GT点云
        fig.add_trace(
            go.Scatter3d(
                x=points[:, 0],
                y=points[:, 1],
                z=points[:, 2],
                mode='markers',
                marker=dict(
                    size=1,
                    color=['rgb({},{},{})'.format(int(c[0]*255), int(c[1]*255), int(c[2]*255)) 
                           for c in gt_colors],
                ),
                text=[f"{class_names[l]}" if 0 <= l < num_classes else "unknown" for l in gt_labels],
                hovertemplate='<b>%{text}</b><br>X: %{x}<br>Y: %{y}<br>Z: %{z}<extra></extra>',
            ),
            row=1, col=1
        )
        
        # Prediction点云
        fig.add_trace(
            go.Scatter3d(
                x=points[:, 0],
                y=points[:, 1],
                z=points[:, 2],
                mode='markers',
                marker=dict(
                    size=1,
                    color=['rgb({},{},{})'.format(int(c[0]*255), int(c[1]*255), int(c[2]*255)) 
                           for c in pred_colors],
                ),
                text=[f"{class_names[l]}" if 0 <= l < num_classes else "unknown" for l in pred_labels],
                hovertemplate='<b>%{text}</b><br>X: %{x}<br>Y: %{y}<br>Z: %{z}<extra></extra>',
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            title="USDNet Semantic Segmentation Results",
            height=800,
            showlegend=False,
        )
        
        # 保存HTML
        fig.write_html(str(output_path))
        logger.info(f"  ✓ 已保存: {output_path}")


# ============================================================================
# 主程序
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="USDNet可视化程序")
    
    parser.add_argument('--checkpoint', type=str, required=True, help='模型checkpoint路径')
    parser.add_argument('--zarr_root', type=str, required=True, help='Zarr数据根目录')
    parser.add_argument('--scene_id', type=str, default=None, help='场景ID（不指定则处理第一个）')
    parser.add_argument('--output_dir', type=str, default='./visualizations', help='输出目录')
    parser.add_argument('--device', type=str, default='cuda:0', help='设备')
    parser.add_argument('--voxel_size', type=float, default=0.05, help='体素大小')
    parser.add_argument('--max_points', type=int, default=None, help='最大显示点数（None=不限制）')
    parser.add_argument('--infer_chunk_size', type=int, default=100000, help='推理分块大小（点数/块）')
    parser.add_argument('--max_points_per_object', type=int, default=50000, help='单个物体最大点数（超过则空间分割）')
    parser.add_argument('--use_open3d', action='store_true', help='使用Open3D可视化')
    
    args = parser.parse_args()
    
    logger.info("="*80)
    logger.info("USDNet 可视化程序")
    logger.info("="*80)
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载全局标签映射
    zarr_root = Path(args.zarr_root)
    mapping_file = zarr_root / "global_label_mapping.json"
    
    if not mapping_file.exists():
        logger.error(f"❌ 未找到全局标签映射: {mapping_file}")
        return
    
    with open(mapping_file, 'r') as f:
        label_data = json.load(f)
    
    class_names = label_data['class_names']
    num_classes = label_data['num_classes']
    
    logger.info(f"✓ 全局类别数: {num_classes}")
    
    # 找到Zarr文件
    zarr_files = sorted(zarr_root.glob("*_dino_patch_level.zarr"))
    
    if not zarr_files:
        logger.error(f"❌ 未找到Zarr文件: {zarr_root}")
        return
    
    # 选择场景
    if args.scene_id:
        zarr_path = zarr_root / f"{args.scene_id}_dino_patch_level.zarr"
        if not zarr_path.exists():
            logger.error(f"❌ 未找到场景: {args.scene_id}")
            return
    else:
        zarr_path = zarr_files[0]
        logger.info(f"使用第一个场景: {zarr_path.stem}")
    
    scene_id = zarr_path.stem.replace('_dino_patch_level', '')
    
    # 加载场景数据
    scene_data = ZarrSceneLoader.load_scene(zarr_path, voxel_size=args.voxel_size)
    
    points = scene_data['points']
    gt_labels = scene_data['semantic_labels']
    
    # 降采样（如果指定了max_points且点太多）
    if args.max_points is not None and len(points) > args.max_points:
        logger.info(f"降采样: {len(points):,} → {args.max_points:,}")
        idx = np.random.choice(len(points), args.max_points, replace=False)
        points = points[idx]
        gt_labels = gt_labels[idx] if gt_labels is not None else None
        
        # 更新scene_data
        scene_data['points'] = points
        if scene_data['colors'] is not None:
            scene_data['colors'] = scene_data['colors'][idx]
        if scene_data['normals'] is not None:
            scene_data['normals'] = scene_data['normals'][idx]
    else:
        logger.info(f"使用全部点: {len(points):,}")
    
    # 加载模型并推理
    engine = InferenceEngine(args.checkpoint, device=args.device)
    
    logger.info(f"\n{'='*80}")
    logger.info("🚀 推理场景语义标签")
    logger.info(f"{'='*80}")
    
    pred_labels = engine.predict(scene_data, chunk_size=args.infer_chunk_size)
    
    # 按物体分组可视化
    Visualizer.visualize_plotly_by_objects(
        points=points,
        gt_labels=gt_labels,
        pred_labels=pred_labels,
        class_names=class_names,
        output_dir=output_dir,
        scene_id=scene_id,
        max_points_per_object=args.max_points_per_object
    )
    
    logger.info("\n✅ 完成！")


if __name__ == "__main__":
    main()
