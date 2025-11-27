"""
================================================================================
ScanNet++ DITR 数据蒸馏程序 - 终极完整版
================================================================================

✅ 完全重新设计，确保正确性：
1. 第一步：扫描所有场景，构建全局标签映射（连续ID: 0, 1, 2, ...）
2. 第二步：提取DINO特征 + 计算法向量 + 映射语义标签到全局ID
3. 第三步：保存为Zarr格式

文件名: batch_ditr_complete_final.py
================================================================================
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import logging
from dataclasses import dataclass
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import time
from datetime import datetime
from collections import Counter

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from plyfile import PlyData
import zarr
from numcodecs import Blosc

# Open3D
try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False
    print("⚠️ Open3D未安装")

# 日志配置
log_file = Path("ditr_complete_final.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ==================== 增量式全局标签映射管理器 ====================

class GlobalLabelMappingManager:
    """
    增量式全局标签映射管理器
    
    每处理一个场景时，扫描其标签并更新全局映射
    无需预先扫描所有场景
    """
    
    def __init__(self, data_root: str, output_root: str):
        self.data_root = Path(data_root)
        self.output_root = Path(output_root)
        self.mapping_file = self.output_root / "global_label_mapping.json"
        
        self.global_label_map: Dict[str, int] = {}
        self.num_classes: int = 0
        self.class_names: List[str] = []
        
        # 如果已存在映射，加载它
        if self.mapping_file.exists():
            self._load_existing_mapping()
            logger.info(f"✓ 加载现有全局标签映射: {self.num_classes} 个类别")
        else:
            logger.info("✓ 创建新的增量式全局标签映射")
    
    def _load_existing_mapping(self):
        """加载已存在的映射"""
        try:
            with open(self.mapping_file, 'r') as f:
                data = json.load(f)
            
            self.num_classes = data['num_classes']
            self.global_label_map = data['mapping']
            self.class_names = data['class_names']
        except Exception as e:
            logger.warning(f"加载映射失败: {e}，将创建新映射")
            self.global_label_map = {}
            self.num_classes = 0
            self.class_names = []
    
    def update_from_scene(self, scene_path: Path) -> Dict[int, str]:
        """
        从场景的annotations更新全局映射
        
        Returns:
            local_to_name: {局部segment_id: 类别名称} 的映射
        """
        anno_file = scene_path / "scans" / "segments_anno.json"
        
        if not anno_file.exists():
            return {}
        
        try:
            with open(anno_file, 'r') as f:
                anno_data = json.load(f)
            
            local_to_name = {}
            new_labels = []
            
            for group in anno_data.get('segGroups', []):
                label_name = group.get('label', 'unknown')
                segments = group.get('segments', [])
                
                # 过滤无效类别
                if label_name and label_name not in ['unknown', 'REMOVE', 'SPLIT']:
                    # 添加到全局映射（如果是新类别）
                    if label_name not in self.global_label_map:
                        self.global_label_map[label_name] = self.num_classes
                        self.class_names.append(label_name)
                        self.num_classes += 1
                        new_labels.append(label_name)
                    
                    # 记录局部segment到类别名的映射
                    for seg_id in segments:
                        local_to_name[seg_id] = label_name
            
            # 如果有新类别，保存更新后的映射
            if new_labels:
                logger.info(f"  + 新增 {len(new_labels)} 个类别")
                self._save_mapping()
            
            return local_to_name
        
        except Exception as e:
            logger.error(f"更新映射失败: {e}")
            return {}
    
    def _save_mapping(self):
        """保存映射到文件"""
        self.output_root.mkdir(parents=True, exist_ok=True)
        
        mapping_data = {
            'num_classes': self.num_classes,
            'mapping': self.global_label_map,
            'class_names': self.class_names,
            'created_at': datetime.now().isoformat(),
            'version': '4.0_incremental'
        }
        
        with open(self.mapping_file, 'w') as f:
            json.dump(mapping_data, f, indent=2)
        
        logger.info(f"  ✓ 全局映射已更新: {self.num_classes} 个类别")
    
    def get_global_mapping(self) -> Dict[str, int]:
        return self.global_label_map
    
    def get_num_classes(self) -> int:
        return self.num_classes


# ==================== 法向量估计器 ====================

class NormalEstimator:
    @staticmethod
    def compute_normals_open3d(points: np.ndarray, k: int = 20) -> np.ndarray:
        """
        使用sklearn KNN + 多线程加速计算法向量
        回退到Open3D如果失败
        """
        try:
            from sklearn.neighbors import NearestNeighbors
            
            # sklearn KNN with all CPU cores
            k_neighbors = min(k + 1, len(points))  # +1因为包含自己
            nbrs = NearestNeighbors(n_neighbors=k_neighbors, algorithm='auto', n_jobs=-1).fit(points)
            distances, indices = nbrs.kneighbors(points)
            
            # 批量协方差矩阵计算
            normals = np.zeros_like(points, dtype=np.float32)
            
            for i in range(len(points)):
                neighbors = points[indices[i]]
                centroid = neighbors.mean(axis=0)
                centered = neighbors - centroid
                
                # 协方差矩阵
                cov = centered.T @ centered
                
                # 特征值分解 - 最小特征值对应的特征向量是法向量
                eigenvalues, eigenvectors = np.linalg.eigh(cov)
                normal = eigenvectors[:, 0]  # 最小特征值对应的向量
                
                # 归一化
                norm = np.linalg.norm(normal)
                if norm > 0:
                    normals[i] = normal / norm
                else:
                    normals[i] = [0, 0, 1]
            
            # 简单方向一致性 - 让法向量指向+Z方向
            flip_mask = normals[:, 2] < 0
            normals[flip_mask] *= -1
            
            return normals
            
        except Exception as e:
            logger.warning(f"sklearn法向量计算失败 ({e}), 回退到Open3D")
            
            # 回退到Open3D
            if not HAS_OPEN3D:
                return NormalEstimator.compute_normals_simple(points)
            
            try:
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
                pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=k))
                pcd.orient_normals_consistent_tangent_plane(k=k)
                
                normals = np.asarray(pcd.normals, dtype=np.float32)
                norms = np.linalg.norm(normals, axis=1, keepdims=True)
                norms = np.where(norms > 0, norms, 1.0)
                normals = normals / norms
                
                return normals
            except:
                return NormalEstimator.compute_normals_simple(points)
    
    @staticmethod
    def compute_normals_simple(points: np.ndarray) -> np.ndarray:
        normals = np.zeros_like(points, dtype=np.float32)
        normals[:, 2] = 1.0
        return normals


# ==================== 相机参数 ====================

@dataclass
class CameraParams:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int


# ==================== DINOv3特征提取器 ====================

class DINOv3Extractor:
    def __init__(self, device: str = 'cuda:0'):
        self.device = device
        self.feature_dim = 768
        
        logger.info(f"加载 DINOv3-ViT-B/16...")
        
        import torch.nn as nn
        
        try:
            import timm
            self.model = timm.create_model(
                'vit_base_patch16_224',
                pretrained=False,
                num_classes=0,
                img_size=512,
                patch_size=16
            ).to(device)
        except:
            self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14', pretrained=False)
            self.model.patch_embed.proj = nn.Conv2d(3, 768, kernel_size=16, stride=16)
            self.model = self.model.to(device)
        
        checkpoint_path = '/workspace/project/utils/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth'
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint.get('model', checkpoint.get('state_dict', checkpoint))
        self.model.load_state_dict(state_dict, strict=False)
        self.model.eval()
        
        if 'cuda' in device:
            torch.backends.cudnn.benchmark = True
        
        logger.info(f"✓ DINOv3加载完成")
    
    def extract_patch_tokens(self, image_path: str) -> Optional[np.ndarray]:
        try:
            image = Image.open(image_path).convert('RGB')
            
            transform = transforms.Compose([
                transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ])
            
            image_tensor = transform(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                output = self.model.forward_features(image_tensor)
                
                if isinstance(output, dict):
                    patch_tokens = output.get('x_norm_patchtokens', output.get('x', output)[:, 1:, :])
                else:
                    patch_tokens = output[:, 1:, :]
                
                _, num_patches, feat_dim = patch_tokens.shape
                num_patches_side = int(np.sqrt(num_patches))
                
                patch_grid = patch_tokens.reshape(1, num_patches_side, num_patches_side, feat_dim)
                patch_grid_norm = F.normalize(patch_grid, dim=-1, p=2)
                
                result = patch_grid_norm.squeeze(0).cpu().numpy().astype(np.float32)
                
                del image_tensor, output, patch_tokens
                torch.cuda.empty_cache()
                
                return result
        except:
            return None


# ==================== Patch坐标映射器 ====================

class PatchCoordinateMapper:
    PATCH_SIZE = 16
    IMAGE_SIZE = 512
    NUM_PATCHES = 32
    
    @staticmethod
    def get_patch_coordinates(projected_2d, valid_mask):
        patch_u = np.floor(projected_2d[:, 0] / 16).astype(np.int32)
        patch_v = np.floor(projected_2d[:, 1] / 16).astype(np.int32)
        
        valid_u = (patch_u >= 0) & (patch_u < 32)
        valid_v = (patch_v >= 0) & (patch_v < 32)
        valid_patch = valid_u & valid_v & valid_mask
        
        patch_u = np.clip(patch_u, 0, 31)
        patch_v = np.clip(patch_v, 0, 31)
        
        return patch_u, patch_v, valid_patch


# ==================== GPU点投影器 ====================

class GPUPointProjector:
    def __init__(self, device: str = 'cuda:0'):
        self.device = device
    
    def project_points_batch_gpu(self, points, T_w2c, K):
        points_torch = torch.from_numpy(points).to(self.device).float()
        T_w2c_torch = torch.from_numpy(T_w2c).to(self.device).float()
        
        ones = torch.ones((points_torch.shape[0], 1), device=self.device)
        points_homo = torch.cat([points_torch, ones], dim=1)
        
        points_cam = torch.matmul(points_homo, T_w2c_torch.T)[:, :3]
        valid_z = points_cam[:, 2] > 0.1
        
        u = K.fx * points_cam[:, 0] / points_cam[:, 2] + K.cx
        v = K.fy * points_cam[:, 1] / points_cam[:, 2] + K.cy
        
        valid_u = (u >= 0) & (u < K.width)
        valid_v = (v >= 0) & (v < K.height)
        valid_mask = valid_z & valid_u & valid_v
        
        projected_2d = torch.stack([u, v], dim=1).cpu().numpy()
        return projected_2d, valid_mask.cpu().numpy()


# ==================== ✅ 语义标签加载器（使用局部到全局映射）====================

class SemanticLabelsLoader:
    @staticmethod
    def load_and_map_labels(scene_path: Path, local_to_name: Dict[int, str], 
                           global_label_map: Dict[str, int]) -> np.ndarray:
        """
        加载语义标签并映射到全局ID
        
        三步映射：
        1. 读取每个点的segment_id (segments.json)
        2. 使用local_to_name查找segment_id对应的类别名称
        3. 将类别名称映射到全局ID (global_label_map)
        
        Args:
            scene_path: 场景路径
            local_to_name: {局部segment_id: 类别名称} 由update_from_scene生成
            global_label_map: {类别名称: 全局ID}
        
        Returns:
            semantic_labels: [N] 全局ID数组，范围[0, num_classes-1]或-1
        """
        
        scans_path = scene_path / "scans"
        
        # Step 1: 读取segment indices
        segments_file = scans_path / "segments.json"
        if not segments_file.exists():
            return None
        
        with open(segments_file, 'r') as f:
            seg_data = json.load(f)
        
        seg_indices = np.array(seg_data['segIndices'], dtype=np.int32)
        
        # Step 2 & 3: 生成全局ID标签
        num_points = len(seg_indices)
        semantic_labels = np.full(num_points, -1, dtype=np.int32)
        
        for i in range(num_points):
            seg_id = seg_indices[i]
            
            # 查找类别名称
            if seg_id in local_to_name:
                class_name = local_to_name[seg_id]
                
                # 映射到全局ID
                if class_name in global_label_map:
                    global_id = global_label_map[class_name]
                    semantic_labels[i] = global_id
        
        return semantic_labels


# ==================== Zarr存储 ====================

class ZarrStorage:
    def __init__(self, output_root: str):
        self.output_root = Path(output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)
    
    def save_zarr(self, mapping: Dict, scene_id: str):
        output_file = self.output_root / f"{scene_id}_dino_patch_level.zarr"
        
        try:
            if output_file.exists():
                import shutil
                shutil.rmtree(output_file)
            
            root = zarr.open(str(output_file), mode='w')
            compressor = Blosc(cname='zstd', clevel=5, shuffle=Blosc.SHUFFLE)
            
            # 元数据
            meta = root.create_group('metadata')
            for key in ['scene_id', 'num_points', 'num_valid', 'num_valid_images', 
                       'dino_feature_dim', 'dino_model', 'feature_extraction_mode',
                       'patch_grid_size', 'patch_size', 'fusion_mode', 'data_source', 
                       'has_semantics', 'global_num_classes']:
                if key in mapping:
                    meta.attrs[key] = mapping[key]
            
            # 数组
            root.create_dataset('points', data=mapping['points'], chunks=(10000, 3), compressor=compressor, dtype=np.float32)
            
            if mapping.get('colors') is not None:
                root.create_dataset('colors', data=mapping['colors'], chunks=(10000, 3), compressor=compressor, dtype=np.float32)
            
            if mapping.get('normals') is not None:
                root.create_dataset('normals', data=mapping['normals'], chunks=(10000, 3), compressor=compressor, dtype=np.float32)
            
            root.create_dataset('dino_features', data=mapping['dino_features'], chunks=(10000, 768), compressor=compressor, dtype=np.float32)
            root.create_dataset('valid_mask', data=mapping['valid_mask'], chunks=(10000,), compressor=compressor, dtype=np.bool_)
            root.create_dataset('point_view_count', data=mapping['point_view_count'], chunks=(10000,), compressor=compressor, dtype=np.int32)
            root.create_dataset('semantic_labels', data=mapping['semantic_labels'], chunks=(10000,), compressor=compressor, dtype=np.int32)
            
            # 标签映射
            if mapping.get('label_mapping'):
                label_group = root.create_group('label_mapping')
                for name, idx in mapping['label_mapping'].items():
                    label_group.attrs[name] = int(idx)
            
            size_mb = output_file.stat().st_size / (1024**2)
            logger.info(f"  ✓ 保存完成 ({size_mb:.1f}MB)")
            
            return True
        except Exception as e:
            logger.error(f"保存失败: {e}")
            return False


# ==================== 主映射器 ====================

class DITRMapper:
    def __init__(self, data_root: str, output_root: str, device: str = 'cuda:0', num_threads: int = 4):
        self.data_root = Path(data_root)
        self.output_root = Path(output_root)
        
        self.device = device
        self.num_threads = num_threads
        
        # 初始化组件
        self.global_mapper = GlobalLabelMappingManager(str(data_root), str(output_root))
        self.dino_extractor = DINOv3Extractor(device)
        self.gpu_projector = GPUPointProjector(device)
        self.zarr_storage = ZarrStorage(str(output_root))
    
    def initialize(self):
        """初始化：准备全局标签映射管理器"""
        logger.info("="*80)
        logger.info("初始化 - 增量式全局标签映射")
        logger.info("="*80)
        
        if self.global_mapper.mapping_file.exists():
            logger.info("✓ 已加载现有全局映射")
        else:
            logger.info("✓ 将在处理场景时增量构建全局映射")
        
        return True
    
    def load_point_cloud(self, ply_file: str):
        try:
            ply_data = PlyData.read(ply_file)
            vertex = ply_data['vertex']
            
            points = np.column_stack([vertex['x'], vertex['y'], vertex['z']]).astype(np.float32)
            
            colors = None
            try:
                if 'red' in vertex.data.dtype.names:
                    colors = np.column_stack([vertex['red'], vertex['green'], vertex['blue']]).astype(np.float32)
                    if colors.max() > 1.0:
                        colors = colors / 255.0
            except:
                pass
            
            return points, colors
        except Exception as e:
            logger.error(f"点云加载错误: {e}")
            return None, None
    
    def load_transforms(self, transforms_json: str):
        try:
            with open(transforms_json, 'r') as f:
                data = json.load(f)
            
            camera = CameraParams(
                fx=float(data['fl_x']),
                fy=float(data['fl_y']),
                cx=float(data.get('cx', data.get('w', 512) / 2)),
                cy=float(data.get('cy', data.get('h', 512) / 2)),
                width=int(data.get('w', 1752)),
                height=int(data.get('h', 1168))
            )
            
            extrinsics = {}
            for frame in data['frames']:
                image_name = Path(frame['file_path']).name
                c2w = np.array(frame['transform_matrix'], dtype=np.float32)
                if c2w.shape == (3, 4):
                    c2w_4x4 = np.eye(4, dtype=np.float32)
                    c2w_4x4[:3, :] = c2w
                    c2w = c2w_4x4
                w2c = np.linalg.inv(c2w)
                extrinsics[image_name] = w2c
            
            return extrinsics, camera
        except Exception as e:
            logger.error(f"相机加载错误: {e}")
            return {}, None
    
    def build_scene(self, scene_id: str):
        """处理单个场景"""
        
        scene_path = self.data_root / scene_id
        data_path = scene_path / "dslr"
        
        if not data_path.exists():
            return None
        
        logger.info(f"\n{'='*70}")
        logger.info(f"场景: {scene_id}")
        logger.info(f"{'='*70}")
        
        # 加载点云
        ply_file = scene_path / "scans" / "mesh_aligned_0.05.ply"
        points, colors = self.load_point_cloud(str(ply_file))
        if points is None:
            return None
        
        num_points = len(points)
        logger.info(f"  点云: {num_points:,}个点")
        
        # 计算法向量
        normals = NormalEstimator.compute_normals_open3d(points, k=20)
        logger.info(f"  ✓ 法向量计算完成")
        
        # ✅ 增量式更新全局标签映射
        local_to_name = self.global_mapper.update_from_scene(scene_path)
        
        # ✅ 加载并映射语义标签到全局ID
        global_label_map = self.global_mapper.get_global_mapping()
        semantic_labels = SemanticLabelsLoader.load_and_map_labels(
            scene_path, local_to_name, global_label_map
        )
        
        if semantic_labels is None:
            semantic_labels = np.full(num_points, -1, dtype=np.int32)
        
        # ✅ 验证标签范围
        valid_labels = semantic_labels[semantic_labels >= 0]
        if len(valid_labels) > 0:
            logger.info(f"  ✓ 语义标签: {len(valid_labels):,}个有效点")
            logger.info(f"    标签范围: [{valid_labels.min()}, {valid_labels.max()}]")
            logger.info(f"    期望范围: [0, {self.global_mapper.get_num_classes()-1}]")
            
            if valid_labels.max() >= self.global_mapper.get_num_classes():
                logger.error(f"  ❌ 标签ID超出范围！")
                return None
        
        # 加载相机
        transforms_json = data_path / "nerfstudio" / "transforms_undistorted.json"
        extrinsics, camera = self.load_transforms(str(transforms_json))
        if not extrinsics:
            return None
        
        # 提取DINO特征
        image_dir = data_path / "resized_undistorted_images"
        logger.info(f"  提取DINO特征...")
        
        dino_cache = {}
        for img_name in tqdm(extrinsics.keys(), desc="    ", leave=False):
            img_path = image_dir / img_name
            if not img_path.exists():
                candidates = list(image_dir.glob(f"{img_name.split('.')[0]}.*"))
                if candidates:
                    img_path = candidates[0]
                else:
                    continue
            
            feats = self.dino_extractor.extract_patch_tokens(str(img_path))
            if feats is not None:
                dino_cache[img_name] = feats
        
        logger.info(f"  ✓ 成功: {len(dino_cache)}张")
        
        # 多视角融合
        logger.info(f"  多视角融合...")
        
        dino_features_gpu = torch.zeros((num_points, 768), dtype=torch.float32, device=self.device)
        point_view_count_gpu = torch.zeros(num_points, dtype=torch.int32, device=self.device)
        lock = threading.Lock()
        
        def process_view(img_name):
            try:
                T_w2c = extrinsics[img_name]
                patch_grid = dino_cache[img_name]
                
                proj_2d, valid_proj = self.gpu_projector.project_points_batch_gpu(points, T_w2c, camera)
                
                proj_scaled = proj_2d.copy()
                proj_scaled[:, 0] *= 512 / camera.width
                proj_scaled[:, 1] *= 512 / camera.height
                
                valid = valid_proj & (proj_scaled[:, 0] >= 0) & (proj_scaled[:, 0] < 512) & \
                        (proj_scaled[:, 1] >= 0) & (proj_scaled[:, 1] < 512)
                
                patch_u, patch_v, valid_patch = PatchCoordinateMapper.get_patch_coordinates(proj_scaled, valid)
                
                valid_idx = np.where(valid_patch)[0]
                if len(valid_idx) == 0:
                    return
                
                valid_idx_gpu = torch.from_numpy(valid_idx).to(self.device).long()
                patch_u_gpu = torch.from_numpy(patch_u[valid_idx]).to(self.device).long()
                patch_v_gpu = torch.from_numpy(patch_v[valid_idx]).to(self.device).long()
                
                grid_gpu = torch.from_numpy(patch_grid).to(self.device)
                feats = grid_gpu[patch_v_gpu, patch_u_gpu, :]
                
                with lock:
                    counts = point_view_count_gpu[valid_idx_gpu]
                    first = (counts == 0)
                    update = (counts > 0)
                    
                    if first.any():
                        dino_features_gpu[valid_idx_gpu[first]] = feats[first]
                        point_view_count_gpu[valid_idx_gpu[first]] = 1
                    
                    if update.any():
                        idx_upd = valid_idx_gpu[update]
                        alpha = counts[update].float().unsqueeze(1)
                        dino_features_gpu[idx_upd] = (dino_features_gpu[idx_upd] * alpha + feats[update]) / (alpha + 1)
                        point_view_count_gpu[idx_upd] += 1
            except:
                pass
        
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            list(tqdm(executor.map(process_view, dino_cache.keys()), 
                     total=len(dino_cache), desc="    ", leave=False))
        
        dino_features = dino_features_gpu.cpu().numpy()
        point_view_count = point_view_count_gpu.cpu().numpy()
        valid_mask = point_view_count > 0
        
        del dino_features_gpu, point_view_count_gpu
        torch.cuda.empty_cache()
        
        logger.info(f"  ✓ 有效点: {np.sum(valid_mask):,}/{num_points:,}")
        
        # 构建映射
        mapping = {
            "scene_id": scene_id,
            "num_points": num_points,
            "num_valid": int(np.sum(valid_mask)),
            "num_valid_images": len(dino_cache),
            "points": points,
            "colors": colors,
            "normals": normals,
            "dino_features": dino_features,
            "valid_mask": valid_mask,
            "point_view_count": point_view_count,
            "dino_feature_dim": 768,
            "dino_model": "dinov3_vitb16",
            "feature_extraction_mode": "patch_level",
            "patch_grid_size": 32,
            "patch_size": 16,
            "fusion_mode": "average",
            "semantic_labels": semantic_labels,
            "label_mapping": global_label_map,
            "has_semantics": (valid_labels.size > 0 if len(valid_labels) > 0 else False),
            "data_source": "dslr",
            "global_num_classes": self.global_mapper.get_num_classes(),
        }
        
        return mapping
    
    def process_all(self, scene_ids: Optional[List[str]] = None):
        """处理所有场景"""
        
        if scene_ids is None:
            all_dirs = sorted([d for d in self.data_root.iterdir() if d.is_dir()])
            scene_ids = [d.name for d in all_dirs]
        
        # 检查已处理的
        processed = []
        for f in self.output_root.glob("*.zarr"):
            scene_id = f.stem.replace('_dino_patch_level', '')
            processed.append(scene_id)
        
        remaining = [s for s in scene_ids if s not in processed]
        
        logger.info(f"\n{'='*80}")
        logger.info(f"总场景: {len(scene_ids)} | 已处理: {len(processed)} | 待处理: {len(remaining)}")
        logger.info(f"{'='*80}")
        
        success = 0
        for i, scene_id in enumerate(remaining, 1):
            logger.info(f"\n[{i}/{len(remaining)}]")
            try:
                mapping = self.build_scene(scene_id)
                if mapping:
                    self.zarr_storage.save_zarr(mapping, scene_id)
                    success += 1
            except Exception as e:
                logger.error(f"错误: {e}")
        
        logger.info(f"\n✅ 完成！成功: {success}/{len(remaining)}")


# ==================== 主程序 ====================

def main():
    logger.info("\n" + "="*80)
    logger.info("ScanNet++ DITR 数据蒸馏 - 增量式全局标签映射版")
    logger.info("="*80)
    
    DATA_ROOT = "/mnt/nas/scannetpp_new/data"
    OUTPUT_ROOT = "/workspace/data_elements/newz"
    DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    mapper = DITRMapper(DATA_ROOT, OUTPUT_ROOT, DEVICE, num_threads=4)
    
    # 初始化（准备增量式映射管理器）
    if not mapper.initialize():
        logger.error("❌ 初始化失败！")
        return
    
    logger.info("\n✅ 使用增量式全局标签映射:")
    logger.info("  - 无需预先扫描所有场景")
    logger.info("  - 每处理一个场景就更新全局映射")
    logger.info("  - 新发现的类别自动添加到全局映射")
    
    # 处理所有场景
    mapper.process_all()
    
    logger.info(f"\n✅ 最终全局类别数: {mapper.global_mapper.get_num_classes()}")


if __name__ == "__main__":
    main()