import os
from typing import OrderedDict

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class MultiModalMRIDataset(Dataset):
    def __init__(self, root_dir, modalities, slice_axis=2,
                 transform=None, use_cache=True, cache_size=64):
        """
        参数:
            root_dir (str): 包含样本子目录的根目录
            modalities (list): 模态名称列表 (对应文件名) ,他
            slice_axis (int): 切片轴向 (0:Sagittal, 1:Coronal, 2:Axial)
            transform (callable): 可选的图像增强变换
            use_cache (bool): 是否缓存加载的3D数据
        """
        self.root_dir = root_dir
        self.modalities = modalities
        self.slice_axis = slice_axis
        self.transform = transform
        self.use_cache = use_cache
        self.cache = OrderedDict()
        self.cache_size = cache_size

        # 收集有效样本
        self.samples = []
        for sample_dir in os.listdir(root_dir):
            sample_path = os.path.join(root_dir, sample_dir)
            if os.path.isdir(sample_path):
                valid = True
                shapes = []

                # 检查模态完整性和形状一致性
                for modality in modalities:
                    modality = os.path.basename(
                        sample_path) + f"_{modality}"
                    file_path = os.path.join(sample_path, f"{modality}.nii.gz")
                    if not os.path.exists(file_path):
                        valid = False
                        break

                    # 快速加载头部信息检查形状
                    img = nib.nifti1.load(file_path)
                    shapes.append(img.shape)

                if valid and len(set(shapes)) == 1:
                    self.samples.append(sample_path)

        # 预计算切片索引映射
        self.slice_indices = []
        for sample_idx, sample_path in enumerate(self.samples):
            # 使用第一个模态获取切片数量
            basename = os.path.basename(sample_path)
            sample_shapes = nib.nifti1.load(os.path.join(
                sample_path, f"{basename}_{modalities[0]}.nii.gz")).shape
            num_slices = sample_shapes[slice_axis]
            self.slice_indices.extend(
                [(sample_idx, slice_idx) for slice_idx in range(num_slices)])

    def __len__(self):
        return len(self.slice_indices)

    def _load_volume(self, sample_path):
        """加载单个样本的所有模态数据"""
        volumes = []
        for modality in self.modalities:
            basename = os.path.basename(sample_path)
            img = nib.nifti1.load(os.path.join(
                sample_path, f"{basename}_{modality}.nii.gz"))
            img = nib.funcs.as_closest_canonical(img)  # 标准化方向
            data = img.get_fdata(dtype=np.float32)
            data = self._normalize(data)
            volumes.append(data)
        return np.stack(volumes, axis=0)  # (M, H, W, D)

    def _normalize(self, volume):
        """3D Z-score归一化"""
        vmin = np.min(volume)
        vmax = np.max(volume)
        return (volume - vmin) / (vmax - vmin) if vmax != 0 else volume

    def __getitem__(self, idx):
        sample_idx, slice_idx = self.slice_indices[idx]
        sample_path = self.samples[sample_idx]

        # 从缓存或磁盘加载数据
        if self.use_cache and sample_idx in self.cache:
            self.cache.move_to_end(sample_idx)
            volume = self.cache[sample_idx]
        else:
            volume = self._load_volume(sample_path)
            if self.use_cache:
                if len(self.cache) > self.cache_size:
                    self.cache.popitem(last=False)  # 移除最旧的缓存项
                self.cache[sample_idx] = volume

        # 提取多模态切片 (M, H, W, 1)
        slice_data = volume[:, slice_idx, :, :] if self.slice_axis == 0 else \
            volume[:, :, slice_idx, :] if self.slice_axis == 1 else \
            volume[:, :, :, slice_idx]

        # 转换为Tensor并应用变换
        slice_data = torch.from_numpy(slice_data.copy()).float()
        slice_data = slice_data.unsqueeze(dim=-1)
        if self.transform:
            slice_data = self.transform(slice_data)

        return slice_data


class MultiModalGenerationDataset(MultiModalMRIDataset):
    def __init__(self, root_dir, modalities, transform=None, use_cache=True,
                 selected_contrasts=None, generated_contrasts=None):
        # using axis=2 to extract slices along the third dimension (depth)
        super().__init__(root_dir, modalities, 2, transform, use_cache)
        self.selected_contrasts = selected_contrasts
        self.generated_contrasts = generated_contrasts

    def collate_fn(self, batch):
        # 将batch中的数据拼接成一个Tensor
        batch = [item for item in batch if item is not None]
        batch = torch.stack(batch, dim=0)

        # crop to 192x160
        batch = batch[:, :, 34:226, 40:-40, :]

        num_contrasts = len(self.modalities)

        if self.selected_contrasts is None:
            selected_contrasts = np.random.choice(
                num_contrasts, np.random.randint(1, num_contrasts), replace=False)
            selected_contrasts = sorted(selected_contrasts)
        else:
            selected_contrasts = self.selected_contrasts

        if self.generated_contrasts is None:
            generated_contrasts = np.random.choice(
                num_contrasts, np.random.randint(1, num_contrasts), replace=False)
            generated_contrasts = sorted(generated_contrasts)
        else:
            generated_contrasts = self.generated_contrasts

        return {
            'x': batch[:, selected_contrasts],
            'y': batch[:, generated_contrasts],
            'selected_contrasts': selected_contrasts,
            'generated_contrasts': generated_contrasts,
        }
