import os
from pathlib import Path
from collections import OrderedDict

import nibabel as nib
import numpy as np
import torch
import torch.distributed
from torch.utils.data import Dataset


class MultiModalMRIDataset(Dataset):
    def __init__(self, root_dir, modalities, slice_axis=2,
                 transform=None, enable_mem_monitor=False,
                 max_cache_size=32):
        """
        参数:
            root_dir (str): 包含样本子目录的根目录
            modalities (list): 模态名称列表 (对应文件名)
            slice_axis (int): 切片轴向 (0:Sagittal, 1:Coronal, 2:Axial)
            transform (callable): 可选的图像增强变换
            enable_mem_monitor (bool): 是否启用内存监控
            max_cache_size (int): 保留参数（兼容旧配置）
        """
        self.root_dir = root_dir
        self.modalities = modalities
        self.slice_axis = slice_axis
        self.transform = transform

        # 收集有效样本路径
        self.sample_paths = [os.path.join(root_dir, d)
                             for d in os.listdir(root_dir)
                             if os.path.isdir(os.path.join(root_dir, d))]

        # DDP环境下共享文件列表
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            world_size = torch.distributed.get_world_size()
            rank = torch.distributed.get_rank()
            self.sample_paths = self.sample_paths[rank::world_size]

        # 多线程预加载所有数据
        from concurrent.futures import ThreadPoolExecutor
        self.data = {}
        with ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(self._load_volume, path): path
                for path in self.sample_paths
            }
            for future in futures:
                path = futures[future]
                try:
                    self.data[path] = future.result()
                except Exception as e:
                    print(f"Error loading {path}: {str(e)}")

        # 动态切片参数
        self.min_slice = 27
        self.max_slice = 127
        self.enable_mem_monitor = enable_mem_monitor
        self.mem_stats = {
            'peak_mem': 0,
            'avg_mem': 0,
            'count': 0
        }

    def __len__(self):
        # 动态计算总切片数（样本数 × 每个样本的有效切片数）
        return len(self.sample_paths) * (self.max_slice - self.min_slice)

    def _load_volume(self, sample_path):
        """加载单个样本的所有模态数据"""
        volumes = []
        basename = os.path.basename(sample_path)

        for modality in self.modalities:
            file_path = os.path.join(
                sample_path, f"{basename}_{modality}.nii.gz")
            img = nib.nifti1.load(file_path)
            img = nib.funcs.as_closest_canonical(img)
            data = img.get_fdata(dtype=np.float32)
            data = self._normalize(data)
            volumes.append(data)

        return np.stack(volumes, axis=0)  # (M, H, W, D)

    def _normalize(self, volume):
        """批量归一化处理"""
        vmin = np.min(volume)
        vmax = np.max(volume)
        volume = (volume - vmin) / (vmax - vmin) * 2 - 1 if vmax != 0 else -1
        return np.clip(volume, -1, 1)

    def __getitem__(self, idx):
        # 动态计算样本索引和切片索引
        sample_idx = idx // (self.max_slice - self.min_slice)
        slice_idx = idx % (self.max_slice - self.min_slice) + self.min_slice
        sample_path = self.sample_paths[sample_idx]

        if self.enable_mem_monitor:
            import psutil
            process = psutil.Process()
            mem = process.memory_info().rss / 1024 / 1024  # MB
            self.mem_stats['peak_mem'] = max(self.mem_stats['peak_mem'], mem)
            self.mem_stats['avg_mem'] = (
                self.mem_stats['avg_mem'] * self.mem_stats['count'] + mem
            ) / (self.mem_stats['count'] + 1)
            self.mem_stats['count'] += 1

        # 获取预加载的数据
        volume = self.data[sample_path]

        # 提取切片数据
        if self.slice_axis == 0:
            slice_data = volume[:, slice_idx, :, :]
        elif self.slice_axis == 1:
            slice_data = volume[:, :, slice_idx, :]
        else:
            slice_data = volume[:, :, :, slice_idx]

        # 转换为Tensor
        slice_data = torch.from_numpy(slice_data.copy()).float().unsqueeze(-1)
        if self.transform:
            slice_data = self.transform(slice_data)

        sample_id = int(Path(sample_path).stem.split('_')[1])
        return slice_data, sample_id, slice_idx


class MultiModalGenerationDataset(MultiModalMRIDataset):
    def __init__(self, root_dir, modalities, transform=None, use_cache=True,
                 selected_contrasts=None, generated_contrasts=None, enable_mem_monitor=False,
                 max_cache_size=32):
        """
        参数:
            root_dir (str): 包含样本子目录的根目录
            modalities (list): 模态名称列表 (对应文件名)
            transform (callable): 可选的图像增强变换
            use_cache (bool): 是否启用缓存
            selected_contrasts (list): 预选对比度列表
            generated_contrasts (list): 生成对比度列表
            enable_mem_monitor (bool): 是否启用内存监控
            max_cache_size (int): 最大缓存样本数量
        """
        super().__init__(root_dir, modalities, 2,
                         transform, enable_mem_monitor, max_cache_size)
        self.selected_contrasts = selected_contrasts
        self.generated_contrasts = generated_contrasts

    def collate_fn(self, x):
        # 将batch中的数据拼接成一个Tensor
        batch = [item[0] for item in x if item is not None]
        batch = torch.stack(batch, dim=0)
        idx = [(item[1], item[2]) for item in x if item is not None]
        idx = torch.tensor(idx)

        # GPU上的crop和rotation
        batch = batch[:, :, 40:-40, -226:-34, :]
        batch = torch.rot90(batch, k=1, dims=[2, 3])

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
            'idx': idx,
        }
