import os
from pathlib import Path
from typing import OrderedDict

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset


class MultiModalMRIDataset(Dataset):
    def __init__(self, root_dir, modalities, slice_axis=2,
                 transform=None):
        """
        参数:
            root_dir (str): 包含样本子目录的根目录
            modalities (list): 模态名称列表 (对应文件名)
            slice_axis (int): 切片轴向 (0:Sagittal, 1:Coronal, 2:Axial)
            transform (callable): 可选的图像增强变换
        """
        self.root_dir = root_dir
        self.modalities = modalities
        self.slice_axis = slice_axis
        self.transform = transform
        self.volume_data = {}

        # 多线程加载所有样本数据
        from concurrent.futures import ThreadPoolExecutor
        
        # 收集有效样本路径
        sample_paths = [os.path.join(root_dir, d) 
                      for d in os.listdir(root_dir) 
                      if os.path.isdir(os.path.join(root_dir, d))]
                      
        # 并行加载数据
        with ThreadPoolExecutor() as executor:
            futures = []
            for path in sample_paths:
                futures.append(executor.submit(self._load_volume, path))
            
            for future, path in zip(futures, sample_paths):
                try:
                    self.volume_data[path] = future.result()
                except Exception as e:
                    print(f"加载失败 {path}: {str(e)}")

        # 预计算切片索引
        self.slice_indices = []
        for sample_path in sample_paths:
            if sample_path not in self.volume_data:
                continue
            num_slices = self.volume_data[sample_path].shape[self.slice_axis+1]
            valid_slices = range(min(num_slices//4, 27), max(num_slices//4*3, 127))
            self.slice_indices.extend([
                (sample_path, slice_idx) 
                for slice_idx in valid_slices
            ])

    def __len__(self):
        return len(self.slice_indices)

    def _load_volume(self, sample_path):
        """并行加载单个样本的所有模态数据"""
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
        sample_path, slice_idx = self.slice_indices[idx]
        volume = self.volume_data[sample_path]

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
                 selected_contrasts=None, generated_contrasts=None):
        # using axis=2 to extract slices along the third dimension (depth)
        super().__init__(root_dir, modalities, 2, transform)
        self.selected_contrasts = selected_contrasts
        self.generated_contrasts = generated_contrasts

    def collate_fn(self, x):
        # 将batch中的数据拼接成一个Tensor
        batch = [item[0] for item in x if item is not None]
        batch = torch.stack(batch, dim=0)
        idx = [(item[1], item[2]) for item in x if item is not None]
        idx = torch.tensor(idx)

        # crop to 192x160
        batch = batch[:, :, 40:-40, -226:-34, :].rot90(k=1, dims=(2, 3))

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
