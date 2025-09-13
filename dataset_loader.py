import os
import requests
import zipfile
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import functional as F
import matplotlib.pyplot as plt
from typing import Tuple, Optional, List
import warnings

class PotsdamVaihingenDataset(Dataset):
    """
    PyTorch Dataset for ISPRS Potsdam and Vaihingen datasets.
    loads 256x256 patches for land cover classification.
    """
    
    
    # Class mappings for semantic segmentation
    POTSDAM_CLASSES = {
        0: 'Impervious surfaces',    # RGB: (255, 255, 255) - White
        1: 'Building',               # RGB: (0, 0, 255) - Blue  
        2: 'Low vegetation',         # RGB: (0, 255, 255) - Cyan
        3: 'Tree',                   # RGB: (0, 255, 0) - Green
        4: 'Car',                    # RGB: (255, 255, 0) - Yellow
        5: 'Clutter/background'      # RGB: (255, 0, 0) - Red
    }
    
    VAIHINGEN_CLASSES = {
        0: 'Impervious surfaces',    # RGB: (255, 255, 255) - White
        1: 'Building',               # RGB: (0, 0, 255) - Blue
        2: 'Low vegetation',         # RGB: (0, 255, 255) - Cyan  
        3: 'Tree',                   # RGB: (0, 255, 0) - Green
        4: 'Car',                    # RGB: (255, 255, 0) - Yellow
        5: 'Clutter/background'      # RGB: (255, 0, 0) - Red
    }
    
    # RGB to class mapping
    RGB_TO_CLASS = {
        (255, 255, 255): 0,  # Impervious surfaces - White
        (0, 0, 255): 1,      # Building - Blue
        (0, 255, 255): 2,    # Low vegetation - Cyan
        (0, 255, 0): 3,      # Tree - Green
        (255, 255, 0): 4,    # Car - Yellow
        (255, 0, 0): 5       # Clutter/background - Red
    }

    def __init__(self, 
                 root_dir: str,
                 dataset: str = 'potsdam',
                 split: str = 'train',
                 patch_size: int = 256,
                 stride: int = 128,
                 transform: Optional[transforms.Compose] = None,
                 target_transform: Optional[transforms.Compose] = None,
                 augment: bool = True):
        """
        Args:
            root_dir: Root directory to store datasets
            dataset: 'potsdam', 'vaihingen', or 'both'
            split: 'train', 'val', or 'test'
            patch_size: Size of patches to extract (default: 256)
            stride: Stride for patch extraction (default: 128)
            transform: Optional transform to be applied on images
            target_transform: Optional transform to be applied on labels
            augment: Whether to apply data augmentation (only for training)
        """
        self.root_dir = root_dir
        self.dataset = dataset.lower()
        self.split = split
        self.patch_size = patch_size
        self.stride = stride
        self.transform = transform
        self.target_transform = target_transform
        self.augment = augment and (split == 'train')  # Only augment training data
        
        # Create directory structure
        os.makedirs(root_dir, exist_ok=True)
        
        # Load image and label paths
        self.image_paths, self.label_paths = self._load_file_paths()
        
        # Extract patches
        self.patches = self._extract_patches()
        
        print(f"Loaded {len(self.patches)} patches from {self.dataset} dataset ({self.split} split)")

    def _load_file_paths(self) -> Tuple[List[str], List[str]]:
        """Load image and label file paths based on dataset and split."""
        image_paths = []
        label_paths = []
        
        if self.dataset in ['potsdam', 'both']:
            potsdam_imgs, potsdam_lbls = self._get_potsdam_paths()
            image_paths.extend(potsdam_imgs)
            label_paths.extend(potsdam_lbls)
            
        if self.dataset in ['vaihingen', 'both']:
            vaihingen_imgs, vaihingen_lbls = self._get_vaihingen_paths()
            image_paths.extend(vaihingen_imgs)
            label_paths.extend(vaihingen_lbls)
            
        return image_paths, label_paths

    def _get_potsdam_paths(self) -> Tuple[List[str], List[str]]:
        """Get Potsdam dataset file paths."""
        img_dir = os.path.join(self.root_dir, 'potsdam', 'images')
        lbl_dir = os.path.join(self.root_dir, 'potsdam', 'labels')
        
        if not os.path.exists(img_dir) or not os.path.exists(lbl_dir):
            warnings.warn(f"Potsdam dataset not found in {self.root_dir}/potsdam/")
            return [], []
        
        # Potsdam train/test split convention
        if self.split == 'train':
            # Areas 2_10, 2_13, 2_14, 3_10, 3_11, 3_12, 3_13, 4_10, 4_12, 4_13, 4_14, 5_10, 5_12, 5_13, 5_14, 5_15, 6_8, 6_9, 6_10, 6_11, 6_12, 6_13, 6_15, 7_7, 7_9, 7_11, 2_11, 2_12, 3_14, 4_11, 7_13
            area_ids = ['2_10', '2_13', '2_14', '3_10', '3_11', '3_12', '3_13', '4_10', '4_12', '4_13', '4_14', '5_10', '5_12', '5_13', '5_14', '5_15', '6_8', '6_9', '6_10', '6_11', '6_12', '6_13', '6_15', '7_7', '7_9', '7_11', '2_11', '2_12', '3_14', '4_11', '7_13']
        elif self.split == 'test':
            # Areas 4_15, 5_11, 6_7, 6_11
            area_ids = ['4_15', '5_11', '6_7', '6_11']
        else:
            # Areas 6_14, 7_8, 7_10, 7_12
            area_ids = ['6_14', '7_8', '7_10', '7_12']
            
        image_paths = []
        label_paths = []
        
        for area_id in area_ids:
            # RGB image
            img_path = os.path.join(img_dir, f'top_potsdam_{area_id}_RGB.tif')
            lbl_path = os.path.join(lbl_dir, f'top_potsdam_{area_id}_label.tif')
            
            if os.path.exists(img_path) and os.path.exists(lbl_path):
                image_paths.append(img_path)
                label_paths.append(lbl_path)
                
        return image_paths, label_paths

    def _get_vaihingen_paths(self) -> Tuple[List[str], List[str]]:
        """Get Vaihingen dataset file paths."""
        img_dir = os.path.join(self.root_dir, 'vaihingen', 'images')
        lbl_dir = os.path.join(self.root_dir, 'vaihingen', 'labels')
        
        if not os.path.exists(img_dir) or not os.path.exists(lbl_dir):
            warnings.warn(f"Vaihingen dataset not found in {self.root_dir}/vaihingen/")
            return [], []
        
        # Vaihingen areas: 1-38
        if self.split == 'train':
            area_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
        elif self.split == 'test':
            area_ids = [31, 32, 33, 34]
        else:  # val
            area_ids = [35, 36, 37, 38]
            
        image_paths = []
        label_paths = []
        
        for area_id in area_ids:
            img_path = os.path.join(img_dir, f'top_mosaic_09cm_area{area_id}.tif')
            lbl_path = os.path.join(lbl_dir, f'top_mosaic_09cm_area{area_id}.tif')
            
            if os.path.exists(img_path) and os.path.exists(lbl_path):
                image_paths.append(img_path)
                label_paths.append(lbl_path)
                
        return image_paths, label_paths

    def _extract_patches(self) -> List[Tuple[str, str, int, int]]:
        """Extract patch coordinates from images."""
        patches = []
        
        for img_path, lbl_path in zip(self.image_paths, self.label_paths):
            try:
                # Load image to get dimensions
                with Image.open(img_path) as img:
                    width, height = img.size
                
                # Extract patches with given stride
                for y in range(0, height - self.patch_size + 1, self.stride):
                    for x in range(0, width - self.patch_size + 1, self.stride):
                        patches.append((img_path, lbl_path, x, y))
                        
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                continue
                
        return patches

    def _rgb_to_class_mask(self, rgb_label: np.ndarray) -> np.ndarray:
        """Convert RGB label to class mask."""
        h, w = rgb_label.shape[:2]
        class_mask = np.zeros((h, w), dtype=np.uint8)
        
        for rgb, class_id in self.RGB_TO_CLASS.items():
            mask = np.all(rgb_label == rgb, axis=2)
            class_mask[mask] = class_id
            
        return class_mask

    def __len__(self) -> int:
        return len(self.patches)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path, lbl_path, x, y = self.patches[idx]
        
        # Load image patch
        with Image.open(img_path) as img:
            img_patch = img.crop((x, y, x + self.patch_size, y + self.patch_size))
            img_patch = np.array(img_patch)
            
        # Load label patch  
        with Image.open(lbl_path) as lbl:
            lbl_patch = lbl.crop((x, y, x + self.patch_size, y + self.patch_size))
            lbl_patch = np.array(lbl_patch)
            
        # Convert RGB labels to class indices
        if len(lbl_patch.shape) == 3 and lbl_patch.shape[2] == 3:
            lbl_patch = self._rgb_to_class_mask(lbl_patch)
            
        # Convert to tensors for augmentation
        img_tensor = transforms.ToTensor()(Image.fromarray(img_patch))
        lbl_tensor = torch.from_numpy(lbl_patch.astype(np.uint8))
        
        # Apply synchronized augmentation if enabled
        if self.augment:
            # Random horizontal flip
            if torch.rand(1) < 0.5:
                img_tensor = F.hflip(img_tensor)
                lbl_tensor = F.hflip(lbl_tensor)
            
            # Random vertical flip
            if torch.rand(1) < 0.5:
                img_tensor = F.vflip(img_tensor)
                lbl_tensor = F.vflip(lbl_tensor)
        
        # Apply normalization to image
        if self.transform:
            # Apply additional transforms (like normalization)
            # But skip ToTensor since we already converted
            for t in self.transform.transforms:
                if not isinstance(t, transforms.ToTensor):
                    img_tensor = t(img_tensor)
        else:
            # Apply default normalization
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                           std=[0.229, 0.224, 0.225])
            img_tensor = normalize(img_tensor)
            
        # Convert label to long tensor
        lbl_tensor = lbl_tensor.long()
            
        return img_tensor, lbl_tensor

    def visualize_sample(self, idx: int, save_path: Optional[str] = None):
        """Visualize a sample from the dataset."""
        img, lbl = self[idx]
        
        # Convert tensor to numpy for visualization
        if isinstance(img, torch.Tensor):
            if img.shape[0] == 3:  # CHW format
                img_np = img.permute(1, 2, 0).numpy()
            else:
                img_np = img.numpy()
        else:
            img_np = img
            
        if isinstance(lbl, torch.Tensor):
            lbl_np = lbl.numpy()
        else:
            lbl_np = lbl
            
        # Create visualization
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        # Show image
        axes[0].imshow(img_np)
        axes[0].set_title('Image')
        axes[0].axis('off')
        
        # Show label with colormap
        im = axes[1].imshow(lbl_np, cmap='tab10', vmin=0, vmax=5)
        axes[1].set_title('Label')
        axes[1].axis('off')
        
        # Add colorbar
        plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            plt.show()


def get_transforms(is_training: bool = True) -> Tuple[Optional[transforms.Compose], Optional[transforms.Compose]]:
    """Get data transforms for training/validation."""
    
    # For images, we just need normalization (ToTensor and augmentation handled in __getitem__)
    img_transform = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225]),
    ])
        
    # For labels, just convert to long tensor (already handled in __getitem__)
    lbl_transform = None
    
    return img_transform, lbl_transform


def create_dataloaders(root_dir: str,
                      dataset: str = 'potsdam',
                      patch_size: int = 256,
                      stride: int = 128,
                      batch_size: int = 16,
                      num_workers: int = 4
                      ) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders.
    
    Args:
        root_dir: Root directory for datasets
        dataset: 'potsdam', 'vaihingen', or 'both'
        patch_size: Size of patches (default: 256)
        stride: Stride for patch extraction (default: 128)
        batch_size: Batch size for dataloaders
        num_workers: Number of worker processes
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    
    # Get transforms
    train_transform, train_target_transform = get_transforms(is_training=True)
    val_transform, val_target_transform = get_transforms(is_training=False)
    
    # Create datasets
    train_dataset = PotsdamVaihingenDataset(
        root_dir=root_dir,
        dataset=dataset,
        split='train',
        patch_size=patch_size,
        stride=stride,
        transform=train_transform,
        target_transform=train_target_transform,
        augment=True
    )
    
    val_dataset = PotsdamVaihingenDataset(
        root_dir=root_dir,
        dataset=dataset,
        split='val',
        patch_size=patch_size,
        stride=stride,
        transform=val_transform,
        target_transform=val_target_transform,
        augment=False
    )
    
    test_dataset = PotsdamVaihingenDataset(
        root_dir=root_dir,
        dataset=dataset,
        split='test',
        patch_size=patch_size,
        stride=stride,
        transform=val_transform,
        target_transform=val_target_transform,
        augment=False
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    root_dir = "./data"
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        root_dir=root_dir,
        dataset='potsdam',  # or 'vaihingen' or 'both'
        patch_size=256,
        stride=128,
        batch_size=8,
        num_workers=2
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Test loading a batch
    if len(train_loader) > 0:
        for images, labels in train_loader:
            print(f"Image batch shape: {images.shape}")
            print(f"Label batch shape: {labels.shape}")
            print(f"Image dtype: {images.dtype}")
            print(f"Label dtype: {labels.dtype}")
            print(f"Label unique values: {torch.unique(labels)}")
            break
