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
import random

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
                 transform: Optional[transforms.Compose] = None,
                 target_transform: Optional[transforms.Compose] = None,
                 augment: bool = True):
        """
        Args:
            root_dir: Root directory to store datasets
            dataset: 'potsdam', 'vaihingen', or 'both'
            split: 'train', 'val', or 'test'
            patch_size: Size of patches to extract (default: 256)
            transform: Optional transform to be applied on images
            target_transform: Optional transform to be applied on labels
            augment: Whether to apply data augmentation (only for training)
        """
        self.root_dir = root_dir
        self.dataset = dataset.lower()
        self.split = split
        self.patch_size = patch_size
        self.transform = transform
        self.target_transform = target_transform
        self.augment = augment and (split == 'train')  # Only augment training data
        
        # Create directory structure
        os.makedirs(root_dir, exist_ok=True)
        
        # Load image and label paths
        self.image_paths, self.label_paths = self._load_file_paths()
        
        # Prepare valid images for dynamic patch sampling
        self.valid_images, self.num_patches = self._prepare_valid_images()
        
        print(f"Prepared {len(self.valid_images)} images for dynamic sampling of {self.num_patches} patches from {self.dataset} dataset ({self.split} split)")

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
            area_ids = ['2_10', '2_11', '2_12', '3_10', '3_11', '3_12', '4_11', '4_12', '5_10', '5_12', '6_7', '6_8', '6_10', '6_11', '6_12', '7_7', '7_9', '7_8', '7_12']
        elif self.split == 'test':
            area_ids = ['5_11', '6_9', '7_11']
        elif self.split == 'val':
            area_ids = ['4_10', '7_10']
        else: # holdout
            area_ids = ['2_13', '2_14', '3_13', '3_14', '4_13', '4_14', '4_15', '5_13', '5_14', '5_15', '6_13', '6_14', '6_15', '7_13']

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
        
        if self.split == 'train':
            area_ids = [1, 3, 7, 9, 11, 13, 17, 18, 19, 23, 25, 26, 28, 32, 34, 36, 37]
        elif self.split == 'test':
            area_ids = [5, 15, 21, 30]
        elif self.split == 'val':
            area_ids = [7, 9]
        else: # holdout
            area_ids = [2, 4, 6, 8, 10, 12, 14, 16, 20, 22, 24, 27, 29, 31, 33, 35, 38]
            
        image_paths = []
        label_paths = []
        
        for area_id in area_ids:
            img_path = os.path.join(img_dir, f'top_mosaic_09cm_area{area_id}.tif')
            lbl_path = os.path.join(lbl_dir, f'top_mosaic_09cm_area{area_id}.tif')
            
            if os.path.exists(img_path) and os.path.exists(lbl_path):
                image_paths.append(img_path)
                label_paths.append(lbl_path)
                
        return image_paths, label_paths

    def _prepare_valid_images(self) -> Tuple[List[Tuple[str, str, int, int]], int]:
        """Prepare valid images for dynamic patch sampling."""
        
        # Determine target number of patches based on paper specifications
        target_patches = {
            'train': 5000,   # Paper specification
            'val': 1000,     # Smaller validation set
            'test': 1000     # Test set for evaluation
        }
        
        num_patches = target_patches.get(self.split, 1000)
        
        print(f"Preparing images for dynamic sampling of {num_patches} patches for {self.split} split...")
        
        # Collect valid image information for random sampling
        valid_images = []
        for img_path, lbl_path in zip(self.image_paths, self.label_paths):
            try:
                # Load image to get dimensions
                with Image.open(img_path) as img:
                    width, height = img.size
                
                # Check if image is large enough for patches
                max_x = width - self.patch_size
                max_y = height - self.patch_size
                
                if max_x > 0 and max_y > 0:
                    valid_images.append((img_path, lbl_path, max_x, max_y))
                    
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                continue
        
        if not valid_images:
            print("No valid images found for patch extraction!")
            return [], num_patches
        
        print(f"Successfully prepared {len(valid_images)} images for dynamic patch sampling")
        return valid_images, num_patches

    def _rgb_to_class_mask(self, rgb_label: np.ndarray) -> np.ndarray:
        """Convert RGB label to class mask."""
        h, w = rgb_label.shape[:2]
        class_mask = np.zeros((h, w), dtype=np.uint8)
        
        for rgb, class_id in self.RGB_TO_CLASS.items():
            mask = np.all(rgb_label == rgb, axis=2)
            class_mask[mask] = class_id
            
        return class_mask

    def __len__(self) -> int:
        return self.num_patches

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Note: Each call to __getitem__ provides uniform random sampling
        # The 'idx' parameter is ignored in favor of true random sampling
        # This ensures uniform random sampling across the entire dataset for each epoch
        
        # Randomly select an image from valid images (uniform sampling)
        img_path, lbl_path, max_x, max_y = random.choice(self.valid_images)
        
        # Randomly select patch position within the image
        x = random.randint(0, max_x)
        y = random.randint(0, max_y)
        
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
            # Random horizontal flip (mirroring) with probability 0.5
            if torch.rand(1) < 0.5:
                img_tensor = F.hflip(img_tensor)
                lbl_tensor = F.hflip(lbl_tensor)
            
            # Random vertical flip with probability 0.5
            if torch.rand(1) < 0.5:
                img_tensor = F.vflip(img_tensor)
                lbl_tensor = F.vflip(lbl_tensor)
        
        if self.transform:
            # Apply additional transforms
            # But skip ToTensor since we already converted
            for t in self.transform.transforms:
                if not isinstance(t, transforms.ToTensor):
                    img_tensor = t(img_tensor)
        
        # ToTensor already converted from [0, 255] to [0.0, 1.0], so we keep this normalization
            
        # Convert label to long tensor
        lbl_tensor = lbl_tensor.long()
            
        return img_tensor, lbl_tensor

    def get_full_image(self, image_idx: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load a full-resolution image and its corresponding label for TTA evaluation.
        
        Args:
            image_idx: Index of the image to load (from available image paths)
            
        Returns:
            Tuple of (full_image_tensor, full_label_tensor)
        """
        if image_idx >= len(self.image_paths):
            raise IndexError(f"Image index {image_idx} out of range. Available: {len(self.image_paths)}")
        
        img_path = self.image_paths[image_idx]
        lbl_path = self.label_paths[image_idx]
        
        print(f"Loading full resolution image: {os.path.basename(img_path)}")
        
        # Load full image
        with Image.open(img_path) as img:
            img_array = np.array(img)
            print(f"Full image size: {img_array.shape}")
            
        # Load full label
        with Image.open(lbl_path) as lbl:
            lbl_array = np.array(lbl)
            
        # Convert RGB labels to class indices if needed
        if len(lbl_array.shape) == 3 and lbl_array.shape[2] == 3:
            lbl_array = self._rgb_to_class_mask(lbl_array)
            
        # Convert to tensors
        img_tensor = transforms.ToTensor()(Image.fromarray(img_array))
        lbl_tensor = torch.from_numpy(lbl_array.astype(np.uint8)).long()
        
        # ToTensor already provides [0.0, 1.0] normalization, so we keep it as is
        
        # Add batch dimension
        img_tensor = img_tensor.unsqueeze(0)  # [1, 3, H, W]
        
        return img_tensor, lbl_tensor

    def get_available_images(self) -> List[str]:
        """Get list of available full-resolution image names."""
        return [os.path.basename(path) for path in self.image_paths]

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


def worker_init_fn(worker_id):
    """Initialize each worker with a different random seed for data augmentation"""
    # Get initial seed and ensure it's within valid range for numpy
    base_seed = torch.initial_seed()
    # Ensure seed is within numpy's valid range (0 to 2^32 - 1)
    worker_seed = (base_seed + worker_id) % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)

# TODO Refactor this
def get_transforms(is_training: bool = True) -> Tuple[Optional[transforms.Compose], Optional[transforms.Compose]]:
    """Get data transforms for training/validation."""
    
    img_transform = None  # No additional transforms needed - images stay in [0.0, 1.0] range
        
    # For labels, just convert to long tensor (already handled in __getitem__)
    lbl_transform = None
    
    return img_transform, lbl_transform


def create_dataloaders(root_dir: str,
                      dataset: str = 'potsdam',
                      patch_size: int = 256,
                      batch_size: int = 16,
                      num_workers: int = 4,
                      pin_memory: bool = True,
                      drop_last: bool = True
                      ) -> Tuple[DataLoader, DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders.
    
    Args:
        root_dir: Root directory for datasets
        dataset: 'potsdam', 'vaihingen', or 'both'
        patch_size: Size of patches (default: 256)
        batch_size: Batch size for dataloaders
        num_workers: Number of worker processes
        pin_memory: Whether to use pinned memory for faster GPU transfer
        drop_last: Whether to drop the last incomplete batch (recommended for training)
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader, holdout_loader)
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
        transform=train_transform,
        target_transform=train_target_transform,
        augment=True
    )
    
    val_dataset = PotsdamVaihingenDataset(
        root_dir=root_dir,
        dataset=dataset,
        split='val',
        patch_size=patch_size,
        transform=val_transform,
        target_transform=val_target_transform,
        augment=False
    )
    
    test_dataset = PotsdamVaihingenDataset(
        root_dir=root_dir,
        dataset=dataset,
        split='test',
        patch_size=patch_size,
        transform=val_transform,
        target_transform=val_target_transform,
        augment=False
    )

    holdout_dataset= PotsdamVaihingenDataset(
        root_dir=root_dir,
        dataset=dataset,
        split='holdout',
        patch_size=patch_size,
        transform=val_transform,
        target_transform=val_target_transform,
        augment=False
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,                    # Essential for training: randomizes batch composition
        num_workers=num_workers,
        pin_memory=pin_memory,          # Faster GPU transfer
        drop_last=drop_last,            # Ensures consistent batch sizes for training
        worker_init_fn=worker_init_fn   # Ensures different seeds for each worker
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,                  # No shuffling needed for validation
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False                 # Keep all validation data
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,                  # No shuffling needed for testing
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False                 # Keep all test data
    )

    holdout_loader = DataLoader(
        holdout_dataset,
        batch_size=batch_size,
        shuffle=False,                  # No shuffling needed for holdout evaluation
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False                 # Keep all holdout data
    )
    
    return train_loader, val_loader, test_loader, holdout_loader


if __name__ == "__main__":
    root_dir = "./data"
    
    # Create dataloaders
    train_loader, val_loader, test_loader, holdout_loader = create_dataloaders(
        root_dir=root_dir,
        dataset='potsdam',  # or 'vaihingen' or 'both'
        patch_size=256,
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
