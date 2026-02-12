"""
Data Preprocessing Module for Image Colorization

This module handles all data preprocessing steps:
1. Image loading and resizing
2. RGB to LAB color space conversion
3. Data augmentation
4. Normalization
5. Train/Validation/Test split
"""

import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms


def rgb2lab(rgb):
    """
    Convert RGB image to LAB color space.

    Args:
        rgb: RGB image as numpy array with values in [0, 255] or [0, 1]

    Returns:
        LAB image as numpy array
        L: [0, 100], A: [-128, 127], B: [-128, 127]
    """
    rgb = np.asarray(rgb, dtype=np.float32)

    # Normalize to [0, 1] if needed
    if rgb.max() > 1.0:
        rgb = rgb / 255.0

    # RGB to XYZ conversion (sRGB with D65 white point)
    mask = rgb > 0.04045
    rgb_linear = np.where(mask, ((rgb + 0.055) / 1.055) ** 2.4, rgb / 12.92)

    xyz_matrix = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041]
    ], dtype=np.float32)

    xyz = np.dot(rgb_linear, xyz_matrix.T)

    # Normalize by D65 white point
    white_point = np.array([0.95047, 1.0, 1.08883], dtype=np.float32)
    xyz = xyz / white_point

    # XYZ to LAB
    epsilon = 0.008856
    kappa = 903.3

    mask = xyz > epsilon
    f_xyz = np.where(mask, np.cbrt(xyz), (kappa * xyz + 16.0) / 116.0)

    L = 116.0 * f_xyz[..., 1] - 16.0
    a = 500.0 * (f_xyz[..., 0] - f_xyz[..., 1])
    b = 200.0 * (f_xyz[..., 1] - f_xyz[..., 2])

    lab = np.stack([L, a, b], axis=-1)
    return lab


def lab2rgb(lab):
    """
    Convert LAB image to RGB color space.

    Args:
        lab: LAB image as numpy array

    Returns:
        RGB image as numpy array with values in [0, 1]
    """
    lab = np.asarray(lab, dtype=np.float32)

    L = lab[..., 0]
    a = lab[..., 1]
    b = lab[..., 2]

    # LAB to XYZ
    fy = (L + 16.0) / 116.0
    fx = a / 500.0 + fy
    fz = fy - b / 200.0

    epsilon = 0.008856
    kappa = 903.3

    x_mask = fx ** 3 > epsilon
    y_mask = L > kappa * epsilon
    z_mask = fz ** 3 > epsilon

    x = np.where(x_mask, fx ** 3, (116.0 * fx - 16.0) / kappa)
    y = np.where(y_mask, ((L + 16.0) / 116.0) ** 3, L / kappa)
    z = np.where(z_mask, fz ** 3, (116.0 * fz - 16.0) / kappa)

    # Denormalize by D65 white point
    white_point = np.array([0.95047, 1.0, 1.08883], dtype=np.float32)
    xyz = np.stack([x, y, z], axis=-1) * white_point

    # XYZ to RGB matrix
    rgb_matrix = np.array([
        [3.2404542, -1.5371385, -0.4985314],
        [-0.9692660, 1.8760108, 0.0415560],
        [0.0556434, -0.2040259, 1.0572252]
    ], dtype=np.float32)

    rgb_linear = np.dot(xyz, rgb_matrix.T)

    # Apply inverse gamma correction
    rgb_linear = np.clip(rgb_linear, 0, None)
    mask = rgb_linear > 0.0031308
    rgb = np.where(mask, 1.055 * (rgb_linear ** (1.0 / 2.4)) - 0.055, 12.92 * rgb_linear)

    rgb = np.clip(rgb, 0.0, 1.0)
    return rgb


class ColorizationDataset(Dataset):
    """
    PyTorch Dataset for Image Colorization.

    Preprocessing steps:
    1. Load RGB image
    2. Resize to target size
    3. Apply data augmentation (training only)
    4. Convert RGB to LAB
    5. Separate L channel (input) and AB channels (target)
    6. Normalize to [-1, 1]
    """

    def __init__(self, root_dir, image_size=256, augment=True):
        """
        Args:
            root_dir: Directory containing images
            image_size: Target size for images
            augment: Whether to apply data augmentation
        """
        self.root_dir = root_dir
        self.image_size = image_size
        self.augment = augment
        self.image_files = self._get_image_files()

        # Data augmentation transforms
        if augment:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
            ])

    def _get_image_files(self):
        """Get all valid image files from directory."""
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        files = []
        for f in os.listdir(self.root_dir):
            ext = os.path.splitext(f)[1].lower()
            if ext in valid_extensions:
                files.append(os.path.join(self.root_dir, f))
        return files

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]

        # Load and transform image
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        img = np.array(img)

        # Convert to LAB color space
        lab = rgb2lab(img).astype(np.float32)

        # Normalize L channel to [-1, 1]
        L = lab[:, :, 0:1]
        L = (L / 50.0) - 1.0

        # Normalize AB channels to [-1, 1]
        AB = lab[:, :, 1:3]
        AB = AB / 110.0

        # Convert to tensors (C, H, W)
        L = torch.from_numpy(L.transpose(2, 0, 1))
        AB = torch.from_numpy(AB.transpose(2, 0, 1))

        return L, AB


def create_dataloaders(data_dir, batch_size=16, image_size=256,
                       val_split=0.1, test_split=0.1, num_workers=4):
    """
    Create train, validation, and test dataloaders.

    Args:
        data_dir: Directory containing training images
        batch_size: Batch size
        image_size: Target image size
        val_split: Fraction for validation
        test_split: Fraction for test
        num_workers: Number of data loading workers

    Returns:
        train_loader, val_loader, test_loader
    """
    # Create full dataset
    full_dataset = ColorizationDataset(data_dir, image_size, augment=True)

    # Calculate split sizes
    total_size = len(full_dataset)
    test_size = int(total_size * test_split)
    val_size = int(total_size * val_split)
    train_size = total_size - val_size - test_size

    # Split dataset
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    # Create dataloaders
    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    print(f"Dataset Split:")
    print(f"  Train: {train_size} images")
    print(f"  Validation: {val_size} images")
    print(f"  Test: {test_size} images")

    return train_loader, val_loader, test_loader
