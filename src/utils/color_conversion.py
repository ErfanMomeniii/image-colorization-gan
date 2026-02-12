"""
Color Conversion Utilities

Pure NumPy implementation of RGB to LAB and LAB to RGB conversion.
No external dependencies required (no scipy, no scikit-image).

LAB Color Space:
- L: Luminance (0-100)
- A: Green-Red axis (-128 to 127)
- B: Blue-Yellow axis (-128 to 127)

This implementation uses sRGB with D65 white point.
"""

import numpy as np


def rgb2lab(rgb):
    """
    Convert RGB image to LAB color space.

    Args:
        rgb: RGB image as numpy array with values in [0, 255] or [0, 1]
             Shape: (H, W, 3) or (B, H, W, 3)

    Returns:
        LAB image as numpy array
        L: [0, 100], A: [-128, 127], B: [-128, 127]
    """
    rgb = np.asarray(rgb, dtype=np.float32)

    # Normalize to [0, 1] if needed
    if rgb.max() > 1.0:
        rgb = rgb / 255.0

    # RGB to XYZ conversion (sRGB with D65 white point)
    # Apply inverse gamma correction
    mask = rgb > 0.04045
    rgb_linear = np.where(mask, ((rgb + 0.055) / 1.055) ** 2.4, rgb / 12.92)

    # RGB to XYZ matrix (sRGB to XYZ)
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
    epsilon = 0.008856  # (6/29)^3
    kappa = 903.3  # (29/3)^3

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
             L: [0, 100], A: [-128, 127], B: [-128, 127]
             Shape: (H, W, 3) or (B, H, W, 3)

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

    epsilon = 0.008856  # (6/29)^3
    kappa = 903.3  # (29/3)^3

    x_mask = fx ** 3 > epsilon
    y_mask = L > kappa * epsilon
    z_mask = fz ** 3 > epsilon

    x = np.where(x_mask, fx ** 3, (116.0 * fx - 16.0) / kappa)
    y = np.where(y_mask, ((L + 16.0) / 116.0) ** 3, L / kappa)
    z = np.where(z_mask, fz ** 3, (116.0 * fz - 16.0) / kappa)

    # Denormalize by D65 white point
    white_point = np.array([0.95047, 1.0, 1.08883], dtype=np.float32)
    xyz = np.stack([x, y, z], axis=-1) * white_point

    # XYZ to RGB matrix (XYZ to sRGB)
    rgb_matrix = np.array([
        [3.2404542, -1.5371385, -0.4985314],
        [-0.9692660, 1.8760108, 0.0415560],
        [0.0556434, -0.2040259, 1.0572252]
    ], dtype=np.float32)

    rgb_linear = np.dot(xyz, rgb_matrix.T)

    # Apply gamma correction
    rgb_linear = np.clip(rgb_linear, 0, None)
    mask = rgb_linear > 0.0031308
    rgb = np.where(mask, 1.055 * (rgb_linear ** (1.0 / 2.4)) - 0.055, 12.92 * rgb_linear)

    rgb = np.clip(rgb, 0.0, 1.0)
    return rgb


def normalize_lab(lab):
    """
    Normalize LAB values to [-1, 1] range for neural network.

    Args:
        lab: LAB image with L in [0, 100], AB in [-128, 127]

    Returns:
        Normalized LAB with all channels in [-1, 1]
    """
    lab = np.asarray(lab, dtype=np.float32)

    # Normalize L: [0, 100] -> [-1, 1]
    L = (lab[..., 0:1] / 50.0) - 1.0

    # Normalize AB: [-128, 127] -> approximately [-1, 1]
    AB = lab[..., 1:3] / 110.0

    return L, AB


def denormalize_lab(L, AB):
    """
    Denormalize LAB values from [-1, 1] to original range.

    Args:
        L: Normalized L channel in [-1, 1]
        AB: Normalized AB channels in [-1, 1]

    Returns:
        LAB image with L in [0, 100], AB in [-128, 127]
    """
    L = np.asarray(L, dtype=np.float32)
    AB = np.asarray(AB, dtype=np.float32)

    # Denormalize L: [-1, 1] -> [0, 100]
    L_denorm = (L + 1.0) * 50.0

    # Denormalize AB: [-1, 1] -> [-110, 110]
    AB_denorm = AB * 110.0

    # Combine
    if L_denorm.ndim == 2:
        L_denorm = L_denorm[..., np.newaxis]
    if AB_denorm.ndim == 2:
        AB_denorm = AB_denorm[..., np.newaxis]

    lab = np.concatenate([L_denorm, AB_denorm], axis=-1)
    return lab


def tensor_to_lab(L_tensor, AB_tensor):
    """
    Convert PyTorch tensors to LAB numpy array.

    Args:
        L_tensor: L channel tensor (B, 1, H, W) or (1, H, W)
        AB_tensor: AB channel tensor (B, 2, H, W) or (2, H, W)

    Returns:
        LAB image as numpy array (H, W, 3) or (B, H, W, 3)
    """
    import torch

    if isinstance(L_tensor, torch.Tensor):
        L = L_tensor.detach().cpu().numpy()
    else:
        L = L_tensor

    if isinstance(AB_tensor, torch.Tensor):
        AB = AB_tensor.detach().cpu().numpy()
    else:
        AB = AB_tensor

    # Handle batch dimension
    if L.ndim == 4:
        # (B, 1, H, W) -> (B, H, W, 1)
        L = L.transpose(0, 2, 3, 1)
        AB = AB.transpose(0, 2, 3, 1)
    else:
        # (1, H, W) -> (H, W, 1)
        L = L.transpose(1, 2, 0)
        AB = AB.transpose(1, 2, 0)

    return denormalize_lab(L, AB)


if __name__ == "__main__":
    # Test conversion
    import numpy as np

    # Create test RGB image
    rgb = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)

    # Convert RGB -> LAB -> RGB
    lab = rgb2lab(rgb)
    rgb_reconstructed = lab2rgb(lab)

    # Check reconstruction error
    rgb_normalized = rgb / 255.0
    error = np.mean(np.abs(rgb_normalized - rgb_reconstructed))
    print(f"RGB -> LAB -> RGB reconstruction error: {error:.6f}")

    # Test normalization
    L, AB = normalize_lab(lab)
    print(f"L range: [{L.min():.2f}, {L.max():.2f}]")
    print(f"AB range: [{AB.min():.2f}, {AB.max():.2f}]")

    # Test denormalization
    lab_reconstructed = denormalize_lab(L, AB)
    error = np.mean(np.abs(lab - lab_reconstructed))
    print(f"LAB -> normalize -> denormalize error: {error:.6f}")
