"""
Visualization Utilities for Image Colorization

Contains functions for visualizing:
- LAB color channels
- Training history
- Comparison grids
- Sample outputs
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from .color_conversion import lab2rgb, denormalize_lab


def visualize_lab_channels(lab_image, save_path=None, show=True):
    """
    Visualize L, A, B channels separately.

    Args:
        lab_image: LAB image array (H, W, 3)
        save_path: Path to save the figure (optional)
        show: Whether to display the figure
    """
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle('LAB Color Space Channels', fontsize=14, fontweight='bold')

    # Original (RGB reconstruction)
    rgb = lab2rgb(lab_image)
    axes[0].imshow(np.clip(rgb, 0, 1))
    axes[0].set_title('Original (RGB)')
    axes[0].axis('off')

    # L channel (grayscale)
    axes[1].imshow(lab_image[:, :, 0], cmap='gray')
    axes[1].set_title(f'L Channel\n(Luminance: {lab_image[:, :, 0].min():.1f}-{lab_image[:, :, 0].max():.1f})')
    axes[1].axis('off')

    # A channel (green-red)
    im_a = axes[2].imshow(lab_image[:, :, 1], cmap='RdYlGn_r')
    axes[2].set_title(f'A Channel\n(Green-Red: {lab_image[:, :, 1].min():.1f}-{lab_image[:, :, 1].max():.1f})')
    axes[2].axis('off')
    plt.colorbar(im_a, ax=axes[2], fraction=0.046)

    # B channel (blue-yellow)
    im_b = axes[3].imshow(lab_image[:, :, 2], cmap='YlGnBu_r')
    axes[3].set_title(f'B Channel\n(Blue-Yellow: {lab_image[:, :, 2].min():.1f}-{lab_image[:, :, 2].max():.1f})')
    axes[3].axis('off')
    plt.colorbar(im_b, ax=axes[3], fraction=0.046)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_training_history(history, save_path=None, show=True):
    """
    Plot training history with multiple metrics.

    Args:
        history: Dictionary with training metrics
        save_path: Path to save the figure (optional)
        show: Whether to display the figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Training History', fontsize=16, fontweight='bold')

    epochs = range(1, len(history['train_g_loss']) + 1)

    # Generator Loss
    axes[0, 0].plot(epochs, history['train_g_loss'], 'b-', label='Train', linewidth=2)
    if 'val_g_loss' in history:
        axes[0, 0].plot(epochs, history['val_g_loss'], 'r--', label='Validation', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Generator Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Discriminator Loss
    axes[0, 1].plot(epochs, history['train_d_loss'], 'b-', label='Train', linewidth=2)
    if 'val_d_loss' in history:
        axes[0, 1].plot(epochs, history['val_d_loss'], 'r--', label='Validation', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_title('Discriminator Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # L1 and GAN Loss Components
    if 'train_l1_loss' in history and 'train_gan_loss' in history:
        axes[1, 0].plot(epochs, history['train_l1_loss'], 'g-', label='L1 Loss', linewidth=2)
        axes[1, 0].plot(epochs, history['train_gan_loss'], 'm-', label='GAN Loss', linewidth=2)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].set_title('Loss Components')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

    # Learning Rate
    if 'lr_g' in history and 'lr_d' in history:
        axes[1, 1].plot(epochs, history['lr_g'], 'b-', label='Generator LR', linewidth=2)
        axes[1, 1].plot(epochs, history['lr_d'], 'r-', label='Discriminator LR', linewidth=2)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_title('Learning Rate Schedule')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_yscale('log')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def create_comparison_grid(grayscale_images, predicted_images, ground_truth_images,
                           save_path=None, show=True, titles=None):
    """
    Create a comparison grid showing grayscale, predicted, and ground truth images.

    Args:
        grayscale_images: List of grayscale images (H, W) or (H, W, 1)
        predicted_images: List of predicted RGB images (H, W, 3)
        ground_truth_images: List of ground truth RGB images (H, W, 3)
        save_path: Path to save the figure (optional)
        show: Whether to display the figure
        titles: Optional list of titles for each row
    """
    n_samples = len(grayscale_images)
    fig, axes = plt.subplots(n_samples, 3, figsize=(12, 4 * n_samples))

    if n_samples == 1:
        axes = axes.reshape(1, -1)

    fig.suptitle('Colorization Results Comparison', fontsize=16, fontweight='bold')

    for i in range(n_samples):
        # Grayscale
        gray = grayscale_images[i]
        if gray.ndim == 3:
            gray = gray.squeeze()
        axes[i, 0].imshow(gray, cmap='gray')
        axes[i, 0].set_title('Grayscale Input' if i == 0 else '')
        axes[i, 0].axis('off')

        # Predicted
        pred = np.clip(predicted_images[i], 0, 1)
        axes[i, 1].imshow(pred)
        axes[i, 1].set_title('Predicted' if i == 0 else '')
        axes[i, 1].axis('off')

        # Ground Truth
        gt = np.clip(ground_truth_images[i], 0, 1)
        axes[i, 2].imshow(gt)
        axes[i, 2].set_title('Ground Truth' if i == 0 else '')
        axes[i, 2].axis('off')

        # Add row title if provided
        if titles and i < len(titles):
            axes[i, 0].set_ylabel(titles[i], fontsize=12, rotation=0, ha='right', va='center')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def save_colorized_image(L_tensor, AB_tensor, save_path):
    """
    Save colorized image from L and AB tensors.

    Args:
        L_tensor: L channel tensor (1, H, W) normalized to [-1, 1]
        AB_tensor: AB channel tensor (2, H, W) normalized to [-1, 1]
        save_path: Path to save the image
    """
    import torch

    # Convert to numpy
    if isinstance(L_tensor, torch.Tensor):
        L = L_tensor.detach().cpu().numpy()
    else:
        L = L_tensor

    if isinstance(AB_tensor, torch.Tensor):
        AB = AB_tensor.detach().cpu().numpy()
    else:
        AB = AB_tensor

    # Handle dimensions
    if L.ndim == 3:
        L = L.transpose(1, 2, 0)  # (1, H, W) -> (H, W, 1)
    if AB.ndim == 3:
        AB = AB.transpose(1, 2, 0)  # (2, H, W) -> (H, W, 2)

    # Denormalize
    lab = denormalize_lab(L, AB)

    # Convert to RGB
    rgb = lab2rgb(lab)
    rgb = np.clip(rgb * 255, 0, 255).astype(np.uint8)

    # Save
    img = Image.fromarray(rgb)
    img.save(save_path)
    print(f"Saved colorized image to: {save_path}")


def visualize_predictions_batch(model, dataloader, device, num_samples=8,
                                 save_path=None, show=True):
    """
    Visualize model predictions on a batch of images.

    Args:
        model: Trained generator model
        dataloader: DataLoader with test data
        device: Computation device
        num_samples: Number of samples to visualize
        save_path: Path to save the figure (optional)
        show: Whether to display the figure
    """
    import torch

    model.eval()

    grayscale_list = []
    predicted_list = []
    ground_truth_list = []

    with torch.no_grad():
        for L, AB_real in dataloader:
            L = L.to(device)
            AB_pred = model(L)

            for i in range(min(L.size(0), num_samples - len(grayscale_list))):
                # Get numpy arrays
                L_np = L[i].cpu().numpy().transpose(1, 2, 0)
                AB_pred_np = AB_pred[i].cpu().numpy().transpose(1, 2, 0)
                AB_real_np = AB_real[i].numpy().transpose(1, 2, 0)

                # Denormalize and convert to RGB
                lab_pred = denormalize_lab(L_np, AB_pred_np)
                lab_real = denormalize_lab(L_np, AB_real_np)

                rgb_pred = lab2rgb(lab_pred)
                rgb_real = lab2rgb(lab_real)

                grayscale_list.append(L_np.squeeze())
                predicted_list.append(rgb_pred)
                ground_truth_list.append(rgb_real)

            if len(grayscale_list) >= num_samples:
                break

    create_comparison_grid(grayscale_list, predicted_list, ground_truth_list,
                           save_path=save_path, show=show)


if __name__ == "__main__":
    print("Visualization utilities loaded successfully.")
