"""
Model Evaluation Module for Image Colorization

This module provides comprehensive evaluation metrics for assessing
the quality of colorized images compared to ground truth.

Metrics implemented:
1. PSNR (Peak Signal-to-Noise Ratio) - measures reconstruction quality
2. SSIM (Structural Similarity Index) - measures structural similarity
3. Colorfulness - measures color richness of generated images
4. L1/L2 Error - measures pixel-wise reconstruction error
"""

import os
import json
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm


def calculate_psnr(pred, target, max_val=1.0):
    """
    Calculate Peak Signal-to-Noise Ratio (PSNR).

    PSNR = 10 * log10(MAX^2 / MSE)

    Higher PSNR indicates better reconstruction quality.
    Typical values: 20-40 dB (higher is better)

    Args:
        pred: Predicted image tensor (B, C, H, W) or (C, H, W)
        target: Ground truth image tensor
        max_val: Maximum pixel value (1.0 for normalized images)

    Returns:
        PSNR value in dB
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()

    mse = np.mean((pred - target) ** 2)
    if mse == 0:
        return float('inf')

    psnr = 10 * np.log10((max_val ** 2) / mse)
    return psnr


def calculate_ssim(pred, target, window_size=11, C1=0.01**2, C2=0.03**2):
    """
    Calculate Structural Similarity Index (SSIM).

    SSIM compares luminance, contrast, and structure between images.
    Range: [-1, 1], where 1 indicates perfect similarity.

    Args:
        pred: Predicted image array (H, W) or (H, W, C)
        target: Ground truth image array
        window_size: Size of the sliding window
        C1, C2: Constants for stability

    Returns:
        SSIM value (higher is better)
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()

    # Handle multi-channel images
    if pred.ndim == 3:
        ssim_vals = []
        for c in range(pred.shape[-1]):
            ssim_vals.append(calculate_ssim(pred[..., c], target[..., c], window_size, C1, C2))
        return np.mean(ssim_vals)

    # Calculate local means
    kernel = np.ones((window_size, window_size)) / (window_size ** 2)

    mu_x = _convolve2d(pred, kernel)
    mu_y = _convolve2d(target, kernel)

    mu_x_sq = mu_x ** 2
    mu_y_sq = mu_y ** 2
    mu_xy = mu_x * mu_y

    sigma_x_sq = _convolve2d(pred ** 2, kernel) - mu_x_sq
    sigma_y_sq = _convolve2d(target ** 2, kernel) - mu_y_sq
    sigma_xy = _convolve2d(pred * target, kernel) - mu_xy

    # SSIM formula
    numerator = (2 * mu_xy + C1) * (2 * sigma_xy + C2)
    denominator = (mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2)

    ssim_map = numerator / denominator
    return np.mean(ssim_map)


def _convolve2d(image, kernel):
    """Simple 2D convolution using numpy."""
    from numpy.lib.stride_tricks import as_strided

    # Pad image
    pad = kernel.shape[0] // 2
    padded = np.pad(image, pad, mode='reflect')

    # Create view for convolution
    shape = (image.shape[0], image.shape[1], kernel.shape[0], kernel.shape[1])
    strides = padded.strides * 2
    view = as_strided(padded, shape=shape, strides=strides)

    return np.einsum('ijkl,kl->ij', view, kernel)


def calculate_colorfulness(image):
    """
    Calculate colorfulness metric (Hasler and Süsstrunk, 2003).

    Measures the richness and variety of colors in an image.
    Higher values indicate more colorful images.

    Args:
        image: RGB image array (H, W, 3) with values in [0, 1]

    Returns:
        Colorfulness metric value
    """
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()

    # Ensure image is in (H, W, 3) format
    if image.ndim == 3 and image.shape[0] == 3:
        image = image.transpose(1, 2, 0)

    R, G, B = image[:, :, 0], image[:, :, 1], image[:, :, 2]

    # Compute rg and yb
    rg = R - G
    yb = 0.5 * (R + G) - B

    # Compute mean and standard deviation
    rg_mean, rg_std = np.mean(rg), np.std(rg)
    yb_mean, yb_std = np.mean(yb), np.std(yb)

    # Combine metrics
    std_root = np.sqrt(rg_std ** 2 + yb_std ** 2)
    mean_root = np.sqrt(rg_mean ** 2 + yb_mean ** 2)

    colorfulness = std_root + 0.3 * mean_root
    return colorfulness


def calculate_l1_error(pred, target):
    """Calculate mean L1 (absolute) error."""
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()
    return np.mean(np.abs(pred - target))


def calculate_l2_error(pred, target):
    """Calculate mean L2 (squared) error."""
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()
    return np.mean((pred - target) ** 2)


def evaluate_batch(pred_batch, target_batch):
    """
    Evaluate a batch of predictions.

    Args:
        pred_batch: Batch of predicted AB channels (B, 2, H, W)
        target_batch: Batch of target AB channels (B, 2, H, W)

    Returns:
        Dictionary with mean metrics over the batch
    """
    batch_size = pred_batch.shape[0]

    psnr_values = []
    ssim_values = []
    l1_values = []
    l2_values = []

    for i in range(batch_size):
        pred = pred_batch[i].transpose(1, 2, 0) if isinstance(pred_batch, np.ndarray) else pred_batch[i].permute(1, 2, 0)
        target = target_batch[i].transpose(1, 2, 0) if isinstance(target_batch, np.ndarray) else target_batch[i].permute(1, 2, 0)

        psnr_values.append(calculate_psnr(pred, target))
        ssim_values.append(calculate_ssim(pred, target))
        l1_values.append(calculate_l1_error(pred, target))
        l2_values.append(calculate_l2_error(pred, target))

    return {
        'psnr': np.mean(psnr_values),
        'ssim': np.mean(ssim_values),
        'l1_error': np.mean(l1_values),
        'l2_error': np.mean(l2_values)
    }


class Evaluator:
    """
    Comprehensive model evaluator for image colorization.

    Evaluates model performance using multiple metrics and
    generates visualization reports.
    """

    def __init__(self, generator, device, save_dir='results/evaluation'):
        """
        Initialize evaluator.

        Args:
            generator: Trained generator model
            device: Computation device (cuda/cpu)
            save_dir: Directory to save evaluation results
        """
        self.generator = generator
        self.device = device
        self.save_dir = save_dir

        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'samples'), exist_ok=True)

        self.generator.eval()

    @torch.no_grad()
    def evaluate_dataset(self, dataloader, num_samples=None):
        """
        Evaluate model on entire dataset.

        Args:
            dataloader: DataLoader with test data
            num_samples: Number of samples to evaluate (None = all)

        Returns:
            Dictionary with evaluation metrics
        """
        all_psnr = []
        all_ssim = []
        all_l1 = []
        all_l2 = []
        all_colorfulness_pred = []
        all_colorfulness_gt = []

        total = num_samples if num_samples else len(dataloader)
        pbar = tqdm(dataloader, total=total, desc="Evaluating")

        for batch_idx, (L, AB_real) in enumerate(pbar):
            if num_samples and batch_idx >= num_samples:
                break

            L = L.to(self.device)
            AB_real = AB_real.to(self.device)

            # Generate predictions
            AB_pred = self.generator(L)

            # Calculate metrics for each sample in batch
            for i in range(L.size(0)):
                pred = AB_pred[i].cpu().numpy().transpose(1, 2, 0)
                target = AB_real[i].cpu().numpy().transpose(1, 2, 0)

                all_psnr.append(calculate_psnr(pred, target))
                all_ssim.append(calculate_ssim(pred, target))
                all_l1.append(calculate_l1_error(pred, target))
                all_l2.append(calculate_l2_error(pred, target))

                # Convert to RGB for colorfulness
                L_np = L[i].cpu().numpy().transpose(1, 2, 0)
                pred_rgb = self._lab_to_rgb_approx(L_np, pred)
                gt_rgb = self._lab_to_rgb_approx(L_np, target)

                all_colorfulness_pred.append(calculate_colorfulness(pred_rgb))
                all_colorfulness_gt.append(calculate_colorfulness(gt_rgb))

            pbar.set_postfix({
                'PSNR': f'{np.mean(all_psnr):.2f}',
                'SSIM': f'{np.mean(all_ssim):.3f}'
            })

        # Compile results
        results = {
            'psnr': {
                'mean': float(np.mean(all_psnr)),
                'std': float(np.std(all_psnr)),
                'min': float(np.min(all_psnr)),
                'max': float(np.max(all_psnr))
            },
            'ssim': {
                'mean': float(np.mean(all_ssim)),
                'std': float(np.std(all_ssim)),
                'min': float(np.min(all_ssim)),
                'max': float(np.max(all_ssim))
            },
            'l1_error': {
                'mean': float(np.mean(all_l1)),
                'std': float(np.std(all_l1))
            },
            'l2_error': {
                'mean': float(np.mean(all_l2)),
                'std': float(np.std(all_l2))
            },
            'colorfulness': {
                'predicted_mean': float(np.mean(all_colorfulness_pred)),
                'ground_truth_mean': float(np.mean(all_colorfulness_gt)),
                'ratio': float(np.mean(all_colorfulness_pred) / np.mean(all_colorfulness_gt))
            },
            'num_samples': len(all_psnr)
        }

        return results

    def _lab_to_rgb_approx(self, L, AB):
        """Approximate LAB to RGB conversion for visualization."""
        # Denormalize
        L_denorm = (L + 1) * 50  # [0, 100]
        AB_denorm = AB * 110  # [-110, 110]

        # Simple approximation (not exact LAB to RGB)
        # This is just for colorfulness metric visualization
        lab = np.concatenate([L_denorm, AB_denorm], axis=-1)

        # Approximate RGB
        rgb = np.zeros_like(lab)
        rgb[:, :, 0] = np.clip(L_denorm[:, :, 0] / 100 + AB_denorm[:, :, 0] / 500, 0, 1)
        rgb[:, :, 1] = np.clip(L_denorm[:, :, 0] / 100, 0, 1)
        rgb[:, :, 2] = np.clip(L_denorm[:, :, 0] / 100 - AB_denorm[:, :, 1] / 200, 0, 1)

        return np.clip(rgb, 0, 1)

    def save_results(self, results, filename='evaluation_results.json'):
        """Save evaluation results to JSON file."""
        filepath = os.path.join(self.save_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {filepath}")

    def plot_metrics_distribution(self, dataloader, num_samples=100):
        """
        Plot distribution of metrics across samples.

        Args:
            dataloader: DataLoader with test data
            num_samples: Number of samples to evaluate
        """
        all_psnr = []
        all_ssim = []

        with torch.no_grad():
            for batch_idx, (L, AB_real) in enumerate(dataloader):
                if batch_idx * L.size(0) >= num_samples:
                    break

                L = L.to(self.device)
                AB_pred = self.generator(L)

                for i in range(L.size(0)):
                    pred = AB_pred[i].cpu().numpy().transpose(1, 2, 0)
                    target = AB_real[i].numpy().transpose(1, 2, 0)

                    all_psnr.append(calculate_psnr(pred, target))
                    all_ssim.append(calculate_ssim(pred, target))

        # Create plots
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle('Evaluation Metrics Distribution', fontsize=14, fontweight='bold')

        # PSNR histogram
        axes[0].hist(all_psnr, bins=20, color='steelblue', edgecolor='black', alpha=0.7)
        axes[0].axvline(np.mean(all_psnr), color='red', linestyle='--',
                        label=f'Mean: {np.mean(all_psnr):.2f} dB')
        axes[0].set_xlabel('PSNR (dB)')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('PSNR Distribution')
        axes[0].legend()

        # SSIM histogram
        axes[1].hist(all_ssim, bins=20, color='coral', edgecolor='black', alpha=0.7)
        axes[1].axvline(np.mean(all_ssim), color='red', linestyle='--',
                        label=f'Mean: {np.mean(all_ssim):.3f}')
        axes[1].set_xlabel('SSIM')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('SSIM Distribution')
        axes[1].legend()

        plt.tight_layout()
        save_path = os.path.join(self.save_dir, 'metrics_distribution.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Metrics distribution saved to: {save_path}")

    @torch.no_grad()
    def visualize_samples(self, dataloader, num_samples=5):
        """
        Visualize sample predictions vs ground truth.

        Args:
            dataloader: DataLoader with test data
            num_samples: Number of samples to visualize
        """
        from ..preprocessing import lab2rgb

        samples_collected = 0
        L_samples = []
        AB_pred_samples = []
        AB_real_samples = []

        for L, AB_real in dataloader:
            L = L.to(self.device)
            AB_pred = self.generator(L)

            for i in range(L.size(0)):
                if samples_collected >= num_samples:
                    break

                L_samples.append(L[i].cpu())
                AB_pred_samples.append(AB_pred[i].cpu())
                AB_real_samples.append(AB_real[i])
                samples_collected += 1

            if samples_collected >= num_samples:
                break

        # Create visualization
        fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))
        fig.suptitle('Colorization Results: Grayscale | Predicted | Ground Truth',
                     fontsize=14, fontweight='bold')

        for i in range(num_samples):
            L_np = L_samples[i].numpy().transpose(1, 2, 0)
            AB_pred_np = AB_pred_samples[i].numpy().transpose(1, 2, 0)
            AB_real_np = AB_real_samples[i].numpy().transpose(1, 2, 0)

            # Denormalize
            L_denorm = (L_np + 1) * 50
            AB_pred_denorm = AB_pred_np * 110
            AB_real_denorm = AB_real_np * 110

            # Combine and convert to RGB
            lab_pred = np.concatenate([L_denorm, AB_pred_denorm], axis=-1)
            lab_real = np.concatenate([L_denorm, AB_real_denorm], axis=-1)

            try:
                rgb_pred = lab2rgb(lab_pred)
                rgb_real = lab2rgb(lab_real)
            except:
                # Fallback if lab2rgb not available
                rgb_pred = np.clip(lab_pred / 100, 0, 1)
                rgb_real = np.clip(lab_real / 100, 0, 1)

            # Plot grayscale
            axes[i, 0].imshow(L_np[:, :, 0], cmap='gray')
            axes[i, 0].set_title('Grayscale Input')
            axes[i, 0].axis('off')

            # Plot prediction
            axes[i, 1].imshow(np.clip(rgb_pred, 0, 1))
            axes[i, 1].set_title('Predicted Color')
            axes[i, 1].axis('off')

            # Plot ground truth
            axes[i, 2].imshow(np.clip(rgb_real, 0, 1))
            axes[i, 2].set_title('Ground Truth')
            axes[i, 2].axis('off')

        plt.tight_layout()
        save_path = os.path.join(self.save_dir, 'sample_predictions.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Sample predictions saved to: {save_path}")

    def generate_report(self, results):
        """
        Generate a text report of evaluation results.

        Args:
            results: Dictionary of evaluation results
        """
        report = []
        report.append("=" * 60)
        report.append("MODEL EVALUATION REPORT")
        report.append("=" * 60)
        report.append("")

        report.append("1. RECONSTRUCTION QUALITY (PSNR)")
        report.append("-" * 40)
        report.append(f"   Mean PSNR: {results['psnr']['mean']:.2f} dB")
        report.append(f"   Std Dev:   {results['psnr']['std']:.2f} dB")
        report.append(f"   Range:     [{results['psnr']['min']:.2f}, {results['psnr']['max']:.2f}] dB")
        report.append("")

        report.append("2. STRUCTURAL SIMILARITY (SSIM)")
        report.append("-" * 40)
        report.append(f"   Mean SSIM: {results['ssim']['mean']:.4f}")
        report.append(f"   Std Dev:   {results['ssim']['std']:.4f}")
        report.append(f"   Range:     [{results['ssim']['min']:.4f}, {results['ssim']['max']:.4f}]")
        report.append("")

        report.append("3. PIXEL-WISE ERROR")
        report.append("-" * 40)
        report.append(f"   Mean L1 Error: {results['l1_error']['mean']:.4f}")
        report.append(f"   Mean L2 Error: {results['l2_error']['mean']:.4f}")
        report.append("")

        report.append("4. COLORFULNESS")
        report.append("-" * 40)
        report.append(f"   Predicted:    {results['colorfulness']['predicted_mean']:.2f}")
        report.append(f"   Ground Truth: {results['colorfulness']['ground_truth_mean']:.2f}")
        report.append(f"   Ratio:        {results['colorfulness']['ratio']:.2f}")
        report.append("")

        report.append("5. EVALUATION SUMMARY")
        report.append("-" * 40)
        report.append(f"   Total Samples: {results['num_samples']}")
        report.append("")
        report.append("=" * 60)

        report_text = "\n".join(report)

        # Save report
        report_path = os.path.join(self.save_dir, 'evaluation_report.txt')
        with open(report_path, 'w') as f:
            f.write(report_text)

        print(report_text)
        print(f"\nReport saved to: {report_path}")

        return report_text


if __name__ == "__main__":
    # Example usage
    print("Evaluator module loaded successfully.")
    print("Available metrics: PSNR, SSIM, Colorfulness, L1/L2 Error")
