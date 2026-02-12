"""
Training Module for Image Colorization GAN

This module handles:
1. Model training loop
2. Loss logging and visualization
3. Checkpoint saving
4. Validation during training
5. Learning rate scheduling
"""

import os
import json
import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime


class Trainer:
    """
    Trainer class for Image Colorization GAN.

    Handles training loop, logging, checkpointing, and validation.
    """

    def __init__(self, generator, discriminator, train_loader, val_loader,
                 device, config, save_dir='results'):
        """
        Initialize trainer.

        Args:
            generator: Generator model (U-Net)
            discriminator: Discriminator model (PatchGAN)
            train_loader: Training data loader
            val_loader: Validation data loader
            device: Training device (cuda/cpu)
            config: Training configuration dict
            save_dir: Directory to save results
        """
        self.generator = generator
        self.discriminator = discriminator
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config
        self.save_dir = save_dir

        # Create directories
        os.makedirs(os.path.join(save_dir, 'checkpoints'), exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'samples'), exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'plots'), exist_ok=True)

        # Initialize optimizers
        self.opt_g = torch.optim.Adam(
            generator.parameters(),
            lr=config['lr_g'],
            betas=(config['beta1'], config['beta2'])
        )
        self.opt_d = torch.optim.Adam(
            discriminator.parameters(),
            lr=config['lr_d'],
            betas=(config['beta1'], config['beta2'])
        )

        # Learning rate schedulers
        self.scheduler_g = torch.optim.lr_scheduler.StepLR(
            self.opt_g, step_size=30, gamma=0.5
        )
        self.scheduler_d = torch.optim.lr_scheduler.StepLR(
            self.opt_d, step_size=30, gamma=0.5
        )

        # Loss functions
        self.criterion_gan = nn.BCEWithLogitsLoss()
        self.criterion_l1 = nn.L1Loss()

        # Training history
        self.history = {
            'train_g_loss': [],
            'train_d_loss': [],
            'val_g_loss': [],
            'val_d_loss': [],
            'train_l1_loss': [],
            'train_gan_loss': [],
            'lr_g': [],
            'lr_d': []
        }

        self.start_epoch = 0

    def train_epoch(self, epoch):
        """Train for one epoch."""
        self.generator.train()
        self.discriminator.train()

        total_g_loss = 0
        total_d_loss = 0
        total_l1_loss = 0
        total_gan_loss = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}")

        for batch_idx, (L, AB_real) in enumerate(pbar):
            L = L.to(self.device)
            AB_real = AB_real.to(self.device)
            batch_size = L.size(0)

            # ---------------------
            # Train Discriminator
            # ---------------------
            self.discriminator.zero_grad()

            # Real samples
            pred_real = self.discriminator(L, AB_real)
            label_real = torch.ones_like(pred_real, device=self.device)
            loss_d_real = self.criterion_gan(pred_real, label_real)

            # Fake samples
            AB_fake = self.generator(L)
            pred_fake = self.discriminator(L, AB_fake.detach())
            label_fake = torch.zeros_like(pred_fake, device=self.device)
            loss_d_fake = self.criterion_gan(pred_fake, label_fake)

            # Total discriminator loss
            loss_d = (loss_d_real + loss_d_fake) * 0.5
            loss_d.backward()
            self.opt_d.step()

            # ---------------------
            # Train Generator
            # ---------------------
            self.generator.zero_grad()

            # GAN loss
            pred_fake = self.discriminator(L, AB_fake)
            loss_g_gan = self.criterion_gan(pred_fake, label_real)

            # L1 loss
            loss_g_l1 = self.criterion_l1(AB_fake, AB_real) * self.config['l1_lambda']

            # Total generator loss
            loss_g = loss_g_gan + loss_g_l1
            loss_g.backward()
            self.opt_g.step()

            # Update metrics
            total_g_loss += loss_g.item()
            total_d_loss += loss_d.item()
            total_l1_loss += loss_g_l1.item()
            total_gan_loss += loss_g_gan.item()

            # Update progress bar
            pbar.set_postfix({
                'G': f'{loss_g.item():.3f}',
                'D': f'{loss_d.item():.3f}',
                'L1': f'{loss_g_l1.item():.3f}'
            })

        # Calculate averages
        num_batches = len(self.train_loader)
        avg_g_loss = total_g_loss / num_batches
        avg_d_loss = total_d_loss / num_batches
        avg_l1_loss = total_l1_loss / num_batches
        avg_gan_loss = total_gan_loss / num_batches

        return avg_g_loss, avg_d_loss, avg_l1_loss, avg_gan_loss

    @torch.no_grad()
    def validate(self):
        """Validate the model."""
        self.generator.eval()
        self.discriminator.eval()

        total_g_loss = 0
        total_d_loss = 0

        for L, AB_real in self.val_loader:
            L = L.to(self.device)
            AB_real = AB_real.to(self.device)

            # Generate fake samples
            AB_fake = self.generator(L)

            # Discriminator loss
            pred_real = self.discriminator(L, AB_real)
            pred_fake = self.discriminator(L, AB_fake)
            label_real = torch.ones_like(pred_real, device=self.device)
            label_fake = torch.zeros_like(pred_fake, device=self.device)

            loss_d = 0.5 * (
                self.criterion_gan(pred_real, label_real) +
                self.criterion_gan(pred_fake, label_fake)
            )

            # Generator loss
            loss_g_gan = self.criterion_gan(pred_fake, label_real)
            loss_g_l1 = self.criterion_l1(AB_fake, AB_real) * self.config['l1_lambda']
            loss_g = loss_g_gan + loss_g_l1

            total_g_loss += loss_g.item()
            total_d_loss += loss_d.item()

        num_batches = len(self.val_loader)
        return total_g_loss / num_batches, total_d_loss / num_batches

    def save_checkpoint(self, epoch, path):
        """Save training checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'optimizer_g_state_dict': self.opt_g.state_dict(),
            'optimizer_d_state_dict': self.opt_d.state_dict(),
            'scheduler_g_state_dict': self.scheduler_g.state_dict(),
            'scheduler_d_state_dict': self.scheduler_d.state_dict(),
            'history': self.history,
            'config': self.config
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path):
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.opt_g.load_state_dict(checkpoint['optimizer_g_state_dict'])
        self.opt_d.load_state_dict(checkpoint['optimizer_d_state_dict'])
        self.scheduler_g.load_state_dict(checkpoint['scheduler_g_state_dict'])
        self.scheduler_d.load_state_dict(checkpoint['scheduler_d_state_dict'])
        self.history = checkpoint['history']
        self.start_epoch = checkpoint['epoch']
        print(f"Loaded checkpoint from epoch {self.start_epoch}")

    def plot_losses(self, save_path=None):
        """Plot training and validation losses."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        epochs = range(1, len(self.history['train_g_loss']) + 1)

        # Generator Loss
        axes[0, 0].plot(epochs, self.history['train_g_loss'], 'b-', label='Train')
        axes[0, 0].plot(epochs, self.history['val_g_loss'], 'r--', label='Validation')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Generator Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # Discriminator Loss
        axes[0, 1].plot(epochs, self.history['train_d_loss'], 'b-', label='Train')
        axes[0, 1].plot(epochs, self.history['val_d_loss'], 'r--', label='Validation')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].set_title('Discriminator Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        # L1 and GAN Loss
        axes[1, 0].plot(epochs, self.history['train_l1_loss'], 'g-', label='L1 Loss')
        axes[1, 0].plot(epochs, self.history['train_gan_loss'], 'm-', label='GAN Loss')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].set_title('L1 vs GAN Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(True)

        # Learning Rate
        axes[1, 1].plot(epochs, self.history['lr_g'], 'b-', label='Generator LR')
        axes[1, 1].plot(epochs, self.history['lr_d'], 'r-', label='Discriminator LR')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_title('Learning Rate Schedule')
        axes[1, 1].legend()
        axes[1, 1].grid(True)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

    def train(self, num_epochs, resume_path=None):
        """
        Full training loop.

        Args:
            num_epochs: Number of epochs to train
            resume_path: Path to checkpoint to resume from
        """
        if resume_path and os.path.exists(resume_path):
            self.load_checkpoint(resume_path)

        print("=" * 60)
        print("TRAINING CONFIGURATION")
        print("=" * 60)
        print(f"Device: {self.device}")
        print(f"Epochs: {num_epochs}")
        print(f"Batch Size: {self.config['batch_size']}")
        print(f"Learning Rate (G): {self.config['lr_g']}")
        print(f"Learning Rate (D): {self.config['lr_d']}")
        print(f"L1 Lambda: {self.config['l1_lambda']}")
        print("=" * 60)

        best_val_loss = float('inf')

        for epoch in range(self.start_epoch, num_epochs):
            # Train
            train_g, train_d, train_l1, train_gan = self.train_epoch(epoch)

            # Validate
            val_g, val_d = self.validate()

            # Update learning rate
            self.scheduler_g.step()
            self.scheduler_d.step()

            # Record history
            self.history['train_g_loss'].append(train_g)
            self.history['train_d_loss'].append(train_d)
            self.history['val_g_loss'].append(val_g)
            self.history['val_d_loss'].append(val_d)
            self.history['train_l1_loss'].append(train_l1)
            self.history['train_gan_loss'].append(train_gan)
            self.history['lr_g'].append(self.opt_g.param_groups[0]['lr'])
            self.history['lr_d'].append(self.opt_d.param_groups[0]['lr'])

            # Print epoch summary
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print(f"  Train - G: {train_g:.4f}, D: {train_d:.4f}, L1: {train_l1:.4f}")
            print(f"  Val   - G: {val_g:.4f}, D: {val_d:.4f}")

            # Save best model
            if val_g < best_val_loss:
                best_val_loss = val_g
                self.save_checkpoint(
                    epoch + 1,
                    os.path.join(self.save_dir, 'checkpoints', 'best_model.pth')
                )
                print(f"  Saved best model (val_g_loss: {val_g:.4f})")

            # Save periodic checkpoint
            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(
                    epoch + 1,
                    os.path.join(self.save_dir, 'checkpoints', f'checkpoint_epoch_{epoch + 1}.pth')
                )

            # Save latest checkpoint
            self.save_checkpoint(
                epoch + 1,
                os.path.join(self.save_dir, 'checkpoints', 'latest.pth')
            )

            # Plot losses
            self.plot_losses(os.path.join(self.save_dir, 'plots', 'training_curves.png'))

        # Save final model
        torch.save(
            self.generator.state_dict(),
            os.path.join(self.save_dir, 'checkpoints', 'generator_final.pth')
        )

        # Save training history
        with open(os.path.join(self.save_dir, 'training_history.json'), 'w') as f:
            json.dump(self.history, f, indent=2)

        print("\n" + "=" * 60)
        print("TRAINING COMPLETE")
        print("=" * 60)

        return self.history
