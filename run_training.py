"""
Main Training Script for Image Colorization GAN

This script provides a command-line interface for training the colorization model
using the modular project structure.

Usage:
    python run_training.py --epochs 50 --batch_size 16 --lr 0.0002

    # Resume from checkpoint
    python run_training.py --resume results/checkpoints/latest.pth

    # Quick test run
    python run_training.py --epochs 5 --batch_size 8
"""

import argparse
import os
import torch
import json
from datetime import datetime

# Import from modular structure
from src.models import UNetGenerator, PatchDiscriminator
from src.preprocessing import create_dataloaders
from src.training import Trainer


def get_device():
    """Auto-detect best available device."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    parser = argparse.ArgumentParser(
        description='Train Image Colorization GAN',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Data arguments
    parser.add_argument('--data_dir', type=str, default='data/train',
                        help='Directory containing training images')
    parser.add_argument('--image_size', type=int, default=256,
                        help='Input image size')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Training batch size')
    parser.add_argument('--val_split', type=float, default=0.1,
                        help='Validation split ratio')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')

    # Training arguments
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=2e-4,
                        help='Learning rate for both G and D')
    parser.add_argument('--lr_g', type=float, default=None,
                        help='Learning rate for generator (overrides --lr)')
    parser.add_argument('--lr_d', type=float, default=None,
                        help='Learning rate for discriminator (overrides --lr)')
    parser.add_argument('--beta1', type=float, default=0.5,
                        help='Adam beta1 parameter')
    parser.add_argument('--beta2', type=float, default=0.999,
                        help='Adam beta2 parameter')
    parser.add_argument('--l1_lambda', type=float, default=100.0,
                        help='Weight for L1 loss')

    # Model arguments
    parser.add_argument('--features', type=int, default=64,
                        help='Base number of features in generator')

    # Checkpoint arguments
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume training')
    parser.add_argument('--save_dir', type=str, default='results',
                        help='Directory to save results')

    args = parser.parse_args()

    # Device
    device = get_device()
    print(f"\n{'='*60}")
    print("IMAGE COLORIZATION GAN - TRAINING")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")

    # Configuration
    config = {
        'data_dir': args.data_dir,
        'image_size': args.image_size,
        'batch_size': args.batch_size,
        'val_split': args.val_split,
        'num_workers': args.num_workers,
        'lr_g': args.lr_g if args.lr_g else args.lr,
        'lr_d': args.lr_d if args.lr_d else args.lr,
        'beta1': args.beta1,
        'beta2': args.beta2,
        'l1_lambda': args.l1_lambda,
    }

    # Save config
    os.makedirs(args.save_dir, exist_ok=True)
    with open(os.path.join(args.save_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    # Print configuration
    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()

    # Create data loaders
    print("Loading data...")
    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir=config['data_dir'],
        batch_size=config['batch_size'],
        image_size=config['image_size'],
        val_split=config['val_split'],
        test_split=0.1,
        num_workers=config['num_workers']
    )

    # Initialize models
    print("\nInitializing models...")
    generator = UNetGenerator(
        in_channels=1,
        out_channels=2,
        features=args.features
    ).to(device)

    discriminator = PatchDiscriminator(
        in_channels=3,
        features=args.features
    ).to(device)

    print(f"Generator parameters: {count_parameters(generator):,}")
    print(f"Discriminator parameters: {count_parameters(discriminator):,}")

    # Initialize trainer
    trainer = Trainer(
        generator=generator,
        discriminator=discriminator,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        config=config,
        save_dir=args.save_dir
    )

    # Train
    print(f"\nStarting training for {args.epochs} epochs...")
    history = trainer.train(
        num_epochs=args.epochs,
        resume_path=args.resume
    )

    # Save final models
    os.makedirs('trained_models', exist_ok=True)
    torch.save(generator.state_dict(), 'trained_models/generator_final.pth')
    torch.save(discriminator.state_dict(), 'trained_models/discriminator_final.pth')

    print(f"\n{'='*60}")
    print("TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"Final Generator Loss: {history['train_g_loss'][-1]:.4f}")
    print(f"Final Discriminator Loss: {history['train_d_loss'][-1]:.4f}")
    print(f"Best Validation Loss: {min(history['val_g_loss']):.4f}")
    print(f"\nModels saved to: trained_models/")
    print(f"Checkpoints saved to: {args.save_dir}/checkpoints/")
    print(f"Plots saved to: {args.save_dir}/plots/")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
