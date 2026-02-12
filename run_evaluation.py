"""
Model Evaluation Script for Image Colorization GAN

This script evaluates the trained model using scientific metrics.

Usage:
    python run_evaluation.py --model trained_models/generator_final.pth

    # Evaluate with specific number of samples
    python run_evaluation.py --model trained_models/generator_final.pth --samples 100
"""

import argparse
import os
import torch
import json

from src.models import UNetGenerator
from src.preprocessing import create_dataloaders
from src.evaluation import Evaluator


def get_device():
    """Auto-detect best available device."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate Image Colorization Model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--model', type=str, default='trained_models/generator_final.pth',
                        help='Path to trained generator model')
    parser.add_argument('--data_dir', type=str, default='data/train',
                        help='Directory containing test images')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for evaluation')
    parser.add_argument('--samples', type=int, default=None,
                        help='Number of samples to evaluate (None = all)')
    parser.add_argument('--save_dir', type=str, default='results/evaluation',
                        help='Directory to save evaluation results')
    parser.add_argument('--visualize', action='store_true',
                        help='Generate visualization plots')

    args = parser.parse_args()

    # Device
    device = get_device()
    print(f"\n{'='*60}")
    print("IMAGE COLORIZATION GAN - EVALUATION")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Model: {args.model}")
    print(f"{'='*60}\n")

    # Load model
    generator = UNetGenerator(in_channels=1, out_channels=2, features=64).to(device)

    if os.path.exists(args.model):
        state_dict = torch.load(args.model, map_location=device)
        # Handle checkpoint vs state_dict
        if 'generator_state_dict' in state_dict:
            generator.load_state_dict(state_dict['generator_state_dict'])
        else:
            generator.load_state_dict(state_dict)
        print(f"Loaded model from: {args.model}")
    else:
        print(f"ERROR: Model not found at {args.model}")
        return

    generator.eval()

    # Load data
    print("\nLoading test data...")
    _, _, test_loader = create_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=256,
        val_split=0.1,
        test_split=0.1,
        num_workers=4
    )

    # Initialize evaluator
    evaluator = Evaluator(
        generator=generator,
        device=device,
        save_dir=args.save_dir
    )

    # Run evaluation
    print("\nRunning evaluation...")
    results = evaluator.evaluate_dataset(test_loader, num_samples=args.samples)

    # Generate report
    evaluator.generate_report(results)
    evaluator.save_results(results)

    # Generate visualizations
    if args.visualize:
        print("\nGenerating visualizations...")
        evaluator.plot_metrics_distribution(test_loader, num_samples=50)
        evaluator.visualize_samples(test_loader, num_samples=5)

    print(f"\n{'='*60}")
    print("EVALUATION COMPLETE")
    print(f"{'='*60}")
    print(f"Results saved to: {args.save_dir}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
