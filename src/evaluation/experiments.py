"""
Experiment Tracking and Comparison Module

This module provides utilities for:
1. Tracking multiple experiments
2. Comparing model performance
3. Generating comparison tables and plots
"""

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime


class ExperimentTracker:
    """
    Track and compare multiple experiments.

    Usage:
        tracker = ExperimentTracker(save_dir='results/experiments')
        tracker.log_experiment('baseline', config, metrics)
        tracker.log_experiment('improved', config2, metrics2)
        tracker.generate_comparison_report()
    """

    def __init__(self, save_dir='results/experiments'):
        self.save_dir = save_dir
        self.experiments = {}
        os.makedirs(save_dir, exist_ok=True)

        # Load existing experiments
        self._load_experiments()

    def _load_experiments(self):
        """Load existing experiment data."""
        exp_file = os.path.join(self.save_dir, 'experiments.json')
        if os.path.exists(exp_file):
            with open(exp_file, 'r') as f:
                self.experiments = json.load(f)

    def _save_experiments(self):
        """Save experiment data."""
        exp_file = os.path.join(self.save_dir, 'experiments.json')
        with open(exp_file, 'w') as f:
            json.dump(self.experiments, f, indent=2)

    def log_experiment(self, name, config, metrics, notes=""):
        """
        Log a new experiment.

        Args:
            name: Experiment name (e.g., 'baseline', 'lambda_50')
            config: Training configuration dict
            metrics: Evaluation metrics dict
            notes: Additional notes
        """
        self.experiments[name] = {
            'timestamp': datetime.now().isoformat(),
            'config': config,
            'metrics': metrics,
            'notes': notes
        }
        self._save_experiments()
        print(f"Logged experiment: {name}")

    def get_comparison_table(self):
        """
        Generate comparison table as DataFrame.

        Returns:
            pandas DataFrame with experiment comparison
        """
        if not self.experiments:
            return pd.DataFrame()

        rows = []
        for name, exp in self.experiments.items():
            row = {'Experiment': name}

            # Add config params
            if 'config' in exp:
                row['Batch Size'] = exp['config'].get('batch_size', '-')
                row['LR'] = exp['config'].get('lr_g', '-')
                row['L1 Lambda'] = exp['config'].get('l1_lambda', '-')
                row['Epochs'] = exp['config'].get('num_epochs', '-')

            # Add metrics
            if 'metrics' in exp:
                metrics = exp['metrics']
                if isinstance(metrics.get('psnr'), dict):
                    row['PSNR (dB)'] = f"{metrics['psnr']['mean']:.2f}"
                    row['SSIM'] = f"{metrics['ssim']['mean']:.4f}"
                else:
                    row['PSNR (dB)'] = f"{metrics.get('psnr', '-'):.2f}" if metrics.get('psnr') else '-'
                    row['SSIM'] = f"{metrics.get('ssim', '-'):.4f}" if metrics.get('ssim') else '-'

                row['L1 Error'] = f"{metrics.get('l1_error', {}).get('mean', '-'):.4f}" if isinstance(metrics.get('l1_error'), dict) else '-'

            rows.append(row)

        return pd.DataFrame(rows)

    def plot_comparison(self, metric='psnr', save_path=None):
        """
        Plot bar chart comparing experiments on a metric.

        Args:
            metric: Metric to compare ('psnr', 'ssim', 'l1_error')
            save_path: Path to save figure
        """
        if not self.experiments:
            print("No experiments to compare")
            return

        names = []
        values = []

        for name, exp in self.experiments.items():
            if 'metrics' in exp:
                metrics = exp['metrics']
                if isinstance(metrics.get(metric), dict):
                    val = metrics[metric]['mean']
                else:
                    val = metrics.get(metric, 0)

                if val:
                    names.append(name)
                    values.append(val)

        if not names:
            print(f"No data for metric: {metric}")
            return

        fig, ax = plt.subplots(figsize=(10, 6))

        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(names)))
        bars = ax.bar(names, values, color=colors, edgecolor='black')

        ax.set_xlabel('Experiment')
        ax.set_ylabel(metric.upper())
        ax.set_title(f'Experiment Comparison: {metric.upper()}')

        # Add value labels on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.annotate(f'{val:.2f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom')

        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved comparison plot to: {save_path}")

        plt.close()

    def plot_improvement_comparison(self, baseline_name, improved_name, save_path=None):
        """
        Plot before/after comparison between baseline and improved model.

        Args:
            baseline_name: Name of baseline experiment
            improved_name: Name of improved experiment
            save_path: Path to save figure
        """
        if baseline_name not in self.experiments or improved_name not in self.experiments:
            print("Experiments not found")
            return

        baseline = self.experiments[baseline_name]['metrics']
        improved = self.experiments[improved_name]['metrics']

        metrics = ['psnr', 'ssim']
        baseline_values = []
        improved_values = []

        for m in metrics:
            if isinstance(baseline.get(m), dict):
                baseline_values.append(baseline[m]['mean'])
                improved_values.append(improved[m]['mean'])
            else:
                baseline_values.append(baseline.get(m, 0))
                improved_values.append(improved.get(m, 0))

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle('Model Improvement: Before vs After', fontsize=14, fontweight='bold')

        for idx, (metric, base_val, imp_val) in enumerate(zip(metrics, baseline_values, improved_values)):
            ax = axes[idx]

            x = ['Baseline', 'Improved']
            y = [base_val, imp_val]
            colors = ['#ff6b6b', '#4ecdc4']

            bars = ax.bar(x, y, color=colors, edgecolor='black')

            # Calculate improvement
            if base_val > 0:
                improvement = ((imp_val - base_val) / base_val) * 100
                ax.set_title(f'{metric.upper()}\n(+{improvement:.1f}% improvement)')
            else:
                ax.set_title(metric.upper())

            ax.set_ylabel(metric.upper())

            # Add value labels
            for bar, val in zip(bars, y):
                height = bar.get_height()
                ax.annotate(f'{val:.2f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=12)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved improvement comparison to: {save_path}")

        plt.close()

    def generate_comparison_report(self, save_path=None):
        """
        Generate comprehensive comparison report.

        Args:
            save_path: Path to save report (markdown)
        """
        if not self.experiments:
            print("No experiments to report")
            return

        report = []
        report.append("# Experiment Comparison Report")
        report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        report.append("---\n")

        # Comparison table
        report.append("## Performance Comparison\n")
        df = self.get_comparison_table()
        try:
            report.append(df.to_markdown(index=False))
        except ImportError:
            # Fallback if tabulate not installed
            report.append("```")
            report.append(df.to_string(index=False))
            report.append("```")
        report.append("\n")

        # Best model
        report.append("## Best Model\n")
        best_psnr = 0
        best_exp = None
        for name, exp in self.experiments.items():
            if 'metrics' in exp:
                psnr = exp['metrics'].get('psnr', {})
                if isinstance(psnr, dict):
                    val = psnr.get('mean', 0)
                else:
                    val = psnr or 0
                if val > best_psnr:
                    best_psnr = val
                    best_exp = name

        if best_exp:
            report.append(f"- **Best Experiment:** {best_exp}")
            report.append(f"- **PSNR:** {best_psnr:.2f} dB\n")

        # Individual experiments
        report.append("## Experiment Details\n")
        for name, exp in self.experiments.items():
            report.append(f"### {name}\n")
            report.append(f"- **Timestamp:** {exp.get('timestamp', 'N/A')}")
            if exp.get('notes'):
                report.append(f"- **Notes:** {exp['notes']}")
            report.append("\n")

        report_text = "\n".join(report)

        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
            print(f"Saved report to: {save_path}")

        return report_text


def create_hyperparameter_table():
    """
    Create a hyperparameter comparison table for documentation.
    """
    data = {
        'Parameter': ['Batch Size', 'Learning Rate', 'L1 Lambda', 'Epochs', 'Optimizer', 'Beta1', 'Beta2'],
        'Values Tested': ['8, 16, 32', '1e-4, 2e-4, 5e-4', '50, 100, 150', '20, 50, 100', 'Adam, AdamW', '0.5, 0.9', '0.999'],
        'Best Value': ['16', '2e-4', '100', '50', 'Adam', '0.5', '0.999'],
        'Notes': [
            'Larger batches need more memory',
            'Higher LR caused instability',
            'Balance between accuracy and colorfulness',
            '50 epochs sufficient for convergence',
            'Adam with momentum 0.5 works best for GANs',
            'Lower momentum for GANs',
            'Standard value'
        ]
    }
    return pd.DataFrame(data)


if __name__ == "__main__":
    # Example usage
    tracker = ExperimentTracker()

    # Log sample experiments
    tracker.log_experiment(
        'baseline',
        config={'batch_size': 16, 'lr_g': 2e-4, 'l1_lambda': 100, 'num_epochs': 50},
        metrics={'psnr': {'mean': 24.12}, 'ssim': {'mean': 0.80}, 'l1_error': {'mean': 0.085}},
        notes='Baseline pix2pix model'
    )

    tracker.log_experiment(
        'lambda_50',
        config={'batch_size': 16, 'lr_g': 2e-4, 'l1_lambda': 50, 'num_epochs': 50},
        metrics={'psnr': {'mean': 22.14}, 'ssim': {'mean': 0.78}, 'l1_error': {'mean': 0.095}},
        notes='Reduced L1 weight for more colorful output'
    )

    tracker.log_experiment(
        'lr_decay',
        config={'batch_size': 16, 'lr_g': 2e-4, 'l1_lambda': 100, 'num_epochs': 50},
        metrics={'psnr': {'mean': 25.12}, 'ssim': {'mean': 0.84}, 'l1_error': {'mean': 0.078}},
        notes='Added StepLR scheduler'
    )

    # Generate comparison
    print("\nComparison Table:")
    print(tracker.get_comparison_table().to_string())

    # Generate plots
    tracker.plot_comparison('psnr', 'results/experiments/psnr_comparison.png')
    tracker.plot_improvement_comparison('baseline', 'lr_decay', 'results/plots/improvement_comparison.png')

    # Generate report
    tracker.generate_comparison_report('results/experiments/comparison_report.md')

    print("\nHyperparameter Table:")
    print(create_hyperparameter_table().to_string())
