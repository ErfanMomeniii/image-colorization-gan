"""
Synthetic Dataset Generator for Image Colorization GAN

This script generates synthetic color images for training the colorization model.
The images contain random gradients and geometric shapes with various colors.

Usage:
    python generate_dataset.py --output data/train --num_images 100
"""

import os
import argparse
import numpy as np
from PIL import Image


def generate_gradient_background(size=256):
    """
    Generate a random gradient background.

    Args:
        size: Image size (width and height)

    Returns:
        numpy array of shape (size, size, 3) with RGB values
    """
    img = np.zeros((size, size, 3), dtype=np.uint8)

    # Random color intensities for gradients
    r_intensity = np.random.randint(100, 255)
    g_intensity = np.random.randint(100, 255)
    b_intensity = np.random.randint(100, 255)

    for y in range(size):
        for x in range(size):
            # Horizontal gradient for R channel
            img[y, x, 0] = int((x / size) * r_intensity)
            # Vertical gradient for G channel
            img[y, x, 1] = int((y / size) * g_intensity)
            # Diagonal gradient for B channel
            img[y, x, 2] = int(((x + y) / (2 * size)) * b_intensity)

    return img


def add_circle(img, cx, cy, radius, color):
    """
    Add a filled circle to the image.

    Args:
        img: numpy array of shape (H, W, 3)
        cx, cy: Center coordinates
        radius: Circle radius
        color: RGB color tuple

    Returns:
        Modified image array
    """
    size = img.shape[0]
    for y in range(max(0, cy - radius), min(size, cy + radius)):
        for x in range(max(0, cx - radius), min(size, cx + radius)):
            if (x - cx) ** 2 + (y - cy) ** 2 < radius ** 2:
                img[y, x] = color
    return img


def add_rectangle(img, x1, y1, x2, y2, color):
    """
    Add a filled rectangle to the image.

    Args:
        img: numpy array of shape (H, W, 3)
        x1, y1: Top-left corner
        x2, y2: Bottom-right corner
        color: RGB color tuple

    Returns:
        Modified image array
    """
    size = img.shape[0]
    x1, x2 = max(0, x1), min(size, x2)
    y1, y2 = max(0, y1), min(size, y2)
    img[y1:y2, x1:x2] = color
    return img


def generate_synthetic_image(size=256):
    """
    Generate a single synthetic training image.

    The image contains:
    1. Random gradient background
    2. 1-3 random geometric shapes (circles or rectangles)

    Args:
        size: Image size (width and height)

    Returns:
        numpy array of shape (size, size, 3) with RGB values
    """
    # Start with gradient background
    img = generate_gradient_background(size)

    # Add 1-3 random shapes
    num_shapes = np.random.randint(1, 4)

    for _ in range(num_shapes):
        # Random color for shape
        color = np.random.randint(0, 255, 3)

        # Randomly choose shape type
        shape_type = np.random.choice(['circle', 'rectangle'])

        if shape_type == 'circle':
            # Random circle parameters
            cx = np.random.randint(50, size - 50)
            cy = np.random.randint(50, size - 50)
            radius = np.random.randint(20, 60)
            img = add_circle(img, cx, cy, radius, color)
        else:
            # Random rectangle parameters
            x1 = np.random.randint(20, size - 80)
            y1 = np.random.randint(20, size - 80)
            width = np.random.randint(30, 80)
            height = np.random.randint(30, 80)
            img = add_rectangle(img, x1, y1, x1 + width, y1 + height, color)

    return img


def generate_dataset(output_dir, num_images, size=256):
    """
    Generate a complete synthetic dataset.

    Args:
        output_dir: Directory to save images
        num_images: Number of images to generate
        size: Image size (width and height)
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    print(f"Generating {num_images} synthetic images...")
    print(f"Output directory: {output_dir}")
    print(f"Image size: {size}x{size}")
    print("-" * 50)

    # Set random seed for reproducibility
    np.random.seed(42)

    for i in range(num_images):
        # Generate image
        img = generate_synthetic_image(size)

        # Save image
        filename = f"synthetic_{i:04d}.jpg"
        filepath = os.path.join(output_dir, filename)
        Image.fromarray(img).save(filepath, quality=95)

        # Progress update
        if (i + 1) % 10 == 0 or i == 0:
            print(f"Generated: {i + 1}/{num_images} images")

    print("-" * 50)
    print(f"Dataset generation complete!")
    print(f"Total images: {num_images}")
    print(f"Location: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate synthetic dataset for image colorization'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/train',
        help='Output directory for generated images'
    )
    parser.add_argument(
        '--num_images',
        type=int,
        default=100,
        help='Number of images to generate'
    )
    parser.add_argument(
        '--size',
        type=int,
        default=256,
        help='Image size (width and height)'
    )

    args = parser.parse_args()

    generate_dataset(args.output, args.num_images, args.size)


if __name__ == "__main__":
    main()
