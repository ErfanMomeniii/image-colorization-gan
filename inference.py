import os
import argparse
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import config
from color_utils import rgb2lab
from models import UNetGenerator
from utils import lab_to_rgb


def load_and_preprocess_image(image_path, image_size=256):
    """Load an image and convert to L channel tensor."""
    img = Image.open(image_path).convert('RGB')
    img = img.resize((image_size, image_size), Image.LANCZOS)
    img = np.array(img)

    # Convert to LAB
    lab = rgb2lab(img).astype(np.float32)

    # Get L channel and normalize
    L = lab[:, :, 0:1]
    L = (L / 50.0) - 1.0

    # Convert to tensor
    L = torch.from_numpy(L.transpose(2, 0, 1)).unsqueeze(0)

    return L, img


def colorize_image(generator, image_path, device, image_size=256):
    """Colorize a single grayscale image."""
    L, original = load_and_preprocess_image(image_path, image_size)
    L = L.to(device)

    generator.eval()
    with torch.no_grad():
        AB_fake = generator(L)

    # Convert to RGB
    rgb = lab_to_rgb(L, AB_fake)[0]

    return rgb, original


def main():
    parser = argparse.ArgumentParser(description='Colorize grayscale images using trained GAN')
    parser.add_argument('--input', type=str, required=True, help='Input image path or directory')
    parser.add_argument('--output', type=str, default='output', help='Output directory')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/latest.pth',
                        help='Path to model checkpoint')
    parser.add_argument('--image_size', type=int, default=256, help='Image size')
    parser.add_argument('--device', type=str, default=config.DEVICE, help='Device (cuda/mps/cpu)')
    parser.add_argument('--show', action='store_true', help='Show results instead of saving')
    args = parser.parse_args()

    # Setup device
    if args.device == 'cuda' and not torch.cuda.is_available():
        args.device = 'cpu'
    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Load model
    generator = UNetGenerator(in_channels=1, out_channels=2).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    generator.load_state_dict(checkpoint['generator_state_dict'])
    print(f"Loaded checkpoint from {args.checkpoint}")

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Get input images
    if os.path.isfile(args.input):
        image_paths = [args.input]
    else:
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        image_paths = [
            os.path.join(args.input, f) for f in os.listdir(args.input)
            if os.path.splitext(f)[1].lower() in valid_extensions
        ]

    print(f"Processing {len(image_paths)} images...")

    for image_path in image_paths:
        print(f"Colorizing: {image_path}")
        colorized, original = colorize_image(generator, image_path, device, args.image_size)

        if args.show:
            # Display results
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            axes[0].imshow(original)
            axes[0].set_title('Original')
            axes[0].axis('off')
            axes[1].imshow(colorized)
            axes[1].set_title('Colorized')
            axes[1].axis('off')
            plt.tight_layout()
            plt.show()
        else:
            # Save result
            filename = os.path.basename(image_path)
            name, ext = os.path.splitext(filename)
            output_path = os.path.join(args.output, f"{name}_colorized.png")

            # Convert to uint8 and save
            colorized_uint8 = (colorized * 255).astype(np.uint8)
            Image.fromarray(colorized_uint8).save(output_path)
            print(f"Saved: {output_path}")

    print("Done!")


if __name__ == "__main__":
    main()
