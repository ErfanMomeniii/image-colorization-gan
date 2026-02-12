"""
Gradio Web UI for Image Colorization GAN
"""
import os
import torch
import numpy as np
from PIL import Image
import gradio as gr

import config
from src.models import UNetGenerator
from src.utils.color_conversion import rgb2lab, lab2rgb


def load_model(checkpoint_path=None):
    """Load the trained generator model."""
    device = torch.device(config.DEVICE)
    generator = UNetGenerator(in_channels=1, out_channels=2).to(device)

    # Try to load checkpoint
    if checkpoint_path is None:
        # Try trained_models first, then checkpoints
        paths_to_try = [
            os.path.join(config.TRAINED_MODELS_DIR, 'best_model.pth'),
            os.path.join(config.TRAINED_MODELS_DIR, 'generator_final.pth'),
            os.path.join(config.TRAINED_MODELS_DIR, 'latest.pth'),
            os.path.join(config.CHECKPOINT_DIR, 'best_model.pth'),
            os.path.join(config.CHECKPOINT_DIR, 'latest.pth'),
        ]
        for path in paths_to_try:
            if os.path.exists(path):
                checkpoint_path = path
                break

    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        if 'generator_state_dict' in checkpoint:
            generator.load_state_dict(checkpoint['generator_state_dict'])
        else:
            generator.load_state_dict(checkpoint)
        print(f"Loaded model from {checkpoint_path}")
    else:
        print("Warning: No trained model found. Using untrained model.")

    generator.eval()
    return generator, device


def colorize_image(input_image, model_info):
    """Colorize a grayscale or color image."""
    generator, device = model_info

    if input_image is None:
        return None

    # Convert to PIL Image if needed
    if isinstance(input_image, np.ndarray):
        input_image = Image.fromarray(input_image)

    # Resize to model input size
    original_size = input_image.size
    input_image = input_image.convert('RGB')
    input_image = input_image.resize((config.IMAGE_SIZE, config.IMAGE_SIZE), Image.LANCZOS)
    img_array = np.array(input_image)

    # Convert to LAB and extract L channel
    lab = rgb2lab(img_array)
    L = lab[:, :, 0:1]
    L_normalized = (L / 50.0) - 1.0

    # Convert to tensor
    L_tensor = torch.from_numpy(L_normalized.transpose(2, 0, 1)).unsqueeze(0).float().to(device)

    # Generate colorization
    with torch.no_grad():
        AB_fake = generator(L_tensor)

    # Convert back to RGB
    AB_fake = AB_fake.cpu().numpy()[0]

    # Denormalize
    L_out = (L_normalized[:, :, 0] + 1.0) * 50.0
    AB_out = AB_fake * 110.0

    # Combine LAB
    lab_out = np.zeros((config.IMAGE_SIZE, config.IMAGE_SIZE, 3), dtype=np.float32)
    lab_out[:, :, 0] = L_out
    lab_out[:, :, 1] = AB_out[0]
    lab_out[:, :, 2] = AB_out[1]

    # Convert to RGB
    rgb_out = lab2rgb(lab_out)
    rgb_out = (rgb_out * 255).astype(np.uint8)

    # Resize back to original size
    result = Image.fromarray(rgb_out)
    result = result.resize(original_size, Image.LANCZOS)

    return np.array(result)


def create_grayscale(input_image):
    """Convert image to grayscale for preview."""
    if input_image is None:
        return None

    if isinstance(input_image, np.ndarray):
        input_image = Image.fromarray(input_image)

    grayscale = input_image.convert('L').convert('RGB')
    return np.array(grayscale)


def create_ui():
    """Create the Gradio interface."""
    # Load model once at startup
    print("Loading model...")
    model_info = load_model()
    print(f"Model loaded on device: {model_info[1]}")

    with gr.Blocks(title="Image Colorization GAN", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # Image Colorization GAN

        Transform grayscale images into colorized versions using a trained GAN model.

        **How to use:**
        1. Upload an image (color or grayscale)
        2. Click "Colorize" to generate the colorized version
        3. Download the result if you like it
        """)

        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(
                    label="Input Image",
                    type="numpy",
                    sources=["upload", "clipboard"],
                )

                with gr.Row():
                    colorize_btn = gr.Button("Colorize", variant="primary", size="lg")
                    grayscale_btn = gr.Button("Show Grayscale", size="lg")

            with gr.Column(scale=1):
                output_image = gr.Image(
                    label="Output Image",
                    type="numpy",
                )

        with gr.Row():
            gr.Markdown("""
            ### Tips:
            - Works best with **256x256** images (automatically resized)
            - Upload any image - the model extracts grayscale and predicts colors
            - Results depend on training data - some subjects colorize better than others
            """)

        # Example images section
        gr.Markdown("### Examples")
        gr.Examples(
            examples=[
                ["data/train/synthetic_000.jpg"],
                ["data/train/synthetic_010.jpg"],
                ["data/train/synthetic_020.jpg"],
            ] if os.path.exists("data/train/synthetic_000.jpg") else [],
            inputs=input_image,
            outputs=output_image,
            fn=lambda x: colorize_image(x, model_info),
            cache_examples=False,
        )

        # Event handlers
        colorize_btn.click(
            fn=lambda x: colorize_image(x, model_info),
            inputs=input_image,
            outputs=output_image,
        )

        grayscale_btn.click(
            fn=create_grayscale,
            inputs=input_image,
            outputs=output_image,
        )

        gr.Markdown("""
        ---
        **Model Architecture:** U-Net Generator + PatchGAN Discriminator
        **Color Space:** LAB (Lightness + A/B color channels)
        """)

    return demo


if __name__ == "__main__":
    demo = create_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )
