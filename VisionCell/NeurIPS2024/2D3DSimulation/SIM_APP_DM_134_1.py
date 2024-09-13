import torch
from denoising_diffusion_pytorch import Unet, GaussianDiffusion
from PIL import Image
import os
import numpy as np

# Check for GPU availability and use it if available, otherwise use the CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the trained model from the specified path
model_path = "SIM_134_bs16_epoch2000_ts1000_ns10000.pt"
model = Unet(
    dim=64,
    dim_mults=(1, 2, 4, 8)
)
model.load_state_dict(torch.load(model_path))
model = model.to(device)

# Create the GaussianDiffusion object with the trained model
diffusion = GaussianDiffusion(
    model,
    image_size=128,
    timesteps=1000,
    loss_type='l1'
)
diffusion = diffusion.to(device)

# Generate and save simulated images
num_simulations = 100  # Increase this number to generate more images per run
batch_size = 10  # Increase this number to generate more images at once

if not os.path.exists("simulated_images134_1"):
    os.mkdir("simulated_images134_1")

for j in range(num_simulations // batch_size):
    torch.manual_seed(j)  # Set a different seed for each batch to increase diversity
    simulated_images = diffusion.sample(batch_size=batch_size).detach().cpu().numpy()
    simulated_images = simulated_images.transpose(0, 2, 3, 1)

    for i, img in enumerate(simulated_images):
        img = (img * 255).astype('uint8')
        img = Image.fromarray(img)
        img.save(f"simulated_images134_1/simulated_image_{j*batch_size+i+1}.png")

