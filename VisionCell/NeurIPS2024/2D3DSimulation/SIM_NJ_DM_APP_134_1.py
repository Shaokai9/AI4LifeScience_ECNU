import torch
from denoising_diffusion_pytorch import Unet, GaussianDiffusion
from PIL import Image
import numpy as np
import os

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
num_simulations = 100
simulated_images = []

for _ in range(num_simulations // 100):  # Adjust the denominator based on your GPU memory
    batch_images = diffusion.sample(batch_size=100).detach().cpu().numpy()  # Adjust the batch_size based on your GPU memory
    batch_images = batch_images.transpose(0, 2, 3, 1)
    simulated_images.append(batch_images)
    torch.cuda.empty_cache()  # Clear CUDA cache

simulated_images = np.concatenate(simulated_images, axis=0)

if not os.path.exists("simulated_images134_1"):
    os.mkdir("simulated_images134_1")

for i, img in enumerate(simulated_images):
    img = (img * 255).astype('uint8')
    img = Image.fromarray(img)
    img.save(f"simulated_images134_1/simulated_image_{i+1}.png")
