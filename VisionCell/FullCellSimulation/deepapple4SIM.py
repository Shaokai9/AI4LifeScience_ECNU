import torch
from video_diffusion_pytorch import Unet3D, GaussianDiffusion
import imageio
import numpy as np

# Define the model architecture
model = Unet3D(
    dim = 64,
    dim_mults = (1, 2, 4, 8)
)

# Define the diffusion model
diffusion = GaussianDiffusion(
    model,
    image_size = 128,
    num_frames = 10,
    timesteps = 1000,
    loss_type = 'l1'
)

# Load the trained model
model_path = 'cell_sim4/cell_1.pth'
model.load_state_dict(torch.load(model_path))
model.eval()

NUM_VIDEOS = 100  # change this value to produce more or fewer videos
SEED_START = 42  # starting seed for randomness

# Generate diverse simulated videos
for i in range(NUM_VIDEOS):
    # Set seed for reproducibility and diversity
    torch.manual_seed(SEED_START + i)

    simulated_video = diffusion.sample(batch_size=1)

    # Convert tensor to numpy array and normalize it to 0-255
    simulated_video = simulated_video.permute(0, 2, 3, 4, 1).detach().cpu().numpy()

    # Clip values to ensure they are in the right range
    simulated_video = np.clip(simulated_video, 0, 1)

    # Scale to 0-255 and convert to uint8
    simulated_video = (simulated_video * 255).astype(np.uint8)

    # Save as gif
    simulated_video = simulated_video.squeeze()
    simulated_video_list = [frame for frame in simulated_video]
    imageio.mimsave(f'cell_sim4_diverse/cell_diverse_{i}.gif', simulated_video_list, 'GIF', duration=0.1)

