import torch
import numpy as np
from denoising_diffusion_pytorch import Unet, GaussianDiffusion
from PIL import Image
import matplotlib.pyplot as plt

# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Unet(dim=64, dim_mults=(1, 2, 4, 8)).to(device)
model.load_state_dict(torch.load("trained_model.pt"))

# Set up the diffusion
diffusion = GaussianDiffusion(model, image_size=1155, timesteps=10, loss_type='l1').to(device)

# Generate an image
images = diffusion.sample(batch_size=1).detach().cpu().numpy()
image = images[0].transpose(1, 2, 0)

# Convert to uint8
image = (image * 255).astype('uint8')

# Compute histogram for each color channel
colors = ('r', 'g', 'b')
for i, color in enumerate(colors):
    histogram, bin_edges = np.histogram(image[:,:,i], bins=256, range=(0, 256))
    plt.plot(bin_edges[:-1], histogram, color=color)

plt.title('Color Histogram')
plt.xlabel('Color Value')
plt.ylabel('Pixels')

plt.show()
