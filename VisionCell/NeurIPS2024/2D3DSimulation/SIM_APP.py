import torch
from denoising_diffusion_pytorch import Unet, GaussianDiffusion
from PIL import Image

# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Unet(dim=64, dim_mults=(1, 2, 4, 8)).to(device)
model.load_state_dict(torch.load("trained_model.pt"))

# Set up the diffusion
diffusion = GaussianDiffusion(model, image_size=1155, timesteps=10, loss_type='l1').to(device)

# Total number of images to generate
total_images = 100

# Number of images to generate in each batch
batch_size = 10

# Generate images in batches
for i in range(0, total_images, batch_size):
    images = diffusion.sample(batch_size=batch_size).detach().cpu().numpy()
    images = images.transpose(0, 2, 3, 1)

    # Save images to disk
    for j, img in enumerate(images):
        img = (img * 255).astype('uint8')
        img = Image.fromarray(img)
        img.save(f"simulated_images/simulated_image_{i+j+1}.png")

print("Done!")
