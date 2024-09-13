import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from denoising_diffusion_pytorch import Unet, GaussianDiffusion
import matplotlib.pyplot as plt
from PIL import Image
import os

# Check for GPU availability and use it if available, otherwise use the CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Replace this with the path to your own images
image_folder = input("Enter path to folder containing training images: ")

# Set the batch size
batch_size = 1

# Set up the dataset and data loader
transform = transforms.Compose([
    transforms.Resize((1609, 1155), Image.ANTIALIAS),
    transforms.ToTensor(),
])

dataset = ImageFolder(image_folder, transform=transform)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Set up the model and diffusion
model = Unet(
    dim=64,
    dim_mults=(1, 2, 4, 8)
).to(device)

diffusion = GaussianDiffusion(
    model,
    image_size=1155,
    timesteps=10,
    loss_type='l1'
).to(device)

# Set up the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

# Training loop
num_epochs = 2

for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    for batch in data_loader:
        images, _ = batch
        images = images.to(device)
        optimizer.zero_grad()
        loss = diffusion(images)
        loss.backward()
        optimizer.step()

    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}')

# Save the trained model
save_path = "trained_model.pt"
torch.save(model.state_dict(), save_path)

# Generate and save simulated images
num_simulations = 100
simulated_images = diffusion.sample(batch_size=num_simulations).detach().cpu().numpy()
simulated_images = simulated_images.transpose(0, 2, 3, 1)

if not os.path.exists("simulated_images"):
    os.mkdir("simulated_images")

for i, img in enumerate(simulated_images):
    img = (img * 255).astype('uint8')
    img = Image.fromarray(img)
    img.save(f"simulated_images/simulated_image_{i+1}.png")

print("Done!")
