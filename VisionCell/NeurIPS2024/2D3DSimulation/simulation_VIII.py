import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from denoising_diffusion_pytorch import Unet, GaussianDiffusion
import matplotlib.pyplot as plt

# Replace this with the path to your own images
image_folder = '/Users/hassanyang/VirtualEnv/PyTorch/CellData/SimDateSet/Test'

# Set the batch size
batch_size = 8

# Set up the dataset and data loader
transform = transforms.Compose([
    transforms.Resize(128),
    transforms.CenterCrop(128),
    transforms.ToTensor(),
])

dataset = ImageFolder(image_folder, transform=transform)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Set up the model and diffusion
model = Unet(
    dim=64,
    dim_mults=(1, 2, 4, 8)
)

diffusion = GaussianDiffusion(
    model,
    image_size=128,
    timesteps=1000,  # number of steps
    loss_type='l1'  # L1 or L2
)

# Set up the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Training loop
num_epochs = 2  # Set the number of epochs

for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    for batch in data_loader:
        images, _ = batch
        optimizer.zero_grad()
        loss = diffusion(images)
        loss.backward()
        optimizer.step()
    #Sample images after each epoch
    #sampled_images = diffusion.sample(batch_size=4)
    
    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}')

# After training, sample images
sampled_images = diffusion.sample(batch_size=4).detach().cpu().numpy()
sampled_images = sampled_images.transpose(0, 2, 3, 1)  # (N, H, W, C)

# Display sampled images
for i, img in enumerate(sampled_images):
    plt.subplot(2, 2, i + 1)
    plt.imshow(img)
    plt.axis('off')

plt.show()
