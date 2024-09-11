#This macro imports several Python modules and libraries such as PyTorch, torchvision, denoising_diffusion_pytorch, matplotlib, and PIL (Python Imaging Library). The main function of the macro is to train a denoising diffusion model on a set of input images, and then generate new images using the trained model.
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from denoising_diffusion_pytorch import Unet, GaussianDiffusion
import matplotlib.pyplot as plt
from PIL import Image
import os

# These above lines import necessary libraries and modules.

# Check for GPU availability and use it if available, otherwise use the CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# This line prompts the user to input the path to the folder containing the images to be used for training.
image_folder = input("files:")

# Set the batch size
batch_size = 16

# Set up the dataset and data loader. These lines define the transformation pipeline that will be applied to the input images. In our case, the images will be resized to 128x128 pixels using antialiasing and then converted to tensors.
transform = transforms.Compose([
    transforms.Resize((128, 128), Image.ANTIALIAS),
    transforms.ToTensor(),
])

#Resizing an image means changing its dimensions, i.e., its width and height, while maintaining its aspect ratio. In this code, the input images are resized to 128x128 pixels using antialiasing to reduce any aliasing artifacts that may occur during the resizing process. This code uses the transforms.Resize() function from the PyTorch transforms module to resize the input images to a 128x128 size. The ANTIALIAS method is used to resample the image, which helps to reduce the appearance of jagged edges or other artifacts that can occur when an image is resized.

#These lines create a PyTorch dataset and data loader using the input images and the transformation pipeline defined above.
dataset = ImageFolder(image_folder, transform=transform)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Set up the model and diffusion. These lines define the denoising diffusion model to be trained. The model architecture is a UNet with 4 layers, where each layer has a number of channels equal to dim multiplied by the corresponding value in dim_mults. The GaussianDiffusion object wraps the model and defines the diffusion process that will be used during training.
# Set up the model and diffusion.
model = Unet(
    dim=64,
    dim_mults=(1, 2, 4, 8)
).to(device)  # Move the model to the selected device (GPU or CPU)

diffusion = GaussianDiffusion(
    model,
    image_size=128,
    timesteps=1000,
    loss_type='l1'
).to(device)  # Move the diffusion object to the selected device (GPU or CPU)


# Set up the optimizer ( Adam optimizer )
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# The following lines define the training loop. For each epoch, the loop iterates over the batches of images in the data loader, computes the loss on each batch using the diffusion object, computes the gradients using backpropagation, and updates the model parameters using the optimizer.
# Training loop
num_epochs = 2000  # Set the number of epochs

for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    for batch in data_loader:
        images, _ = batch
        images = images.to(device)  # Move the input images to the selected device (GPU or CPU)
        optimizer.zero_grad()
        loss = diffusion(images)
        loss.backward()
        optimizer.step()

    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}')

# After training, sample images. The following lines generate new images using the trained denoising diffusion model.
sampled_images = diffusion.sample(batch_size=4).detach().cpu().numpy()
sampled_images = sampled_images.transpose(0, 2, 3, 1)  # (N, H, W, C)

# Resize the generated images to the desired dimensions (720x1200)
resized_images = [Image.fromarray((img * 255).astype('uint8')).resize((1200, 720), Image.ANTIALIAS) for img in sampled_images]

# Display resized sampled images
for i, img in enumerate(resized_images):
    plt.subplot(2, 2, i + 1)
    plt.imshow(img)
    plt.axis('off')

plt.show()

# Save the trained model
save_path = "SIM_134_bs16_epoch2000_ts1000_ns10000.pt"
torch.save(model.state_dict(), save_path)

# Generate and save simulated images
num_simulations = 10000
simulated_images = diffusion.sample(batch_size=num_simulations).detach().cpu().numpy()
simulated_images = simulated_images.transpose(0, 2, 3, 1)

if not os.path.exists("simulated_images134_1"):
    os.mkdir("simulated_images134_1")

for i, img in enumerate(simulated_images):
    img = (img * 255).astype('uint8')
    img = Image.fromarray(img)
    img.save(f"simulated_images134_1/simulated_image_{i+1}.png")
