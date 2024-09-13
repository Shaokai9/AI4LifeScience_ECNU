import os
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from denoising_diffusion_pytorch import Unet, GaussianDiffusion
import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = 500_000_000

# Function to save the trained model
def save_model(model, save_path):
    torch.save(model.state_dict(), save_path)

# Function to generate and save images using the diffusion model
def generate_and_save_images(diffusion, num_images, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    batch_size = 4

    # Generate images in batches
    for i in range(0, num_images, batch_size):
        num_samples = min(batch_size, num_images - i)
        sampled_images = diffusion.sample(batch_size=num_samples).detach().cpu().numpy()
        sampled_images = (sampled_images * 255).astype(np.uint8)
        sampled_images = sampled_images.transpose(0, 2, 3, 1)  # (N, H, W, C)

        # Resize the generated images to the desired dimensions (1200x720)
        resized_images = [Image.fromarray(img).resize((image_width, image_height), Image.ANTIALIAS) for img in sampled_images]

        # Save the generated images
        for j, img in enumerate(sampled_images):
            img_path = os.path.join(save_dir, f"image_{i + j}.png")
            Image.fromarray(img).save(img_path)

# Set device to GPU if available, else CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Get the path to the image folder
image_folder = input("Enter the path to the image folder:")

batch_size = 4
#image_size = 128

# Set the new image dimensions
image_width = 1200
image_height = 720
image_size = (image_height, image_width)

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((image_height, image_width)), # Resize the image to the desired dimensions
    transforms.ToTensor(),
])

# Define image transformations
#transform = transforms.Compose([
#    transforms.Resize(image_size),
#    transforms.CenterCrop(image_size),
#    transforms.ToTensor(),
#])

# Create dataset and data loader
dataset = ImageFolder(image_folder, transform=transform)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Define the U-Net model
# The U-Net is a type of convolutional neural network (CNN) architecture that was originally designed for biomedical image segmentation tasks. It was introduced by Olaf Ronneberger, Philipp Fischer, and Thomas Brox in their paper "U-Net: Convolutional Networks for Biomedical Image Segmentation." The U-Net architecture is particularly effective at handling small datasets and producing high-quality segmentation results with a small number of training images. The name "U-Net" comes from the fact that the architecture visually resembles the letter "U" when illustrated. The U-Net consists of an encoder (downsampling) path and a decoder (upsampling) path, which are connected by a series of skip connections. These skip connections help retain the spatial information from the encoder path and improve the localization accuracy of the segmentation results.
model = Unet(
    dim=64,#dim is the base number of channels (or filters) in the first layer of the U-Net
    dim_mults=(1, 2, 4, 8) #dim_mults is a tuple that specifies the growth factor of channels as we go deeper into the U-Net. The dim_mults values of (1, 2, 4, 8) indicate that the number of channels doubles with each downsampling layer in the encoder path.
)

# The U-Net is used as the denoising function for the diffusion model. Here's a brief explanation of the U-Net architecture used in our macro:

# 1.Encoder (Downsampling) Path: The encoder path consists of a series of convolutional layers followed by max-pooling layers.  Each convolutional layer is followed by an activation function (e.g., ReLU). The purpose of the encoder path is to extract features from the input image and reduce its spatial dimensions.
# 2.Decoder (Upsampling) Path: The decoder path consists of a series of up-convolution (transposed convolution) layers followed by concatenation with the corresponding skip connection from the encoder path. Each up-convolution layer is followed by an activation function (e.g., ReLU). The purpose of the decoder path is to upsample the feature maps and combine them with the feature maps from the encoder path to generate the final output.
# 3.Skip Connections: These are the connections between the encoder and decoder paths that allow the network to retain spatial information from the input image. Skip connections help combine low-level features (fine details) with high-level features (semantic information) to produce accurate segmentation results.
# 4.Output Layer: The final layer of the U-Net produces the denoised output image. This layer typically uses a sigmoid or softmax activation function to generate the final denoised image.

#The loss is calculated as the difference between the input image and the reconstructed image generated by the diffusion model. The goal of the training process is to minimize this loss. The diffusion model is trained using the GaussianDiffusion class, which takes the model (in this case, a U-Net), image size, timesteps, and loss type as input parameters. During training, the GaussianDiffusion class computes the loss for each input image by comparing it with the reconstructed image generated by the model at each timestep. The loss type can be either L1 or L2 loss. L1 loss is the absolute difference between the input and the reconstructed image, while L2 loss is the squared difference.
# Define the diffusion model
#diffusion = GaussianDiffusion(
#    model,
#    (image_height, image_width), # Pass the new image dimensions
    #image_size=image_size,
#    timesteps=10,  # number of steps
#    loss_type='l1'  # L1 or L2
#)

config = GaussianDiffusionConfig(
    model=model,
    image_size=image_size,
    num_timesteps=10,
    loss_type='l1'
)

diffusion = GaussianDiffusion(config)

# Move the models to the specified device
model = model.to(device)
diffusion = diffusion.to(device)

# Define the optimizer
# The following lines of code calculate the loss for each batch of images:
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Set the number of training epochs
num_epochs = 5

# Initialize lists to store loss values for each epoch
loss_values = []

# Training loop
for epoch in range(num_epochs):
    epoch_loss = 0
    num_batches = 0

    #The diffusion(images) call computes the loss between the input images and the reconstructed images generated by the diffusion model. The loss is then backpropagated through the model using loss.backward() to update the model's weights. Finally, the optimizer performs a weight update using optimizer.step().
    print(f"Epoch {epoch + 1}/{num_epochs}")
    for batch in data_loader:
        images, _ = batch
        images = images.to(device)
        optimizer.zero_grad()
        loss = diffusion(images)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        num_batches += 1

    # Calculate average loss for the current epoch
    avg_loss = epoch_loss / num_batches
    loss_values.append(avg_loss)

    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}')

# Save the trained model
save_model_path = "trained_model.pth"
save_model(model, save_model_path)

# Generate and save simulated images
num_simulated_images = 10
save_images_directory = "simulated_images"
generate_and_save_images(diffusion, num_simulated_images, save_images_directory)

#In a diffusion model, the goal is to learn a denoising function that reconstructs the original images from noisy versions of them. The loss function is a mathematical way to measure the discrepancy between the reconstructed images and the original images. The lower the loss value, the closer the reconstructed images are to the original images.

#In the our macro, the loss function is defined by the GaussianDiffusion class, and it can be either L1 or L2 loss. Here's a brief explanation of each:

    #L1 Loss: The L1 loss, also known as Mean Absolute Error (MAE), calculates the absolute difference between each pixel of the original image and the corresponding pixel in the reconstructed image, and then takes the average of these absolute differences across all pixels. The L1 loss is defined as:
    #    L1_loss = (1/N) * Σ|y_true - y_pred|
    #    where y_true is the original image, y_pred is the reconstructed image, and N is the total number of pixels.

    #L2 Loss: The L2 loss, also known as Mean Squared Error (MSE), calculates the squared difference between each pixel of the original image and the corresponding pixel in the reconstructed image, and then takes the average of these squared differences across all pixels. The L2 loss is defined as:
    #    L2_loss = (1/N) * Σ(y_true - y_pred)^2
    #     where y_true is the original image, y_pred is the reconstructed image, and N is the total number of pixels.

# Here is an example to illustrate the concept of loss:

#Assume we have a 2x2 grayscale image, and the diffusion model generates a reconstructed image as follows:

#  Original Image (y_true):
#   [[10, 20],
#   [30, 40]]

#  Reconstructed Image (y_pred):
#   [[12, 22],
#   [28, 38]]

#For L1 loss, we calculate the absolute difference between each corresponding pixel:
# Absolute Differences:
#   [[2, 2],
#   [2, 2]]
#   L1_loss = (2+2+2+2) / 4 = 8 / 4 = 2

#For L2 loss, we calculate the squared difference between each corresponding pixel:
# Squared Differences:
#   [[4, 4],
#   [4, 4]]
#   L2_loss = (4+4+4+4) / 4 = 16 / 4 = 4

# During training, the diffusion model tries to minimize this loss by adjusting its parameters (weights) to generate better reconstructed images. A smaller loss value indicates that the reconstructed images are closer to the original images, which means the model is doing a better job at denoising and reconstructing the images.
