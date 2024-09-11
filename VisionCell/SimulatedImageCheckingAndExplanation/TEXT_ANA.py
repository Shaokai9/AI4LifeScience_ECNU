#For texture analysis, one common method used is the Gray-Level Co-Occurrence Matrix (GLCM). The GLCM is a histogram of co-occurring grayscale values at a given offset over an image. From the GLCM, we can extract texture features like contrast, dissimilarity, homogeneity, energy, correlation, etc.

#There's no built-in function in PyTorch for GLCM, so we'll use the greycomatrix and greycoprops functions from the skimage.feature module in the scikit-image library.

#This script assumes that we have our model's .pt file and a dataset of images. The script will generate new images, convert them to grayscale, calculate the GLCM, and then compute several texture features.

#Here's an example of how we could do this:

import torch
from denoising_diffusion_pytorch import Unet, GaussianDiffusion
from PIL import Image
from skimage.feature import greycomatrix, greycoprops
from skimage.color import rgb2gray
import numpy as np
import os

# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Unet(dim=64, dim_mults=(1, 2, 4, 8)).to(device)
model.load_state_dict(torch.load("trained_model.pt"))

# Set up the diffusion
diffusion = GaussianDiffusion(model, image_size=1155, timesteps=10, loss_type='l1').to(device)

# Generate a batch of images
num_images = 100
images = diffusion.sample(batch_size=num_images).detach().cpu().numpy()

# Compute texture features for each image
for i, img in enumerate(images):
    # Convert to grayscale and scale to the range 0-255
    img_gray = rgb2gray(img)
    img_gray = (img_gray * 255).astype('uint8')

    # Compute the GLCM
    glcm = greycomatrix(img_gray, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)

    # Compute texture features
    contrast = greycoprops(glcm, 'contrast')[0, 0]
    dissimilarity = greycoprops(glcm, 'dissimilarity')[0, 0]
    homogeneity = greycoprops(glcm, 'homogeneity')[0, 0]
    energy = greycoprops(glcm, 'energy')[0, 0]
    correlation = greycoprops(glcm, 'correlation')[0, 0]

    print(f"Image {i+1}:")
    print(f"  Contrast: {contrast}")
    print(f"  Dissimilarity: {dissimilarity}")
    print(f"  Homogeneity: {homogeneity}")
    print(f"  Energy: {energy}")
    print(f"  Correlation: {correlation}\n")

#This script will print out the texture features for each generated image. If we want to compare these to the texture features of our training images, we could compute the same features for a sample of our training images and compare the distributions. We can use these texture features to compare the original and generated images to see how well the model has learned to capture the textures in our data.
