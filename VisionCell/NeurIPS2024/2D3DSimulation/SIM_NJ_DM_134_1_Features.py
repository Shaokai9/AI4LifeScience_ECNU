import torch
from torchvision.models import resnet50
from torchvision.transforms import ToTensor, Normalize, Compose
from denoising_diffusion_pytorch import Unet, GaussianDiffusion
from PIL import Image
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Unet(dim=64, dim_mults=(1, 2, 4, 8)).to(device)
model.load_state_dict(torch.load("trained_model.pt"))

# Set up the diffusion
diffusion = GaussianDiffusion(model, image_size=1155, timesteps=10, loss_type='l1').to(device)

# Generate a batch of images
num_images = 100
images = diffusion.sample(batch_size=num_images).detach()

# Set up the pre-processing for ResNet
preprocess = Compose([
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load a pre-trained ResNet model
resnet = resnet50(pretrained=True)
resnet = resnet.to(device)
resnet.eval()

# Remove the last layer to get the features instead of the classification
resnet = torch.nn.Sequential(*list(resnet.children())[:-1])

# Initialize empty lists to hold the feature values
feature1_values = []
feature2_values = []
feature3_values = []

# Extract features for each image
for i, img in enumerate(images):
    # Preprocess the image and add a batch dimension
    img_preprocessed = preprocess(img).unsqueeze(0).to(device)

    # Pass the image through ResNet to get the features
    features = resnet(img_preprocessed)

    # Flatten the output tensor to get a 1D tensor of features
    features = features.view(features.size(0), -1)

    # Select three features
    feature1 = features[0, 0].item()
    feature2 = features[0, 1].item()
    feature3 = features[0, 2].item()

    # Add the feature values to the respective lists
    feature1_values.append(feature1)
    feature2_values.append(feature2)
    feature3_values.append(feature3)

# Plot the features in 3D space
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(feature1_values, feature2_values, feature3_values)

ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.set_zlabel('Feature 3')

plt.show()
