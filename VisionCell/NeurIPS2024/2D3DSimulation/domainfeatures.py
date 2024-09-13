import torch
from torchvision.models import resnet50
from torchvision.transforms import ToTensor, Normalize, Compose
from denoising_diffusion_pytorch import Unet, GaussianDiffusion
from PIL import Image
import os

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

# Extract features for each image
for i, img in enumerate(images):
    # Preprocess the image and add a batch dimension
    img_preprocessed = preprocess(img).unsqueeze(0).to(device)

    # Pass the image through ResNet to get the features
    features = resnet(img_preprocessed)

    # The output is a batch of feature vectors, so take the first one
    features = features[0]

    print(f"Image {i+1} features:")
    print(features)
