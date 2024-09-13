import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.models import inception_v3
import numpy as np
from scipy.linalg import sqrtm

# Setting the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Loading the trained model and removing the final layer
model = inception_v3(pretrained=False, aux_logits=False)
model.fc = torch.nn.Identity()  # Remove the last layer
model.load_state_dict(torch.load('trained_inception_model.pth'))
model = model.to(device)
model.eval()

# Data transformations (assuming the data is in the same format as before)
transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Dataloaders for real and generated (or another set of real) images
real_dataset = datasets.ImageFolder(root='path_to_real_images_folder', transform=transform)
real_dataloader = torch.utils.data.DataLoader(real_dataset, batch_size=32, shuffle=False)

fake_dataset = datasets.ImageFolder(root='path_to_fake_images_folder', transform=transform)
fake_dataloader = torch.utils.data.DataLoader(fake_dataset, batch_size=32, shuffle=False)

# Function to extract features from images
def extract_features(dataloader, model):
    features_list = []
    with torch.no_grad():
        for inputs, _ in dataloader:
            inputs = inputs.to(device)
            features = model(inputs)
            features_list.append(features.cpu())
    return torch.cat(features_list, dim=0)

# FID computation
def compute_fid(features1, features2):
    mu1, sigma1 = features1.mean(axis=0), np.cov(features1, rowvar=False)
    mu2, sigma2 = features2.mean(axis=0), np.cov(features2, rowvar=False)
    
    diff = mu1 - mu2
    cov_mean = sqrtm(sigma1.dot(sigma2))
    
    if np.iscomplexobj(cov_mean):
        cov_mean = cov_mean.real
    
    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * cov_mean)
    return fid

# Extracting features from the images
real_features = extract_features(real_dataloader, model).numpy()
fake_features = extract_features(fake_dataloader, model).numpy()

# Calculating the FID score
fid_score = compute_fid(real_features, fake_features)
print(f"FID Score between real and fake images: {fid_score}")
