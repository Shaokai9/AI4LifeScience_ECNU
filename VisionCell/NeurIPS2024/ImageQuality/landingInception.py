import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
from PIL import Image
import numpy as np
from scipy.linalg import sqrtm
from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from torchvision.models import inception_v3

# Preprocessing Function
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize(299),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)

# FID Calculation
def calculate_fid(act1, act2):
    mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)
    ssdiff = np.sum((mu1 - mu2)**2.0)
    covmean = sqrtm(sigma1.dot(sigma2))
    if iscomplexobj(covmean):
        covmean = covmean.real
    fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid

# Activation Extraction
def get_activation(image, model, device):
    with torch.no_grad():
        image = image.to(device)
        # Assuming using the pre-logits activations (before the final fully connected layer)
        features = model(image)[0]
        features = F.adaptive_avg_pool2d(features, (1, 1))
        features = features.view(features.shape[0], -1).cpu().numpy()
    return features

# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 10  # Make sure to update this based on your dataset classes count
model = inception_v3(pretrained=False, num_classes=num_classes).to(device)
model.load_state_dict(torch.load('trained_inception_model.pth'))
model.eval()

# Paths for your images
image_path1 = 'path/to/image1.jpg'
image_path2 = 'path/to/image2.jpg'

# Process the images
image1 = preprocess_image(image_path1)
image2 = preprocess_image(image_path2)

# Extract activations
act1 = get_activation(image1, model, device)
act2 = get_activation(image2, model, device)

# Calculate FID
fid_same = calculate_fid(act1, act1)
fid_different = calculate_fid(act1, act2)

print('FID (same): %.3f' % fid_same)
print('FID (different): %.3f' % fid_different)
