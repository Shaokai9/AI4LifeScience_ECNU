import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
from PIL import Image
import numpy as np
from scipy.linalg import sqrtm
from numpy import cov
from numpy import trace
from numpy import iscomplexobj

def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize(299),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)

def calculate_fid(act1, act2):
    mu1, sigma1 = act1.mean(dim=0), cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(dim=0), cov(act2, rowvar=False)
    ssdiff = torch.sum((mu1 - mu2)**2.0)
    covmean = sqrtm(sigma1.dot(sigma2))
    if iscomplexobj(covmean):
        covmean = covmean.real
    fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid

def get_activation(image, model):
    with torch.no_grad():
        if torch.cuda.is_available():
            image = image.cuda()
            model = model.cuda()
        features = model(image)
        features = F.avg_pool2d(features, kernel_size=(features.shape[2], features.shape[3]))
        features = features.view(features.shape[0], -1)
        features = features.cpu().numpy()
    return features

model = torch.hub.load('pytorch/vision:v0.9.0', 'inception_v3', pretrained=True, aux_logits=False)
model.eval()

image_path1 = 'path/to/image1.jpg'
image_path2 = 'path/to/image2.jpg'

image1 = preprocess_image(image_path1)
image2 = preprocess_image(image_path2)

act1 = get_activation(image1, model)
act2 = get_activation(image2, model)

fid_same = calculate_fid(torch.from_numpy(act1), torch.from_numpy(act1))
fid_different = calculate_fid(torch.from_numpy(act1), torch.from_numpy(act2))

print('FID (same): %.3f' % fid_same)
print('FID (different): %.3f' % fid_different)
