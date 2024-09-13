import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image  # For loading images
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Preprocessing pipeline
# Convert images to tensors and normalize them
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Ensuring the input images are resized to 224x224
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Grayscale images have just one channel
])

# Specify image paths
image_path1 = 'path_to_image1.png'
image_path2 = 'path_to_image2.png'

# Load and preprocess images
img1 = Image.open(image_path1)  # Grayscale images, so no need to convert to RGB
img2 = Image.open(image_path2)
img1 = transform(img1).unsqueeze(0)  # Add a batch dimension
img2 = transform(img2).unsqueeze(0)  # Add a batch dimension

# Load VGG16 model (use only its features module)
vgg16 = models.vgg16(pretrained=True).features.eval()  # Set to evaluation mode

# Custom decoder
# This will map the concatenated VGG16 features to a smaller representation and then decode it back to an image-like tensor
decoder = nn.Sequential(
    nn.Linear(2 * 7 * 7 * 512, 256),  # Multiply by 2 since we're concatenating features of two images
    nn.ReLU(),
    nn.Linear(256, 7 * 7 * 128),
    nn.ReLU(),
    nn.Linear(7 * 7 * 128, 7 * 7 * 3),  # The output is an image-like tensor with 3 channels
    nn.Tanh()
)

# Extract features for both images using VGG16
features1 = vgg16(img1).view(1, -1)
features2 = vgg16(img2).view(1, -1)

# Concatenate the features
features = torch.cat((features1, features2), 1)

# Decode the concatenated features
decoded_imgs = decoder(features)

# Visualize in 3D space
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(decoded_imgs[0][0], decoded_imgs[0][1], decoded_imgs[0][2])
ax.set_xlabel('Channel 1')
ax.set_ylabel('Channel 2')
ax.set_zlabel('Channel 3')
plt.show()
