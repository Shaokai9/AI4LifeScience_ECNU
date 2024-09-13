import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def load_and_resize_image(image_path):
    # Load the image and resize to 224x224
    image = Image.open(image_path).convert('RGB')
    image = image.resize((10, 10))
    return image

def image_to_tensor(image):
    # Convert image to tensor
    transform = transforms.ToTensor()
    return transform(image)

def tensor_to_image(tensor, cmap=None):
    # Convert the tensor to a PIL Image and return it
    transform = transforms.ToPILImage()
    image = transform(tensor)
    if cmap:
        plt.imshow(np.asarray(image), cmap=cmap)
        plt.axis('off')
        plt.savefig(f'{cmap}_channel.jpg', bbox_inches='tight', pad_inches=0)
        plt.close()
    return image

def split_rgb_channels(image_tensor):
    # Split the image tensor into R, G, B channels
    red_channel = image_tensor[0,:,:].unsqueeze(0)
    green_channel = image_tensor[1,:,:].unsqueeze(0)
    blue_channel = image_tensor[2,:,:].unsqueeze(0)
    return red_channel, green_channel, blue_channel

def save_tensor_to_file(tensor, filename):
    # Save a tensor to a text file
    np.savetxt(filename, tensor.numpy().squeeze(), fmt='%f')

# Load your image
image_path = 'sca1.png'  # Replace with your image path
image = load_and_resize_image(image_path)

# Convert to tensor
image_tensor = image_to_tensor(image)

# Split into RGB channels
red_tensor, green_tensor, blue_tensor = split_rgb_channels(image_tensor)

# Save channel images in original visualization style
tensor_to_image(red_tensor,   cmap='Reds')
tensor_to_image(green_tensor, cmap='Greens')
tensor_to_image(blue_tensor,  cmap='Blues')

# Save tensors to text files
save_tensor_to_file(red_tensor,   'red_channel.txt')
save_tensor_to_file(green_tensor, 'green_channel.txt')
save_tensor_to_file(blue_tensor,  'blue_channel.txt')

print("Channel images and tensor matrices saved.")
