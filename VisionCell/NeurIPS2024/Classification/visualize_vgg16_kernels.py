import torch
import numpy as np
import torchvision.models as models
import matplotlib.pyplot as plt

# Check for GPU availability and use CPU as a fallback
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Replace this with the path to your model's state dictionary
model_path = 'trained_vgg16_base_bestmodel_20231111_110420.pth'

# Number of classes your model was trained on
num_classes = 2

# Initialize a VGG16 model
vgg16 = models.vgg16(pretrained=False)
vgg16.classifier[6] = torch.nn.Linear(vgg16.classifier[6].in_features, num_classes)

# Load your trained weights into the model
vgg16.load_state_dict(torch.load(model_path, map_location=device))

# Move the model to the selected device
vgg16 = vgg16.to(device)

# Channel labels and colors for annotations
channel_labels = ['R', 'G', 'B']
annotation_colors = ['red', 'green', 'blue']

# Function to normalize the kernel for visualization
def normalize_kernel(kernel):
    kernel -= kernel.min()
    kernel /= kernel.max()
    return kernel

# Function to save the first three kernels from each convolutional layer as images with annotations
def save_kernel_images(layer_kernels, layer_index, num_kernels=3):
    for i in range(num_kernels):
        fig, axes = plt.subplots(1, 3, figsize=(9, 3))  # One subplot for each channel
        for j in range(3):
            ax = axes[j]
            kernel = normalize_kernel(layer_kernels[i, j].cpu().numpy())
            ax.imshow(kernel, cmap='gray')
            ax.set_title(f'Layer {layer_index+1} - Kernel {i+1} - Channel {channel_labels[j]}', fontsize=6)
            ax.axis('off')
            # Annotate each element with colored text
            for (x, y), val in np.ndenumerate(kernel):
                ax.text(y, x, f'{val:.2f}', ha='center', va='center', fontsize=6, color=annotation_colors[j])
        plt.savefig(f'Layer_{layer_index+1}_Kernel_{i+1}_Annotated.png')
        plt.close(fig)

# Function to save the first three kernels from each convolutional layer as number-only images
def save_kernel_number_images(layer_kernels, layer_index, num_kernels=3):
    for i in range(num_kernels):
        fig, axes = plt.subplots(1, 3, figsize=(9, 3))  # One subplot for each channel
        for j in range(3):
            ax = axes[j]
            ax.set_facecolor('white')
            ax.axis('off')
            kernel = layer_kernels[i, j].cpu().numpy()
            # Annotate each element with black text
            for (x, y), val in np.ndenumerate(kernel):
                ax.text(y, x, f'{val:.2f}', ha='center', va='center', fontsize=6, color='black')
        plt.savefig(f'Layer_{layer_index+1}_Kernel_{i+1}_NumbersOnly.png')
        plt.close(fig)

# Iterate through the convolutional layers and save both types of images
layer_index = 0
for layer in vgg16.features:
    if isinstance(layer, torch.nn.Conv2d):
        layer_kernels = layer.weight.data[:3].to(device)  # Extract first three kernels
        save_kernel_images(layer_kernels, layer_index)  # Save with annotations
        save_kernel_number_images(layer_kernels, layer_index)  # Save with only numbers
        layer_index += 1
