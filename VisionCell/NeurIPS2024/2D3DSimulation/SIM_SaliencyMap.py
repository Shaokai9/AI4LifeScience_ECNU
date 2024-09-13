import torch
from torchvision.transforms import transforms
from denoising_diffusion_pytorch import Unet, GaussianDiffusion
import matplotlib.pyplot as plt
from PIL import Image


# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Unet(dim=64, dim_mults=(1, 2, 4, 8)).to(device)
#model.load_state_dict(torch.load("SIM_134_bs16_epoch2000_ts1000_ns10000.pt"))
model.load_state_dict(torch.load("SIM_134_bs16_epoch2000_ts1000_ns10000.pt", map_location=torch.device('cpu')))
model.eval()

# Load and preprocess the image
image_path = "/Users/hassanyang/VirtualEnv/ChatGPT_PyTorch/CellSIM/Data/02/Prefix_OnPoint_roi_00_slice_0684.png"
#image = Image.open(image_path).convert("RGB")

transform = transforms.Compose([
    transforms.Resize((1609, 1155), Image.ANTIALIAS),
    transforms.ToTensor(),
])
image = Image.open(image_path)
image = transform(image).unsqueeze(0).to(device)

# Ensure the image requires gradient
image = image.float()
image.requires_grad_()

# Forward pass
# Placeholder value for 'time'. This will need to be replaced with actual data.
time = 100
output = model(image, time)


# Compute the gradients of the output with respect to the input
output.backward()
saliency_map = image.grad.data

# Convert the saliency map from a 3D tensor to a 2D tensor by taking the maximum value across all color channels
saliency_map, _ = torch.max(saliency_map.abs(), dim=1)

# Display the saliency map
plt.imshow(saliency_map.squeeze().cpu(), cmap='hot')
plt.show()
