import torch
from denoising_diffusion_pytorch import Unet

# Define the path of your .pt file
MODEL_PATH = 'SIM_134_bs16_epoch2000_ts1000_ns10000.pt'

# Initialize the model
# Now, model is an instance of Unet class
model = Unet(dim=64, dim_mults=(1, 2, 4, 8))

# Load the state dict previously saved
state_dict = torch.load(MODEL_PATH, map_location=torch.device('cpu'))

# Load the state dict to the model
model.load_state_dict(state_dict)

# Switch to evaluation mode
model.eval()

# Print the architecture of your model
print(model)

# Print names of all modules
for name, module in model.named_modules():
    print(name)
