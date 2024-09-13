import os
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from denoising_diffusion_pytorch import Unet, GaussianDiffusion
import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = 500_000_000

def pad_to_square(img):
    w, h = img.size
    max_dim = max(w, h)
    padded_img = Image.new("RGB", (max_dim, max_dim), (0, 0, 0))
    padded_img.paste(img, ((max_dim - w) // 2, (max_dim - h) // 2))
    return padded_img

def save_model(model, save_path):
    torch.save(model.state_dict(), save_path)

def generate_and_save_images(diffusion, num_images, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    batch_size = 2

    for i in range(0, num_images, batch_size):
        num_samples = min(batch_size, num_images - i)
        sampled_images = diffusion.sample(batch_size=num_samples).detach().cpu().numpy()
        sampled_images = (sampled_images * 255).astype(np.uint8)
        sampled_images = sampled_images.transpose(0, 2, 3, 1)

        resized_images = [Image.fromarray(img).resize((image_width, image_height), Image.LANCZOS) for img in sampled_images]

        for j, img in enumerate(resized_images):
            img_path = os.path.join(save_dir, f"image_{i + j}.png")
            img.save(img_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

image_folder = input("Enter the path to the image folder:")

batch_size = 2
image_width = 1155
image_height = 1609
image_size = max(image_width, image_height)

transform = transforms.Compose([
    transforms.Lambda(pad_to_square),
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
])

dataset = ImageFolder(image_folder, transform=transform)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

model = Unet(
    dim=64,
    dim_mults=(1, 2, 4, 8)
)

diffusion = GaussianDiffusion(
    model=model,
    image_size=image_size,
    #num_timesteps=10,
    loss_type='l1'
)

model = model.to(device)
diffusion = diffusion.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

num_epochs = 5

loss_values = []

for epoch in range(num_epochs):
    epoch_loss = 0
    num_batches = 0

    print(f"Epoch {epoch + 1}/{num_epochs}")
    for batch in data_loader:
        images, _ = batch
        images = images.to(device)
        optimizer.zero_grad()
        loss = diffusion(images)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        num_batches += 1

    avg_loss = epoch_loss / num_batches
    loss_values.append(avg_loss)

    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}')

save_model_path = "trained_model.pth"
save_model(model, save_model_path)

num_simulated_images = 10
save_images_directory = "simulated_images"
generate_and_save_images(diffusion, num_simulated_images, save_images_directory)

