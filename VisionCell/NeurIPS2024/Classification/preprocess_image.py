import torch
from torchvision import transforms
from PIL import Image

# Define the preprocessing transformation
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize the image to 224x224 pixels
    transforms.ToTensor(),  # Convert the image to a PyTorch tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Normalize using the ImageNet mean and std
                         std=[0.229, 0.224, 0.225])
])

def save_preprocessed_image(image_path, save_path):
    # Load the image
    image = Image.open(image_path).convert('RGB')
    
    # Apply the preprocessing pipeline
    preprocessed_image = preprocess(image)
    
    # Convert back to PIL image and save
    image_to_save = transforms.ToPILImage()(preprocessed_image)
    image_to_save.save(save_path)

# Path to your input image
image_path = '300099_013_4974377.png'

# Path where you want to save the preprocessed image
save_path = '300099_013_4974377_preprocessed.jpg'

# Apply preprocessing to the input image and save it
save_preprocessed_image(image_path, save_path)
