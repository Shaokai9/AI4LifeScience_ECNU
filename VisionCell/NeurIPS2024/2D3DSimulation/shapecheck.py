import cv2
import os

# Define the path to your image folder
folder_path = ''

# Get a list of all the image files in the folder
image_files = [f for f in os.listdir(folder_path) if f.endswith('.png')]

# Loop through all the images
for image_file in image_files:
    # Load the image
    img = cv2.imread(os.path.join(folder_path, image_file))

    # Print the image shape
    print(f'{image_file}: {img.shape}')
