import os
from PIL import Image

def check_image_dimensions(folder_path):
    png_files = [file for file in os.listdir(folder_path) if file.lower().endswith(".png")]

    for file in png_files:
        img_path = os.path.join(folder_path, file)
        img = Image.open(img_path)
        width, height = img.size
        print(f"Image: {file}, Width: {width}, Height: {height}")

folder_path = input("Enter the path to the folder containing PNG images: ")
check_image_dimensions(folder_path)
