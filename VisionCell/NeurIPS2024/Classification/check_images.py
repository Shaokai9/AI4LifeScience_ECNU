from PIL import Image
import os

def check_images(image_dir):
    """
    Check images in the specified directory and print their properties.
    
    :param image_dir: Directory with images to check
    """
    # Store problematic images for reporting
    problematic_images = []

    for filename in os.listdir(image_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
            try:
                with Image.open(os.path.join(image_dir, filename)) as img:
                    # Get basic details of the image
                    mode = img.mode
                    format_ = img.format
                    size = img.size

                    # Check if image is RGB (for Inception V3)
                    if mode not in ['RGB', 'RGBA']:
                        problematic_images.append((filename, f"Non-RGB mode: {mode}"))

                    # Print details of the image
                    print(f"{filename}: Mode={mode}, Format={format_}, Size={size}")

            except Exception as e:
                problematic_images.append((filename, str(e)))

    # Report problematic images
    if problematic_images:
        print("\nProblematic images:")
        for file, issue in problematic_images:
            print(f"{file}: {issue}")

# Replace 'path_to_your_images' with the path to your actual image directory.
check_images('/Users/hassanyang/VirtualEnv/ChatGPT_PyTorch/SNOPlus/PileUp/SeparationMethod/NeuralNetWork/training/abnormal')
