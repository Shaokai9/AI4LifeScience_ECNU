import imageio
import os

def create_gif_from_images(image_folder, output_folder):
    # Get list of subfolders in the main directory
    apples = sorted([os.path.join(image_folder, f) for f in os.listdir(image_folder) if os.path.isdir(os.path.join(image_folder, f))])

    # For each apple
    for i, apple in enumerate(apples):
        # Get list of PNG images for this apple
        images = sorted([os.path.join(apple, f) for f in os.listdir(apple) if f.endswith('.png')])

        # Read images into memory
        image_list = [imageio.imread(image) for image in images]

        # Save as gif
        imageio.mimsave(os.path.join(output_folder, f'apple_{i+1}.gif'), image_list, duration=0.1)

# Usage
create_gif_from_images('/Users/hassanyang/VirtualEnv/ChatGPT_PyTorch/CellSIM/ObjSeg/cropped', '/Users/hassanyang/VirtualEnv/ChatGPT_PyTorch/CellSIM/Simulation/VideoSim')
