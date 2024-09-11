import cv2
import os
import numpy as np
from pathlib import Path

def load_frames_from_gif(gif_path):
    """
    Load frames from a GIF file.

    Args:
    - gif_path (str): Path to the GIF file.

    Returns:
    - List of grayscale frames.
    """
    frames = []
    cap = cv2.VideoCapture(gif_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(gray_frame)
    cap.release()
    return frames

def interpolate(img1, img2, alpha):
    """
    Interpolate between two frames using a specified weight.

    Args:
    - img1 (array): First image/frame.
    - img2 (array): Second image/frame.
    - alpha (float): Interpolation weight.

    Returns:
    - Interpolated image.
    """
    return cv2.addWeighted(img1, 1-alpha, img2, alpha, 0)

def save_gif(frames, output_path):
    """
    Save frames as a GIF file.

    Args:
    - frames (list): List of frames to be saved.
    - output_path (str): Path to save the GIF.
    """
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    height, width = frames[0].shape
    out = cv2.VideoWriter(output_path, fourcc, 10, (width, height), isColor=False)
    for frame in frames:
        out.write(frame)
    out.release()

def main(input_gif_path, output_gif_path):
    """
    Main function to load a GIF, perform interpolation, and save the interpolated GIF.

    Args:
    - input_gif_path (str): Path to the input GIF.
    - output_gif_path (str): Path to save the output GIF.
    """
    frames = load_frames_from_gif(input_gif_path)

    interpolated_frames = []
    for i in range(len(frames) - 1):
        interpolated_frames.append(frames[i])
        for j in range(1, 10):  # Create 9 interpolated frames between each original pair
            alpha = j / 10
            interpolated_frame = interpolate(frames[i], frames[i+1], alpha)
            interpolated_frames.append(interpolated_frame)
    interpolated_frames.append(frames[-1])

    save_gif(interpolated_frames, output_gif_path)

if __name__ == '__main__':
    input_gif_path = 'path_to_your_input.gif'
    # Extract name of the input gif, append "100" and set as the output name
    output_name = Path(input_gif_path).stem + "_100.gif"
    output_gif_path = os.path.join(os.path.dirname(input_gif_path), output_name)
    main(input_gif_path, output_gif_path)
