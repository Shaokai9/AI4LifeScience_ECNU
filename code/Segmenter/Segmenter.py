import numpy as np
import cv2
import os
from glob import glob

def segment_and_annotate_images(image_folder, mask_file, output_dir):
    # Load the npy file
    data = np.load(mask_file, allow_pickle=True).item()

    # Get the masks
    masks = data['masks']

    # Get the unique labels in the masks, excluding the background (label 0)
    labels = np.unique(masks)[1:]  # [1:] to exclude background

    # Get all the images in the folder
    image_files = glob(os.path.join(image_folder, '*.png'))  # Change to your image extension

    # Create a directory for each label
    label_dirs = [os.path.join(output_dir, f'label_{label}') for label in labels]
    for label_dir in label_dirs:
        os.makedirs(label_dir, exist_ok=True)

    # Create directory for annotated images
    annotated_dir = os.path.join(output_dir, 'annotated')
    os.makedirs(annotated_dir, exist_ok=True)

    # For each image in the folder
    for image_file in image_files:
        # Load the image
        image = cv2.imread(image_file)

        # Create a copy for annotation
        annotated_image = image.copy()

        # Get the base name of the image for naming the segments
        image_name = os.path.splitext(os.path.basename(image_file))[0]

        # For each label, create a segment, annotate it, and save it in the corresponding directory
        for label, label_dir in zip(labels, label_dirs):
            binary_mask = (masks == label).astype(np.uint8)
            segment = cv2.bitwise_and(image, image, mask=binary_mask)

            # Find the bounding box of the segment and crop the image to this bounding box
            x, y, w, h = cv2.boundingRect(binary_mask)
            segment = segment[y:y+h, x:x+w]

            # Find the center of the segment for annotation
            moments = cv2.moments(binary_mask)
            if moments['m00'] != 0:
                center_x = int(moments['m10'] / moments['m00'])
                center_y = int(moments['m01'] / moments['m00'])

                # Annotate the center of the segment
                cv2.putText(annotated_image, str(label), (center_x, center_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Save the segment
            cv2.imwrite(os.path.join(label_dir, f'segment_{image_name}.png'), segment)

        # Save the annotated image
        cv2.imwrite(os.path.join(annotated_dir, f'annotated_{image_name}.png'), annotated_image)

# Usage
segment_and_annotate_images(r'F:\fenge\s', r'F:\fenge\185_seg.npy', r'F:\fenge\segments')










