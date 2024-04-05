import os
import cv2
import numpy as np

def compute_iou(box1, box2):
    """
    Compute the Intersection-Over-Union of two given boxes.

    Args:
    box1 -- first box, list object with coordinates (x_center, y_center, width, height)
    box2 -- second box, list object with coordinates (x_center, y_center, width, height)
    """

    xi1 = max(box1[0] - box1[2] / 2.0, box2[0] - box2[2] / 2.0)
    yi1 = max(box1[1] - box1[3] / 2.0, box2[1] - box2[3] / 2.0)
    xi2 = min(box1[0] + box1[2] / 2.0, box2[0] + box2[2] / 2.0)
    yi2 = min(box1[1] + box1[3] / 2.0, box2[1] + box2[3] / 2.0)
    inter_area = max(xi2 - xi1, 0) * max(yi2 - yi1, 0)

    box1_area = box1[2] * box1[3]
    box2_area = box2[2] * box2[3]
    union_area = box1_area + box2_area - inter_area

    iou = inter_area / union_area

    return iou

# Set your own paths here
txt_folder = r'C:\Users\SuQun\Desktop\ss\exp94\labels'
output_txt_path = r'C:\Users\SuQun\Desktop\ss\exp94\1'
img_folder = r'C:\Users\SuQun\Desktop\ss\exp94\p'
output_img_folder = r'C:\Users\SuQun\Desktop\ss\exp94\2'

# Initialize a list to hold all the targets
targets = []
new_targets = []

# Process each txt file in the folder
for txt_filename in sorted(os.listdir(txt_folder)):
    if txt_filename.endswith('.txt'):
        img_filename = txt_filename.replace('.txt', '.png')  # assuming the images are .jpg files
        img = cv2.imread(os.path.join(img_folder, img_filename))
        height, width, _ = img.shape

        # Read the current txt file
        with open(os.path.join(txt_folder, txt_filename), 'r') as f:
            for line in f:
                line = line.strip().split()
                if line[0] == '0':  # assuming the class ID for your targets is 0
                    # Parse the normalized bounding box coordinates
                    x, y, w, h = map(float, line[1:5])

                    # Compute the IOU with each target
                    ious = [compute_iou([x, y, w, h], target['bbox']) for target in targets]
                    if ious:
                        # Find the target with the highest IOU
                        best_match_index = np.argmax(ious)
                        best_match_iou = ious[best_match_index]

                        if best_match_iou <= 0.3:  # this is the IOU threshold
                            # Create a new target
                            target = {'id': len(targets) + 1, 'bbox': [x, y, w, h], 'source': txt_filename}
                            targets.append(target)
                            new_targets.append(target)

                            # Draw the bounding box and ID on the image
                            left = int((x - w / 2) * width)
                            top = int((y - h / 2) * height)
                            right = int((x + w / 2) * width)
                            bottom = int((y + h / 2) * height)
                            cv2.rectangle(img, (left, top), (right, bottom), (0, 0, 255), 2)
                            cv2.putText(img, str(target['id']), (left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                    else:
                        # Create a new target
                        target = {'id': len(targets) + 1, 'bbox': [x, y, w, h], 'source': txt_filename}
                        targets.append(target)
                        new_targets.append(target)

                        # Draw the bounding box and ID on the image
                        left = int((x - w / 2) * width)
                        top = int((y - h / 2) * height)
                        right = int((x + w / 2) * width)
                        bottom = int((y + h / 2) * height)
                        cv2.rectangle(img, (left, top), (right, bottom), (0, 0, 255), 2)
                        cv2.putText(img, str(target['id']), (left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                # Save the image
                cv2.imwrite(os.path.join(output_img_folder, img_filename), img)

        # Write the results to the output txt file
        with open(output_txt_path, 'w') as f:
            for target in new_targets:
                f.write(f"ID: {target['id']}, BBox (normalized): {target['bbox']}, Source: {target['source']}\n")

