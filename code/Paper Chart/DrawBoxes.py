import cv2

def draw_boxes(image_path, txt_path, output_path, color=(0, 255, 0)):
    # Read the image
    img = cv2.imread(image_path)
    height, width = img.shape[:2]

    # Read the YOLO txt file
    with open(txt_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        try:
            class_id, x_center, y_center, w, h = map(float, line.split())
        except ValueError:
            continue  # Skip this line if it can't be converted to a bounding box

        # Convert coordinates and size to pixel values in the actual image
        x_center *= width
        y_center *= height
        w *= width
        h *= height

        # Calculate the coordinates of the top left and bottom right corners of the rectangle
        x1 = int(x_center - w / 2)
        y1 = int(y_center - h / 2)
        x2 = int(x_center + w / 2)
        y2 = int(y_center + h / 2)

        # Draw the rectangle on the image, with the color specified by the color parameter
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

    # Save the image
    cv2.imwrite(output_path, img)

# Define paths
image_path = r'C:\Users\SuQun\Desktop\ss\sss\283.png'
txt_path = r'C:\Users\SuQun\Desktop\ss\output3\284.txt'
output_path = r'C:\Users\SuQun\Desktop\ss\sss\3.png'
# Use the function
draw_boxes(image_path, txt_path, output_path, color=(255, 0, 0))


