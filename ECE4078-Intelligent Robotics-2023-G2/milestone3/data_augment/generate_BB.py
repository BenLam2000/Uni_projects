import cv2
import numpy as np
import os

FRUIT_LABELS = {
        0: 'bg',
        1: 'redapple',
        2: 'greenapple',
        3: 'orange',
        4: 'mango',
        5: 'capsicum'
    }
COLOURS = {
        0: (0, 0, 0),
        1: (0, 255, 0),
        2: (227, 127, 27),
        3: (204, 0, 255),
        4: (255, 0, 0),
        5: (255, 102, 0)
}

img_folder = 'dataset/images'  # folder containing augmented images
labels_BB_folder = 'dataset/images_BB'  # output folder for image with bounding boxes and labels
labels_YOLO_folder = 'dataset/labels_YOLO'  # output folder to store YOLO compatible txt labels
final_total_image_count = len(os.listdir(labels_YOLO_folder))
img_count = 0

for filename in os.listdir(labels_YOLO_folder):
    # Load the labeled image
    image_name = filename.split('.txt')[0]  # 'image_0'
    image = cv2.imread(f'{img_folder}/{image_name}.png')
    # print(image.shape)

    # Read the data from text file
    with open(f'{labels_YOLO_folder}/{filename}', 'r') as f:
        lines = [line.strip() for line in f.readlines()]  # store all lines into array and remov '\n' at end of each line

        for line in lines:
            data = line.split(' ')
            label = int(data[0])

            fruit_mid_x_ratio = float(data[1])
            fruit_mid_y_ratio = float(data[2])
            fruit_width_ratio = float(data[3])
            fruit_height_ratio = float(data[4])

            image_width = image.shape[1]
            image_height = image.shape[0]

            fruit_mid_x = int(fruit_mid_x_ratio * image_width)
            fruit_mid_y = int(fruit_mid_y_ratio * image_height)
            fruit_width = int(fruit_width_ratio * image_width)
            fruit_height = int(fruit_height_ratio * image_height)

            fruit_left = fruit_mid_x - fruit_width//2
            fruit_right = fruit_mid_x + fruit_width//2
            fruit_top = fruit_mid_y - fruit_height//2
            fruit_bottom = fruit_mid_y + fruit_height//2

            # Draw the bounding box on the image
            cv2.rectangle(image, (fruit_left, fruit_top), (fruit_right, fruit_bottom), COLOURS[label], 2)

            # Display the class label on top of the box
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(image, f'{FRUIT_LABELS[label]}', (fruit_left+3, fruit_top+15), font, 0.6, COLOURS[label], 2)

    # Save the annotated image after placing all bounding boxes
    cv2.imwrite(f'{labels_BB_folder}/{image_name}.png', image)

    # increment image count
    img_count += 1
    print(f"Image count: {img_count} / {final_total_image_count} \t {round((img_count / final_total_image_count) * 100, 2)}% complete\n")

    # # Display the image (optional)
    # cv2.imshow('Annotated Image', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()





