import cv2
import numpy as np
import os, random
import itertools
import math

# control inputs
MAX_NUM_OF_FRUITS = 4
# MAX_NUM_OF_FRUITS = 1
FRUIT_COUNT = [16, 9, 12, 7, 12]  # Number of fruits in each category
TOTAL_IMAGE_COUNT = 10000
# TOTAL_IMAGE_COUNT = 1000
FRUIT_NUM_DISTRIBUTION = {1: 0.35, 2: 0.35, 3: 0.2, 4: 0.1}  # for this number of fruits, what is the percentage of the total image count?
OCCLUSION_DISTRIBUTION = {'aruco': 0.2, 'fruit': 0.2, 'both': 0.1, 'none': 0.5}
# augmentation hyperparameters
ARENA_CROP_SCALE = (1.0, 1.5)  # 1.0= ori size  # DONE
FRUIT_ROTATE_ANGLE = (-45, 45)  # degrees  # DONE
FRUIT_SCALE = (0.2, 0.8)
# colour augmentation
CONTRAST_RANGE = (0.9, 1.1)  # DONE
BRIGHTNESS_RANGE = (-20, 20)  # DONE
SATURATION_RANGE = (0.9, 1.1)  # DONE

# constants
FRUITS = ['red_apple', 'green_apple', 'orange', 'mango', 'capsicum']
FRUIT_LABELS = {
        # 'bg': 0,
        'red_apple': 1,
        'green_apple': 2,
        'orange': 3,
        'mango': 4,
        'capsicum': 5
    }
FRUIT_FILENAMES = [filename for filename in os.listdir('pics/fruits') if filename.endswith('.png')]
ARENA_FILENAMES = [filename for filename in os.listdir('pics/arena') if filename.endswith('.png')]  # ignore folders
ARUCO_FILENAMES = [filename for filename in os.listdir('pics/aruco') if filename.endswith('.png')]
SHOE_FILENAMES = [filename for filename in os.listdir('pics/shoes') if filename.endswith('.png')]

# image num calculation
NUM_OF_BG_IMG = len(ARENA_FILENAMES)
NUM_OF_IMG_PER_BG_IMG = int(TOTAL_IMAGE_COUNT/NUM_OF_BG_IMG)
FRUIT_TYPE_COMBINATIONS = {1: 0, 2: 0, 3: 0, 4: 0}  # {1: 5, 2: 10, 3: 10, 4: 5} # for this number of fruits how many possible combination of different types are there?
for num_of_fruits in range(1, MAX_NUM_OF_FRUITS+1):
    FRUIT_TYPE_COMBINATIONS[num_of_fruits] = len(list(itertools.combinations(range(len(FRUITS)), num_of_fruits)))
FILE_NUM_COMBINATIONS = {1: 0, 2: 0, 3: 0, 4: 0}  # {1: 20, 2: 17, 3: 17, 4: 10}
for num_of_fruits in range(1, MAX_NUM_OF_FRUITS+1):
    FILE_NUM_COMBINATIONS[num_of_fruits] = int((NUM_OF_IMG_PER_BG_IMG*FRUIT_NUM_DISTRIBUTION[num_of_fruits])/FRUIT_TYPE_COMBINATIONS[num_of_fruits])
print(f'FRUIT_TYPE_COMBINATIONS: {FRUIT_TYPE_COMBINATIONS}')
print(f'FILE_NUM_COMBINATIONS: {FILE_NUM_COMBINATIONS}\n')
final_total_image_count = 0
for num_of_fruits in range(1, MAX_NUM_OF_FRUITS+1):
    final_total_image_count += FILE_NUM_COMBINATIONS[num_of_fruits]*FRUIT_TYPE_COMBINATIONS[num_of_fruits]
final_total_image_count *= NUM_OF_BG_IMG

# folder paths
images_folder = 'dataset/images'  # folder containing images for training
images_BB_folder = 'dataset/images_BB'  # output folder for images with bounding boxes and labels
labels_segment_folder = 'dataset/labels_segment'  # output folder for segmented images
labels_segment_visible_folder = 'dataset/labels_segment_visible'
labels_YOLO_folder = 'dataset/labels_YOLO'  # output folder to store YOLO compatible txt labels

# Initialize a variable to keep track of the total number of combinations
img_count = 0


def augment_single_image(fruit_im):
    # 3 - flip fruit image only (hori/vert/both)
    flip_fruit = random.choice([True, False])
    if flip_fruit:
        flip_type = random.choice([-1, 0, 1])  # 1-horizontal flip, 0-vertical, -1-both
        fruit_im = cv2.flip(fruit_im, flip_type)
        # print(f'flipped fruit: {flip_type}')

    # 4 - rotate fruit within a range
    # Create rotation matrix and Apply the rotation to the image
    angle = random.randint(FRUIT_ROTATE_ANGLE[0], FRUIT_ROTATE_ANGLE[1])
    rotation_matrix = cv2.getRotationMatrix2D((fruit_im.shape[1] / 2, fruit_im.shape[0] / 2), angle, 1)
    rotated_image = cv2.warpAffine(fruit_im, rotation_matrix, (fruit_im.shape[1], fruit_im.shape[0]))
    fruit_im = rotated_image

    # 5 - scale fruit within a range
    scale = round(random.uniform(FRUIT_SCALE[0], FRUIT_SCALE[1]), 1)  # 0.1-1.0, 1 d.p.
    scale_matrix = cv2.getRotationMatrix2D((fruit_im.shape[1] / 2, fruit_im.shape[0] / 2), 0, scale)
    # scale_matrix = cv2.getRotationMatrix2D((fruit_im.shape[1] / 2, fruit_im.shape[0] / 2), 0, 0.8)
    scaled_image = cv2.warpAffine(fruit_im, scale_matrix, (fruit_im.shape[1], fruit_im.shape[0]))
    fruit_im = scaled_image

    # check fruit height
    fruit_im_sum_channels = np.sum(fruit_im, axis=2)  # sum up all channels to get 2D matrix
    fruit_im_sum_rows = np.sum(fruit_im_sum_channels, axis=1)
    fruit_top = np.where(fruit_im_sum_rows > 0)[0][0]
    fruit_bottom = np.where(fruit_im_sum_rows > 0)[0][-1]
    fruit_height = fruit_bottom - fruit_top

    # scale the fruit down until < half of the image height
    while fruit_height > 250:
        scale_matrix = cv2.getRotationMatrix2D((fruit_im.shape[1] / 2, fruit_im.shape[0] / 2), 0, 0.9)
        fruit_im = cv2.warpAffine(fruit_im, scale_matrix, (fruit_im.shape[1], fruit_im.shape[0]))

        # check fruit height
        fruit_im_sum_channels = np.sum(fruit_im, axis=2)  # sum up all channels to get 2D matrix
        fruit_im_sum_rows = np.sum(fruit_im_sum_channels, axis=1)
        fruit_top = np.where(fruit_im_sum_rows > 0)[0][0]
        fruit_bottom = np.where(fruit_im_sum_rows > 0)[0][-1]
        fruit_height = fruit_bottom - fruit_top

    return fruit_im


def augment_non_colour(bg_im, fruit_im_list):
    # @@@@@@@@@@@@@@@@@@@@@@@@@ ARENA @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    # 1 - flip arena image only (hori)  / no flip(ori)
    flip_arena = random.choice([True, False])
    if flip_arena:
        # print('Flipped arena')
        bg_im_augmented = cv2.flip(bg_im, 1)
    else:
        bg_im_augmented = bg_im

    # 2 - crop arena image
    scale = round(random.uniform(ARENA_CROP_SCALE[0], ARENA_CROP_SCALE[1]), 1)  # 1.0-1.8, 1 d.p.
    # print(scale)
    scale_matrix = cv2.getRotationMatrix2D((bg_im.shape[1] / 2, bg_im.shape[0] / 2), 0, scale)
    cropped_image = cv2.warpAffine(bg_im_augmented, scale_matrix, (bg_im.shape[1], bg_im.shape[0]))
    bg_im_augmented = cropped_image

    # print(f'crop scale:{scale}')

    # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ FRUITS @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    fruit_im_list_augmented = []
    for i in range(len(fruit_im_list)):
        fruit_im = fruit_im_list[i]
        fruit_im = augment_single_image(fruit_im)

        # add new fruit augmented fruit image into list
        fruit_im_list_augmented.append(fruit_im)

    return bg_im_augmented, fruit_im_list_augmented


def get_obj_boundaries(im):
    img_sum_channels = np.sum(im, axis=2)  # sum up all channels to get 2D matrix
    img_sum_columns = np.sum(img_sum_channels, axis=0)
    left = np.where(img_sum_columns > 0)[0][0]
    right = np.where(img_sum_columns > 0)[0][-1]
    img_sum_rows = np.sum(img_sum_channels, axis=1)
    top = np.where(img_sum_rows > 0)[0][0]
    bottom = np.where(img_sum_rows > 0)[0][-1]

    width = right - left
    height = bottom - top

    return left, right, top, bottom, width, height


def get_occlusion_boundaries(bg_im, back_img, front_img, back_img_top, back_img_left):
    _, _, _, _, back_img_width, back_img_height = get_obj_boundaries(back_img)
    _, _, _, _, front_img_width, front_img_height = get_obj_boundaries(front_img)
    bg_im_width = bg_im.shape[1]
    bg_im_height = bg_im.shape[0]
    # print(f'\nback_img_width: {back_img_width} \t back_img_height: {back_img_height}')
    # print(f'front_img_width: {front_img_width} \t front_img_height: {front_img_height}')
    # print(f'back_img_top: {back_img_top} \t back_img_left: {back_img_left}')

    coverage = 0.05
    x_min = back_img_left + int(coverage*back_img_width) - front_img_width
    if x_min < 0:
        x_min = 0
    elif x_min > bg_im_width - front_img_width:
        x_min = bg_im_width - front_img_width
    x_max = back_img_left + back_img_width - int(coverage*back_img_width)
    if x_max > bg_im_width - front_img_width:
        x_max = bg_im_width - front_img_width
    elif x_max < 0:
        x_max = 0
    y_min = back_img_top + int(coverage*back_img_height)
    if y_min < 0:
        y_min = 0
    elif y_min > bg_im_height - front_img_height:
        y_min = bg_im_height - front_img_height
    y_max = back_img_top + back_img_height - int(coverage * back_img_height)
    if y_max > bg_im_height - front_img_height:
        y_max = bg_im_height - front_img_height
    elif y_max < 0:
        y_max = 0
    # print(f'\nx_min: {x_min} \t x_max: {x_max} \t y_min: {y_min} \t y_max:{y_max}\n')

    return x_min, x_max, y_min, y_max


def is_out_of_frame(bg_im, front_im, front_img_top, front_img_left):
    _, _, _, _, front_img_width, front_img_height = get_obj_boundaries(front_im)
    bg_im_width = bg_im.shape[1]
    bg_im_height = bg_im.shape[0]
    front_img_right = front_img_left + front_img_width
    front_img_bottom = front_img_top + front_img_height

    if front_img_left < 0:
        return True
    if front_img_right > bg_im_width:
        return True
    if front_img_top < 0:
        return True
    if front_img_bottom > bg_im_height:
        return True

    return False


# place object in black canvas size of bg img
def place_obj(bg_im, front_im, pos):
    front_im_left, front_im_right, front_im_top, front_im_bottom, front_im_width, front_im_height = get_obj_boundaries(front_im)

    front_im_left_new = pos[1]
    front_im_right_new = pos[1] + front_im_width
    front_im_top_new = pos[0]
    front_im_bottom_new = pos[0] + front_im_height
    obj_placed = np.zeros_like(bg_im, np.uint8)
    obj_placed[front_im_top_new:front_im_bottom_new, front_im_left_new:front_im_right_new] = front_im[front_im_top:front_im_bottom,
                                                                                                        front_im_left:front_im_right]
    return obj_placed, front_im_left_new, front_im_right_new, front_im_top_new, front_im_bottom_new, front_im_width, front_im_height


def get_yolo_prop(bg_im, obj_left_new, obj_right_new, obj_top_new, obj_bottom_new, obj_width, obj_height):
    labeled_image_width = bg_im.shape[1]
    labeled_image_height = bg_im.shape[0]
    obj_mid_x = (obj_left_new + obj_right_new) // 2
    obj_mid_y = (obj_bottom_new + obj_top_new) // 2

    obj_mid_x_ratio = round(obj_mid_x / labeled_image_width, 7)
    obj_mid_y_ratio = round(obj_mid_y / labeled_image_height, 7)
    obj_width_ratio = round(obj_width / labeled_image_width, 7)
    # if obj_left_new > 0 and obj_right_new < labeled_image_width:  # fruit not touching edges, add 1px padding to make sure fruit is IN BB
    #     obj_width_ratio = round((obj_width + 2) / labeled_image_width, 7)
    obj_height_ratio = round(obj_height / labeled_image_height, 7)
    # if obj_top_new > 0 and obj_bottom_new < labeled_image_height:
    #     obj_height_ratio = round((obj_height + 2) / labeled_image_height, 7)

    return obj_mid_x_ratio, obj_mid_y_ratio, obj_width_ratio, obj_height_ratio


def get_mask(fruit_positioned):
    # Create a mask of the fruit (true at fruit pixels and false at transparent parts)
    fruit_mask = np.sum(fruit_positioned, axis=2)
    fruit_mask = fruit_mask > 0  # only transparent pixels are 0

    # Reduce the fruit mask along its outline to remove black line artifact
    # tests: no erode # (3,3) 1  # (5,5) 1 (BEST)  # (3,3) 2  # (5,5) 2
    kernel = np.ones((5, 5), np.uint8)
    fruit_mask = cv2.erode(fruit_mask.astype(np.uint8), kernel, iterations=1)
    fruit_mask = fruit_mask.astype(bool)

    # # Feathering mask
    # # Apply Gaussian blur to the mask to create the feathered edge
    # mask_blurred = cv2.GaussianBlur(fruit_mask.astype(np.uint8)*255, (11, 11), 0)
    # mask_blurred_3chan = cv2.cvtColor(mask_blurred, cv2.COLOR_GRAY2BGR).astype('float') / 255.

    # # Normalize the blurred mask to ensure values are in the range [0, 255]
    # fruit_mask = cv2.normalize(blurred_mask, None, 0, 255, cv2.NORM_MINMAX)
    # # Convert the blurred mask back to binary (if needed)
    # _, fruit_mask = cv2.threshold(fruit_mask, 1, 255, cv2.THRESH_BINARY)
    # fruit_mask = fruit_mask.astype(bool)

    # # Display the superimposed image
    # cv2.imshow('mask2', fruit_mask.astype(np.uint8)*255)
    # cv2.imshow('mask', blurred_mask)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return fruit_mask


def position_fruits(bg_im, fruit_im_list, fruit_label_array, occlusion_type='none'):
    non_colour_augmented_im = bg_im.copy()  # bg image that will be layered on with multiple fruits
    img_list = []
    pos_list = []
    label_list = []

    # 1. choose which fruit to be 'aruco', 'fruit' or 'none'
    fruits_occlusion = ['none']*len(fruit_im_list)
    if occlusion_type == 'aruco':  # aruco can be first one
        aruco_id = random.choice(list(range(0, len(fruit_im_list))))  # compulsory aruco
        fruits_occlusion[aruco_id] = 'aruco'
    elif occlusion_type == 'fruit':  # fruit cannot be first one
        if len(fruit_im_list) > 1:  # one don't have fruit occlusion
            fruit_id = random.choice(list(range(1, len(fruit_im_list))))
            fruits_occlusion[fruit_id] = 'fruit'
    elif occlusion_type == 'both':
        if len(fruit_im_list) > 1:  # one don't have fruit occlusion
            id_list = list(range(0, len(fruit_im_list)))
            fruit_id = random.choice([i for i in id_list if i > 0])
            id_list_rem = [i for i in id_list if i != fruit_id]
            # print(id_list_rem)
            aruco_id = random.choice(id_list_rem)
            fruits_occlusion[fruit_id] = 'fruit'
            fruits_occlusion[aruco_id] = 'aruco'

    # 2. obtain positions, images and labels for all objects
    for i, fruit_im in enumerate(fruit_im_list):
        # get boundaries of fruit
        fruit_left, fruit_right, fruit_top, fruit_bottom, fruit_width, fruit_height = get_obj_boundaries(fruit_im)
        # obtain fruit position
        if fruits_occlusion[i] == 'none':  # randomly place
            pos = (random.randint(bg_im.shape[0] // 2 - 20, bg_im.shape[0] - fruit_height),  # restrict to bottom half
                   random.randint(0, bg_im.shape[1] - fruit_width))  # position to place top left of fruit
            pos_list.append(pos)
            img_list.append(fruit_im)
            label_list.append(fruit_label_array[i])
        elif fruits_occlusion[i] == 'aruco':  # randomly placed with 95% aruco on top
            # placing fruit
            pos = (random.randint(bg_im.shape[0] // 2 - 20, bg_im.shape[0] - fruit_height),  # restrict to bottom half
                   random.randint(0, bg_im.shape[1] - fruit_width))  # position to place top left of fruit
            pos_list.append(pos)
            img_list.append(fruit_im)
            label_list.append(fruit_label_array[i])

            # placing aruco
            aruco_img_num = random.choice(list(range(1, len(ARUCO_FILENAMES)+1)))
            aruco_img = cv2.imread(f'pics/aruco/aruco_{aruco_img_num}.png')
            aruco_img = augment_single_image(aruco_img)
            x_min, x_max, y_min, y_max = get_occlusion_boundaries(bg_im, fruit_im, aruco_img, pos[0], pos[1])
            aruco_pos = (random.randint(y_min, y_max), random.randint(x_min, x_max))
            # while is_out_of_frame(bg_im, aruco_img, aruco_pos[0], aruco_pos[1]):
            #     aruco_pos = (random.randint(y_min, y_max), random.randint(x_min, x_max))
            pos_list.append(aruco_pos)
            img_list.append(aruco_img)
            label_list.append(0)
        elif fruits_occlusion[i] == 'fruit':  # randomly placed with 95% fruit on top
            # placing occlusion fruit
            x_min, x_max, y_min, y_max = get_occlusion_boundaries(bg_im, img_list[-1], fruit_im, pos_list[-1][0], pos_list[-1][1])
            # print(f'\nx_min: {x_min} \t x_max: {x_max} \t y_min: {y_min} \t y_max:{y_max}\n')
            pos = (random.randint(y_min, y_max), random.randint(x_min, x_max))
            # while is_out_of_frame(bg_im, fruit_im, pos[0], pos[1]):
            #     pos = (random.randint(y_min, y_max), random.randint(x_min, x_max))
            pos_list.append(pos)
            img_list.append(fruit_im)
            label_list.append(fruit_label_array[i])

    # 3. add shoe randomly (max 1 per image)
    if random.choice([0,1,0,0]) == 1:
        ind = random.randint(0, len(img_list))
        shoe_img_num = random.choice(list(range(1, len(SHOE_FILENAMES) + 1)))
        shoe_img = cv2.imread(f'pics/shoes/shoe_{shoe_img_num}.png')
        shoe_img = augment_single_image(shoe_img)
        shoe_left, shoe_right, shoe_top, shoe_bottom, shoe_width, shoe_height = get_obj_boundaries(shoe_img)
        shoe_pos = (random.randint(bg_im.shape[0] // 2 - 20, bg_im.shape[0] - shoe_height),  # restrict to bottom half
                    random.randint(0, bg_im.shape[1] - shoe_width))
        pos_list.insert(ind, shoe_pos)
        img_list.insert(ind, shoe_img)
        label_list.insert(ind, 0)

    # 4. superimpose objects with bg, obtain YOLO labels and save in text files
    output_filename = f'{labels_YOLO_folder}/image_{img_count}.txt'
    with open(output_filename, 'w') as f:
        for i, obj_im in enumerate(img_list):
            obj_positioned, front_im_left_new, front_im_right_new, front_im_top_new, front_im_bottom_new, front_im_width, front_im_height = \
                place_obj(bg_im, obj_im, pos_list[i])

            # get properties for YOLO txt file
            obj_mid_x_ratio, obj_mid_y_ratio, obj_width_ratio, obj_height_ratio = \
                get_yolo_prop(bg_im, front_im_left_new, front_im_right_new, front_im_top_new, front_im_bottom_new,
                              front_im_width, front_im_height)

            # write fruit class and properties to YOLO txt file
            f.write(f"{label_list[i]} {obj_mid_x_ratio} {obj_mid_y_ratio} {obj_width_ratio} {obj_height_ratio}\n")

            obj_mask = get_mask(obj_positioned)

            # Superimpose fruit image onto bg image (updates as it goes)
            non_colour_augmented_im[obj_mask] = obj_positioned[obj_mask]
            # fruit_positioned = fruit_positioned.astype('float') / 255.
            # non_colour_augmented_im = non_colour_augmented_im.astype('float') / 255.
            # non_colour_augmented_im = non_colour_augmented_im * (1 - mask_blurred_3chan) + fruit_positioned * mask_blurred_3chan

            # # Display the superimposed image
            # cv2.imshow('mask2', non_colour_augmented_im)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

    return non_colour_augmented_im


def augment_colour(non_colour_augmented_im):
    contrast = round(random.uniform(CONTRAST_RANGE[0], CONTRAST_RANGE[1]), 1)  # Adjust the contrast (1.0 means no change)
    brightness = random.randint(BRIGHTNESS_RANGE[0], BRIGHTNESS_RANGE[1])  # Adjust the brightness (positive or negative value)
    saturation = round(random.uniform(SATURATION_RANGE[0], SATURATION_RANGE[1]), 1)  # Adjust the saturation (1.0 means no change)
    # print(f'contrast:{contrast},\t brightness:{brightness},\t saturation:{saturation}\n')
    # adjust contrast & brightness
    colour_augmented_im = cv2.convertScaleAbs(non_colour_augmented_im, alpha=contrast, beta=brightness)
    # adjust saturation
    hsv_image = cv2.cvtColor(colour_augmented_im, cv2.COLOR_BGR2HSV)  # convert to HSV space
    hsv_image[:, :, 1] = np.clip(hsv_image[:, :, 1] * saturation, 0, 255).astype(np.uint8)
    colour_augmented_im = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)  # convert back to RGB

    return colour_augmented_im


def load_images(bg_im_name, fruit_type_comb, file_num_comb):
    # Load the background image
    bg_im = cv2.imread(f'pics/arena/{bg_im_name}')
    # print(bg_img.shape)  # (height, width, channel) / (row, columns, channel)

    # load required number of fruit images for combo
    fruit_im_list = []
    fruit_label_array = []  # used for labels

    for j in range(len(file_num_comb)):
        # Load the fruit image with an alpha channel (transparency)
        fruit_type = FRUITS[fruit_type_comb[j]]
        file_num = file_num_comb[j]
        fruit_img_filename = f'{fruit_type}_{file_num}.png'
        fruit_img = cv2.imread(f'pics/fruits/{fruit_img_filename}')

        # Resize the fruit image to match the dimensions of the background image
        fruit_img = cv2.resize(fruit_img, (bg_im.shape[1], bg_im.shape[0]))

        # add to image list and label list
        fruit_im_list.append(fruit_img)
        fruit_label_array.append(FRUIT_LABELS[fruit_type])

    return bg_im, fruit_im_list, fruit_label_array


if __name__ == "__main__":
    # Iterate through all background images
    for bg_img_name in ARENA_FILENAMES:
        # Iterate through all possible combinations of 1, 2, and 3 fruits
        for num_of_fruits in range(1, MAX_NUM_OF_FRUITS+1):
            # Generate combinations of fruit categories with different fruits
            fruit_type_combinations = list(itertools.combinations(range(len(FRUITS)), num_of_fruits))  # combination of fruit categories

            # Iterate through each combination of fruit types
            for fruit_type_combo in fruit_type_combinations:
                # get all possible combinations between image file names
                file_num_combinations = []
                if num_of_fruits == 1:
                    fruit_type1 = range(1, FRUIT_COUNT[fruit_type_combo[0]] + 1)
                    file_num_combinations = list(itertools.product(fruit_type1))
                elif num_of_fruits == 2:
                    fruit_type1 = range(1, FRUIT_COUNT[fruit_type_combo[0]] + 1)
                    fruit_type2 = range(1, FRUIT_COUNT[fruit_type_combo[1]] + 1)
                    file_num_combinations = list(itertools.product(fruit_type1, fruit_type2))
                    # print(file_num_combinations)
                elif num_of_fruits == 3:
                    fruit_type1 = range(1, FRUIT_COUNT[fruit_type_combo[0]] + 1)
                    fruit_type2 = range(1, FRUIT_COUNT[fruit_type_combo[1]] + 1)
                    fruit_type3 = range(1, FRUIT_COUNT[fruit_type_combo[2]] + 1)
                    file_num_combinations = list(itertools.product(fruit_type1, fruit_type2, fruit_type3))
                elif num_of_fruits == 4:
                    fruit_type1 = range(1, FRUIT_COUNT[fruit_type_combo[0]] + 1)
                    fruit_type2 = range(1, FRUIT_COUNT[fruit_type_combo[1]] + 1)
                    fruit_type3 = range(1, FRUIT_COUNT[fruit_type_combo[2]] + 1)
                    fruit_type4 = range(1, FRUIT_COUNT[fruit_type_combo[3]] + 1)
                    file_num_combinations = list(itertools.product(fruit_type1, fruit_type2, fruit_type3, fruit_type4))

                # add additional combinations until required number of combinations for this num of fruit is met
                final_file_num_combinations = []
                if len(file_num_combinations) < FILE_NUM_COMBINATIONS[num_of_fruits]:
                    final_file_num_combinations = file_num_combinations.copy()
                    while len(final_file_num_combinations) < FILE_NUM_COMBINATIONS[num_of_fruits]:
                        final_file_num_combinations.append(random.choice(file_num_combinations))
                elif len(file_num_combinations) == FILE_NUM_COMBINATIONS[num_of_fruits]:
                    final_file_num_combinations = file_num_combinations.copy()
                else:  # len(file_num_combinations) > FILE_NUM_COMBINATIONS[num_of_fruits]
                    final_file_num_combinations = random.sample(file_num_combinations, FILE_NUM_COMBINATIONS[num_of_fruits])

                occluded_aruco_list = []
                occluded_fruit_list = []
                occluded_both_list = []
                no_occlusion_list = []
                if num_of_fruits == 1:  # 1 type of fruit cannot have occlusion by fruits
                    occluded_aruco_list = random.sample(final_file_num_combinations, round(OCCLUSION_DISTRIBUTION['aruco'] * len(final_file_num_combinations)))
                    remaining = [combo for combo in final_file_num_combinations if combo not in occluded_aruco_list]
                    no_occlusion_list = remaining
                elif num_of_fruits > 1:
                    occluded_aruco_list = random.sample(final_file_num_combinations, round(OCCLUSION_DISTRIBUTION['aruco']*len(final_file_num_combinations)))
                    remaining = [combo for combo in final_file_num_combinations if combo not in occluded_aruco_list]
                    occluded_fruit_list = random.sample(remaining, round(OCCLUSION_DISTRIBUTION['fruit']*len(final_file_num_combinations)))
                    remaining = [combo for combo in remaining if combo not in occluded_fruit_list]
                    occluded_both_list = random.sample(remaining, round(OCCLUSION_DISTRIBUTION['both']*len(final_file_num_combinations)))
                    remaining = [combo for combo in remaining if combo not in occluded_both_list]
                    no_occlusion_list = remaining

                # loop through each possible combination of file numbers
                for file_num_combo in final_file_num_combinations:  # length of fruit type combo == length of file num combo
                    occlusion = ''
                    if file_num_combo in occluded_aruco_list:
                        occlusion = 'aruco'
                    elif file_num_combo in occluded_fruit_list:
                        occlusion = 'fruit'
                    elif file_num_combo in occluded_both_list:
                        occlusion = 'both'
                    elif file_num_combo in no_occlusion_list:
                        occlusion = 'none'

                    # load images to superimpose and their labels
                    bg_img, fruit_img_list, fruit_label_list = load_images(bg_img_name, fruit_type_combo, file_num_combo)

                    # non-colour augmentation
                    bg_img, fruit_img_list = augment_non_colour(bg_img, fruit_img_list)

                    # position fruits in bg image
                    non_colour_augmented_img = position_fruits(bg_img, fruit_img_list, fruit_label_list, occlusion_type=occlusion)

                    # colour augmentation
                    colour_augmented_img = augment_colour(non_colour_augmented_img)

                    # save image
                    cv2.imwrite(f'{images_folder}/image_{img_count}.png', colour_augmented_img)

                    # # Display the superimposed image
                    # cv2.imshow('Final Augmented Image', colour_augmented_img)
                    # cv2.waitKey(0)
                    # cv2.destroyAllWindows()

                    # increment image count
                    img_count += 1
                    print(f"Image count: {img_count} / {final_total_image_count} \t {round((img_count/final_total_image_count)*100,2)}% complete\n")





