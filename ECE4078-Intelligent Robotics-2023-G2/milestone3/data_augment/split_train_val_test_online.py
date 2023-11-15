import os
import shutil
import random

# folder paths
images_folder = 'dataset/images'  # folder containing images for training
labels_YOLO_folder = 'dataset/labels_YOLO'  # output folder to store YOLO compatible txt labels
test_images_folder = 'YOLO_dataset_5/test/images'
test_labels_folder = 'YOLO_dataset_5/test/labels'
train_images_folder = 'YOLO_dataset_5/train/images'
train_labels_folder = 'YOLO_dataset_5/train/labels'
valid_images_folder = 'YOLO_dataset_5/valid/images'
valid_labels_folder = 'YOLO_dataset_5/valid/labels'
online_bg_set_range = [4950, 5939]

# CONSTANTS
train_ratio = 0.8
valid_ratio = 0.1
test_ratio = 0.1

total_img_count = len(os.listdir(images_folder))
train_img_count = int(train_ratio * total_img_count)
valid_img_count = int(valid_ratio * total_img_count)
test_img_count = int(test_ratio * total_img_count)



# Splitting filenames into train, test, valid
# online background: 6370 - 7349
train_online = [filename.split('.png')[0] for filename in os.listdir(images_folder) if
                online_bg_set_range[0] <= int(filename.split('.png')[0].split('_')[1]) <= online_bg_set_range[1]]
rem_arena = [filename.split('.png')[0] for filename in os.listdir(images_folder)
             if filename.split('.png')[0] not in train_online]  # images without online bg
train_arena_count = train_img_count - len(train_online)
train_arena = random.sample(rem_arena, train_arena_count)
train = train_arena + train_online
rem_arena = [filename.split('.png')[0] for filename in os.listdir(images_folder)
             if filename.split('.png')[0] not in train]  # images without online bg and arena train set
val = random.sample(rem_arena, valid_img_count)
test = [filename for filename in rem_arena if filename not in val]
train.sort()
val.sort()
test.sort()
# print(train, '\n')
# print(val, '\n')
# print(test, '\n')

img_count = 0
# populate train folder
for filename in train:  # filename not including ext, 'image_0'
    # copy train data
    src_x = os.path.join(images_folder, f'{filename}.png')
    dst_x = os.path.join(train_images_folder, f'{filename}.png')
    shutil.copy(src_x, dst_x)

    # copy train labels
    src_y = os.path.join(labels_YOLO_folder, f'{filename}.txt')
    dst_y = os.path.join(train_labels_folder, f'{filename}.txt')
    shutil.copy(src_y, dst_y)

    img_count += 1
    print(
        f"Image count: {img_count} / {total_img_count} \t {round((img_count / total_img_count) * 100, 2)}% complete\n")

# populate validation folder
for filename in val:  # filename not including ext, 'image_0'
    # copy val data
    src_x = os.path.join(images_folder, f'{filename}.png')
    dst_x = os.path.join(valid_images_folder, f'{filename}.png')
    shutil.copy(src_x, dst_x)

    # copy val labels
    src_y = os.path.join(labels_YOLO_folder, f'{filename}.txt')
    dst_y = os.path.join(valid_labels_folder, f'{filename}.txt')
    shutil.copy(src_y, dst_y)

    img_count += 1
    print(
        f"Image count: {img_count} / {total_img_count} \t {round((img_count / total_img_count) * 100, 2)}% complete\n")

# populate test folder
for filename in test:  # filename not including ext, 'image_0'
    # copy test data
    src_x = os.path.join(images_folder, f'{filename}.png')
    dst_x = os.path.join(test_images_folder, f'{filename}.png')
    shutil.copy(src_x, dst_x)

    # copy test labels
    src_y = os.path.join(labels_YOLO_folder, f'{filename}.txt')
    dst_y = os.path.join(test_labels_folder, f'{filename}.txt')
    shutil.copy(src_y, dst_y)

    img_count += 1
    print(
        f"Image count: {img_count} / {total_img_count} \t {round((img_count / total_img_count) * 100, 2)}% complete\n")
