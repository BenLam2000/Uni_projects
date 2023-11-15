import ast
import json


def parse_poses(input_filename):
    with open(input_filename, 'r') as f:
        try:
            gt_dict = json.load(f)
        except ValueError as e:
            with open(input_filename, 'r') as f:
                gt_dict = ast.literal_eval(f.readline())

        poses = []
        image_filenames = []
        for key in gt_dict:
            # [[1.2], [1.2], [3.141593]]
            poses.append([[gt_dict[key]["x"]], [gt_dict[key]["y"]], [gt_dict[key]["theta"]]])

        with open(f'lab_output/images.txt') as fp:
            for line in fp.readlines():
                pose_dict = ast.literal_eval(line)
                image_filenames.append(pose_dict['imgfname'])

    return poses, image_filenames  # [[[1.2], [1.2], [3.141593]], [[1.2], [1.2], [3.141593]], ...]


def modify_poses(poses, img_filenames, output_filename):
    f = open(output_filename, 'w')
    for i in range(len(poses)):
        img_dict = {"pose": poses[i],
                    "imgfname": img_filenames[i]}
        img_line = json.dumps(img_dict)
        f.write(img_line + '\n')
        f.flush()


if __name__ == '__main__':
    robot_poses, img_fnames = parse_poses('TRUE_ROBOT_POSE.txt')
    modify_poses(robot_poses, img_fnames, "lab_output/images.txt")
