# M4 - Autonomous fruit searching

# basic python packages
import sys, os
import cv2
import numpy as np
import json
import ast
import argparse
import time
import math

# import SLAM components
# sys.path.insert(0, "{}/slam".format(os.getcwd()))
# from slam.ekf import EKF
# from slam.robot import Robot
# import slam.aruco_detector as aruco

# import utility functions
sys.path.insert(0, "util")
from pibot import Alphabot
import measure as measure
from a_star import a_star
from network.scripts.detector import Detector
from TargetPoseEst import estimate_pose

def read_true_map(fname):
    """Read the ground truth map and output the pose of the ArUco markers and 3 types of target fruit to search

    @param fname: filename of the map
    @return:
        1) list of target fruits, e.g. ['redapple', 'greenapple', 'orange']
        2) locations of the target fruits, [[x1, y1], ..... [xn, yn]]
        3) locations of ArUco markers in order, i.e. pos[9, :] = position of the aruco10_0 marker
    """
    with open(fname, 'r') as f:
        try:
            gt_dict = json.load(f)                   
        except ValueError as e:
            with open(fname, 'r') as f:
                gt_dict = ast.literal_eval(f.readline())   
        fruit_list = []
        fruit_true_pos = []
        aruco_true_pos = np.empty([10, 2])

        taglist = []
        markers = [[], []]  # 2 rows n columns

        # remove unique id of targets of the same type
        for key in gt_dict:
            x = np.round(gt_dict[key]['x'], 1)
            y = np.round(gt_dict[key]['y'], 1)

            if key.startswith('aruco'):
                marker_id = int(key.split('_0')[0].split('aruco')[1]) - 1
                aruco_true_pos[marker_id][0] = x
                aruco_true_pos[marker_id][1] = y

                # add for generating slam.txt
                taglist.append(marker_id+1)
                markers[0].append(x)
                markers[1].append(y)

            else:  # append 3 fruit names from true map to fruit list
                fruit_list.append(key[:-2])
                if len(fruit_true_pos) == 0:
                    fruit_true_pos = np.array([[x, y]])
                else:
                    fruit_true_pos = np.append(fruit_true_pos, [[x, y]], axis=0)

        # for marker_id in taglist:
        #     markers[0][marker_id] = aruco_true_pos[marker_id][0]
        #     markers[1][marker_id] = aruco_true_pos[marker_id][1]
        #     taglist[marker_id] += 1

        # print(taglist)
        # print(markers)

        map_attributes = {"taglist": taglist,
                          "markers": markers}

        with open('slam.txt', 'w') as map_file:
            json.dump(map_attributes, map_file, indent=2)

        return fruit_list, fruit_true_pos, aruco_true_pos, gt_dict


def read_search_list():
    """Read the search order of the target fruits

    @return: search order of the target fruits
    """
    search_list = []
    with open('search_list.txt', 'r') as fd:
        fruits = fd.readlines()

        for fruit in fruits:
            search_list.append(fruit.strip())

    return search_list


def print_target_fruits_pos(search_list, fruit_list, fruit_true_pos):
    """Print out the target fruits' pos in the search order (fruits_list from true map may not be same order as search list)

    @param search_list: search order of the fruits
    @param fruit_list: list of target fruits
    @param fruit_true_pos: positions of the target fruits
    """
    target_fruits = {}

    print("Search order:")
    n_fruit = 1
    for fruit in search_list:
        for i in range(3):
            if fruit == fruit_list[i]:
                target_fruits[fruit] = fruit_true_pos[i]
                print('{}) {} at [{}, {}]'.format(n_fruit,
                                                  fruit,
                                                  np.round(fruit_true_pos[i][0], 1),
                                                  np.round(fruit_true_pos[i][1], 1)))
        n_fruit += 1

    return target_fruits


# Waypoint navigation
# the robot automatically drives to a given [x,y] coordinate
# additional improvements:
# you may use different motion model parameters for robot driving on its own or driving while pushing a fruit
# try changing to a fully automatic delivery approach: develop a path-finding algorithm that produces the waypoints
def drive_to_point(waypoint, robot_pose):
    # imports camera / wheel calibration parameters 
    fileS = "calibration/param/scale.txt"
    scale = np.loadtxt(fileS, delimiter=',')
    fileB = "calibration/param/baseline.txt"
    baseline = np.loadtxt(fileB, delimiter=',')
    fileBL = "calibration/param/baseline_left.txt"
    baseline_left = np.loadtxt(fileBL, delimiter=',')
    fileBR = "calibration/param/baseline_right.txt"
    baseline_right = np.loadtxt(fileBR, delimiter=',')

    drive_ticks = 25
    turn_ticks = 15
    # get distance and angle to travel
    print(robot_pose)
    x_diff = waypoint[0] - robot_pose[0]
    y_diff = waypoint[1] - robot_pose[1]
    angle = np.arctan2(y_diff, x_diff) - robot_pose[2]
    desired_heading = (angle + np.pi) % (2 * np.pi) - np.pi # limits range of angle to the range [-pi,pi]
    print(desired_heading)
    if desired_heading == -np.pi:
        desired_heading = np.pi

    # turn towards the waypoint
    dir = desired_heading/abs(desired_heading)
    # turn_time = 0.0
    # if dir == 1:
    turn_time = baseline*abs(desired_heading)/(2*scale*turn_ticks)  # replace with your calculation
    # elif dir == -1:
    #     turn_time = baseline*abs(desired_heading)/(2*scale*turn_ticks)  # replace with your calculation

    # if desired_heading == (0/180)*np.pi:
    #     turn_time = 0.0
    # if desired_heading == (45/180)*np.pi:
    #     turn_time = 0.2
    # if desired_heading == (90/180)*np.pi:
    #     turn_time = 0.35
    # if desired_heading == (135/180)*np.pi:
    #     turn_time = 0.515
    # if desired_heading == (180/180)*np.pi:
    #     turn_time = 0.665
    # if desired_heading == -(45/180)*np.pi:
    #     turn_time = 0.2
    # if desired_heading == (-90/180)*np.pi:
    #     turn_time = 0.351
    # if desired_heading == (-135/180)*np.pi:
    #     turn_time = 0.515
    # if desired_heading == (-180 / 180) * np.pi:
    #     turn_time = 0.665

    print("Turning for {:.2f} seconds".format(turn_time))
    pibot.set_velocity([0, dir], turning_tick=turn_ticks, time=turn_time)
    robot_pose[2] = (np.arctan2(y_diff, x_diff) + np.pi) % (
                2 * np.pi) - np.pi  # limits range of angle to the range [-pi,pi]

    # Detection
    # new_fruit_found = detect()
    # new_fruit_found = False
    # if new_fruit_found:
    #     return robot_pose, new_fruit_found

    # Drive straight to the waypoint
    distance_to_goal = np.sqrt(x_diff ** 2 + y_diff ** 2)
    drive_time = distance_to_goal/(drive_ticks*scale) # replace with your calculation
    print("Driving for {:.2f} seconds".format(drive_time))
    pibot.set_velocity([1, 0], tick=drive_ticks, time=drive_time)

    robot_pose[0] = waypoint[0]
    robot_pose[1] = waypoint[1]
    # robot_pose = get_robot_pose()
    # print("Current pose [{}, {}, {}]".format(robot_pose[0], robot_pose[1], robot_pose[2]))
    # print("Arrived at [{}, {}]".format(waypoint[0], waypoint[1]))
    return robot_pose
    ####################################################


def round_nearest(x, a):
    return round(round(x / a) * a, -int(math.floor(math.log10(a))))


def detect():
    # after turning, turn on detection
    yolo_input_img = cv2.cvtColor(pibot.get_image(), cv2.COLOR_RGB2BGR)
    bboxes, yolo_vis = detector.detect_single_image(yolo_input_img)
    new_fruit_found = False
    if bboxes:  # if fruit is detected
        for bbox in bboxes:
            # check if fruit detected is not in fruit list
            label = bbox[0]
            if label not in search_list:
                new_fruit_found = True
                target_pose = estimate_pose(cam_matrix, bbox, relative_pose_only=False)
                target_pose['x'] = round_nearest(target_pose['x'], 0.4)
                target_pose['y'] = round_nearest(target_pose['y'], 0.4)
                # add fruit in M4_TRUEMAP then generate new path
                gt_dict[f"{label}_0"] = target_pose
                with open('M4_true_map_Copy.txt', 'w') as file:
                    json.dump(gt_dict, file)

    # reset waypoint id and find path
    return new_fruit_found


def generate_move_path(robot_pose):
    input(f"press enter to search for {fruit}")
    path = a_star(f'{args.map}', target_fruits[fruit], robot_pose[0:2])
    # path = path[1:-1]
    # path.reverse()
    print(path)
    for waypoint in path:
        input(f"press enter to go next waypoint")
        # robot drives to the waypoint
        robot_pose = drive_to_point(waypoint, robot_pose)
        print("Finished driving to waypoint: {}; New robot pose: {}".format(waypoint, robot_pose))

def get_robot_pose():
    ####################################################
    # TODO: replace with your codes to estimate the pose of the robot
    # We STRONGLY RECOMMEND you to use your SLAM code from M2 here

    # update the robot pose [x,y,theta]
    robot_pose = [0.0,0.0,0.0] # replace with your calculation
    ####################################################
    return robot_pose

# main loop
if __name__ == "__main__":
    parser = argparse.ArgumentParser("Fruit searching")
    # parser.add_argument("--map", type=str, default='TRUEMAP_ALIGNED.txt')
    parser.add_argument("--ip", metavar='', type=str, default='192.168.137.191')
    parser.add_argument("--port", metavar='', type=int, default=8000)
    # parser.add_argument("--yolo_model", default='model.best.pt')
    args, _ = parser.parse_known_args()

    pibot = Alphabot(args.ip,args.port)

    # read in the true map
    # fruits_list, fruits_true_pos, aruco_true_pos, gt_dict = read_true_map(args.map)
    # search_list = read_search_list()
    # print(fruits_list)
    # target_fruits = print_target_fruits_pos(search_list, fruits_list, fruits_true_pos)
    # print(target_fruits)

    # detector = Detector(f'network/scripts/model/{args.yolo_model}')

    # pibot.set_velocity([0, 1], turning_tick=15, time=0.5)

    waypoint = [0.0,0.0]
    robot_pose = [0.0,0.0,0.0]
    level = 1
    # drive_to_point((-0.2,0.2), robot_pose)

    if level == 1:
        # The following code is only a skeleton code the semi-auto fruit searching task
        while True:
            # enter the waypoints
            # instead of manually enter waypoints in command line, you can get coordinates by clicking on a map (GUI input), see camera_calibration.py
            x,y = 0.0,0.0
            x = input("X coordinate of the waypoint: ")
            try:
                x = float(x)
            except ValueError:
                print("Please enter a number.")
                continue
            y = input("Y coordinate of the waypoint: ")
            try:
                y = float(y)
            except ValueError:
                print("Please enter a number.")
                continue

            # robot drives to the waypoint
            waypoint = [x,y]
            robot_pose = drive_to_point(waypoint,robot_pose)
            print("Finished driving to waypoint: {}; New robot xy: {}".format(waypoint,[robot_pose[0],robot_pose[1],np.rad2deg(robot_pose[2])]))

            # exit
            uInput = input("Add a new waypoint? [Y/N]")
            if uInput == 'N' or uInput == 'n':
                break

    elif level == 2:
        for fruit in target_fruits:
            # input(f"press enter to search for {fruit}")
            path = a_star(f'{args.map}', target_fruits[fruit], robot_pose[0:2])
            # path = path[1:-1]
            # path.reverse()
            print(path)
            for waypoint in path:
                # input(f"press enter to go next waypoint")
                # robot drives to the waypoint
                robot_pose = drive_to_point(waypoint,robot_pose)
                print("Finished driving to waypoint: {}; New robot pose: {}".format(waypoint,robot_pose))
                time.sleep(1)

            time.sleep(3)

