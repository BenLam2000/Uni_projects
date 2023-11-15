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


class AutoFruitSearch:
    def __init__(self, pibot, scale, baseline, baseline_left, baseline_right, drive_ticks, turn_left_ticks, turn_right_ticks, turn360_angle, detector=None):
        # import camera / wheel calibration parameters
        self.pibot = pibot
        self.scale = scale
        self.baseline = baseline
        self.baseline_left = baseline_left
        self.baseline_right = baseline_right
        self.drive_ticks = drive_ticks
        self.turn_left_ticks = turn_left_ticks
        self.turn_right_ticks = turn_right_ticks
        self.desired_state = np.zeros((3,1))
        self.robot_state = np.zeros((3,1))
        self.basic_angle = turn360_angle
        self.detector = detector

    def read_true_map(self, fname):
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

            # map_attributes = {"taglist": taglist,
            #                   "markers": markers}

            # with open('slam.txt', 'w') as map_file:
            #     json.dump(map_attributes, map_file, indent=2)

            return fruit_list, fruit_true_pos, aruco_true_pos, gt_dict


    def read_search_list(self):
        """Read the search order of the target fruits

        @return: search order of the target fruits
        """
        search_list = []
        with open('search_list.txt', 'r') as fd:
            fruits = fd.readlines()

            for fruit in fruits:
                search_list.append(fruit.strip())

        return search_list


    def print_target_fruits_pos(self, search_list, fruit_list, fruit_true_pos):
        """Print out the target fruits' pos in the search order (fruits_list from true map may not be same order as search list)

        @param search_list: search order of the fruits
        @param fruit_list: list of target fruits
        @param fruit_true_pos: positions of the target fruits
        """
        target_fruits = {}

        print("Search order:")
        n_fruit = 1
        for fruit in search_list:
            for i in range(5):
                if fruit == fruit_list[i]:
                    target_fruits[fruit] = fruit_true_pos[i]
                    print('{}) {} at [{}, {}]'.format(n_fruit,
                                                      fruit,
                                                      np.round(fruit_true_pos[i][0], 1),
                                                      np.round(fruit_true_pos[i][1], 1)))
            n_fruit += 1

        return target_fruits

    def clip_angle(self, angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi

    # Waypoint navigation
    # the robot automatically drives to a given [x,y] coordinate
    # additional improvements:
    # you may use different motion model parameters for robot driving on its own or driving while pushing a fruit
    # try changing to a fully automatic delivery approach: develop a path-finding algorithm that produces the waypoints
    def drive_to_point(self, waypoint):
        # get distance and angle to travel
        # print(robot_pose)
        x_diff = waypoint[0] - self.robot_state[0, 0]
        y_diff = waypoint[1] - self.robot_state[1, 0]
        angle = np.arctan2(y_diff, x_diff) - self.robot_state[2, 0]
        desired_heading = self.clip_angle(angle)  # limits range of angle to the range [-pi,pi]
        # print(desired_heading)
        if desired_heading == -np.pi:
            desired_heading = np.pi

        # turn towards the waypoint
        self.turn_combine(np.rad2deg(desired_heading))

        # Detection
        # new_fruit_found = detect()
        # new_fruit_found = False
        # if new_fruit_found:
        #     return robot_pose, new_fruit_found

        # Drive straight to the waypoint
        distance_to_goal = np.sqrt(x_diff ** 2 + y_diff ** 2)
        self.drive(distance_to_goal)

        # desired end state
        self.desired_state[0,0] = waypoint[0]
        self.desired_state[1,0] = waypoint[1]
        self.desired_state[2,0] = self.clip_angle(
            np.arctan2(y_diff, x_diff))  # limits range of angle to the range [-pi,pi]
        ####################################################

    # break big angles into basic small angles
    def turn_combine(self, angle_deg):
        if angle_deg > self.basic_angle:
            sections = abs(angle_deg)//self.basic_angle
            for i in range(sections):
                self.turn(math.copysign(self.basic_angle, angle_deg))
            self.turn(math.copysign(abs(angle_deg) % self.basic_angle, angle_deg))
        else:
            self.turn(angle_deg)

    def turn(self, angle_deg, turn_ticks=None, return_turn_time_lv_rv=False):
        angle_rad = np.deg2rad(angle_deg)
        direction = math.copysign(1, angle_deg)
        # direction = angle_rad / abs(angle_rad)

        motion = [0, direction]
        # turn 30 degrees
        turn_time = 0
        if turn_ticks is None:
            if motion == [0, 1]: # turn left
                turn_time = self.baseline * abs(angle_rad) / (2 * self.scale * self.turn_left_ticks)
            elif motion == [0, -1]:  # turn right
                turn_time = self.baseline * abs(angle_rad) / (2 * self.scale * self.turn_right_ticks)
        else:
            if motion == [0, 1]:  # turn left
                turn_time = self.baseline * abs(angle_rad) / (2 * self.scale * turn_ticks)
            elif motion == [0, -1]:  # turn right
                turn_time = self.baseline * abs(angle_rad) / (2 * self.scale * turn_ticks)

        print("Turning for {:.2f} seconds".format(turn_time))
        lv, rv = 0, 0
        if turn_ticks is None:
            if motion == [0, 1]:  # turn left
                lv, rv = self.pibot.set_velocity(
                    [0, 1], turning_tick=self.turn_left_ticks, tick=self.drive_ticks, time=turn_time)
            elif motion == [0, -1]:  # turn right
                lv, rv = self.pibot.set_velocity(
                    [0, -1], turning_tick=self.turn_right_ticks, tick=self.drive_ticks, time=turn_time)
        else:
            if motion == [0, 1]:  # turn left
                lv, rv = self.pibot.set_velocity(
                    [0, 1], turning_tick=turn_ticks, tick=self.drive_ticks, time=turn_time)
            elif motion == [0, -1]:  # turn right
                lv, rv = self.pibot.set_velocity(
                    [0, -1], turning_tick=turn_ticks, tick=self.drive_ticks, time=turn_time)

        if return_turn_time_lv_rv:
            return turn_time, lv, rv

    def drive(self, dist):
        drive_time = dist / (self.drive_ticks * self.scale)  # replace with your calculation
        print("Driving for {:.2f} seconds".format(drive_time))
        self.pibot.set_velocity([1, 0], tick=self.drive_ticks, turning_tick=self.turn_ticks, time=drive_time)

    def round_nearest(self, x, a):
        return round(round(x / a) * a, -int(math.floor(math.log10(a))))

    def detect(self):
        # after turning, turn on detection
        yolo_input_img = cv2.cvtColor(self.pibot.get_image(), cv2.COLOR_RGB2BGR)
        bboxes, yolo_vis = self.detector.detect_single_image(yolo_input_img)
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

    @staticmethod
    def generate_path(self, map_, goal, start):
        path = a_star(f'{map_}', goal, start)
        # path = path[1:-1]
        # path.reverse()
        return path

    def check_pose_error(self):
        error = self.robot_state - self.desired_state
        x_error = error[0,0]
        y_error = error[1,0]
        dist_error = np.sqrt(x_error ** 2 + y_error ** 2)
        theta_error = error[2,0]
        return dist_error, theta_error

    def destination_reached(self, waypoint, robot_pose):
        stop_criteria_met = False
        dist_threshold = 0.1
        heading_threshold = 0.1

        x_diff = waypoint[0] - robot_pose[0]
        y_diff = waypoint[1] - robot_pose[1]
        distance_to_goal = np.sqrt((x_diff) ** 2 + (y_diff) ** 2)
        angle = np.arctan2(y_diff, x_diff) - robot_pose[2]
        desired_heading = (angle + np.pi) % (2 * np.pi) - np.pi  # limits range of angle to the range [-pi,pi]
        if desired_heading == -np.pi:
            desired_heading = np.pi

        if (distance_to_goal <= dist_threshold and abs(desired_heading) <= heading_threshold):
            stop_criteria_met = True
            print("Arrived at [{}, {}]".format(waypoint[0], waypoint[1]))

        return stop_criteria_met

    def run(self):
        fruits_list, fruits_true_pos, aruco_true_pos, gt_dict = operate.fruit_search.read_true_map(args.map)
        search_list = operate.fruit_search.read_search_list()  # TODO: change search list filename
        print(fruits_list)
        target_fruits = operate.fruit_search.print_target_fruits_pos(search_list, fruits_list, fruits_true_pos)
        print(target_fruits)

        # flag to check if robot pose is within 0.1 away from the waypoint
        # reached = True

        for fruit in target_fruits:
            input(f"press enter to search for {fruit}")
            robot_pose = operate.ekf.robot.state
            fruit_position = np.round(target_fruits[fruit], 1)
            start_point = np.round(operate.fruit_search.robot_state[0:2, 0], 1)
            path = operate.fruit_search.generate_path(args.map, fruit_position, start_point)
            print(f"Path from {start_point} to {fruit} at {fruit_position}: {path}")
            for waypoint in path:
                # input(f"press enter to go next waypoint")
                operate.fruit_search.drive_to_point(waypoint, operate.fruit_search.robot_state)

                # RESUME SLAM + 360 turn HERE

                # update robot pose
                operate.fruit_search.robot_state = self.ekf.robot.state

                # check if robot pose is within 0.1 away from the waypoint
                reached = self.destination_reached(waypoint, robot_pose)

                if reached:
                    print("Finished driving to waypoint: {}; New robot pose: {}".format(waypoint, robot_pose))
                    operate.ekf_on = True
                    # turn 360 deg to update pose
                    for i in range(12):
                        fruit_search.automate_turn(robot_pose=robot_pose)  # turn 30 deg each time
                        # TODO: new update slam function for ekf
                        update_slam()
                else:
                    # generate new path
                    fruit_position = round(target_fruits[fruit], 1)
                    start_point = round(robot_pose[0:2], 1)
                    path = a_star(f'{args.map}', fruit_position, start_point)
                    print('New path', path)

                time.sleep(1)

            time.sleep(3)


# main loop
if __name__ == "__main__":
    parser = argparse.ArgumentParser("Fruit searching")
    parser.add_argument("--map", type=str, default='TRUEMAP_ALIGNED.txt')
    parser.add_argument("--ip", metavar='', type=str, default='192.168.137.191')
    parser.add_argument("--port", metavar='', type=int, default=8000)
    parser.add_argument("--yolo_model", default='model.best.pt')
    args, _ = parser.parse_known_args()

    pibot = Alphabot(args.ip,args.port)
    fruit_search = AutoFruitSearch()

    # # read in the true map
    # fruits_list, fruits_true_pos, aruco_true_pos, gt_dict = fruit_search.read_true_map(args.map)
    # search_list = fruit_search.read_search_list()  # TODO: change search list filename
    # print(fruits_list)
    # target_fruits = fruit_search.print_target_fruits_pos(search_list, fruits_list, fruits_true_pos)
    # print(target_fruits)

    # detector = Detector(f'network/scripts/model/{args.yolo_model}')

    # pibot.set_velocity([0, 1], turning_tick=15, time=0.5)

    # waypoint = [0.0,0.0]
    # robot_pose = [0.0,0.0,0.0]
    # level = 3
    # # drive_to_point((-0.2,0.2), robot_pose)
    #
    # if level == 1:
    #     # The following code is only a skeleton code the semi-auto fruit searching task
    #     while True:
    #         # enter the waypoints
    #         # instead of manually enter waypoints in command line, you can get coordinates by clicking on a map (GUI input), see camera_calibration.py
    #         x,y = 0.0,0.0
    #         x = input("X coordinate of the waypoint: ")
    #         try:
    #             x = float(x)
    #         except ValueError:
    #             print("Please enter a number.")
    #             continue
    #         y = input("Y coordinate of the waypoint: ")
    #         try:
    #             y = float(y)
    #         except ValueError:
    #             print("Please enter a number.")
    #             continue
    #
    #         # robot drives to the waypoint
    #         waypoint = [x,y]
    #         robot_pose = drive_to_point(waypoint,robot_pose)
    #         print("Finished driving to waypoint: {}; New robot pose: {}".format(waypoint,robot_pose))
    #
    #         # exit
    #         uInput = input("Add a new waypoint? [Y/N]")
    #         if uInput == 'N' or uInput == 'n':
    #             break
    #
    # elif level == 2:
    #     for fruit in target_fruits:
    #         # input(f"press enter to search for {fruit}")
    #         path = a_star(f'{args.map}', target_fruits[fruit], robot_pose[0:2])
    #         # path = path[1:-1]
    #         # path.reverse()
    #         print(path)
    #         for waypoint in path:
    #             # input(f"press enter to go next waypoint")
    #             # robot drives to the waypoint
    #             robot_pose = drive_to_point(waypoint,robot_pose)
    #             print("Finished driving to waypoint: {}; New robot pose: {}".format(waypoint,robot_pose))
    #             time.sleep(1)
    #
    #         time.sleep(3)

