# teleoperate the robot and perform SLAM
# will be extended in following milestones for system integration

# basic python packages
import numpy as np
import cv2 
import os, sys
import time
import json
import ast
import argparse
import random
import math

# import utility functions
sys.path.insert(0, "{}/utility".format(os.getcwd()))
from util.pibot import Alphabot # access the robot
import util.DatasetHandler as dh # save/load functions
import util.measure as measure # measurements
import pygame # python package for GUI
import shutil # python package for file operations

# import M2 SLAM components
sys.path.insert(0, "{}/slam".format(os.getcwd()))
from slam.ekf import EKF
from slam.robot import Robot
import slam.aruco_detector as aruco
import SLAM_eval

# import M3 CV components
sys.path.insert(0,"{}/network/".format(os.getcwd()))
sys.path.insert(0,"{}/network/scripts".format(os.getcwd()))
from network.scripts.detector import Detector
from ultralytics.utils import ops
import TargetPoseEst

# import M4 components
from auto_fruit_search import AutoFruitSearch
from a_star import a_star

class Operate:
    def __init__(self, args):
        self.mode = 1

        self.folder = 'pibot_dataset/'
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)
        else:
            shutil.rmtree(self.folder)
            os.makedirs(self.folder)

        # initialise data parameters
        if args.play_data:
            self.pibot = dh.DatasetPlayer("record")
        else:
            self.pibot = Alphabot(args.ip, args.port)

        # calibration params
        fileK = "calibration/param/intrinsic.txt"
        self.camera_matrix = np.loadtxt(fileK, delimiter=',')
        fileD = "calibration/param/distCoeffs.txt"
        self.dist_coeffs = np.loadtxt(fileD, delimiter=',')
        fileS = "calibration/param/scale.txt"
        self.scale = np.loadtxt(fileS, delimiter=',')
        fileB = "calibration/param/baseline.txt"
        self.baseline = np.loadtxt(fileB, delimiter=',')
        fileBL = "calibration/param/baseline_left.txt"
        self.baseline_left = np.loadtxt(fileBL, delimiter=',')
        fileBR = "calibration/param/baseline_right.txt"
        self.baseline_right = np.loadtxt(fileBR, delimiter=',')

        # initialise SLAM parameters
        self.ekf = self.init_ekf()
        self.aruco_det = aruco.aruco_detector(
            self.ekf.robot, marker_length = 0.06) # size of the ARUCO markers

        if args.save_data:
            self.data = dh.DatasetWriter('record')
        else:
            self.data = None
        self.output = dh.OutputWriter('lab_output')
        self.command = {'motion':[0, 0], 
                        'inference': False,
                        'output': False,
                        'save_inference': False,
                        'save_image': False,
                        'measure_marker': False,
                        'pose_est': False}
        self.quit = False
        self.pred_fname = ''
        self.request_recover_robot = False
        self.file_output = None
        self.ekf_on = False
        self.double_reset_comfirm = 0
        self.image_id = 0
        self.notification = 'Press ENTER to start SLAM'

        # 5min timer
        self.count_down = 300
        self.start_time = time.time()
        self.control_clock = time.time()

        # initialise images
        self.img = np.zeros([240,320,3], dtype=np.uint8)
        self.aruco_img = np.zeros([240,320,3], dtype=np.uint8)
        self.bg = pygame.image.load('pics/gui_mask.jpg')

        # wheel ticks & robot movement
        self.drive_ticks = 25
        # self.turn_ticks = 15
        self.turn_left_ticks = 15
        self.turn_right_ticks = 15
        self.turn_left_ticks_auto = 15
        self.turn_right_ticks_auto = 15
        self.small_angle = 10
        self.small_dist = 0.01
        self.turn360_angle = 30
        self.turn360_delay = 0.3  # secs delay after each 360 turn
        self.start_360_count = 0

        # M3
        self.bboxes = [['redapple', np.asarray([100, 100, 200, 200]), 0.95]]
        if args.yolo_model == "":
            self.detector = None
            self.yolo_vis = cv2.imread('pics/8bit/detector_splash.png')
        else:
            self.detector = Detector(f'network/scripts/model/{args.yolo_model}')
            self.yolo_vis = np.ones((240, 320, 3)) * 100
        self.yolo_vis_line = np.ones((240, 320, 3)) * 100

        # M4
        self.fruit_search = AutoFruitSearch(self.pibot, self.scale, self.baseline, self.baseline_left, self.baseline_right,
                                            self.drive_ticks, self.turn_left_ticks_auto, self.turn_right_ticks_auto, self.turn360_angle,
                                            self.turn360_delay)
        self.dist_threshold = 0.10
        self.marker_taglist = []
        self.marker_dist = []
        self.marker_rel = []

    # wheel control
    def control(self):       
        if args.play_data:
            lv, rv = self.pibot.set_velocity()            
        else:
            if self.command['motion'] == [1, 0]:  # forward
                lv, rv = self.pibot.set_velocity(
                    [0.97, 0], tick=self.drive_ticks)
            elif self.command['motion'] == [-1, 0]:  # back
                lv, rv = self.pibot.set_velocity(
                    [-0.81, 0], tick=self.drive_ticks)
            elif self.command['motion'] == [0, 1]:  # turn left
                lv, rv = self.pibot.set_velocity(
                    [0, 1.17], turning_tick=self.turn_left_ticks, tick=self.drive_ticks)
            elif self.command['motion'] == [0, -1]:  # turn right
                lv, rv = self.pibot.set_velocity(
                    [0, -1.2], turning_tick=self.turn_right_ticks, tick=self.drive_ticks)
            else:  # stop
                lv, rv = self.pibot.set_velocity(
                    [0, 0])
        if self.data is not None:
            self.data.write_keyboard(lv, rv)
        dt = time.time() - self.control_clock
        if self.command['motion'] == [1, 0]:  # forward
            drive_meas = measure.Drive(1.0*lv, 1.0*rv, dt)
        elif self.command['motion'] == [-1, 0]:  # back
            drive_meas = measure.Drive(1.0*lv, 1.0*rv, dt)
        elif self.command['motion'] == [0, 1]:  # turn left
            drive_meas = measure.Drive(1.0*lv, 1.0*rv, dt)
        elif self.command['motion'] == [0, -1]:  # turn right
            drive_meas = measure.Drive(1.0*lv, 1.0*rv, dt)
        else:  # stop
            drive_meas = measure.Drive(lv, rv, dt)  # always 0,0
        self.control_clock = time.time()
        return drive_meas

    def move_small_step(self):
        if self.command['motion'] == [1, 0]:  # move forward
            self.fruit_search.drive(self.small_dist)
        elif self.command['motion'] == [-1, 0]:  # move backward
            self.fruit_search.drive(-self.small_dist)
        elif self.command['motion'] == [0, 1]:  # turn left
            self.fruit_search.turn(self.small_angle)
        elif self.command['motion'] == [0, -1]:  # turn right
            self.fruit_search.turn(-self.small_angle)

    # camera control
    def take_pic(self):
        self.img = self.pibot.get_image()
        if not self.data is None:
            self.data.write_image(self.img)

    # SLAM with ARUCO markers       
    def update_slam(self, drive_meas=None):
        lms, self.aruco_img = self.aruco_det.detect_marker_positions(self.img)
        lms = [lm for lm in lms if lm.tag <= 10]  # filter unknown markers

        # if len(lms) > 0:
        #     for lm in lms:
        #         print(f"tag: {lm.tag}")
        #         print(f"position: {lm.position}")

        if not self.ekf_on:
            pass
            # if self.request_recover_robot:
            #     is_success = self.ekf.recover_from_pause(lms)
            #     if is_success:
            #         self.notification = 'Robot pose is successfuly recovered'
            #         self.ekf_on = True
            #     else:
            #         self.notification = 'Recover failed, need >2 landmarks!'
            #         self.ekf_on = False
            #     self.request_recover_robot = False
        elif self.ekf_on:  # and not self.debug_flag:
            self.ekf.predict(drive_meas)
            # print("Robot state after prediction: ", self.ekf.robot.state.reshape(-1))
            if self.mode == 2:  # only add landmarks for M2 SLAM
                self.ekf.add_landmarks(lms)
            self.ekf.update(lms)
            # print("Robot state after correction: ", self.ekf.robot.state.reshape(-1), "\n")

    @staticmethod
    def draw_rel_dist(bboxes, yolo_vis, cam_matrix):
        for bbox in bboxes:
            relative_pose, pixel_height, distance = TargetPoseEst.estimate_pose(cam_matrix, bbox, return_all=True)
            vertical_rel_dist = relative_pose["y"]
            horizontal_rel_dist = relative_pose["x"]

            xyxy = ops.xywh2xyxy(bbox[1])
            x1 = int(xyxy[0])
            y1 = int(xyxy[1])
            x2 = int(xyxy[2])
            y2 = int(xyxy[3])

            yolo_vis = cv2.putText(yolo_vis, f'v:{round(vertical_rel_dist, 3)} h:{round(horizontal_rel_dist, 3)}',
                                   (x1, y2-30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            yolo_vis = cv2.putText(yolo_vis, f'px_h:{round(pixel_height,3)}',
                                   (x1, y2-5), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        return yolo_vis

    # draw vertical lines through centre of bounding boxes and frame centre
    @staticmethod
    def draw_centre_lines(bboxes, yolo_vis):
        for bbox in bboxes:
            #  translate bounding box info from [x, y, w, h] back to the format of [x1,y1,x2,y2] topleft and bottomright corner
            xyxy = ops.xywh2xyxy(bbox[1])
            x1 = int(xyxy[0])
            y1 = int(xyxy[1])
            x2 = int(xyxy[2])
            y2 = int(xyxy[3])

            mid_x = int((x1 + x2) / 2)

            # draw line through centre of bbox
            yolo_vis = cv2.line(yolo_vis, (mid_x, y1), (mid_x, y2), (0, 0, 255), thickness=2)

        # draw line through centre of frame
        yolo_vis = cv2.line(yolo_vis, (320, 0), (320, 480), (255, 0, 0), thickness=2)

        return yolo_vis

    # using computer vision to detect targets
    def detect_target(self):
        if self.command['inference'] and self.detector is not None and (self.mode==3 or self.mode==4):
            # need to convert the colour to BGR before passing to YOLO (opencv accepts BGR)
            yolo_input_img = cv2.cvtColor(self.img, cv2.COLOR_RGB2BGR)
            self.bboxes, self.yolo_vis = self.detector.detect_single_image(yolo_input_img)
            self.yolo_vis = self.draw_centre_lines(self.bboxes, self.yolo_vis)
            self.yolo_vis = self.draw_rel_dist(self.bboxes, self.yolo_vis, self.camera_matrix)
            self.file_output = (self.yolo_vis, self.ekf)  # prediction image(BGR), slam pose; doesn't save yet
            # self.notification = f'{len(np.unique(self.bboxes))-1} target type(s) detected'

            # covert the colour back to RGB for display purpose in GUI
            self.yolo_vis = cv2.cvtColor(self.yolo_vis, cv2.COLOR_BGR2RGB)
            self.command['inference'] = False  # uncomment this for continuous detection

    # save images taken by the camera
    def save_image(self):
        f_ = os.path.join(self.folder, f'img_{self.image_id}.png')
        if self.command['save_image']:
            image = self.pibot.get_image()
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(f_, image)
            self.image_id += 1
            self.notification = f'{f_} is saved'
            self.command['save_image'] = False

            return f_

    # wheel and camera calibration for SLAM
    def init_ekf(self):
        robot = Robot(self.baseline, self.scale, self.camera_matrix, self.dist_coeffs)
        return EKF(robot)

    # save SLAM map
    def record_data(self):
        if self.command['output']:  # 'S'
            self.output.write_map(self.ekf)
            self.notification = 'Map is saved'
            self.mode = 3  # next mode (M3 SLAM)
            self.command['output'] = False

        # save inference with the matching robot pose and detector labels
        if self.command['save_inference']:  # 'N'
             # self.detect_target()
            if self.file_output is not None:
                self.command['save_image'] = True
                f_ = self.save_image()
                # print(f_)
                # image = cv2.cvtColor(self.file_output[0], cv2.COLOR_RGB2BGR)
                self.pred_fname = self.output.write_image(f_, self.file_output[0],
                                                          self.file_output[1])
                self.notification = f'Prediction is saved to {self.pred_fname}'
            else:
                self.notification = f'No prediction in buffer, save ignored'
            self.command['save_inference'] = False

        if self.command['pose_est']:  # 'B'
            # next mode (M4 autonomous)
            self.mode = 4
            self.command['pose_est'] = False

    def run_target_update_truemap(self):
        target_est = TargetPoseEst.run(self.camera_matrix, self.detector)

        # modify TRUEMAP.txt
        with open('TRUEMAP_ALIGNED.txt', 'r') as f:
            try:
                gt_dict = json.load(f)
            except ValueError as e:
                with open(fname, 'r') as f:
                    gt_dict = ast.literal_eval(f.readline())
        gt_dict.update(target_est)
        with open('TRUEMAP_ALIGNED.txt', 'w') as file:
            json.dump(gt_dict, file, indent=4)

    # limit angle to [-pi, pi]
    def clip_angle(self, angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def measure_marker(self):
        if self.command['measure_marker']:
            self.start_360_count += 1

            # completed 360
            if self.start_360_count == 360//self.turn360_angle + 1:
                # sort markers into quadrants
                q1_markers, q2_markers, q3_markers, q4_markers = [], [], [], []
                q1_taglist, q2_taglist, q3_taglist, q4_taglist = [], [], [], []

                for i, marker in enumerate(self.ekf.measured_markers):
                    tag = self.ekf.measured_taglist[i]
                    if 0.0 < marker[0] < 1.6 and 0.0 < marker[1] < 1.6:
                        q1_markers.append(marker)
                        q1_taglist.append(tag)
                    elif -1.6 < marker[0] < 0.0 and 0.0 < marker[1] < 1.6:
                        q2_markers.append(marker)
                        q2_taglist.append(tag)
                    elif -1.6 < marker[0] < 0.0 and -1.6 < marker[1] < 0.0:
                        q3_markers.append(marker)
                        q3_taglist.append(tag)
                    elif 0.0 < marker[0] < 1.6 and -1.6 < marker[1] < 0.0:
                        q4_markers.append(marker)
                        q4_taglist.append(tag)

                # store randomly chosen marker from each quadrant
                chosen_markers = []
                chosen_taglist = []
                if len(q1_markers) > 0:
                    q1_chosen_id = random.randint(0,len(q1_markers)-1)
                    chosen_markers.append(q1_markers[q1_chosen_id])
                    chosen_taglist.append(q1_taglist[q1_chosen_id])
                if len(q2_markers) > 0:
                    q2_chosen_id = random.randint(0,len(q2_markers)-1)
                    chosen_markers.append(q2_markers[q2_chosen_id])
                    chosen_taglist.append(q2_taglist[q2_chosen_id])
                if len(q3_markers) > 0:
                    q3_chosen_id = random.randint(0, len(q3_markers) - 1)
                    chosen_markers.append(q3_markers[q3_chosen_id])
                    chosen_taglist.append(q3_taglist[q3_chosen_id])
                if len(q4_markers) > 0:
                    q4_chosen_id = random.randint(0,len(q4_markers)-1)
                    chosen_markers.append(q4_markers[q4_chosen_id])
                    chosen_taglist.append(q4_taglist[q4_chosen_id])

                # fill up remaining measured markers to keep
                desired_num_measured_markers = 4
                unused_tags = [tag for tag in self.ekf.measured_taglist if tag not in chosen_taglist]
                remaining_num_markers_to_add = desired_num_measured_markers-len(chosen_taglist)
                for j in range(remaining_num_markers_to_add):
                    extra_chosen_tag = random.choice(unused_tags)
                    extra_chosen_tag_id = self.ekf.measured_taglist.index(extra_chosen_tag)
                    extra_chosen_marker = self.ekf.measured_markers[extra_chosen_tag_id]
                    chosen_markers.append(extra_chosen_marker)
                    chosen_taglist.append(extra_chosen_tag)

                self.ekf.measured_markers = np.concatenate(chosen_markers, axis=1)
                self.ekf.measured_taglist = chosen_taglist
                print(f"Measured Markers: {self.ekf.measured_markers}")
                print(f"Measured Taglist: {self.ekf.measured_taglist}")

                # generate true map
                measured_aruco = {}
                markers = self.ekf.measured_markers
                taglist = self.ekf.measured_taglist
                # print(self.ekf.measured_markers)
                for i in range(markers.shape[1]):
                    measured_aruco[f"aruco{taglist[i]}_0"] = {'x': markers[0, i], 'y': markers[1, i]}
                with open('TRUEMAP.txt', 'w') as file:
                    json.dump(measured_aruco, file, indent=4)

                # next mode (M2 SLAM)
                self.ekf.robot.state = np.zeros((3,1))  # reset to origin
                self.mode = 2

            else:
                # measure rel x and y of seen markers
                lms, self.aruco_img = self.aruco_det.detect_marker_positions(self.img)
                lms = [lm for lm in lms if lm.tag <= 10]

                # get robot state and rotation matrix
                th = self.ekf.robot.state[2]
                robot_xy = self.ekf.robot.state[0:2, :]
                R_theta = np.block([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])

                # add all newly seen marker measurements (world frame)
                for lm in lms:
                    if lm.tag in self.ekf.measured_taglist:
                        # ignore known tags
                        continue

                    # transform marker robot -> world frame
                    new_measured_marker = robot_xy + R_theta @ lm.position

                    self.ekf.measured_taglist.append(int(lm.tag))
                    self.ekf.measured_markers.append(new_measured_marker)
                    # self.ekf.measured_markers = np.concatenate((self.ekf.measured_markers, new_measured_marker), axis=1)

                print(f"Measured Markers: {self.ekf.measured_markers}")
                print(f"Measured Taglist: {self.ekf.measured_taglist}")
                print(f"Robot pose: {self.ekf.robot.state}")

                # turn robot update robot state manually
                # self.fruit_search.turn(-self.turn360_angle)
                # time.sleep(2)
                self.ekf.robot.state[2,0] += np.deg2rad(self.turn360_angle)
                self.ekf.robot.state[2,0] = self.clip_angle(self.ekf.robot.state[2,0])
                print(f"{self.start_360_count} iteration(s) done. "
                      f"{(360//self.turn360_angle + 1) - self.start_360_count} more to go. "
                      f"Ready for next key press 'M'")

            # run this function only once
            self.command['measure_marker'] = False

    @ staticmethod
    def align_markers():
        # align markers after completing slam
        gt_aruco = SLAM_eval.parse_groundtruth("TRUEMAP.txt")
        us_aruco = SLAM_eval.parse_user_map("lab_output/slam.txt")

        taglist, extra_taglist, us_vec, gt_vec, extra_vec = SLAM_eval.match_aruco_points(us_aruco, gt_aruco)

        # sort taglist and markers
        idx = np.argsort(taglist)
        taglist = np.array(taglist)[idx]
        us_vec = us_vec[:, idx]
        gt_vec = gt_vec[:, idx]

        theta, x = SLAM_eval.solve_umeyama2d(us_vec, gt_vec)

        # us_vec_all = np.concatenate((us_vec, extra_vec), axis=1)
        us_vec_aligned = SLAM_eval.apply_transform(theta, x, us_vec)
        # print(us_vec_aligned)
        if extra_vec is not None:
            extra_vec_aligned = SLAM_eval.apply_transform(theta, x, extra_vec)
        else:
            extra_vec_aligned = None

        SLAM_eval.display_stats_plot(us_vec, gt_vec, extra_vec, us_vec_aligned, extra_vec_aligned, taglist, extra_taglist, theta, x)

        # generate aligned truemap
        aligned_aruco = {}
        for i in range(us_vec_aligned.shape[1]):
            aligned_aruco[f"aruco{taglist[i]}_0"] = {'x':us_vec_aligned[0,i], 'y':us_vec_aligned[1,i]}

        for j in range(extra_vec_aligned.shape[1]):
            aligned_aruco[f"aruco{extra_taglist[j]}_0"] = {'x':extra_vec_aligned[0,j], 'y':extra_vec_aligned[1,j]}

        with open('TRUEMAP_ALIGNED.txt', 'w') as file:
            json.dump(aligned_aruco, file, indent=4)

    def turn360_slam(self, canvas):
        time.sleep(self.turn360_delay)
        sections = 360//self.turn360_angle
        for i in range(sections):
            turn_time, lv, rv = self.fruit_search.turn(self.turn360_angle, return_turn_time_lv_rv=True)
            drive_meas = measure.Drive(lv, rv, turn_time)
            time.sleep(self.turn360_delay)  # wait for robot image to stabilize
            # update self.img to be used by update_slam()
            self.take_pic()
            # run predict and update in SLAM
            self.update_slam(drive_meas)
            # update GUI
            self.draw(canvas)
            pygame.display.update()
        time.sleep(self.turn360_delay)

    # A function to apply trilateration formulas to return the (x,y) intersection point of three circles
    @staticmethod
    def trilaterate(x1, y1, r1, x2, y2, r2, x3, y3, r3):
        A = 2 * x2 - 2 * x1
        B = 2 * y2 - 2 * y1
        C = r1 ** 2 - r2 ** 2 - x1 ** 2 + x2 ** 2 - y1 ** 2 + y2 ** 2
        D = 2 * x3 - 2 * x2
        E = 2 * y3 - 2 * y2
        F = r2 ** 2 - r3 ** 2 - x2 ** 2 + x3 ** 2 - y2 ** 2 + y3 ** 2
        x = (C * E - F * B) / (E * A - B * D)
        y = (C * D - A * F) / (B * D - A * E)
        return x, y

    def measure_single_marker(self):
        # measure rel x and y of seen markers
        lms, self.aruco_img = self.aruco_det.detect_marker_positions(self.img)
        lms = [lm for lm in lms if lm.tag <= 10]

        # add all newly seen marker measurements (world frame)
        for lm in lms:
            # if marker seen before in current 360 turn
            if lm.tag in self.marker_taglist:
                # ignore known tags
                continue

            print(f"Taglist: {lm.tag}")
            print(f"Markers: {lm.position.reshape(1,-1)}")
            dist = np.hypot(lm.position[0, 0], lm.position[1, 0])

            self.marker_taglist.append(int(lm.tag))
            self.marker_dist.append(dist)
            self.marker_rel.append(lm.position)

    def turn360_measure(self, canvas):
        self.marker_dist = []
        self.marker_taglist = []
        self.marker_rel = []

        print("Getting robot x y")
        # time.sleep(self.turn360_delay)
        sections = 360 // self.turn360_angle
        for i in range(sections):
            # input("press ENTER")
            turn_time, lv, rv = self.fruit_search.turn(self.turn360_angle, return_turn_time_lv_rv=True)
            # time.sleep(self.turn360_delay)  # wait for robot image to stabilize
            # update self.img to be used by update_slam()
            self.take_pic()
            # measure markers in current frame
            self.measure_single_marker()
            # update GUI
            self.draw(canvas)
            pygame.display.update()
        # time.sleep(self.turn360_delay)

        print(f"Measured Taglist: {self.marker_taglist}")
        print(f"Measured Markers distances: {self.marker_dist}")

        # trilaterate robot position to obtain robot x and y
        combined = list(zip(self.marker_dist, self.marker_taglist, self.marker_rel))
        sorted_combined = sorted(combined, key=lambda x: x[0])
        sorted_values = sorted_combined
        sorted_taglist = [tag for distance, tag, rel in sorted_values]
        sorted_dist = [distance for distance, tag, rel in sorted_values]
        sorted_rel = [rel for distance, tag, rel in sorted_values]
        sorted_markers = []
        for tag in sorted_taglist:
            index = self.ekf.taglist.index(tag)
            sorted_markers.append(self.ekf.markers[:, index])
        print(f"sorted taglist: {sorted_taglist}")
        print(f"sorted markers: {sorted_markers}")
        print(f"sorted dist: {sorted_dist}")

        robot_x, robot_y = self.trilaterate(sorted_markers[0][0], sorted_markers[0][1], sorted_dist[0],
                                            sorted_markers[1][0], sorted_markers[1][1], sorted_dist[1],
                                            sorted_markers[2][0], sorted_markers[2][1], sorted_dist[2])

        # rare case if all 3 closest markers are collinear, will have divide by 0 error
        i = 3
        while robot_x < -1.6 or robot_x > 1.6 or robot_y < -1.6 or robot_y > 1.6:
            robot_x, robot_y = self.trilaterate(sorted_markers[0][0], sorted_markers[0][1], sorted_dist[0],
                                                sorted_markers[1][0], sorted_markers[1][1], sorted_dist[1],
                                                sorted_markers[i][0], sorted_markers[i][1], sorted_dist[i])
            i += 1
            if i >= len(sorted_taglist):
                print("no extra markers to fix collinear error!")
                while True:
                    pass

        self.ekf.robot.state[0, 0] = robot_x
        self.ekf.robot.state[1, 0] = robot_y
        print(f"Robot State from SLAM: {self.ekf.robot.state.reshape(1, -1)}")

        # align robot theta with nearest marker
        print("Aligning robot theta")
        filtered_sorted_taglist = [tag for distance, tag, rel in sorted_values if distance >= 0.5]
        chosen_tag = filtered_sorted_taglist[0]
        chosen_id = sorted_taglist.index(chosen_tag)
        chosen_marker = sorted_markers[chosen_id]
        x1 = chosen_marker[0]
        y1 = chosen_marker[1]
        print(f"chosen tag: {chosen_tag}")

        # measure rel x and y of seen markers
        chosen_rel = np.zeros((2,1))
        smallest_marker_found = False
        while not smallest_marker_found:
            # time.sleep(self.turn360_delay)
            # input("press ENTER")
            self.take_pic()
            lms, self.aruco_img = self.aruco_det.detect_marker_positions(self.img)
            lms = [lm for lm in lms if lm.tag <= 10]

            for lm in lms:
                print(f"Taglist: {lm.tag}")
                print(f"Markers: {lm.position}")
                # if marker seen before in current 360 turn
                if lm.tag == chosen_tag:
                    smallest_marker_found = True
                    chosen_rel = lm.position
                else:
                    # ignore tags other than smallest tag
                    continue

            if not smallest_marker_found:
                self.fruit_search.turn(self.turn360_angle)

            # update GUI
            self.draw(canvas)
            pygame.display.update()

        rel_x = -chosen_rel[1,0]
        rel_y = chosen_rel[0,0]
        beta = math.atan2(y1-robot_y, x1-robot_x)
        alpha = math.atan2(rel_x, rel_y)
        robot_theta = beta + alpha

        self.ekf.robot.state[2, 0] = robot_theta

        print(f"Robot State from SLAM: [{self.ekf.robot.state[0,0]},{self.ekf.robot.state[1,0]},{np.rad2deg(self.ekf.robot.state[2,0])}]")

    def run_fruit_search(self, canvas):
        fruits_list, fruits_true_pos, aruco_true_pos, gt_dict = self.fruit_search.read_true_map(args.map)
        search_list = self.fruit_search.read_search_list()  # TODO: change search list filename
        print(fruits_list)
        target_fruits = self.fruit_search.print_target_fruits_pos(search_list, fruits_list, fruits_true_pos)
        print(target_fruits)

        # flag to check if robot pose is within 0.1 away from the waypoint
        # reached = True

        for fruit in target_fruits:
            # input(f"press enter to search for {fruit}")
            # robot_pose = self.ekf.robot.state
            fruit_position = np.round(target_fruits[fruit], 1)
            start_point = np.round(self.fruit_search.robot_state[0:2, 0], 1)
            path = self.fruit_search.generate_path(args.map, fruit_position, start_point)
            print(f"Path from {start_point} to {fruit} at {fruit_position}: {path}")
            for waypoint in path:
                print(f"Start driving to waypoint: {waypoint}; Current robot pose: "
                      f"[{np.round(self.ekf.robot.state[0,0],3)},"
                      f"{np.round(self.ekf.robot.state[1,0],3)},"
                      f"{np.round(np.rad2deg(self.ekf.robot.state[2,0]),3)}]")
                # input("press ENTER")
                self.fruit_search.drive_to_point(waypoint)

                # RESUME SLAM + 360 turn HERE
                # self.ekf_on = True
                # self.notification = 'SLAM is running'
                self.turn360_measure(canvas)
                # robot state in ekf will be updated from SLAM

                # update robot pose
                self.fruit_search.robot_state = self.ekf.robot.state

                # # if robot pose far away from desired waypoint, try drive to waypoint again
                # dist_error, _ = self.fruit_search.check_pose_error()
                # if dist_error > self.dist_threshold:
                #     self.fruit_search.drive_to_point(waypoint)
                #     self.turn360_measure(canvas)
                #     # update robot pose
                #     self.fruit_search.robot_state = self.ekf.robot.state
                #     print(f"Robot State from SLAM after correction: {self.ekf.robot.state.reshape(1, -1)}")

                print(f"Finished driving to waypoint: {waypoint}; New robot pose: "
                      f"[{np.round(self.ekf.robot.state[0,0],3)},"
                      f"{np.round(self.ekf.robot.state[1,0],3)},"
                      f"{np.round(np.rad2deg(self.ekf.robot.state[2,0]),3)}]")
                # time.sleep(1)

            print(f"\nFOUND {fruit}!\n")
            time.sleep(3)

        self.mode = 0

    # paint the GUI            
    def draw(self, canvas):
        canvas.blit(self.bg, (0, 0))
        text_colour = (220, 220, 220)
        v_pad = 40
        h_pad = 20

        # paint SLAM outputs
        ekf_view = self.ekf.draw_slam_state(res=(320, 480+v_pad),
                                            not_pause = self.ekf_on)
        canvas.blit(ekf_view, (2*h_pad+320, v_pad))
        robot_view = cv2.resize(self.aruco_img, (320, 240))
        self.draw_pygame_window(canvas, robot_view, position=(h_pad, v_pad))

        # for target detector (M3)
        detector_view = cv2.resize(self.yolo_vis, (320, 240), cv2.INTER_NEAREST)
        self.draw_pygame_window(canvas, detector_view, position=(h_pad, 240 + 2 * v_pad))

        # canvas.blit(self.gui_mask, (0, 0))
        self.put_caption(canvas, caption='SLAM', position=(2*h_pad+320, v_pad)) # M2
        self.put_caption(canvas, caption='Detector (M3)',
                         position=(h_pad, 240+2*v_pad)) # M3
        self.put_caption(canvas, caption='PiBot Cam', position=(h_pad, v_pad))

        notifiation = TEXT_FONT.render(self.notification, False, text_colour)
        canvas.blit(notifiation, (h_pad+10, 596))

        time_remain = self.count_down - time.time() + self.start_time
        if time_remain > 0:
            time_remain = f'Count Down: {time_remain:03.0f}s'
        elif int(time_remain)%2 == 0:
            time_remain = "Time Is Up !!!"
        else:
            time_remain = ""
        count_down_surface = TEXT_FONT.render(time_remain, False, (50, 50, 50))
        canvas.blit(count_down_surface, (2*h_pad+320+5, 530))
        return canvas

    @staticmethod
    def draw_pygame_window(canvas, cv2_img, position):
        cv2_img = np.rot90(cv2_img)
        view = pygame.surfarray.make_surface(cv2_img)
        view = pygame.transform.flip(view, True, False)
        canvas.blit(view, position)

    @staticmethod
    def put_caption(canvas, caption, position, text_colour=(200, 200, 200)):
        caption_surface = TITLE_FONT.render(caption,
                                          False, text_colour)
        canvas.blit(caption_surface, (position[0], position[1]-25))

    # keyboard teleoperation        
    def update_keyboard(self):
        for event in pygame.event.get():
            ########### replace with your M1 codes ###########
            # drive forward
            if event.type == pygame.KEYDOWN and event.key == pygame.K_UP:
                self.command['motion'] = [1, 0]
            # drive backward
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_DOWN:
                self.command['motion'] = [-1, 0]
            # turn left
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_LEFT:
                self.command['motion'] = [0, 1]
            # drive right
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_RIGHT:
                self.command['motion'] = [0, -1]
            # stop robot when one of arrow keys is released
            elif event.type == pygame.KEYUP and event.key in [pygame.K_UP, pygame.K_DOWN, pygame.K_LEFT, pygame.K_RIGHT]:
                self.command['motion'] = [0, 0]
            ####################################################
            # stop
            # elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
            #     self.command['motion'] = [0, 0]
            # save image
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_i:
                self.command['save_image'] = True
            # save SLAM map
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_s:
                self.command['output'] = True
            # reset SLAM map
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                if self.double_reset_comfirm == 0:
                    self.notification = 'Press again to confirm CLEAR MAP'
                    self.double_reset_comfirm += 1
                elif self.double_reset_comfirm == 1:
                    self.notification = 'SLAM Map is cleared'
                    self.double_reset_comfirm = 0
                    self.ekf.reset()
            # run SLAM
            elif (event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN and
                  (self.mode == 2 or self.mode == 3)):
                # n_observed_markers = len(self.ekf.taglist)
                # if n_observed_markers == 0:
                #     if not self.ekf_on:
                #         self.notification = 'SLAM is running'
                #         self.ekf_on = True
                #     else:
                #         self.notification = '> 2 landmarks is required for pausing'
                # elif n_observed_markers < 3:
                #     self.notification = '> 2 landmarks is required for pausing'
                # else:
                #     if not self.ekf_on:
                #         self.request_recover_robot = True
                self.ekf_on = not self.ekf_on
                if self.ekf_on:
                    self.notification = 'SLAM is running'
                else:
                    self.notification = 'SLAM is paused'
            # run object detector
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_p:
                self.command['inference'] = True
            # save object detection outputs
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_n:
                self.command['save_inference'] = True
            # save measured markers
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_m:
                self.command['measure_marker'] = True
            # run target pose est to get targets.txt
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_b:
                self.command['pose_est'] = True
            # quit
            elif event.type == pygame.QUIT:
                self.quit = True
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                self.quit = True
        if self.quit:
            pygame.quit()
            sys.exit()

        
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", metavar='', type=str, default='localhost')
    parser.add_argument("--port", metavar='', type=int, default=8000)
    parser.add_argument("--calib_dir", type=str, default="calibration/param/")
    parser.add_argument("--save_data", action='store_true')
    parser.add_argument("--play_data", action='store_true')
    parser.add_argument("--yolo_model", default='model.best.pt')
    parser.add_argument("--map", type=str, default='TRUEMAP_ALIGNED.txt')  # TODO: change filename
    args, _ = parser.parse_known_args()

    operate = Operate(args)

    # init pygame parameters
    pygame.font.init()
    TITLE_FONT = pygame.font.Font('pics/8-BitMadness.ttf', 35)
    TEXT_FONT = pygame.font.Font('pics/8-BitMadness.ttf', 40)

    width, height = 700, 660
    canvas = pygame.display.set_mode((width, height))
    pygame.display.set_caption('ECE4078 2022 Lab')
    pygame.display.set_icon(pygame.image.load('pics/8bit/pibot5.png'))
    canvas.fill((0, 0, 0))
    splash = pygame.image.load('pics/loading.png')
    pibot_animate = [pygame.image.load('pics/8bit/pibot1.png'),
                     pygame.image.load('pics/8bit/pibot2.png'),
                     pygame.image.load('pics/8bit/pibot3.png'),
                     pygame.image.load('pics/8bit/pibot4.png'),
                     pygame.image.load('pics/8bit/pibot5.png')]
    pygame.display.update()

    mode = input("Operation Mode (1-Initial 360 turn, 2-M2, 3-M3, 4-M4):")
    operate.mode = int(mode)

    while True:
        if operate.mode == 1:  # initial 360
            input("MODE 1: Initial 360 turn -> PRESS ENTER TO START")
            start = False

            # GUI loading screen
            counter = 40
            while not start:
                for event in pygame.event.get():
                    if event.type == pygame.KEYDOWN:
                        start = True
                canvas.blit(splash, (0, 0))
                x_ = min(counter, 600)
                if x_ < 600:
                    canvas.blit(pibot_animate[counter%10//2], (x_, 565))
                    pygame.display.update()
                    counter += 2

            while start and operate.mode == 1:
                operate.update_keyboard()
                operate.measure_marker()
                operate.take_pic()
                operate.move_small_step()
                # drive_meas = operate.control()
                operate.update_slam()
                operate.record_data()
                operate.save_image()
                # print("Robot state: ", operate.ekf.robot.state)
                # print("Markers: ", operate.ekf.markers, "\n")
                # visualise
                operate.draw(canvas)
                pygame.display.update()

        elif operate.mode == 2:  # M2 SLAM
            input("MODE 2: M2 SLAM -> PRESS ENTER TO START")

            # start SLAM
            operate.ekf_on = True
            operate.notification = 'SLAM is running'
            operate.draw(canvas)
            pygame.display.update()

            while operate.mode == 2:
                operate.update_keyboard()
                operate.take_pic()
                drive_meas = operate.control()
                operate.update_slam(drive_meas)
                operate.record_data()
                operate.save_image()
                # print(f"Robot state: ({operate.ekf.robot.state[0,0]}, {operate.ekf.robot.state[1,0]}, {np.rad2deg(operate.ekf.robot.state[2,0])})")
                # print("Markers: ", operate.ekf.markers, "\n")
                # visualise
                operate.draw(canvas)
                pygame.display.update()

            # pause SLAM
            operate.ekf_on = False
            operate.notification = 'SLAM is paused'
            operate.draw(canvas)
            pygame.display.update()

            # align marker estimations to measured markers
            operate.align_markers()

        elif operate.mode == 3:  # M3 search fruit pos
            input("MODE 3: M3 Search Fruit Pos -> PRESS ENTER TO START")

            # switch to EKF TRUEMAP mode
            operate.ekf.switch_mode()

            # # start SLAM
            # operate.ekf_on = True
            # operate.notification = 'SLAM is running'
            # operate.draw(canvas)
            # pygame.display.update()

            while operate.mode == 3:
                operate.update_keyboard()
                operate.take_pic()
                drive_meas = operate.control()
                operate.update_slam(drive_meas)
                operate.record_data()
                operate.save_image()
                operate.detect_target()
                print("Robot state: ", operate.ekf.robot.state.reshape(1,-1))
                # print("Markers: \n", operate.ekf.markers.T)
                # print("Taglist: ", operate.ekf.taglist, "\n")
                # visualise
                operate.draw(canvas)
                pygame.display.update()

            # # pause SLAM
            # operate.ekf_on = False
            # operate.notification = 'SLAM is paused'
            # operate.draw(canvas)
            # pygame.display.update()

            # update TRUEMAP_ALIGNED with fruits from M3
            operate.run_target_update_truemap()

        elif operate.mode == 4:  # autonomous fruit search
            input("MODE 4: M4 Autonomous Fruit Search -> PRESS ENTER TO START")

            # switch to EKF TRUEMAP mode
            operate.ekf.switch_mode()

            # start SLAM
            operate.ekf_on = True
            operate.notification = 'SLAM is running'
            operate.draw(canvas)
            pygame.display.update()

            # auto fruit search
            operate.run_fruit_search(canvas)
            print("\nFruit search complete!\n")

        elif operate.mode == 5:  # for wheel calibration
            command1_list = []

            for direction in [1,-1]:
                print(f"Calibrate turning direction: {direction}:")
                while True:
                    command1 = input("Input command 1 (~1.0) for turning: ")
                    command1 = float(command1)

                    operate.fruit_search.turn(direction*operate.fruit_search.basic_angle, command1)

                    uInput = input(f"Did the robot turn {operate.fruit_search.basic_angle} deg?[y/N]")
                    if uInput == 'y':
                        command1_list.append(command1)
                        break

            # while True:
            #     input("sdada")
            #     lv, rv = operate.pibot.set_velocity(
            #         [1, 0], turning_tick=operate.turn_left_ticks, tick=operate.drive_ticks, time=2)

            print(f"Turn ticks for turning left and right {operate.fruit_search.basic_angle} deg is: {command1}")


        elif operate.mode == 6:  # for wheel calibration

            command1_list = []

            for direction in [1, -1]:

                print(f"Calibrate driving direction: {direction}:")

                while True:

                    command1 = input("Input command 1 (~1.0) for driving: ")

                    command1 = float(command1)

                    operate.fruit_search.drive(direction * 0.20, command1)

                    uInput = input(f"Did the robot drive 20 cm?[y/N]")

                    if uInput == 'y':
                        command1_list.append(command1)

                        break

            # while True:

            #     input("sdada")

            #     lv, rv = operate.pibot.set_velocity(

            #         [1, 0], turning_tick=operate.turn_left_ticks, tick=operate.drive_ticks, time=2)

            print(f"Command1 for driving straight and backward 20cm is: {command1}")



        elif operate.mode == 7:  # checking trilateration

            operate.ekf.switch_mode()

            operate.turn360_measure(canvas)

            while True:
                pass


        elif operate.mode == 8:

            robot_x, robot_y = operate.trilaterate(-0.4, 0.4, 0.5712, 0, 1.2, 0.9418, 0.4, 0.4, 1.2885)

            print(f"x:{robot_x}  y:{robot_y}")

            while True:
                pass


