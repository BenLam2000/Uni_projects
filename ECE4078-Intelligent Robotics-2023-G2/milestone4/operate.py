# teleoperate the robot, perform SLAM and object detection

# basic python packages
import numpy as np
import cv2 
import os, sys
import time
import json
import ast
import math

# import utility functions
sys.path.insert(0, "{}/utility".format(os.getcwd()))
from util.pibot import Alphabot # access the robot
import util.DatasetHandler as dh # save/load functions
import util.measure as measure # measurements
import pygame # python package for GUI
import shutil # python package for file operations

# import SLAM components you developed in M2
sys.path.insert(0, "{}/slam".format(os.getcwd()))
from slam.ekf import EKF
from slam.robot import Robot
import slam.aruco_detector as aruco

# import CV components
sys.path.insert(0,"{}/network/".format(os.getcwd()))
sys.path.insert(0,"{}/network/scripts".format(os.getcwd()))
from network.scripts.detector import Detector
from ultralytics.utils import ops

# for M3
from TargetPoseEst import estimate_pose

# for M4
from a_star import a_star


class Operate:
    def __init__(self, args):
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

        # M4
        self.robot_state = np.array((3,1))
        self.drive_ticks = 15
        self.turn_ticks = 15
        self.start_turn360_id = 0
        self.start_move_id = 0
        self.mode = 1
        self.start = False
        self.fruit_id = 0
        self.waypoint_id = 0
        self.search_list = []
        self.path = [[0,0],[0,0]]
        # self.turn = True
        self.move_start = 0
        self.move_time = 0.0
        self.move_id = 1
        # self.move_end = time.time()
        # self.lms = []
        self.target_fruits = {}
        self.init_M4()
        # self.get_true_measurements()  # get true LMS

        # initialise SLAM parameters
        self.ekf = self.init_ekf(args.calib_dir, args.ip)
        self.aruco_det = aruco.aruco_detector(
            self.ekf.robot, marker_length=0.06)  # size of the ARUCO markers

        if args.save_data:
            self.data = dh.DatasetWriter('record')
        else:
            self.data = None
        self.output = dh.OutputWriter('lab_output')
        self.command = {'motion': [0, 0],
                        'inference': False,
                        'output': False,
                        'save_inference': False,
                        'save_image': False,
                        'move': False}
        self.quit = False
        self.pred_fname = ''
        self.raw_img_fname = ''
        self.request_recover_robot = False
        self.file_output = None
        self.ekf_on = False
        self.double_reset_comfirm = 0
        self.image_id = 0
        self.notification = 'Press ENTER to start SLAM'
        # a 5min timer
        self.count_down = 300
        self.start_time = time.time()
        self.control_clock = time.time()
        # initialise images
        self.img = np.zeros([240, 320, 3], dtype=np.uint8)
        self.aruco_img = np.zeros([240, 320, 3], dtype=np.uint8)
        self.bboxes = [['redapple', np.asarray([100, 100, 200, 200]), 0.95]]
        if args.yolo_model == "":
            self.detector = None
            self.yolo_vis = cv2.imread('pics/8bit/detector_splash.png')
        else:
            self.detector = Detector(f'network/scripts/model/{args.yolo_model}')
            self.yolo_vis = np.ones((240, 320, 3)) * 100
        self.yolo_vis_line = np.ones((240, 320, 3)) * 100
        self.bg = pygame.image.load('pics/gui_mask.jpg')

    # wheel control
    def control(self):
        if args.play_data:
            lv, rv = self.pibot.set_velocity()
        else:
            lv, rv = self.pibot.set_velocity(
                self.command['motion'], tick=self.drive_ticks, turning_tick=self.turn_ticks, time=0)
        if self.data is not None:
            self.data.write_keyboard(lv, rv)
        dt = time.time() - self.control_clock  # every loop time
        drive_meas = measure.Drive(lv, rv, dt)
        self.control_clock = time.time()
        # if sum(self.)
        if self.command['motion'] != [0,0]:
            if time.time() - self.move_start >= self.move_time:
                self.command['motion'] = [0, 0]

        return drive_meas

    # camera control
    def take_pic(self):
        self.img = self.pibot.get_image()
        if self.data is not None:
            self.data.write_image(self.img)

    # SLAM with ARUCO markers
    def update_slam(self, drive_meas):
        # print(self.ekf_on)
        # create ground truth LMS based on TRUE aruco pos (world frame) &
        lms, self.aruco_img = self.aruco_det.detect_marker_positions(self.img)
        lms = [lm for lm in lms if lm.tag <= 10]  # remove unknowns
        print(f"robot state: {self.ekf.robot.state.reshape([1,-1])}")
        # print(f"number of detected arucos: {len(lms)}")
        # for lm in lms:
        #     print(f"pos: [{lm.position[0]}, {lm.position[1]}]  id: {lm.tag}")
        print("\n")
        if self.request_recover_robot:
            is_success = self.ekf.recover_from_pause(lms)
            if is_success:
                self.notification = 'Robot pose is successfuly recovered'
                self.ekf_on = True
            else:
                self.notification = 'Recover failed, need >2 landmarks!'
                self.ekf_on = False
            self.request_recover_robot = False
        elif self.ekf_on:  # and not self.debug_flag:
            self.ekf.predict(drive_meas)
            self.ekf.add_landmarks(lms)  # CHANGE THIS
            self.ekf.update(lms)

    # print out relative x and y distances near BB
    @staticmethod
    def draw_rel_dist(bboxes, yolo_vis, cam_matrix):
        for bbox in bboxes:
            relative_pose = estimate_pose(cam_matrix, bbox, relative_pose_only=True)
            vertical_rel_dist = relative_pose["y"]
            horizontal_rel_dist = relative_pose["x"]

            xyxy = ops.xywh2xyxy(bbox[1])
            x1 = int(xyxy[0])
            y1 = int(xyxy[1])
            x2 = int(xyxy[2])
            y2 = int(xyxy[3])

            yolo_vis = cv2.putText(yolo_vis, f'v:{round(vertical_rel_dist, 3)} h:{round(horizontal_rel_dist, 3)}',
                                   (x1, y2), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

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

            # draw line through centre of each bbox
            yolo_vis = cv2.line(yolo_vis, (mid_x, y1), (mid_x, y2), (0, 0, 255), thickness=2)

        # draw line through centre of frame
        yolo_vis = cv2.line(yolo_vis, (320, 0), (320, 480), (255, 0, 0), thickness=2)

        return yolo_vis

    # using computer vision to detect targets
    def detect_target(self, cam_matrix):
        if self.command['inference'] and self.detector is not None:
            # need to convert the colour to BGR before passing to YOLO (opencv accepts BGR)
            yolo_input_img = cv2.cvtColor(self.img, cv2.COLOR_RGB2BGR)
            self.bboxes, self.yolo_vis = self.detector.detect_single_image(yolo_input_img)
            self.yolo_vis = self.draw_centre_lines(self.bboxes, self.yolo_vis)
            self.yolo_vis = self.draw_rel_dist(self.bboxes, self.yolo_vis, cam_matrix)
            self.file_output = (self.yolo_vis, self.ekf)  # prediction image(BGR), robot slam pose; doesn't save yet
            # self.notification = f'{len(np.unique(self.bboxes))-1} target type(s) detected'

            # covert the colour back to RGB for display purpose in GUI
            self.yolo_vis = cv2.cvtColor(self.yolo_vis, cv2.COLOR_BGR2RGB)
            self.command['inference'] = False  # uncomment this for continuous detection

    # save raw images taken by the camera
    def save_image(self):
        self.raw_img_fname = os.path.join(self.folder, f'img_{self.image_id}.png')
        if self.command['save_inference']:  # press N
            image = self.pibot.get_image()
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(self.raw_img_fname, image)
            self.image_id += 1
            self.notification = f'{self.raw_img_fname} is saved'
            # self.command['save_image'] = False

    # initialize EKF/SLAM with Robot consisting of parameters obtained from wheel and camera calibration
    def init_ekf(self, datadir, ip):
        fileK = "{}intrinsic.txt".format(datadir)
        camera_matrix = np.loadtxt(fileK, delimiter=',')
        fileD = "{}distCoeffs.txt".format(datadir)
        dist_coeffs = np.loadtxt(fileD, delimiter=',')
        fileS = "{}scale.txt".format(datadir)
        scale = np.loadtxt(fileS, delimiter=',')
        if ip == 'localhost':
            scale /= 2
        fileB = "{}baseline.txt".format(datadir)
        baseline = np.loadtxt(fileB, delimiter=',')
        robot = Robot(baseline, scale, camera_matrix, dist_coeffs)
        return EKF(robot)

    # save SLAM map / prediction image & robot pose
    def record_data(self):
        if self.command['output']:  # press S
            self.output.write_map(self.ekf)
            self.notification = 'Map is saved'
            self.command['output'] = False
        # save inference with the matching robot pose and detector labels
        if self.command['save_inference']:  # press N
            # self.detect_target()
            if self.file_output is not None:
                # self.save_image()
                self.pred_fname = self.output.write_image(self.raw_img_fname, self.file_output[0],
                                                          self.file_output[1])
                self.notification = f'Prediction is saved to {self.pred_fname}'
            else:
                self.notification = f'No prediction in buffer, save ignored'
            self.command['save_inference'] = False

    # paint the GUI
    def draw(self, canvas):
        canvas.blit(self.bg, (0, 0))
        text_colour = (220, 220, 220)
        v_pad = 40
        h_pad = 20

        # paint SLAM outputs
        ekf_view = self.ekf.draw_slam_state(res=(320, 480 + v_pad),
                                            not_pause=self.ekf_on)
        canvas.blit(ekf_view, (2 * h_pad + 320, v_pad))
        robot_view = cv2.resize(self.aruco_img, (320, 240))
        self.draw_pygame_window(canvas, robot_view,
                                position=(h_pad, v_pad)
                                )

        # for target detector (M3)
        detector_view = cv2.resize(self.yolo_vis, (320, 240), cv2.INTER_NEAREST)
        self.draw_pygame_window(canvas, detector_view,
                                position=(h_pad, 240 + 2 * v_pad)
                                )

        # canvas.blit(self.gui_mask, (0, 0))
        self.put_caption(canvas, caption='SLAM', position=(2 * h_pad + 320, v_pad))
        self.put_caption(canvas, caption='Detector',
                         position=(h_pad, 240 + 2 * v_pad))
        self.put_caption(canvas, caption='PiBot Cam', position=(h_pad, v_pad))

        notifiation = TEXT_FONT.render(self.notification,
                                       False, text_colour)
        canvas.blit(notifiation, (h_pad + 10, 596))

        time_remain = self.count_down - time.time() + self.start_time
        if time_remain > 0:
            time_remain = f'Count Down: {time_remain:03.0f}s'
        elif int(time_remain) % 2 == 0:
            time_remain = "Time Is Up !!!"
        else:
            time_remain = ""
        count_down_surface = TEXT_FONT.render(time_remain, False, (50, 50, 50))
        canvas.blit(count_down_surface, (2 * h_pad + 320 + 5, 530))
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
        canvas.blit(caption_surface, (position[0], position[1] - 25))

    # keyboard teleoperation
    def update_keyboard(self):
        for event in pygame.event.get():
            ########### replace with your M1 codes ###########
            # drive forward
            if event.type == pygame.KEYDOWN and event.key == pygame.K_UP:
                self.command['motion'] = [2, 0]
            # drive backward
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_DOWN:
                self.command['motion'] = [-2, 0]
            # turn left
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_LEFT:
                self.command['motion'] = [0, 2]
            # drive right
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_RIGHT:
                self.command['motion'] = [0, -2]
            # stop robot when one of arrow keys is released
            elif event.type == pygame.KEYUP and event.key in [pygame.K_UP, pygame.K_DOWN, pygame.K_LEFT,
                                                              pygame.K_RIGHT]:
                self.command['motion'] = [0, 0]
            ####################################################
            # stop
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                # self.command['motion'] = [0, 0]
                self.start = True

                # SLAM
                n_observed_markers = len(self.ekf.taglist)
                # n_observed_markers = 0  # CHANGE THIS
                if n_observed_markers == 0:
                    if not self.ekf_on:
                        self.notification = 'SLAM is running'
                        self.ekf_on = True
                    else:
                        self.notification = '> 2 landmarks is required for pausing'
                elif n_observed_markers < 3:
                    self.notification = '> 2 landmarks is required for pausing'
                else:
                    if not self.ekf_on:
                        self.request_recover_robot = True
                    self.ekf_on = not self.ekf_on
                    if self.ekf_on:
                        self.notification = 'SLAM is running'
                    else:
                        self.notification = 'SLAM is paused'
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
            # run object detector
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_p:
                self.command['inference'] = True
            # save object detection outputs
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_n:
                self.command['save_inference'] = True
            # quit
            elif event.type == pygame.QUIT:
                self.quit = True
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                self.quit = True
        if self.quit:
            pygame.quit()
            sys.exit()

    # @@@@@@@@@@@@@@@@@ AUTO_FRUIT_SEARCH @@@@@@@@@@@@@@@@@@@@@
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
                    taglist.append(marker_id + 1)
                    markers[0].append(x)
                    markers[1].append(y)

                else:  # append 3 fruit names from true map to fruit list
                    fruit_list.append(key[:-2])
                    if len(fruit_true_pos) == 0:
                        fruit_true_pos = np.array([[x, y]])
                    else:
                        fruit_true_pos = np.append(fruit_true_pos, [[x, y]], axis=0)

            cov = [[0] * 20] * 20
            map_attributes = {"taglist": taglist,
                              "markers": markers,
                              "covariance": cov}

            output_fname = "slam"
            with open(output_fname + '.txt', 'w') as map_file:
                json.dump(map_attributes, map_file, indent=2)

            return fruit_list, fruit_true_pos, aruco_true_pos

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

        print("Search order:")
        n_fruit = 1
        for fruit in search_list:
            for i in range(3):
                if fruit == fruit_list[i]:
                    self.target_fruits[fruit] = fruit_true_pos[i]
                    print('{}) {} at [{}, {}]'.format(n_fruit,
                                                      fruit,
                                                      np.round(fruit_true_pos[i][0], 1),
                                                      np.round(fruit_true_pos[i][1], 1)))
            n_fruit += 1

    # Waypoint navigation
    # the robot automatically drives to a given [x,y] coordinate
    # additional improvements:
    # you may use different motion model parameters for robot driving on its own or driving while pushing a fruit
    # try changing to a fully automatic delivery approach: develop a path-finding algorithm that produces the waypoints
    def drive_to_point(self, waypoint):
        if self.move_id == 1:
            self.turn(waypoint)
        elif self.move_id == 2:
            pass
            # turn detector on and check fruit
        elif self.move_id == 3:
            self.drive_straight(waypoint)

    def turn(self, waypoint):
        # Compute distance & angle towards goal position ---------------
        robot_pose = self.robot_state
        x_diff = waypoint[0] - robot_pose[0,:][0]
        y_diff = waypoint[1] - robot_pose[1,:][0]
        angle = np.arctan2(y_diff, x_diff) - robot_pose[2,:][0]
        # print(x_diff)
        # print(y_diff)
        # print(angle)
        desired_heading = (angle + np.pi) % (2 * np.pi) - np.pi  # limits range of angle to the range [-pi,pi]
        if desired_heading == -np.pi:
            desired_heading = np.pi

        # print(desired_heading)
        self.turn_angle(desired_heading)
        self.robot_state[2,:][0] = (np.arctan2(y_diff, x_diff) + np.pi) % (2 * np.pi) - np.pi

    def turn_angle(self, angle):
        # imports camera / wheel calibration parameters
        fileS = "calibration/param/scale.txt"
        scale = np.loadtxt(fileS, delimiter=',')
        fileB = "calibration/param/baseline.txt"
        baseline = np.loadtxt(fileB, delimiter=',')
        fileBL = "calibration/param/baseline_left.txt"
        baseline_left = np.loadtxt(fileBL, delimiter=',')
        fileBR = "calibration/param/baseline_right.txt"
        baseline_right = np.loadtxt(fileBR, delimiter=',')


        # Apply control to robot
        # turn towards the waypoint
        dir = angle / abs(angle)
        if dir == 1:
            self.move_time = baseline * abs(angle) / (
                        2 * scale * self.turn_ticks)  # replace with your calculation
        elif dir == -1:
            self.move_time = baseline * abs(angle) / (
                    2 * scale * self.turn_ticks)  # replace with your calculation

        print(f"Turning for {self.move_time} seconds")
        # self.command['motion'] = [0, dir]
        self.turn_ticks = 15
        lv, rv = self.pibot.set_velocity([0, dir], turning_tick=self.turn_ticks, time=self.move_time)
        self.move_start = time.time()

        # print("Current pose [{}, {}, {}]".format(self.robot.state[0], self.robot.state[1], self.robot.state[2]))
        # print("Arrived at [{}, {}]".format(waypoint[0], waypoint[1]))
        ####################################################

    def drive_straight(self, waypoint):
        # Compute distance & angle towards goal position ---------------
        robot_pose = self.robot_state
        x_diff = waypoint[0] - robot_pose[0,:][0]
        y_diff = waypoint[1] - robot_pose[1,:][0]
        distance_to_goal = np.sqrt(x_diff ** 2 + y_diff ** 2)
        self.drive_dist(distance_to_goal)
        self.robot_state[0,:][0] = waypoint[0]
        self.robot_state[1,:][0] = waypoint[1]

    def drive_dist(self, dist):
        # imports camera / wheel calibration parameters
        fileS = "calibration/param/scale.txt"
        scale = np.loadtxt(fileS, delimiter=',')

        # Apply control to robot
        # after turning, drive straight to the waypoint
        self.move_time = dist / (self.drive_ticks * scale)  # replace with your calculation
        print(f"Driving for {self.move_time} seconds")
        # self.command['motion'] = [1, 0]
        self.drive_ticks = 15
        self.pibot.set_velocity([1, 0], tick=self.drive_ticks, time=self.move_time)
        self.move_start = time.time()

        # print("Current pose [{}, {}, {}]".format(robot_pose[0], robot_pose[1], robot_pose[2]))
        # print("Arrived at [{}, {}]".format(waypoint[0], waypoint[1]))
        ####################################################

    def get_robot_pose(self):
        ####################################################
        # TODO: replace with your codes to estimate the pose of the robot
        # We STRONGLY RECOMMEND you to use your SLAM code from M2 here

        # update the robot pose [x,y,theta]
        # robot_pose = self.ekf.robot.state  # replace with your calculation
        ####################################################

        return robot_pose

    def detect(self):
        # after turning, turn on detection
        self.command['inference'] = True
        self.detect_target(camera_matrix)
        # if no fruit is detected
        if self.bboxes == []:
            # TODO: drive robot to waypoint
            drive_time = distance_to_goal / (drive_ticks * scale)
            print("Driving for {:.2f} seconds".format(drive_time))
            self.pibot.set_velocity([1, 0], tick=drive_ticks, time=drive_time)

            # TODO: update robot pose
            robot_pose[0] = waypoint[0]
            robot_pose[1] = waypoint[1]
            robot_pose[2] = (np.arctan2(y_diff, x_diff) + np.pi) % (
                    2 * np.pi) - np.pi  # limits range of angle to the range [-pi,pi]
            self.robot.state = robot_pose
            # robot_pose = get_robot_pose()
            print("Current pose [{}, {}, {}]".format(robot_pose[0], robot_pose[1], robot_pose[2]))
            print("Arrived at [{}, {}]".format(waypoint[0], waypoint[1]))

        else:  # if fruit is detected
            # check if fruit detected is in fruit list
            if self.bboxes[0] == self.target_fruits[self.fruit_id]:
                # TODO: drive robot to waypoint
                wheel_vel = 15
                drive_time = distance_to_goal / (wheel_vel * scale)
                print("Driving for {:.2f} seconds".format(drive_time))
                # ppi.set_velocity([1, 0], tick=wheel_vel, time=drive_time)

            else:  # if fruit is not in fruit list
                # add fruit into search list and true_map then find new path
                self.target_fruits.append(self.bboxes[0])
                output_filename = "search_list"
                with open(output_filename + '.txt', 'a') as f:
                    f.write(self.bboxes[0])

                output_filename = "search_list"
                with open(output_filename + '.txt', 'a') as f:
                    f.write('\n')
                    f.write(self.bboxes[0])

                # Read the entire file, excluding the last line
                output_filename = "M4_true_map_Copy"
                with open(output_filename + '.txt', 'r') as file:
                    lines = file.readlines()[:-1]

                # TODO: where to get fruit_pose
                line1 = "    \"{}\":".format(self.bboxes[0])
                line1 = ''.join([line1, " {\n"])
                line2 = "\t\"x\": {},\n".format(fruit_pose[0])
                line3 = "\t\"y\": {},\n".format(fruit_pose[1])
                # Open the file in write mode (this will overwrite the existing content)
                with open(output_filename + '.txt', 'w') as file:
                    file.writelines(lines[:-1])
                    file.writelines("    },\n")
                    file.write(line1)
                    file.write(line2)
                    file.write(line3)
                    file.write('    }\n')
                    file.write('}')
                # reset waypoint id and find path
                self.waypoint_id = 0
                self.run_fruit_search(level=2)

    def round_nearest(self, x, a):
        return round(round(x / a) * a, -int(math.floor(math.log10(a))))

    def generate_path(self):
        # input(f"press enter to search for {self.target_fruits[self.search_list[self.fruit_id]]}")
        start_pt = tuple(self.ekf.robot.state[0:2].reshape(-1))
        start_pt = tuple(self.round_nearest(x, 0.2) for x in start_pt)
        print(start_pt)
        self.path = a_star(f'{args.map}', self.target_fruits[self.search_list[self.fruit_id]], start_pt)
        self.path = self.path[1:-1]
        self.path.reverse()
        print(self.path)

    def run_fruit_search(self, level=2):
        if self.command['move']:
            if level == 1:
                # waypoint = [0.0, 0.0]
                # robot_pose = [0.0, 0.0, 0.0]
                # The following code is only a skeleton code the semi-auto fruit searching task
                while True:
                    # enter the waypoints
                    # instead of manually enter waypoints in command line, you can get coordinates by clicking on a map (GUI input), see camera_calibration.py
                    x, y = 0.0, 0.0
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

                    # estimate the robot's pose
                    # robot_pose = [0.0,0.0,0.0]
                    # robot_pose = get_robot_pose()

                    # robot drives to the waypoint
                    waypoint = [x, y]
                    self.drive_to_point(waypoint)
                    print("Finished driving to waypoint: {}; New robot pose: {}".format(waypoint, robot_pose))

                    # exit
                    # ppi.set_velocity([0, 0])
                    uInput = input("Add a new waypoint? [Y/N]")
                    if uInput == 'N' or uInput == 'n':
                        break

            elif level == 2:
                if self.waypoint_id < len(self.path):
                    if self.waypoint_id == 0:  # at the start of a path
                        self.generate_path()
                    # input(f"press enter to go next waypoint")
                    # robot drives to the waypoint
                    print(self.waypoint_id)
                    self.drive_to_point(self.path[self.waypoint_id])
                    print("Finished driving to waypoint: {}; New robot pose: {}".format(self.path[self.waypoint_id],self.ekf.robot.state.reshape(1,-1)))

            self.command["move"] = False

    def check_time(self):
        # check time and change if time up
        cur_time = time.time()
        # print(cur_time - self.move_start)
        # print(self.move_id)

        if cur_time - self.move_start >= self.move_time:
            self.move_id += 1
            if self.move_id > 3:
                # reset
                self.move_id = 1

                # check dist between waypoint & robot pose from SLAM
                # move to next waypoint in path
                self.waypoint_id += 1
                if self.waypoint_id >= len(self.path):  # move to next fruit
                    self.waypoint_id = 0
                    self.fruit_id += 1
                    if self.fruit_id > len(self.search_list):
                        self.quit = True
                        self.start = False
            self.command['move'] = True
            # print(self.command['move'])

    # get ground truth measurements /lms
    def get_true_measurements(self):
        with open("slam.txt", 'r') as map_file:
            map_attributes = json.load(map_file)  # need to import json
        taglist = map_attributes["taglist"]
        markers = np.array(map_attributes["markers"])

        for i, tag in enumerate(taglist):
            position = np.array([markers[:, i]])  # [x, y]
            lm = measure.Marker(position, tag)
            lms.append(lm)

    def init_M4(self):
        fruits_list, fruits_true_pos, aruco_true_pos = self.read_true_map(args.map)
        self.search_list = self.read_search_list()
        print(fruits_list)
        self.print_target_fruits_pos(self.search_list, fruits_list, fruits_true_pos)
        print(self.target_fruits)

    def start_sequence(self):
        # start_pts = [(0.6,0.6),(0.6,-0.6),(-0.6,-0.6),(-0.6,0.6)]
        start_pts = [(0.2,0.2),(0.2,-0.2),(-0.2,-0.2),(-0.2,0.2)]
        angle = 30
        sections = 360 // angle
        n = 1

        if len(self.path) > 0:
            cur_time = time.time()
            if (cur_time - self.move_start) >= 2:
                self.turn_angle((angle/180)*np.pi)
                self.start_turn360_id += 1
                if self.start_turn360_id >= sections:
                    self.start_turn360_id = 0
                    # generate new path
                    # self.path = a_star(f'{args.map}', start_pts[self.start_move_id//2], robot_pose)
                    # self.path = self.path[:-1]
                    # self.path.reverse()
                    self.path = []
                    self.move_start = 0.0
        else:
            cur_time = time.time()
            if (cur_time - self.move_start) >= math.ceil(self.move_time):
                if self.start_move_id % 2 == 0:  # even
                    if self.start_move_id == 0:
                        self.turn_angle((45/180)*np.pi)
                    elif self.start_move_id == 2:
                        self.turn_angle((-135/180)*np.pi)
                    elif self.start_move_id > 2:
                        self.turn_angle((-90 / 180) * np.pi)
                else:  # odd
                    if self.start_move_id == 1:
                        self.drive_dist(math.sqrt(2*(0.2**2)))
                    elif self.start_move_id > 1:
                        self.drive_dist(0.4)
                    self.path = [(0, 0), (0, 0)]
                self.start_move_id += 1
                if self.start_move_id == 8:
                    self.mode = 2








if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("Fruit searching")
    parser.add_argument("--map", type=str, default='M4_true_map.txt')
    parser.add_argument("--ip", metavar='', type=str, default='localhost')
    parser.add_argument("--port", metavar='', type=int, default=8000)
    parser.add_argument("--calib_dir", type=str, default="calibration/param/")
    parser.add_argument("--save_data", action='store_true')
    parser.add_argument("--play_data", action='store_true')
    parser.add_argument("--yolo_model", default='model.best.pt')
    args, _ = parser.parse_known_args()

    # read in camera matrix for target pose estimate
    fileK = f'calibration/param/intrinsic.txt'
    camera_matrix = np.loadtxt(fileK, delimiter=',')

    pygame.font.init()
    TITLE_FONT = pygame.font.Font('pics/8-BitMadness.ttf', 35)
    TEXT_FONT = pygame.font.Font('pics/8-BitMadness.ttf', 40)

    width, height = 700, 660
    canvas = pygame.display.set_mode((width, height))
    pygame.display.set_caption('ECE4078 2021 Lab')
    pygame.display.set_icon(pygame.image.load('pics/8bit/pibot5.png'))
    canvas.fill((0, 0, 0))
    splash = pygame.image.load('pics/loading.png')
    pibot_animate = [pygame.image.load('pics/8bit/pibot1.png'),
                     pygame.image.load('pics/8bit/pibot2.png'),
                     pygame.image.load('pics/8bit/pibot3.png'),
                     pygame.image.load('pics/8bit/pibot4.png'),
                     pygame.image.load('pics/8bit/pibot5.png')]
    pygame.display.update()

    start = False

    counter = 40
    while not start:
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                start = True
        canvas.blit(splash, (0, 0))
        x_ = min(counter, 600)
        if x_ < 600:
            canvas.blit(pibot_animate[counter % 10 // 2], (x_, 565))
            pygame.display.update()
            counter += 2

    operate = Operate(args)

    # path = a_star(f'{args.map}', (0.8, 0.4), (-1.8, 0.0))
    # print(path)
    # operate.turn((0.4, 0.4))
    # operate.drive_straight((0.4, 0.4))
    # while True:
    #     input("Press enter")
    #     operate.turn_angle((45/180)*np.pi)
    #     operate.pibot.set_velocity(operate.command['motion'], tick=operate.drive_ticks, turning_tick=operate.turn_ticks, time=operate.move_time)
    # operate.pibot.set_velocity([0, 1], turning_tick=15, time=2)


    while True:
        # start_time =
        operate.update_keyboard()
        # operate.take_pic()
        # drive_meas = operate.control()
        # operate.update_slam(drive_meas)
        # operate.save_image()
        # operate.record_data()
        # operate.detect_target(camera_matrix)

        if operate.start:
            # if operate.mode == 1:
            #     operate.start_sequence()
            # elif operate.mode == 2:
            operate.check_time()
            operate.run_fruit_search(level=2)
        # visualise
        # operate.draw(canvas)
        # pygame.display.update()




