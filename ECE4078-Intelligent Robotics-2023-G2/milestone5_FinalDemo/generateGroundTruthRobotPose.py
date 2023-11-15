
############### run this scrpt using the command: ###################################
#-------------- python generateGroundTruth.py <output file name> -------------------#
############### without the angle brackets ##########################################

import numpy as np
import matplotlib.pyplot as plt
import math
import json
import ast


def parse_groundtruth(fname, fname2=''):
    fruit = ['redapple', 'greenapple', 'orange', 'mango', 'capsicum']

    aruco_x = []
    aruco_y = []
    aruco_num = []

    aruco_x_2 = []
    aruco_y_2 = []
    aruco_num2 = []

    fruit_x = []
    fruit_y = []
    fruit_list = []

    with open(fname, 'r') as f:
        try:
            gt_dict = json.load(f)
        except ValueError as e:
            with open(fname, 'r') as f:
                gt_dict = ast.literal_eval(f.readline())

        for key in gt_dict:
            if key.startswith("aruco"):
                aruco_num.append(key.split('_0')[0].split('aruco')[1])
                aruco_x.append(gt_dict[key]["x"])
                aruco_y.append(gt_dict[key]["y"])

    if fname2 != '':
        with open(fname2, 'r') as f2:
            try:
                gt_dict2 = json.load(f2)
            except ValueError as e:
                with open(fname2, 'r') as f2:
                    gt_dict2 = ast.literal_eval(f2.readline())

            for key in gt_dict2:
                if key.startswith("aruco"):
                    aruco_num2.append(key.split('_0')[0].split('aruco')[1])
                    aruco_x_2.append(gt_dict2[key]["x"])
                    aruco_y_2.append(gt_dict2[key]["y"])

            for key in gt_dict2:
                obj = key.strip('_0')
                if obj in fruit:
                    fruit_list.append(obj)
                    fruit_x.append(gt_dict2[key]["x"])
                    fruit_y.append(gt_dict2[key]["y"])

    return aruco_x, aruco_y, aruco_num, aruco_x_2, aruco_y_2, aruco_num2, fruit_x, fruit_y, fruit_list


def plot_map(aruco_x_list, aruco_y_list, aruco_num, title_, aruco_x_list2=[],aruco_y_list2=[], aruco_num2=[], fruit_x_list=[], fruit_y_list=[], fruit_list=[]):
    fruit_color = [[128, 0, 0], [155, 255, 70], [255, 85, 0], [255, 180, 0], [0, 128, 0]]

    # plt only intakes 0-1 RGB values
    for i in range(len(fruit_color)):
        for j in range(3):
            fruit_color[i][j] /= 255

    # Variables, p will contains clicked points, idx contains current point that is being selected
    space = np.array([-1.6, -1.2, -0.8, -0.4, 0, 0.4, 0.8, 1.2, 1.6])

    fig = plt.figure()
    plt.plot(0, 0, 'rx')  # plot red 'x' to mark centre
    # plot markers
    for i in range(len(aruco_num)):
        plt.scatter(aruco_x_list[i], aruco_y_list[i], color='C0')
        if not aruco_num2:
            plt.text(aruco_x_list[i] + 0.05, aruco_y_list[i] + 0.05, aruco_num[i], color='C0', size=12)
    # plot markers 2
    print(aruco_num2)
    if aruco_num2:
        for i in range(len(aruco_num2)):
            plt.scatter(aruco_x_list2[i], aruco_y_list2[i], color='C1', marker='x', s=100)
            plt.text(aruco_x_list2[i] + 0.05, aruco_y_list2[i] + 0.05, aruco_num2[i], color='C0', size=12)
        # plot fruits
        for i in range(len(fruit_x_list)):
            plt.scatter(fruit_x_list[i], fruit_y_list[i], color=fruit_color[i])
            # plt.text(fruit_x_list[i] + 0.05, fruit_y_list[i] + 0.05, fruit_list[i], color='C0', size=12)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(f'{title_}')
    plt.xticks(space)
    plt.yticks(space)
    plt.grid()
    plt.show()


if __name__ == '__main__':
    aruco_x_list, aruco_y_list, aruco_num, aruco_x_2, aruco_y_2, aruco_num2, fruit_x_list, fruit_y_list, fruit_list = parse_groundtruth('TRUEMAP.txt', 'TRUEMAP_ALIGNED.txt')
    # print(aruco_num)

    plot_map(aruco_x_list, aruco_y_list, aruco_num, 'Initial 4 Measured Markers (x,y)', aruco_x_2, aruco_y_2, aruco_num2, fruit_x_list, fruit_y_list, fruit_list)

    # def round_nearest(x, a):
    #     return round(round(x / a) * a, -int(math.floor(math.log10(a))))
    #
    # # pick points
    # def onclick(event):
    #     global idx
    #
    #     if event.button == 1:
    #         # left mouse click
    #         if idx % 2 == 0:  # odd num of clicks
    #             x = round_nearest(event.xdata, 0.4)
    #             y = round_nearest(event.ydata, 0.4)
    #             px.append(x)
    #             py.append(y)
    #         else:  # even num of clicks
    #             fruit_x = round_nearest(event.xdata, 0.4)
    #             fruit_y = round_nearest(event.ydata, 0.4)
    #             theta = round(math.atan2((fruit_y - py[-1]), (fruit_x - px[-1])), 6)
    #             p_theta.append(theta)
    #         idx += 1
    #     elif event.button == 3:
    #         # right click, delete point
    #         if (idx-1) % 2 == 0:  # last click was x,y
    #             del px[-1]
    #             del py[-1]
    #         else:
    #             del p_theta[-1]
    #         idx -= 1
    #
    #     # update plot
    #     plt.clf()  # clear figure, but keep window open
    #     plt.plot(0,0,'rx')  # replot centre point
    #     # plot markers
    #     for i in range(len(aruco_x_list)):
    #         plt.scatter(aruco_x_list[i], aruco_y_list[i], color='C0')
    #     # plot fruits
    #     for i in range(len(fruit_x_list)):
    #         plt.scatter(fruit_x_list[i], fruit_y_list[i], color=fruit_color[i])
    #     # plot robot poses
    #     for i in range(len(px)):
    #         plt.scatter(px[i], py[i], color='m')
    #         if not (len(p_theta) < len(px) and i >= len(p_theta)):
    #             plt.plot([px[i], px[i]+0.3*math.cos(p_theta[i])], [py[i], py[i]+0.3*math.sin(p_theta[i])], color='m')
    #             plt.text(px[i] + 0.2*math.cos(p_theta[i]) - 0.05, py[i] + 0.2*math.sin(p_theta[i]) - 0.05, i + 1, color='C0', size=12)
    #         else:
    #             plt.text(px[i] + 0.05, py[i] + 0.05, i + 1, color='C0', size=12)
    #
    #     plt.xlabel("X"); plt.ylabel("Y")
    #     if idx % 2 == 0:
    #         plt.title(f'Click on robot pose {int(idx/2)+1} (x,y)')
    #     else:
    #         plt.title(f"Click on fruit robot for robot pose {int(idx/2)+1} (theta)")
    #     plt.xticks(space); plt.yticks(space)
    #     plt.grid()
    #     plt.show()
    #
    # print("Specify points on the grid, close figure when done.")
    # ka = fig.canvas.mpl_connect('button_press_event', onclick)
    # plt.show()
    #
    # # updating the robot poses from images.txt
    # d = {}
    # for i in range(len(px)):
    #     d["pose" + str(i+1) + "_0"] = {"x": px[i], "y":py[i], "theta": p_theta[i]}
    #
    # output_filename = "TRUE_ROBOT_POSE"
    # with open(output_filename+'.txt', 'w') as f:
    #     json.dump(d, f, indent=4)