
############### run this scrpt using the command: ###################################
#-------------- python generateGroundTruth.py <output file name> -------------------#
############### without the angle brackets ##########################################

import numpy as np
import matplotlib.pyplot as plt
import math
import json
import os.path

def updateTruemap(old_truemap, new_point, new_obstacle):
    # Make a new truemap txt file
    # If TRUEMAP_new exist (means the TRUEMAP has been updated before), overwrite that file
    if os.path.exists('./TRUEMAP_new.txt'):
        new_truemap = 'TRUEMAP_new'
        old_truemap = new_truemap + '.txt'
    # If it does not exist (means this is the first time the TRUEMAP is being updated), make a TRUEMAP_new file
    else:
        new_truemap = old_truemap.rsplit(".")[0] + "_new"    

    # Read through the truemap
    with open(old_truemap) as f:
        data = f.read()

    # Reconstruct the data as a dictionary
    data_dict = json.loads(data) 

    # Rename the new_obstacle's name
    new_obstacle = new_obstacle + "_0"

    # Add the new_obstacle and its coordinate to js
    data_dict[new_obstacle] = {}
    data_dict[new_obstacle]['x'] = new_point[0]
    data_dict[new_obstacle]['y'] = new_point[1]

    # Write into new truemap
    with open(new_truemap +'.txt', 'w') as f:
        json.dump(data_dict, f, indent=4)    

# EG:
# updateTruemap('TRUEMAP.txt',(0.4,1.2),"orange")
# updateTruemap('TRUEMAP.txt',(-0.8,-0.8),"capsicum")
# updateTruemap('TRUEMAP.txt',(-0.8,1.2),"mango")



# if __name__ == '__main__':
#     idx = 0
#     space = np.array([-1.6, -1.2, -0.8, -0.4, 0, 0.4, 0.8, 1.2, 1.6])    
    
#     fig = plt.figure()
#     plt.plot(0,0,'rx')  # plot red 'x' to mark centre
#     plt.xlabel("X"); plt.ylabel("Y")
#     plt.title(f'Click on aruco marker {idx + 1} (x,y)')
#     plt.xticks(space); plt.yticks(space)
#     plt.grid()
    
#     # Variables, p will contains clicked points, idx contains current point that is being selected
#     px, py = [], []
#     fruit = ['redapple', 'greenapple', 'orange', 'mango', 'capsicum']
#     fruit_color = [[128, 0, 0], [155, 255, 70], [255, 85, 0], [255, 180, 0], [0, 128, 0]]

#     # plt only intakes 0-1 RGB values
#     for i in range(len(fruit_color)):
#         for j in range(3):
#             fruit_color[i][j] /= 255
    
#     def round_nearest(x, a):
#         return round(round(x / a) * a, -int(math.floor(math.log10(a))))
    
#     # pick points
#     def onclick(event):
#         global idx

#         x = round_nearest(event.xdata, 0.4)
#         y = round_nearest(event.ydata, 0.4)

#         if event.button == 1:
#             # left mouse click
#             px.append(x)
#             py.append(y)
#             idx += 1
#         elif event.button == 3:
#             # right click, delete point
#             del px[-1]
#             del py[-1]
#             idx -= 1

#         plt.clf()  # clear figure, but keep window open
#         plt.plot(0,0,'rx')  # replot centre point
#         # plt.scatter(px[:idx],py[:idx], color='C0')
#         for i in range(len(px)):
#             if i < 10:
#                 plt.scatter(px[i],py[i], color='C0')
#                 plt.text(px[i]+0.05, py[i]+0.05, i+1, color='C0', size=12)
#             else:
#                 plt.scatter(px[i],py[i], color=fruit_color[i-10])
#                 plt.text(px[i]+0.05, py[i]+0.05, i+1, color=fruit_color[i-10], size=12)
#         plt.xlabel("X"); plt.ylabel("Y")
#         # if idx < 10:
#         #     plt.title(f'Click on aruco marker {idx + 1} (x,y)')
#         # else:
#         #     print(idx)
#         #     plt.title(f'Click on {fruit[idx-10]} (x,y)')
#         plt.xticks(space); plt.yticks(space)
#         plt.grid()
#         plt.show()
    
#     print("Specify points on the grid, close figure when done.") 
#     ka = fig.canvas.mpl_connect('button_press_event', onclick)
#     plt.show()
    
#     d = {}
#     for i in range(len(px)):
#         if i < 10:
#             d["aruco" + str(i+1) + "_0"] = {"x": px[i], "y":py[i]}
#         else:
#             d[fruit[i-10] + "_0"] = {"x": px[i], "y":py[i]}

#     output_filename = "TRUEMAP"
#     with open(output_filename +'.txt', 'w') as f:
#         json.dump(d, f, indent=4)