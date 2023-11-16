# Overview
The goal of this group-based semester-long project was to build a robot that can locate 3 specified fruits in an arena in sequence autonomously while avoiding obstacles.

# Rules and restrictions
<a href="" target="blank"><img align="center" src="https://github.com/BenLam2000/Uni_projects/blob/main/ECE4078-Intelligent%20Robotics-2023-G2/pics/Alphabot2-pizero-8.jpg?raw=true" height="200" /></a>
1. All groups had to use the provided robot - Alphabot2, which had a RaspberryPi as its core. Python was the language to program the robot.
<a href="" target="blank"><img align="center" src="https://github.com/BenLam2000/Uni_projects/blob/main/ECE4078-Intelligent%20Robotics-2023-G2/pics/arena_labelled.png?raw=true" height="400" /></a>
3. All groups had to use the provided arena for the robot demonstration, with 10 AruCo markers and 5 fruits that would be randomly placed on it.
4. The only sensor allowed to use is the RaspberryPi camera. All other sensors onboard the Alphabot such as IR and ultrasonic sensors were strictly prohibited. Since this robot didn't have wheel encoders, we had to regularly recalibrate the robot's driving parameters to ensure the robot's actual motion did not deviate too much from its predicted trajectory. 

Based on this [video of the final robot demo](https://youtu.be/f8h7jwVJRQ0?si=i3vBQcp5ochfqqM5), the following are the main parts shown in the video:
# Part 1: Obtaining Marker References
In an ideal scenario, we would be provided with a map of the ground truth locations of all ArUco markers and fruits on the arena, but not for the case during final demo. We came up with a solution to comput

# Part 2: SLAM
<a href="" target="blank"><img align="center" src="https://github.com/BenLam2000/Uni_projects/blob/main/ECE4078-Intelligent%20Robotics-2023-G2/pics/M2.gif?raw=true" height="400" /></a>

When the robot is moving about on the arena, it has no clue where it currently is and what its surrounding looks like. To achieve this, the robot performs Simultaneous Localization and Mapping (SLAM). In part 2 of the video, the robot is being manually driven using keyboard to roam around the arena in a strategic path. While roaming around, the robot estimates its current location based on its speed and direction of movement at regular time intervals. However, the robot's movement will never be able to match the predicted trajectory exactly (i.e. robot will not drive perfectly straight when told to do so due to slippage on the floor surface). To help improve the estimate of current robot location, the robot will use estimated distance of ArUco markers observed from its camera feed to correct the previously predicted robot location. This process of prediction and correction is an algorithm called Extended Kalman Filter (EKF), which is crucial to the localization part of SLAM. As for mapping, as new markers are observed, they will populate the map in the GUI. Not only is the robot location constantly updating, the ArUco markers' location estimates are also being updated. The circles around the markers and robot indicate the level of uncertainty in their locations. As the robot's and ArUco markers' location estimates improve, the circles will get smaller, indicating the robot is now much more certain of their locations. These estimations will never be 100% accurate, but based on lots of test runs we know when the estimates of the marker locations are good enough to proceed with the following stages of the demo. The thing is in this part we only need the ArUco marker estimates, not the robot location.










    

