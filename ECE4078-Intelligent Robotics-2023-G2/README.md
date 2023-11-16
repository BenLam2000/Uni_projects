# Overview
The goal of this group-based semester-long project was to build a robot that can locate 3 specified fruits in an arena in sequence autonomously while avoiding obstacles.

# Rules and restrictions
<a href="" target="blank"><img align="center" src="https://github.com/BenLam2000/Uni_projects/blob/main/ECE4078-Intelligent%20Robotics-2023-G2/pics/Alphabot2-pizero-8.jpg?raw=true" height="200" /></a>
1. All groups had to use the provided robot - Alphabot2, which had a RaspberryPi as its core. Python was the language to program the robot.
<a href="" target="blank"><img align="center" src="https://github.com/BenLam2000/Uni_projects/blob/main/ECE4078-Intelligent%20Robotics-2023-G2/pics/arena_labelled.png?raw=true" height="400" /></a>
3. All groups had to use the provided arena for the robot demonstration, with 10 AruCo markers and 5 fruits that would be randomly placed on it.
4. The only sensor allowed to use is the RaspberryPi camera. All other sensors onboard the Alphabot such as IR and ultrasonic sensors were strictly prohibited. Since this robot didn't have wheel encoders, we had to regularly recalibrate the robot's driving parameters to ensure the robot's actual motion did not deviate too much from its predicted trajectory. 

Based on this [video of the final robot demo](https://youtu.be/f8h7jwVJRQ0?si=i3vBQcp5ochfqqM5), the following are the main parts shown in the video:

# Part 2: SLAM
When the robot is moving about on the arena, it has no clue where it currently is and what its surrounding looks like. To achieve this, the robot performs Simultaneous Localization and Mapping (SLAM). 

## 2.1: Localization
In part 2 of the video, the robot is manually driven using keyboard to roam around the arena in a strategic path. While roaming around, the robot estimates its current location based on its speed and direction of movement at regular time intervals. However, the robot's actual movement will never be exactly same as the predicted trajectory (i.e. robot will not drive perfectly straight due to slippage on the floor surface). To improve the estimate of current robot location, the robot will use estimated distance of ArUco markers observed from its camera feed to correct the previous robot location prediction. This process of prediction and correction is an algorithm called Extended Kalman Filter (EKF), which is crucial to the localization part of SLAM. 

<a href="" target="blank"><img align="center" src="https://github.com/BenLam2000/Uni_projects/blob/main/ECE4078-Intelligent%20Robotics-2023-G2/pics/M2.gif?raw=true" height="400" /></a>

## 2.2: Mapping
As for mapping, as new markers are observed, they will populate the map in the GUI. Not only is the robot location constantly updating, the ArUco markers' location estimates are also being updated. The shrinking circles around the markers and robot indicate that the robot is becoming more certain of their location estimatios. These estimations will never be 100% accurate, but based on lots of test runs we know when the estimates of the marker locations are good enough to proceed with the following stages of the demo. Even though localization is a part of SLAM, the goal of this part is to obtain only the map of the ArUco marker estimations.

# Part 1: Obtaining Marker References
The SLAM map obtained only captures the relative relationship between markers, not their absolute locations. In an ideal scenario, we would be provided with a true map of actual locations of all ArUco markers, which can be used to align (rotate and translate) the estimations to obtain the best possible absolute locations. To mimic practical scenarios, we were not provided this true map during the final demo. Also, using just the raw SLAM map for autonomous fruit searching was not an option as it would result in a lot of collisions. Therefore, we came up with a solution to rotate on the spot at the starting point, take distance measurements of 4 random markers, treating this as our own "true map", which is used to align the 10 marker estimations from SLAM.

*SLAM aligned using true map*
*SLAM aligned using 4 markers* 

# Part 3: Obtaining Fruit Poses
Once we have the aligned map of all ArUco markers, we now need to obtain all 5 fruit locations on the arena by taking picures of the 5 fruits using the robot's camera. There are 2 important parts for this to work: knowing the robot's current location and the relative distance to a fruit captured by the camera.

## 3.1: Localization Using Trilateration


## 3.2: Obtaining Relative Fruit Distance Using Object Recognition









    

