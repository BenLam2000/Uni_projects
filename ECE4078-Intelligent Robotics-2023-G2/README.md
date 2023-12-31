# Overview
<p align="justify">
The goal of this group-based semester-long project was to develop an autonomous robot that can locate 3 fruits in an arena in a specified sequence while avoiding obstacles. <a href="https://youtu.be/f8h7jwVJRQ0?si=PVQi02WeHcUocjod" target="_blank">This</a> is a video of the final robot demonstration.
</p>

# Rules and restrictions
<div align="center">
    <a href="" target="blank"><img align="center" src="https://github.com/BenLam2000/Uni_projects/blob/main/ECE4078-Intelligent%20Robotics-2023-G2/pics/Alphabot2-pizero-8.jpg?raw=true" height="300" /></a>
</div>
<div align="center">
    <i>Alphabot2</i>
</div>
<div align="justify">
1. All groups had to use the provided robot - Alphabot2, which had a RaspberryPi as its core. Python was the language to program the robot.
</div>
<div align="center">
    <a href="" target="blank"><img align="center" src="https://github.com/BenLam2000/Uni_projects/blob/main/ECE4078-Intelligent%20Robotics-2023-G2/pics/arena_labelled.png?raw=true" height="500" /></a>
</div>
<div align="center">
    <i>Arena for with fruits and ArUco markers</i>
</div>
<div align="justify">
2. All groups had to use the provided arena for the robot demonstration, with 10 ArUco markers and 5 fruits that would be randomly placed on it.
</div>
<div align="justify">
3. The only sensor allowed to use is the RaspberryPi camera. All other sensors onboard the Alphabot such as IR and ultrasonic sensors were strictly prohibited. Since this robot didn't have wheel encoders, we had to regularly recalibrate the robot's driving parameters to ensure the robot's actual motion did not deviate too much from its predicted trajectory.
</div>

- - - - 
The following are the main parts shown in the demo video:

# Part 2: SLAM
<div align="justify">
When the robot is moving about on the arena, it has no clue where it currently is and what its surrounding looks like. To achieve this, the robot performs Simultaneous Localization and Mapping (SLAM). 
</div>

## 2.1: Localization
<div align="justify">
In part 2 of the video, the robot is manually driven using keyboard to roam around the arena in a strategic path. While roaming around, the robot estimates its current location based on its speed and direction of movement at regular time intervals. However, the robot's actual movement will never be exactly same as the predicted trajectory (i.e. robot will not drive perfectly straight due to slippage on the floor surface). To improve the estimate of current robot location, the robot will use estimated distance of ArUco markers observed from its camera feed to correct the previous robot location prediction. This process of prediction and correction is an algorithm called Extended Kalman Filter (EKF), which is crucial to the localization part of SLAM. 
</div>

## 2.2: Mapping
<div align="center">
    <a href="" target="blank"><img align="center" src="https://github.com/BenLam2000/Uni_projects/blob/main/ECE4078-Intelligent%20Robotics-2023-G2/pics/M2.gif?raw=true" height="500" /></a>
</div>
<div align="center">
    <i>Different stages of SLAM</i>
</div>
<div align="justify">
As for mapping, as new markers are observed, they will populate the map in the GUI. Not only is the robot location constantly updating, the ArUco markers' location estimates are also being updated. The shrinking circles around the markers and robot indicate that the robot is becoming more certain of their location estimatios. These estimations will never be 100% accurate, but based on lots of test runs we know when the estimates of the marker locations are good enough to proceed with the following stages of the demo. Even though localization is a part of SLAM, the goal of this part is to obtain only the map of the ArUco marker estimations.
</div>

# Part 1: Obtaining Marker References
<div align="justify">
The SLAM map obtained only captures the relative relationship between markers, not their absolute locations. In an ideal scenario, we would be provided with a true map of actual locations of all ArUco markers, which can be used to align (rotate and translate) the estimations to obtain the best possible absolute locations. To mimic practical scenarios, we were not provided this true map during the final demo. Also, using just the raw SLAM map for autonomous fruit searching was not an option as it would result in a lot of collisions. Therefore, we came up with a solution to rotate on the spot at the starting point, take distance measurements of 4 random markers, treating this as our own "true map", which is used to align the 10 marker estimations from SLAM.
</div>

<div align="center">
    <a href="" target="blank"><img align="center" src="https://github.com/BenLam2000/Uni_projects/blob/main/ECE4078-Intelligent%20Robotics-2023-G2/pics/SLAM_eval.png?raw=true" height="400" /></a>
</div>
<div align="center">
    <i>SLAM map aligned using true map (ideal case)</i>
</div>

<div align="center">
    <a href="" target="blank"><img align="center" src="https://github.com/BenLam2000/Uni_projects/blob/main/ECE4078-Intelligent%20Robotics-2023-G2/pics/TRUEMAP_ALIGNED.png?raw=true" height="400" /></a>
</div>
<div align="center">
    <i>SLAM map aligned using 4 measured markers</i>
</div>

# Part 3: Obtaining Fruit Poses
<div align="justify">
Once we have the aligned map of all ArUco markers, we now need to obtain all 5 fruit locations on the arena by taking picures of the 5 fruits using the robot's camera. There are 2 important parts for this to work: knowing the robot's current location and the relative distance to a fruit captured by the camera.
</div>

## 3.1: Localization Using Trilateration
<div align="justify">
After attempting to use SLAM for this localization with different parameter settings, we realized it would take too long to find the right combination of settings and even then it would be very inconsistent between attempts.
</div>

<div align="center">
    <a href="" target="blank"><img align="center" src="https://github.com/BenLam2000/Uni_projects/blob/main/ECE4078-Intelligent%20Robotics-2023-G2/pics/trilateration.png?raw=true" height="400" /></a>
</div>
<div align="center">
    <i>Trilateration between robot and markers</i>
</div>

<div align="justify">
Therefore, we switched to localization using <a href="https://www.101computing.net/cell-phone-trilateration-algorithm/" target="_blank">trilateration</a>, where the robot is able to estimate its location with just distance measurements of any 3 observed markers while rotating on the spot. We found this method to be more robust and consistent because it is independent of previous location estimates along the path, unlike SLAM, whose error will accumulate as the robot deviates more from its desired trajectory.
</div>

## 3.2: Obtaining Relative Fruit Distance Using Object Recognition
<div align="justify">
To recognize the fruits, we have trained a YOLOv8 object detection model with our custom generated dataset (10000 images) consisting of superimposed fruits and other obstacles onto background images of the arena. Data augmentation techniques were used to increase variation and dataset size to make the model recognize the fruits under different lighting conditions and from differnt angles. 
</div>

<div align="center">
    <a href="" target="blank"><img align="center" src="https://github.com/BenLam2000/Uni_projects/blob/main/ECE4078-Intelligent%20Robotics-2023-G2/pics/augment.png?raw=true" height="300" /></a>
</div>
<div align="center">
    <i>Example of superimposed image</i>
</div>

The bounding box dimensions detected via the model would be used to estimate the relative distance from the robot to the fruit.

<div align="center">
    <a href="" target="blank"><img align="center" src="https://github.com/BenLam2000/Uni_projects/blob/main/ECE4078-Intelligent%20Robotics-2023-G2/pics/M3.png?raw=true" height="300" /></a>
</div>
<div align="center">
    <i>Object recognition with bounding box surrounding fruit</i>
</div>

# Part 4: Autonomous fruit search
<div align="justify">
The aligned SLAM map and fruit locations from previous parts are used in this final part where the robot is tasked to visit 3 fruits (stop within 50cm radius) in a particular order, while the 2 remaining fruits and the 10 ArUco markers will be treated as obstacles. These obstacles must be avoided otherwise a penalty will be incurred. The most crucial aspect to achieve this task is path planning.
</div>

## 4.1: Path Planning
<div align="justify">
The robot uses the A* (A-star) search algorithm to plan the path from starting point to the final fruit by setting multiple waypoints in between. A-star was the preferred algorithm because it implements a grid-based approach, which is very compatible with our arena.
</div>

<div align="center">
    <a href="" target="blank"><img align="center" src="https://github.com/BenLam2000/Uni_projects/blob/main/ECE4078-Intelligent%20Robotics-2023-G2/pics/astar.png?raw=true" height="400" /></a>
</div>
<div align="center">
    <i>A-star search algorithm</i>
</div>

<div align="justify">
After the robot reaches each waypoint, it will rotate on the spot and perform trilateration similar to part 3 (<a href="https://youtu.be/f8h7jwVJRQ0?si=wPZ15ErTzgAfGS_a&t=883" target="_blank">from video</a>) to estimate its current location. The path will be adjusted so that it always uses the current location estimate and the next waypoint, so it will be robust even if the robot does not reach precisely at the desired waypoint.
</div>

<div align="justify">
The maximum allowable distance between any 2 waypoints is set to a relatively small value compared to each grid of the arena so that it checks its current location often enough that it will not deviate from the desired waypoint until it collides with an obstacle or go out of arena bounds. In a way this is considered in-built obstacle avoidance and a dedicated algorithm is no longer needed for this.
</div>











    

