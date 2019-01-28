# Self-Driving Car using ROS  
#### Udacity Self-Driving Car Engineer Nanodegree --- Capstone Project


## Team 4Tzones
![LOGO](./imgs/4Tzones_logo.svg)  

|     Team Member     |            Email             |                    LinkedIn                            |  
|        :---:        |            :---:             |                      :---:                             |  
| Mohamed Elgeweily   | mohamed.elgeweily@gmail.com  | https://www.linkedin.com/in/mohamed-elgeweily-05372377 |  
| Jerry Tan Si Kai    | jerrytansk@gmail.com         | https://www.linkedin.com/in/thejerrytan                |  
| Karthikeya Subbarao | karthikeya108@gmail.com      | https://www.linkedin.com/in/karthikeyasubbarao         |  
| Pradeep Korivi      | pradeepkorivi@gmail.com      | https://www.linkedin.com/in/pradeepkorivi              |  
| Sergey Morozov      | ser94mor@gmail.com           | https://www.linkedin.com/in/aoool                      |  

All team members contributed equally to the project.

*4Tzones* means "Four Time Zones," indicating that team members were located in 4 different time zones 
while working on this project. The time zones range from UTC+1 to UTC+8.


## Software Architecture
![Software architecture](./imgs/final-project-ros-graph-v2.png)  

Note that obstacle detection is not implemented for this project.


## Traffic Light Detection Node

A large part of the project is to implement a traffic light detector/classifier that recognizes 
the color of nearest upcoming traffic light and publishes it to /waypoint_updater node so it can prepare 
the car to speed up or slow down accordingly. Because the real world images differ substantially from simulator images, 
we tried out different approaches for both. The approaches which worked best are described below.

### Simulator (Highway) --- OpenCV Approach
In this approach we used the basic features of OpenCV to solve the problem, the steps are described below.
* Image is transformed to HSV colorspace, as the color feature can be extracted easily in this colorspace.
* Mask is applied to isolate red pixels in the image. 
* Contour detection is performed on the masked image.
* For each contour, area is checked, and, if it falls under the approximate area of traffic light, 
polygon detection is performed and checked if the the number of sides is more than minimum required closed loop polygon. 
* If all the above conditions satisfy there is a red sign in the image. 

#### Pros
* This approach is very fast.
* Uses minimum resources.

#### Cons
* This is not robust enough, the thresholds need to be adjusted always.
* Doesnt work properly on real world data as there is lot of noise. 

### Real World (Test Lot) --- YOLOv3-tiny (You Only Look Once)
We used this approach for real world.
TODO:write about it

### Real World (Test Lot) --- SSD (Single Shot Detection)
We need to solve both object detection - where in the image is the object, 
and object classification --- given detections on an image, classify traffic lights. 
While there are teams who approached it as 2 separate problems to be solved, 
recent advancements in Deep Learning has developed models that attempt to solve both at once.
For example, SSD (Single Shot Multibox Detection) and YOLO (You Only Look Once).

We attempted transfer learning using the pre-trained SSD_inception_v2 model trained on COCO dataset, 
and retrain it on our own dataset for NUM_EPOCHS, achieving a final loss of FINAL_LOSS.

Here is a sample of the dataset.
![Udacity Test Site training images](report/udacity_visualization.png)

Sample dataset for simulator images
![simulator training images](report/sim_visualization.png)

Here are the results of our trained model.
(Insert image here!)


### Dataset

#### Image Collection
We used images from 3 ROS bags provided by Udacity.
* [traffic_lights.bag](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/traffic_light_bag_file.zip)
* [just_traffic_light.bag](https://drive.google.com/file/d/0B2_h37bMVw3iYkdJTlRSUlJIamM/view?usp=sharing) 
* [loop_with_traffic_light.bag](https://drive.google.com/file/d/0B2_h37bMVw3iYkdJTlRSUlJIamM/view?usp=sharing)


### Other approaches for traffic light detection

We experimented with few other (unsuccessful) approaches to detect traffic light. 

#### Idea

The idea is to use the entire image with a given traffic light color as an individual class. This means we will have 4 classes

 1. Entire image showing `yellow` traffic sign 
 2. Entire image showing `green` traffic sign 
 3. Entire image showing `red` traffic sign 
 4. Entire image showing `no` traffic sign 

#### Dataset

We created a dataset combining two other datasets that was already made available [here](https://github.com/alex-lechner/Traffic-Light-Classification) and [here](https://github.com/coldKnight/TrafficLight_Detection-TensorFlowAPI#get-the-dataset).

The combined dataset can be found [here](https://www.dropbox.com/s/k8l0aeopw544lud/simulator.tgz?dl=0).

#### Models

We trained couple of models:

1. A simple CNN with two convolutional layers, a fully connected layer and an output layer. The initial results looked promising with `training accuracy > 97%` and `test accuracy > 90%`. However when we deployed and tested the model, the results were not consistent. The car did not always stop at red lights and sometimes it did not move even when the lights were green. Efforts to achieve higher accuracies were in vain. 

2. Used transfer learning for multi-class classification approach using `VGG19` and `InceptionV3` models, using `imagenet` weights. The network did not learn anything after `1-2` epochs and hence the training accuracy never exceeded `65%`.


### Learning Points


### Future Work


### Acknowledgements

- We would like to thank Udacity for providing the instructional videos and learning resources.
- We would like to thank Alex Lechner for his wonderful tutorial on how to do transfer learning on TensorFlow Object Detection API research models and get it to run on older tensorflow versions, as well as providing datasets. You can view his readme here: https://github.com/alex-lechner/Traffic-Light-Classification/blob/master/README.md#1-the-lazy-approach
