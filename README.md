# 41X - Deep Learning Air Hockey Robot
Real-time air hockey robot controlled by a convolutional neural network.

### Main Components
1. [Air Hockey Game Simulator](https://github.com/arakhmat/air-hockey) is implemented using Pygame library and is used to generate images that are as close as possible to the frames that will be captured by an Android phone during an actual game. The simulator has real-life physics and serves two main purposes:
    1. Produce frames labeled with the action of programmed AI in order to pretrain convolutional layers of the neural network.
    2. Act as an environment for a Reinforcement Learning Agent.
2. [gym-air-hockey](https://github.com/arakhmat/gym-air-hockey) is an OpenAI Gym Environment Wrapper around [Air Hockey Game Simulator](https://github.com/arakhmat/air-hockey). It is used to determine the rewards, as well as process actions and observations.
3. [Perception](https://github.com/arakhmat/perception) is an Android Application that is used to control the robot during the game. It captures and processes the frames, infers the prediction of the CNN and sends it to Arduino via BluetoothLE.
4. This repository contains scripts used to generate labeled frames, pretrain CNN using supervised learning, further train CNN using reinforcement learning, convert keras model to caffe2 using [keras-to-caffe2 converter](https://github.com/arakhmat/keras-to-caffe2), visualize CNN filters and layer activations using [unveiler](https://github.com/arakhmat/unveiler) and etc.
### Training Neural Network
![alt text](https://github.com/arakhmat/41X/blob/master/images/network.png)
### System Overview
![alt text](https://github.com/arakhmat/41X/blob/master/images/system.jpg)
#### Screenshot of Frames Generated Using [Air Hockey Game Simulator](https://github.com/arakhmat/air-hockey)
![alt text](https://github.com/arakhmat/41X/blob/master/images/simulation.png)
#### Screenshot of Frames Captured Using [Perception](https://github.com/arakhmat/perception)
![alt text](https://github.com/arakhmat/41X/blob/master/images/phone.png)
### Prerequisites
[Python3](https://www.anaconda.com/download/)  
[Android Studio](https://developer.android.com/studio/index.html)  
[keras-to-caffe2](https://github.com/arakhmat/keras-to-caffe2)  
##### Optional:
[unveiler](https://github.com/arakhmat/unveiler)
### Download and Install
```
git clone --recursive https://github.com/arakhmat/41X
cd 41X
cd air-hockey; pip install -e .; cd ..;
cd gym-air-hockey; pip install -e .; cd ..;
```
