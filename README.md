# 41X - Air Hockey Robot Capstone Project
Undegraduate capstone project the goal of which is to build a real-time air hockey robot controlled by neural network.

### How it works
Neural network is pretrained with labeled frames generated using [Air Hockey Game Simulator](https://github.com/arakhmat/air-hockey), and then trained via reinforcement learning techniques using [gym-air-hockey](https://github.com/arakhmat/gym-air-hockey) as the environment. Then, the model is converted from keras to caffe2 using [keras-to-caffe2 converter](https://github.com/arakhmat/keras-to-caffe2). After that, the model is copied over to the [Android application](https://github.com/arakhmat/perception). Next, arduino is turned on. Finally, application is launched and after connecting to Arduino via Bluetooth LE model, it sends actions to Arduino to control the robot.
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
