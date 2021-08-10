# pixel-aware gyro aided KLT feature tracker
**Authors:** [Weibo Huang, Hong Liu](http://robotics.pkusz.edu.cn/en)

**10 Aug 2021**: Initial upload.

Pixel-Aware Gyro-aided KLT Feature Tracker is a feature tracker that remains accurate and robust under fast camera-ego motion conditions. The goal is to cope with the tracking fail problem in the conventional KLT tracker causing by the inadequate initial condition that falls out of the convergence region. In particular, we develop a pixel-aware gyro-aided feature prediction algorithm to predict the initial optical flow and obtain the path deformation matrix of each feature point. It increases the probability of initial estimates to locate in its convergence region. Different from the existing methods that assume all the tracked feature pairs were constrained by the same homography prediction matrix, our prediction matrix is flexible for each feature as it considers the pixel coordinates in the prediction process.

In this repository, we provide examples to run the feature tracker in the self-collected sequences named [PKUSZ_RealSenseD435i_sequence](https://drive.google.com/drive/folders/1oBaiijQvzDb9SezgaVPm1ABosvTcztJ7?usp=sharing), which was recorded by a hand-held [RealSense-D435i](https://www.intelrealsense.com/depth-camera-d435i/) camera. We also provide a ROS node to process live feature tracking in the V3_01_difficult sequence of the [EuRoC dataset](http://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets). **The library can be compiled without ROS.** The frame keypoints can be real-time detected by a modified [ORBextractor](https://github.com/raulmur/ORB_SLAM2/blob/master/src/ORBextractor.cc) that used in [ORB_SLAM2](https://github.com/raulmur/ORB_SLAM2) system, or load from files that detected by other methods like [SuperPoint](https://github.com/magicleap/SuperPointPretrainedNetwork). The user can set the **LoadDetectedKeypoints** parameter in **.yaml** to choose keypoint generation mode.


# 1. License

pixel_aware_gyro_aided_klt_feature_tracker is released under a [GPLv3 license - gitee](https://gitee.com/weibohuang/pixel_aware_gyro_aided_klt_feature_tracker/blob/master/LICENSE) or [GPLv3 license - github](https://github.com/weibohuang0314/pixel_aware_gyro_aided_klt_feature_tracker/blob/master/LICENSE).

If you use our method in an academic work, please cite:
```
    TODO: add citation
```

# 2. Prerequisites
We have tested the library in **Ubuntu 18.04**, with an Intel CPU i7-9750H (12 cores @2.60GHz) laptop computer with 16GB RAM. It should be easy to compile in other platforms. 

## C++11 or C++0x Compiler
We use the new thread and chrono functionalities of C++11.

## OpenCV
We use [OpenCV](http://opencv.org) to manipulate images and features. Dowload and install instructions can be found at: http://opencv.org. **Required at least 3.4. Tested with OpenCV 3.4**.

## Eigen3
Download and install instructions can be found at: http://eigen.tuxfamily.org. **Required at least 3.1.0**.

## glog
The glog source code is provided in /Thirdparty/glog-master/. See following Section 3 for installation.

## ROS (optional)
We provide some examples to process the live input or read from rosbag files using [ROS](ros.org). Building these examples is optional. In case you want to use ROS, a version Hydro or newer is needed. Tested on **melodic**.

# 3. Building library and Demo examples

Clone the repository:
```
git clone https://github.com/weibohuang0314/pixel_aware_gyro_aided_klt_feature_tracker.git
```
For the user from China Mainland, you can clone the repository from gitee:
```
git clone https://gitee.com/weibohuang/pixel_aware_gyro_aided_klt_feature_tracker.git
```

We provide a script `build.sh` to build the *Thirdparty* libraries and *pixel_aware_gyro_aided_klt_feature_tracker*. Please make sure you have installed all required dependencies (see section 2). Execute:
```
cd pixel_aware_gyro_aided_klt_feature_tracker
chmod +x build.sh
./build.sh
```

# 4 . Running Demo examples (RealSenseD435i)
1. `cd Examples/Demo/`
2. Uncompress `data/PKUSZ_RealSenseD435i_sequence/sequence_1.zip` into `data/PKUSZ_RealSenseD435i_sequence/` folder. The other sequences of PKUSZ_RealSenseD435i dataset can be downloaded from [Google Share](https://drive.google.com/drive/folders/1oBaiijQvzDb9SezgaVPm1ABosvTcztJ7?usp=sharing) or [BaiduNetDisk](https://pan.baidu.com/s/1f3RIVcMniJs0Z0apdtYRtw) (code: qjd2) for users from China Mainland.
   
3. Execute the following command.
```
./RealSenseD435i RealSenseD435i.yaml
```

4. By default, the keypoints are automatically detected using the `ORBextractor`. 
   
   If you want to load keypoints from file, we give an example as follows. First, uncompress `data/PKUSZ_RealSenseD435i_sequence/SuperPoints.zip` into `data/PKUSZ_RealSenseD435i_sequence/` folder. Second, in the `RealSenseD435i.yaml` file, change `LoadDetectedKeypoints` to `1` and set `DetectedKeypointsFile` to `"/data/PKUSZ_RealSenseD435i_sequence/SuperPoints/sequence_1"`. Third, run `./RealSenseD435i RealSenseD435i.yaml` in the terminal.

# 5. ROS Examples

### Building the nodes for ROS_Demo_Feature_Tracking
1. Add the path including *Examples/ROS/ROS_Demo_Feature_Tracking* to the ROS_PACKAGE_PATH environment variable. Open .bashrc file and add at the end the following line. Replace PATH by the folder where you cloned pixel_aware_gyro_aided_klt_feature_tracker:
```
export ROS_PACKAGE_PATH=${ROS_PACKAGE_PATH}:PATH/pixel_aware_gyro_aided_klt_feature_tracker/Examples/ROS
```

2. Execute `build_ros.sh` script:
```
chmod +x build_ros.sh
./build_ros.sh
```

### Running ROS_Demo_Feature_Tracking Node
1. Run the node using default configuration.
```
roslaunch ROS_Demo_Feature_Tracking EuRoC.launch
```
2. Check the configuration file `EuRoC.yaml`. If the parameter `readFromRosBag` is `0`, you need to play the rosbag file using ```rosbag play sequence.bag```; else if the parameter is `1`, you need to set the `rosBag` argument in the `EuRoC.launch` file.
3. By default, the keypoints are automatically detected using the `ORBextractor`. You can also load keypoints from file, as mentioned in Section 4.4.
