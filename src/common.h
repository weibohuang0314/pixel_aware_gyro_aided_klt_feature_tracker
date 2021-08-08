#ifndef COMMON_H
#define COMMON_H

#include<iostream>
#include<algorithm>
#include<fstream>
#include<chrono>
#include <stdio.h>
//#include <queue>
//#include <map>
//#include <thread>
#include <mutex>
#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <std_msgs/Header.h>
#include <sensor_msgs/Imu.h>

#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <boost/foreach.hpp>

#include "../Thirdparty/glog/include/glog/logging.h"
#include "../src/imu_types.h"
//#include "../src/frame.h"

using namespace std;

queue<sensor_msgs::ImuConstPtr> imu_buf;
queue<sensor_msgs::ImageConstPtr> image_buf;
std::mutex mMutexImage;
std::mutex mMutexIMU;
//cv::Mat M1l,M2l,M1r,M2r;


unsigned long cnt = 0;
int downSampleRate = 1;
//ORB_SLAM3::System* mpSLAM;

//bool bSLAMTracking = false;

string imu_topic, image_topic;
CameraParams camera_params;
IMU::Calib imuCalib;

bool read_from_rosbag = false; //false;
string rosbag_file;

float MANUALLY_ADD_TIME_DELAY = 0;
bool bEstimateTD = false;
float estimated_time_delay = 0; // the time delay estimated by algorithm

string output_file;


int keypoint_number;
float threshold_of_predict_new_keypoint;
int half_patch_size = 5;

void loadConfigureFile(const std::string &file)
{
    cv::FileStorage fSettings(file, cv::FileStorage::READ);
    cv::FileNode node;
    imu_topic = string(fSettings["imuTopic"]);
    image_topic = string(fSettings["leftImageTopic"]);

    // read from rosbag
    read_from_rosbag = int(fSettings["readFromRosBag"]) == 1;
    std::cout << "read_from_rosbag: " << read_from_rosbag << std::endl;

//    rosbag_file = string(fSettings["rosBag"]);
//    std::cout << "rosbag_file: " << rosbag_file << std::endl;
//    if(rosbag_file == "")
//        read_from_rosbag = false;

    // downsample rate for image topics
    node = fSettings["downSampleRate"];
    if (!node.empty())  downSampleRate = int(node);
    std::cout << "downSampleRate: " << downSampleRate << std::endl;

//    MANUALLY_ADD_TIME_DELAY = float(fSettings["manuallyAddTimeDelay"]);
//    std::cout << "manuallyAddTimeDelay: " << MANUALLY_ADD_TIME_DELAY << std::endl;
    bEstimateTD = int(fSettings["estimateTD"]) == 1;
    std::cout << "bEstimateTD: " << bEstimateTD << std::endl;

//    output_file = string(fSettings["output_file"]);

    // camera parameters
    string type = fSettings["Camera.type"];
    float fx = fSettings["Camera.fx"]; float fy = fSettings["Camera.fy"];
    float cx = fSettings["Camera.cx"]; float cy = fSettings["Camera.cy"];
    float k1 = fSettings["Camera.k1"]; float k2 = fSettings["Camera.k2"];
    float p1 = fSettings["Camera.p1"]; float p2 = fSettings["Camera.p2"];
    float k3 = 0;
    node = fSettings["Camera.k3"];
    if(!node.empty() && node.isReal()) k3 = float(node);
    int width = fSettings["Camera.width"]; int height = fSettings["Camera.height"];
    int fps = fSettings["Camera.fps"];
    camera_params = CameraParams(type, fx, fy, cx, cy, k1, k2, p1, p2, k3, width, height, fps);

    // IMU calibration (Tbc, Tcb, noise)
    float ng = fSettings["IMU.NoiseGyro"]; float na = fSettings["IMU.NoiseAcc"];
    float ngw = fSettings["IMU.GyroWalk"]; float naw = fSettings["IMU.AccWalk"];
    int imuFPS = fSettings["IMU.Frequency"];

    node = fSettings["Tbc"];
    cv::Mat Tbc = (cv::Mat_<float>(4,4) << node[0], node[1], node[2], node[3],
            node[4], node[5], node[6], node[7],
            node[8], node[9], node[10], node[11],
            node[12], node[13], node[14], node[15]);

    imuCalib = IMU::Calib(Tbc, ng, na, ngw, naw, imuFPS);

    keypoint_number = fSettings["KeyPointNumber"];
    threshold_of_predict_new_keypoint = fSettings["ThresholdOfPredictNewKeyPoint"];
    half_patch_size = fSettings["HalfPatchSize"];
}

void imuCallback(const sensor_msgs::ImuConstPtr &imu_msg)
{
    unique_lock<mutex> lock(mMutexIMU);

    if(!imu_buf.empty() && imu_msg->header.stamp.toSec() - imu_buf.back()->header.stamp.toSec() > imuCalib.dt + 0.001)
        std::cout << "[WARNING] IMU Data is not continue. last.t: " << std::to_string( imu_buf.back()->header.stamp.toSec() )
                  << ", cur.t: " << std::to_string(imu_msg->header.stamp.toSec()) << ", IMU dt: " << imuCalib.dt << std::endl;

    imu_buf.push(imu_msg);
}

void imageCallback(const sensor_msgs::ImageConstPtr &img_msg)
{
    unique_lock<mutex> lock(mMutexImage);

    if(!image_buf.empty() && img_msg->header.stamp.toSec() - image_buf.back()->header.stamp.toSec() > camera_params.dt + 0.001)
        std::cerr << "[WARNING] Image Data is not continue. last.t: " << std::to_string( image_buf.back()->header.stamp.toSec() )
                  << ", cur.t: " << std::to_string(img_msg->header.stamp.toSec()) << ", camera dt: " << camera_params.dt << std::endl;

    cnt ++;
    if (cnt%downSampleRate == 0)
        image_buf.push(img_msg);
}

cv::Mat getImageFromMsg(const sensor_msgs::ImageConstPtr &img_msg)
{
    // Copy the ros image message to cv::Mat.
    cv_bridge::CvImageConstPtr cv_ptr;
    try
    {
        cv_ptr = cv_bridge::toCvShare(img_msg);
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
    }

    cv::Mat img = cv_ptr->image.clone();
    return img;
}

#endif // COMMON_H
