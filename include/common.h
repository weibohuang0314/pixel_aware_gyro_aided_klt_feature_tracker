/**
* This file is part of pixel_aware_gyro_aided_klt_feature_tracker.
*
* Copyright (C) 2015-2022 Weibo Huang <weibohuang@pku.edu.cn> (Peking University)
* For more information see <https://gitee.com/weibohuang/pixel_aware_gyro_aided_klt_feature_tracker>
* or <https://github.com/weibohuang/pixel_aware_gyro_aided_klt_feature_tracker>
*
* pixel_aware_gyro_aided_klt_feature_tracker is a free software:
* you can redistribute it and/or modify it under the terms of the GNU General
* Public License as published by the Free Software Foundation, either version 3
* of the License, or (at your option) any later version.
*
* pixel_aware_gyro_aided_klt_feature_tracker is distributed in the hope that
* it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with pixel_aware_gyro_aided_klt_feature_tracker.
* If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef COMMON_H
#define COMMON_H

#include <iostream>
#include <algorithm>
#include <fstream>
#include <stdio.h>

#include "imu_types.h"

using namespace std;

bool read_from_rosbag = false; //false;
int downSampleRate = 1;
string imu_topic, image_topic;
CameraParams* pCameraParams;
IMU::Calib imuCalib;


int keypoint_number;
float threshold_of_predict_new_keypoint;
int half_patch_size = 5;

bool loadDetectedKeypoints = false;
string detectedKeypointsFile;

void loadConfigureFile(const std::string &file)
{
    cv::FileStorage fSettings(file, cv::FileStorage::READ);
    cv::FileNode node;
    imu_topic = string(fSettings["imuTopic"]);
    image_topic = string(fSettings["leftImageTopic"]);

    // read from rosbag
    read_from_rosbag = int(fSettings["readFromRosBag"]) == 1;
    std::cout << "read_from_rosbag: " << read_from_rosbag << std::endl;

    // downsample rate for image topics
    node = fSettings["downSampleRate"];
    if (!node.empty())  downSampleRate = int(node);
    std::cout << "downSampleRate: " << downSampleRate << std::endl;

    // loadDetectedKeypoints
    node = fSettings["LoadDetectedKeypoints"];
    if (!node.empty())  {
        loadDetectedKeypoints = int(node)==1;
        std::cout << "loadDetectedKeypoints: " << loadDetectedKeypoints << std::endl;
        detectedKeypointsFile = string(fSettings["DetectedKeypointsFile"]);
        std::cout << "detectedKeypointsFile: " << detectedKeypointsFile << std::endl;
    }

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
    pCameraParams = new CameraParams(type, fx, fy, cx, cy, k1, k2, p1, p2, k3, width, height, fps);

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

int findTimeCorrespondenIndex(std::vector<std::pair<double, std::string>>& vpTimeString, double& t) // only used to compare with SuperGlue
{
    for(int i = 0; i < vpTimeString.size(); i++)
    {
        auto p = vpTimeString[i];
        if(std::abs(t - p.first) < 0.0001)
            return i;   // found
    }
    return -1;  // not found
}

#endif // COMMON_H
