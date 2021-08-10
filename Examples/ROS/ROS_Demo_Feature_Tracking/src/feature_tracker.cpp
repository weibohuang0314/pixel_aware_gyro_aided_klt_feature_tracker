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

#include <ros/ros.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <cv_bridge/cv_bridge.h>
#include <message_filters/subscriber.h>
#include <std_msgs/Header.h>
#include <sensor_msgs/Imu.h>
#include <boost/foreach.hpp>

#include <iostream>
#include <algorithm>
#include <fstream>
#include <chrono>
#include <stdio.h>
#include <time.h>
#include <mutex>

#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>

#include "glog/logging.h"

#include "frame.h"
#include "gyro_aided_tracker.h"
#include "ORBDetectAndDespMatcher.h"
#include "ORBextractor.h"

#include "common.h"

unsigned long cnt = 0;
float MANUALLY_ADD_TIME_DELAY = 0;  // parameter in .launch
string rosbag_file;                 // parameter in .launch
queue<sensor_msgs::ImuConstPtr> imu_buf;
queue<sensor_msgs::ImageConstPtr> image_buf;
std::mutex mMutexImage;
std::mutex mMutexIMU;

double time_cur = 0;
double time_prev = 0;
bool data_valid = true;

cv::Mat image_cur, image_cur_distort;
Frame curFrame, lastFrame;
Frame curFrameWithoutGeometryValid;  // used to display temporal tracked featrues
vector<IMU::Point> vImuMeas;        // IMU measurements from previous image to current image

std::string saveFolderPath;

bool step_mode = false;
bool do_rectify = true;

bool test_orb_detect_and_desp_matcher = false;
ORB_SLAM2::ORBextractor *pORBextractorLeft, *pORBextractorRight;

std::vector<std::pair<double, std::string>> vpTimeCorrespondens;


void setFrameWithoutGeometryValid(Frame& frame, GyroAidedTracker& matcher)
{
    curFrameWithoutGeometryValid = Frame(frame);
    matcher.SetBackToFrame(curFrameWithoutGeometryValid);
    frame.SetCurFrameWithoutGeometryValid(&curFrameWithoutGeometryValid);
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

void sensorProcessTimer(const ros::TimerEvent& event)
{
    // deal with historical measurements
    if(!image_buf.empty() && !imu_buf.empty()){
        double old_imu_t = imu_buf.front()->header.stamp.toSec();
        double old_img_t = image_buf.front()->header.stamp.toSec();
        if(std::abs(old_imu_t - old_img_t) > 3){
            {
                unique_lock<mutex> lock(mMutexImage);
                LOG(INFO) << "Erasing image_buf ...";
                while (!image_buf.empty())
                    image_buf.pop();
            }
            {
                unique_lock<mutex> lock(mMutexIMU);
                LOG(INFO) << "Erasing imu_buf ...";
                while (!imu_buf.empty())
                    imu_buf.pop();
            }
        }
    }

    if(!image_buf.empty() && !imu_buf.empty()
            && imu_buf.back()->header.stamp.toSec() > image_buf.front()->header.stamp.toSec() - MANUALLY_ADD_TIME_DELAY)
    {
        {
            unique_lock<mutex> lock(mMutexImage);
            time_cur = image_buf.front()->header.stamp.toSec();

            image_cur_distort = getImageFromMsg(image_buf.front());
            image_buf.pop();

            if(do_rectify){
                cv::remap(image_cur_distort, image_cur, pCameraParams->M1, pCameraParams->M2, cv::INTER_LINEAR);

                /*// save rectified images
                std::string::size_type pos = rosbag_file.rfind(".bag");
                std::string dir = rosbag_file.substr(0,pos) + "/cam0_rectify/data/";
                std::string command = "mkdir -p " + dir;
                int a = system(command.c_str());
                std::string path = dir + std::to_string(long(time_cur * 1e9)) + ".png";
                cv::imwrite(path, image_cur);

                std::ofstream fp_image_file_list(rosbag_file.substr(0,pos)+"/image_file_list.txt", std::ofstream::app);
                if(!fp_image_file_list.is_open())
                    std::cout << "cannot open: " << rosbag_file.substr(0,pos)+"/image_file_list.txt" << std::endl;
                else {
                    fp_image_file_list << "/cam0_rectify/data/" + std::to_string(long(time_cur * 1e9)) << ".png" << std::endl;
                }
                // end save rectified images
                */
            }
            else
                image_cur = image_cur_distort.clone();
        }

        // Load imu measurements
        {
            // Skip the imu measurement before pretime.
            // This may happen when the IMU message is delay.
            while(!imu_buf.empty() && imu_buf.front()->header.stamp.toSec() < time_prev
                  && std::abs(imu_buf.front()->header.stamp.toSec() - time_prev) > 3.0){
                unique_lock<mutex> lock(mMutexIMU);
                LOG(INFO) << "pop up imu_buf, time_prev: " << std::to_string(time_prev)
                          << ", imu.t: " << std::to_string(imu_buf.front()->header.stamp.toSec());
                imu_buf.pop();
            }

            // Load imu measurements from previous frame
            while(!imu_buf.empty() && imu_buf.front()->header.stamp.toSec() < time_cur - MANUALLY_ADD_TIME_DELAY)
            {
                unique_lock<mutex> lock(mMutexIMU);
                sensor_msgs::ImuConstPtr imuConst = imu_buf.front();
                double t = imuConst->header.stamp.toSec();
                double dx = imuConst->linear_acceleration.x;
                double dy = imuConst->linear_acceleration.y;
                double dz = imuConst->linear_acceleration.z;
                double rx = imuConst->angular_velocity.x;
                double ry = imuConst->angular_velocity.y;
                double rz = imuConst->angular_velocity.z;
                vImuMeas.push_back(IMU::Point(dx,dy,dz,rx,ry,rz,t));
                imu_buf.pop();

                /*// save imu measurements to file
                std::string::size_type pos = rosbag_file.rfind(".bag");
                std::ofstream fp_imu_data(rosbag_file.substr(0,pos)+"/imu.txt", std::ofstream::app);
                if(!fp_imu_data.is_open())
                    std::cout << "cannot open: " << rosbag_file.substr(0,pos)+"/imu.txt" << std::endl;
                else {
                    fp_imu_data <<  std::to_string(long(t * 1e9)) << " "
                                 << dx << " " << dy << " " << dz << " "
                                 << rx << " " << ry << " " << rz << std::endl;
                }
                // end save imu measurements to file
                */
            }

            data_valid = true;
            if (vImuMeas.empty())
                data_valid = false;
        }
    }

    // Feature tracking
    if (data_valid)
    {
        curFrame = Frame(time_cur, image_cur, image_cur_distort, &lastFrame, pCameraParams, pORBextractorLeft, vImuMeas, keypoint_number, threshold_of_predict_new_keypoint);
        if(lastFrame.mGray.empty()){    // if lastFrame is empty, then detect keypoints
            if(!loadDetectedKeypoints){ // Default: detect keypoints using ORBextractor
                curFrame.DetectKeyPoints(pORBextractorLeft);
            }else{  // else: load keypoints from file. Default: not execute
                int idx = findTimeCorrespondenIndex(vpTimeCorrespondens, curFrame.mTimeStamp);
                if (idx < 0){
                    LOG(ERROR) << "Can not find detected keypoints !!! Please check the setting of parameter - 'DetectedKeypointsFile'. curFrame.T: " << std::to_string(curFrame.mTimeStamp);
                    return;
                }
                else {
                    std::string path = detectedKeypointsFile + "/" + vpTimeCorrespondens[idx].second + ".txt";
                    curFrame.LoadDetectedKeypointFromFile(detectedKeypointsFile);
                }
            }
        }

        int n_predict = 0;
        if(!lastFrame.mGray.empty())
        {
            Timer timer;
            cv::Point3f biasg(0,0,0);

            /// Pixel-Aware Gyro-Aided KLT Feature Tracking
            GyroAidedTracker gyroPredictMatcher(lastFrame, curFrame, imuCalib, biasg, cv::Mat(),
                                                GyroAidedTracker::GYRO_PREDICT_WITH_OPTICAL_FLOW_REFINED_CONSIDER_ILLUMINATION_DEFORMATION,
                                                GyroAidedTracker::PIXEL_AWARE_PREDICTION,
                                                saveFolderPath, half_patch_size);

            double t_instant_construct = timer.runTime_s(); timer.freshTimer();

            n_predict = gyroPredictMatcher.TrackFeatures();
            setFrameWithoutGeometryValid(curFrame, gyroPredictMatcher); // save temporal states
            gyroPredictMatcher.GeometryValidation();
            gyroPredictMatcher.SetBackToFrame(curFrame);
            double t_track_features = timer.runTime_s(); timer.freshTimer();

            if(!loadDetectedKeypoints){ // Default: detcet new keypoint ORBextractorLeft
                curFrame.DetectKeyPoints(pORBextractorLeft);
                double t_detect_features = timer.runTime_s(); timer.freshTimer();

                int drawFlowType = 0;   // 0: draw flows on the current frame; 1: draw match line across two frames
                bool bDrawPatch = true, bDrawMistracks = true, bDrawGyroPredictPosition = true;
                curFrame.Display("Pixel-Aware Gyro-Aided KLT Feature Tracking", drawFlowType, bDrawPatch, bDrawMistracks, bDrawGyroPredictPosition);

            }else{// else load keypoints from file. Default: not execute
                double t_detect_features = timer.runTime_s(); timer.freshTimer();
                curFrame.SetPredictKeyPointsAndMask();

                int drawFlowType = 0;   // 0: draw flows on the current frame; 1: draw match line across two frames
                bool bDrawPatch = false, bDrawMistracks = true, bDrawGyroPredictPosition = false;
                curFrame.Display("Pixel-Aware Gyro-Aided KLT Feature Tracking", drawFlowType, bDrawPatch, bDrawMistracks, bDrawGyroPredictPosition);

                int idx = findTimeCorrespondenIndex(vpTimeCorrespondens, curFrame.mTimeStamp);
                std::string path = detectedKeypointsFile + "/" + vpTimeCorrespondens[idx].second + ".txt";
                curFrame.Reset();
                curFrame.LoadDetectedKeypointFromFile(path);
            }

            // test ORB feature detect and match for comparison
            if(test_orb_detect_and_desp_matcher)
            {
                ORBDetectAndDespMatcher matcher(lastFrame, curFrame, pORBextractorLeft, pORBextractorRight, saveFolderPath);
                matcher.FindFeatureMatches();
                matcher.PoseEstimation2d2d();
                matcher.Display();
            }
        }

        // update states
        {
            lastFrame = Frame(curFrame);
            time_prev = time_cur;
            vImuMeas.clear();
            data_valid = false;
        }

    } // End feature tracking
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

    cnt ++;
    if (cnt%downSampleRate == 0)
        image_buf.push(img_msg);
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "feature_tracker");
    ros::NodeHandle nh("~");
    ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);

    nh.getParam("rosBag", rosbag_file);
    nh.getParam("manuallyAddTimeDelay", MANUALLY_ADD_TIME_DELAY);
    std::string output_file;
    nh.getParam("output_file", output_file);

    std::cout << "rosbag_file: " << rosbag_file << std::endl;
    std::cout << "manuallyAddTimeDelay: " << MANUALLY_ADD_TIME_DELAY << std::endl;
    std::cout << "output_file: " << output_file << std::endl;

    cout << "running feature_tracker ..." << endl;

    if(argc != 2)
    {
        cerr << endl << "[Usage]: roslaunch ROS_Demo_Feature_Tracking EuRoC.launch" << endl;
        cerr << endl << "If you enable the 'readFromRosBag' flag in setting file, "
                        "then you should specify the 'rosBag' parameter in the launch file. "
                        "The software will automatically read messages from the bag file." << endl;
        cerr << endl << "Else, you should publish the messages through imuTopic and imageTopic." << endl;

        ros::shutdown();
        return 1;
    }

    loadConfigureFile(argv[1]);
    cout << "Tbc: " << imuCalib.Tbc << endl;

    // create folder for store the processing results
    char *path = getcwd(NULL, 0);
    saveFolderPath = path + output_file;
    std::cout << "saveFolderPath: " << saveFolderPath << std::endl;

    detectedKeypointsFile = path + detectedKeypointsFile;
    if(loadDetectedKeypoints){ // if Load keypoints from file. Default: not execute
        std::string path = detectedKeypointsFile + "/corresponds.txt";
        std::ifstream fin(path.c_str());
        if(!fin.is_open()){
            std::cout << "open file failed. file: " << path << std::endl;
        }else {
            std::string line;
            while(getline(fin, line)){
                std::string::size_type p_dot = line.find(",");
                std::string t1_str = line.substr(0, p_dot), t2_str = line.substr(p_dot+2, line.size()-p_dot);
                vpTimeCorrespondens.push_back(std::make_pair(std::atof(t1_str.c_str()), t2_str));
            }
        }
        fin.close();
    }

    // initial ORBextractor
    int nFeatures = keypoint_number;
    float fScaleFactor = 1.2;
    int nLevels = 1;    // 8
    int fIniThFAST = 20;
    int fMinThFAST = 7;
    pORBextractorLeft = new ORB_SLAM2::ORBextractor(nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);
    if(test_orb_detect_and_desp_matcher){
        pORBextractorRight = new ORB_SLAM2::ORBextractor(nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);
    }

    ros::Rate r(1000);    // 1000
    ros::Timer process_timer = nh.createTimer(ros::Duration(0.005), sensorProcessTimer);

    ////////////////////////////////////////////////////////////
    if(read_from_rosbag)
    {
        rosbag::Bag bag;
        std::cout << endl << "Reading meassage directly from rosbag file. rosbag_file is: " << rosbag_file << endl;
        bag.open(rosbag_file, rosbag::bagmode::Read);

        std::vector<std::string> topics;
        topics.push_back(imu_topic);
        topics.push_back(image_topic);

        LOG(INFO) << "rosbag::View ...";
        rosbag::View view(bag, rosbag::TopicQuery(topics));

        BOOST_FOREACH(rosbag::MessageInstance const m, view)
        {
            if(m.getTopic() == imu_topic || ("/"+m.getTopic() == imu_topic)){
                sensor_msgs::ImuConstPtr simu = m.instantiate<sensor_msgs::Imu>();
                if(simu != NULL) imuCallback(simu);
            }

            if(m.getTopic() == image_topic || ("/"+m.getTopic() == image_topic)){
                sensor_msgs::ImageConstPtr sImage = m.instantiate<sensor_msgs::Image>();
                if(sImage != NULL) imageCallback(sImage);

                // button control
                if(cv::waitKey(1) == 's')
                {
                    LOG(INFO) << "Enable step mode, please select the image show window "
                                 "and then press null space button to step-continue or press 'q' to return to normal mode.";
                    step_mode = true;
                }
                while (step_mode) {
                    int key = cv::waitKey(1);
                    if(key == 'q') {
                        step_mode = false;
                        LOG(INFO) << "Noraml mode ...";
                        break;
                    }else if(key == 32){ // Capture next frame.
                        break;
                    }else{
                        r.sleep();
                        if(!ros::ok())
                            break;
                    }
                }
            }

            ros::spinOnce();
            r.sleep();
            if(!ros::ok())
                break;
        }
        bag.close();
        return 0;
    }
    ////////////////////////////////////////////////////////////

    // subscribe messages from topics
    std::cout << endl << "Waiting to subscribe messages from topics ..." << endl;
    ros::Subscriber sub_imu = nh.subscribe(imu_topic, 2000, imuCallback);
    ros::Subscriber sub_img0 = nh.subscribe(image_topic, 200, imageCallback);

    ros::spin();
    return 0;
}


