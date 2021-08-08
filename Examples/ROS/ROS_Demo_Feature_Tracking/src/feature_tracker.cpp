/**
* This file is added by xiefei2929@126.com
*/


#include<iostream>
#include<algorithm>
#include<fstream>
#include<chrono>
#include <stdio.h>
#include <time.h>
//#include <queue>
//#include <map>
//#include <thread>
#include <mutex>
#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <message_filters/subscriber.h>

#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <std_msgs/Header.h>
#include <sensor_msgs/Imu.h>

#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <boost/foreach.hpp>

#include "../src/common.h"
#include "../src/frame.h"
#include "../src/gyro_predict_matcher.h"
#include "../src/ORBDetectAndDespMatcher.h"
#include "../src/ORBextractor.h"
double time_cur = 0;
double time_prev = 0;
bool data_valid = true;

cv::Mat image_cur, image_cur_distort;
vector<IMU::Point> vImuMeas;    // IMU measurements from previous image to current image
Frame lastLastFrame;
Frame lastFrame;
Frame curFrame;
Frame curFrameWithoutGeometryValid;  // used to display temporal tracked featrues


//vector<cv::Point2f> mvPtGyroPredict_tmp;

std::string saveFolderPath;

bool step_mode = false;

bool do_rectify = true;

bool test_orb_feature = false;
ORB_SLAM2::ORBextractor *pORBextractorLeft, *pORBextractorRight;

bool compare_with_SuperGlue = false;
std::vector<std::pair<double, std::string>> vpTimeCorrespondens;

int findTimeCorrespondenIndex(double t) // only used to compare with SuperGlue
{
    for(int i = 0; i < vpTimeCorrespondens.size(); i++)
    {
        auto p = vpTimeCorrespondens[i];
        if(std::abs(t - p.first) < 0.0001)
            return i;   // found
    }
    return -1;  // not found
}

void setFrameWithoutGeometryValid(Frame& frame, GyroPredictMatcher& matcher);

/**
 * @brief timeDelayEstimation
 * @param frame0 lastLastFrame.
 * @param frame1 lastFrame
 * Design for pure rotation.
 * TODO:
 *     trick-1: RANSAC for estimating time delay
 *     trick-2: use second-order of pixel moving acceleration
 *              for better descriping the pixel movement.
 */
double timeDelayEstimation(const Frame& frame0, const Frame& frame1, const Frame& frame2)
{
    cv::Mat R10_ = frame1.mRcl;
    float r11 = R10_.at<float>(0,0), r12 = R10_.at<float>(0,1), r13 = R10_.at<float>(0,2);
    float r21 = R10_.at<float>(1,0), r22 = R10_.at<float>(1,1), r23 = R10_.at<float>(1,2);
    float r31 = R10_.at<float>(2,0), r32 = R10_.at<float>(2,1), r33 = R10_.at<float>(2,2);

    bool bRANSAC = true;
    bool bHuber = true;
    double delta = 0.01; //0.045;
    bool display = false;
    int h, w;
    cv::Mat im_out;

    if(display){
        h = camera_params.height; w = camera_params.width;
        im_out = cv::Mat(h, 2 * w, CV_8UC1, cv::Scalar(0));
        frame0.mGray.copyTo(im_out.rowRange(0, h).colRange(0, w));
        frame1.mGray.copyTo(im_out.rowRange(0, h).colRange(w, 2*w));
        if(im_out.channels() < 3) //this should be always true
            cv::cvtColor(im_out, im_out, CV_GRAY2BGR);
    }

    std::vector<cv::Point2f> vKPi, vVi, vKPj, vVj;
    for(size_t k = 0, kend = frame2.mvPtIndexInLastFrame.size(); k < kend; k++){
        // Note that only the keypoints that be successfully tracked in at least three consecutive frames
        // can be used to calculate the time delay.
        if(frame2.mvPtIndexInLastFrame[k] > 0){
            int idx_j = frame2.mvPtIndexInLastFrame[k]; // keypoint index in frame1
            //cout << "idx_j: " << idx_j << endl;
            if(frame1.mvPtIndexInLastFrame[idx_j] > 0){
                int idx_i = frame1.mvPtIndexInLastFrame[idx_j]; // keypoint index in frame0
                //cout << "----idx_i: " << idx_i << endl;
                cv::Point2f kpi = frame0.mvKeysNormal[idx_i].pt;
                cv::Point2f vi = frame0.mvFlowVelocityInNormalPlane[idx_i];
                cv::Point2f kpj = frame1.mvKeysNormal[idx_j].pt;
                cv::Point2f vj = frame1.mvFlowVelocityInNormalPlane[idx_j];

                vKPi.push_back(kpi);
                vVi.push_back(vi);
                vKPj.push_back(kpj);
                vVj.push_back(vj);

                if(display){
                    cv::circle(im_out, frame0.mvKeys[idx_i].pt, 2, cv::Scalar(0, 255, 0), -1);
                    cv::circle(im_out, frame1.mvKeys[idx_j].pt + cv::Point2f(w, 0), 2, cv::Scalar(0, 255,0),-1);
                }
            }
        }
    }



    /// use the normalized keypoints and flow velocity to estimate the time delay.
    // method: Gauss-Newton Method
    double dt = 0;  // initial
    vector<double> vErrorNorm;
    double error_norm_max, error_norm_min, error_sum;
    if(vKPi.size() > 10){
        int Iterations = 10;
        double H = 0;
        double b = 0;
        double cost = 0, lastCost = 0;
        Eigen::Vector2d J;

        for(size_t iter = 0; iter < Iterations; iter++){
            H = 0; b = 0;
            vErrorNorm.clear(); vErrorNorm.reserve(vKPi.size());
            error_norm_max = 0; error_norm_min = 10000; error_sum = 0;
            for(size_t i = 0, iend = vKPi.size(); i < iend; i++){
                cv::Point2f vi = vVi[i];
                cv::Point2f vj = vVj[i];
                cv::Point2f kpi_dt = vKPi[i] + vi * dt;
                cv::Point2f kpj_dt = vKPj[i] + vj * dt;
                double tmp = r31 * kpi_dt.x + r32 * kpi_dt.y + r33;
                double e1 = kpj_dt.x * tmp - r11 * kpi_dt.x - r12 * kpi_dt.y - r13;
                double e2 = kpj_dt.y * tmp - r21 * kpi_dt.x - r22 * kpi_dt.y - r23;
                Eigen::Vector2d error(e1, e2);
                double error_norm = std::sqrt( error.transpose() * error);
                if(bHuber && error_norm > delta){
                    cost += delta * error_norm - 0.5 * delta * delta;
                }else {
                    cost += error.transpose() * error;
                }

                vErrorNorm.push_back(error_norm);
                error_norm_max = error_norm_max < error_norm? error_norm: error_norm_max;
                error_norm_min = error_norm_min > error_norm? error_norm: error_norm_min;
                error_sum += error_norm;

                // Je1/dtd, Je2/dtd
                double Je1_dt = vj.x * tmp + kpj_dt.x * (r31 * vi.x + r32 * vi.y) - r11 * vi.x - r12 * vi.y;
                double Je2_dt = vj.y * tmp + kpj_dt.y * (r31 * vi.x + r32 * vi.y) - r21 * vi.x - r22 * vi.y;
                J = Eigen::Vector2d(Je1_dt, Je2_dt);

                if(bHuber && error_norm > delta){
                    H += delta / (error_norm + 1e-6) * J.transpose() * J;
                    b += - delta / (error_norm + 1e-6) * J.transpose() * error;
                }else{
                    H += J.transpose() * J;
                    b += - J.transpose() * error;
                }
            }

            double update = b / (H + 1e-6);
            if(iter > 0 && cost > lastCost)
                break;

            // update dt
            dt += update;

            if(std::abs(update) < 0.001)
                break;  // converge
        }
    }

    if(display){
        cv::imshow("frame0 vs frame1", im_out);
        cv::waitKey(1);
    }

    { // debug
        LOG(INFO) << "error_norm max: " << error_norm_max << ", min: " << error_norm_min << ", avg: " << error_sum / vErrorNorm.size();
    }

    return dt;   // return estimated timeDelay
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
            && imu_buf.back()->header.stamp.toSec() > image_buf.front()->header.stamp.toSec() - MANUALLY_ADD_TIME_DELAY + estimated_time_delay)
    {
        {
            unique_lock<mutex> lock(mMutexImage);
            time_cur = image_buf.front()->header.stamp.toSec();

            image_cur_distort = getImageFromMsg(image_buf.front());
            image_buf.pop();

            if(do_rectify){
                // camera_params.ResetDistCoef();
                cv::remap(image_cur_distort, image_cur, camera_params.M1, camera_params.M2, cv::INTER_LINEAR);

                // save rectified images
                std::string::size_type pos = rosbag_file.rfind(".bag");
                std::string dir = rosbag_file.substr(0,pos) + "/cam0_rectify/data/";
                std::string command = "mkdir -p " + dir;
                int a = system(command.c_str());
                std::string path = dir + std::to_string(long(time_cur * 1e9)) + ".png";
                //LOG(INFO) << "path: " << path;
                cv::imwrite(path, image_cur);

                std::ofstream fp_image_file_list(rosbag_file.substr(0,pos)+"/image_file_list.txt", std::ofstream::app);
                if(!fp_image_file_list.is_open())
                    std::cout << "cannot open: " << rosbag_file.substr(0,pos)+"/image_file_list.txt" << std::endl;
                else {
                    fp_image_file_list << "/cam0_rectify/data/" + std::to_string(long(time_cur * 1e9)) << ".png" << std::endl;
                }
                // end save rectified images
            }
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
            while(!imu_buf.empty() && imu_buf.front()->header.stamp.toSec() < time_cur - MANUALLY_ADD_TIME_DELAY + estimated_time_delay)
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

                // save imu measurements to file
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

            }

            data_valid = true;
            if (vImuMeas.empty())
                data_valid = false;
        }

        // TODO: processing measurements
        if (data_valid)
        {
            curFrame = Frame(time_cur, image_cur, image_cur_distort, &lastFrame, &camera_params, vImuMeas, keypoint_number, threshold_of_predict_new_keypoint);
            if(lastFrame.mGray.empty()){    // if lastFrame is empty, then detect keypoints
                if(compare_with_SuperGlue){ // Load keypoint from SuperPoint
                    int idx = findTimeCorrespondenIndex(curFrame.mTimeStamp);
                    if (idx < 0){
                        LOG(ERROR) << "can not find correspondent features, shouldn't! curFrame.T: " << std::to_string(curFrame.mTimeStamp);
                        return;
                    }
                    else {
                        std::string::size_type p0 = rosbag_file.rfind("/");
                        std::string::size_type p1 = rosbag_file.find(".bag");
                        std::string seq = rosbag_file.substr(p0, p1-p0);
                        char *curDir = getcwd(NULL, 0);

                        std::string path = std::string(curDir) + "/../scripts/data/SuperGluePretrainedNetwork/detected_points/"
                                + seq + "/" + vpTimeCorrespondens[idx].second + ".txt";
                        LOG(INFO) << "path: " << path << ", curFrame.T: " << std::to_string(curFrame.mTimeStamp) << ", idx: " << idx;
                        curFrame.LoadKeypointFromSuperPoint(path);
                    }
                }else{
                    curFrame.DetectKeyPoints();   // Default
                }
            }

            int n_predict = 0;
            if(!lastFrame.mGray.empty())
            {
                Timer timer;
                cv::Point3f biasg(0,0,0);

                /*if(0)   // IMAGE_ONLY_OPTICAL_FLOW_CONSIDER_ILLUMINATION
                {
                    //LOG(INFO) << BLUE"--- IMAGE_ONLY_OPTICAL_FLOW_CONSIDER_ILLUMINATION: ";
                    timer.freshTimer();
                    Frame curFrameTmp = Frame(time_cur, image_cur, image_cur_distort, &lastFrame, &camera_params, vImuMeas, keypoint_number, threshold_of_predict_new_keypoint); //Frame(curFrame);
                    GyroPredictMatcher gyroPredictMatcherTmp(lastFrame, curFrameTmp, imuCalib, biasg, cv::Mat(),
                                                             GyroPredictMatcher::IMAGE_ONLY_OPTICAL_FLOW_CONSIDER_ILLUMINATION, saveFolderPath,
                                                             half_patch_size);

                    gyroPredictMatcherTmp.TrackFeatures(); setFrameWithoutGeometryValid(curFrameTmp, gyroPredictMatcherTmp);
                    gyroPredictMatcherTmp.GeometryValidation();
                    gyroPredictMatcherTmp.SetBackToFrame(curFrameTmp);
                    curFrameTmp.DetectKeyPoints();
                    curFrameTmp.Display("IMAGE_ONLY_OPTICAL_FLOW_CONSIDER_ILLUMINATION");
                }*/

//                if(2)   // for Ours - homography
//                {
//                    timer.freshTimer();
//                    Frame curFrameTmp = Frame(time_cur, image_cur, image_cur_distort, &lastFrame, &camera_params, vImuMeas, keypoint_number, threshold_of_predict_new_keypoint); //Frame(curFrame);
//                    GyroPredictMatcher gyroPredictMatcherTmp(lastFrame, curFrameTmp, imuCalib, biasg, cv::Mat(),
//                                                             GyroPredictMatcher::GYRO_PREDICT_WITH_OPTICAL_FLOW_REFINED_CONSIDER_ILLUMINATION_DEFORMATION, saveFolderPath,
//                                                             half_patch_size, 2);
//                    gyroPredictMatcherTmp.TrackFeatures(); setFrameWithoutGeometryValid(curFrameTmp, gyroPredictMatcherTmp);
//                    gyroPredictMatcherTmp.GeometryValidation();
//                    gyroPredictMatcherTmp.SetBackToFrame(curFrameTmp);

//                    curFrameTmp.DetectKeyPoints();
//                    curFrameTmp.Display("GA KLT - Ours (homography)");

//                    // temp, to be deleted
//                    /*mvPtGyroPredict_tmp = vector<cv::Point2f>(curFrameWithoutGeometryValid.mvPtGyroPredict.begin(), curFrameWithoutGeometryValid.mvPtGyroPredict.end());*/
//                }

                /*if(4)   // for Ours - not deformation
                {
                    timer.freshTimer();
                    Frame curFrameTmp = Frame(time_cur, image_cur, image_cur_distort, &lastFrame, &camera_params, vImuMeas, keypoint_number, threshold_of_predict_new_keypoint); //Frame(curFrame);
                    GyroPredictMatcher gyroPredictMatcherTmp(lastFrame, curFrameTmp, imuCalib, biasg, cv::Mat(),
                                                             GyroPredictMatcher::GYRO_PREDICT_WITH_OPTICAL_FLOW_REFINED_CONSIDER_ILLUMINATION, saveFolderPath,
                                                             half_patch_size, 1);
                    gyroPredictMatcherTmp.TrackFeatures(); setFrameWithoutGeometryValid(curFrameTmp, gyroPredictMatcherTmp);
                    gyroPredictMatcherTmp.GeometryValidation();
                    gyroPredictMatcherTmp.SetBackToFrame(curFrameTmp);

                    curFrameTmp.DetectKeyPoints();
                    curFrameTmp.Display("GA KLT - Ours (not deformation)");

                }

                if(2)   // for Ours - not deformation, not illumination
                {
                    timer.freshTimer();
                    Frame curFrameTmp = Frame(time_cur, image_cur, image_cur_distort, &lastFrame, &camera_params, vImuMeas, keypoint_number, threshold_of_predict_new_keypoint); //Frame(curFrame);
                    GyroPredictMatcher gyroPredictMatcherTmp(lastFrame, curFrameTmp, imuCalib, biasg, cv::Mat(),
                                                             GyroPredictMatcher::GYRO_PREDICT_WITH_OPTICAL_FLOW_REFINED, saveFolderPath,
                                                             half_patch_size, 1);
                    gyroPredictMatcherTmp.TrackFeatures(); setFrameWithoutGeometryValid(curFrameTmp, gyroPredictMatcherTmp);
                    gyroPredictMatcherTmp.GeometryValidation();
                    gyroPredictMatcherTmp.SetBackToFrame(curFrameTmp);

                    curFrameTmp.DetectKeyPoints();
                    curFrameTmp.Display("GA KLT - Ours (not deformation, not illumination)");
                }
                */

                /////////////////////////////////////////////////
                /* // for patch_size_comparison
                if(2)
                {
                    timer.freshTimer();
                    Frame curFrameTmp = Frame(time_cur, image_cur, image_cur_distort, &lastFrame, &camera_params, vImuMeas, keypoint_number, threshold_of_predict_new_keypoint); //Frame(curFrame);
                    GyroPredictMatcher gyroPredictMatcherTmp(lastFrame, curFrameTmp, imuCalib, biasg, cv::Mat(),
                                                             GyroPredictMatcher::GYRO_PREDICT_WITH_OPTICAL_FLOW_REFINED_CONSIDER_ILLUMINATION_DEFORMATION, saveFolderPath+"/HalfPatchSize_0.5/",
                                                             half_patch_size*0.5, 2);
                    gyroPredictMatcherTmp.TrackFeatures(); setFrameWithoutGeometryValid(curFrameTmp, gyroPredictMatcherTmp);
                    gyroPredictMatcherTmp.GeometryValidation();
                    gyroPredictMatcherTmp.SetBackToFrame(curFrameTmp);

                    curFrameTmp.DetectKeyPoints();
                    curFrameTmp.Display("GA KLT - Ours (homography, 0.5*half_patch_size)");
                }

                if(2)
                {
                    timer.freshTimer();
                    Frame curFrameTmp = Frame(time_cur, image_cur, image_cur_distort, &lastFrame, &camera_params, vImuMeas, keypoint_number, threshold_of_predict_new_keypoint); //Frame(curFrame);
                    GyroPredictMatcher gyroPredictMatcherTmp(lastFrame, curFrameTmp, imuCalib, biasg, cv::Mat(),
                                                             GyroPredictMatcher::GYRO_PREDICT_WITH_OPTICAL_FLOW_REFINED_CONSIDER_ILLUMINATION_DEFORMATION, saveFolderPath+"/HalfPatchSize_1.5/",
                                                             half_patch_size*1.5, 2);
                    gyroPredictMatcherTmp.TrackFeatures(); setFrameWithoutGeometryValid(curFrameTmp, gyroPredictMatcherTmp);
                    gyroPredictMatcherTmp.GeometryValidation();
                    gyroPredictMatcherTmp.SetBackToFrame(curFrameTmp);

                    curFrameTmp.DetectKeyPoints();
                    curFrameTmp.Display("GA KLT - Ours (homography, 1.5*half_patch_size)");
                }
                if(2)
                {
                    timer.freshTimer();
                    Frame curFrameTmp = Frame(time_cur, image_cur, image_cur_distort, &lastFrame, &camera_params, vImuMeas, keypoint_number, threshold_of_predict_new_keypoint); //Frame(curFrame);
                    GyroPredictMatcher gyroPredictMatcherTmp(lastFrame, curFrameTmp, imuCalib, biasg, cv::Mat(),
                                                             GyroPredictMatcher::GYRO_PREDICT_WITH_OPTICAL_FLOW_REFINED_CONSIDER_ILLUMINATION_DEFORMATION, saveFolderPath+"/HalfPatchSize_2.0/",
                                                             half_patch_size*2.0, 2);
                    gyroPredictMatcherTmp.TrackFeatures(); setFrameWithoutGeometryValid(curFrameTmp, gyroPredictMatcherTmp);
                    gyroPredictMatcherTmp.GeometryValidation();
                    gyroPredictMatcherTmp.SetBackToFrame(curFrameTmp);

                    curFrameTmp.DetectKeyPoints();
                    curFrameTmp.Display("GA KLT - Ours (homography, 2.0*half_patch_size)");
                }
                if(2)
                {
                    timer.freshTimer();
                    Frame curFrameTmp = Frame(time_cur, image_cur, image_cur_distort, &lastFrame, &camera_params, vImuMeas, keypoint_number, threshold_of_predict_new_keypoint); //Frame(curFrame);
                    GyroPredictMatcher gyroPredictMatcherTmp(lastFrame, curFrameTmp, imuCalib, biasg, cv::Mat(),
                                                             GyroPredictMatcher::GYRO_PREDICT_WITH_OPTICAL_FLOW_REFINED_CONSIDER_ILLUMINATION_DEFORMATION, saveFolderPath+"/HalfPatchSize_2.5/",
                                                             half_patch_size*2.5, 2);
                    gyroPredictMatcherTmp.TrackFeatures(); setFrameWithoutGeometryValid(curFrameTmp, gyroPredictMatcherTmp);
                    gyroPredictMatcherTmp.GeometryValidation();
                    gyroPredictMatcherTmp.SetBackToFrame(curFrameTmp);

                    curFrameTmp.DetectKeyPoints();
                    curFrameTmp.Display("GA KLT - Ours (homography, 2.5*half_patch_size)");
                }*/

                /////////////////////////////////////////////////

                if(0)   // for OPENCV_OPTICAL_FLOW_PYR_LK
                {
                    //LOG(INFO) << BLUE"--- OPENCV_OPTICAL_FLOW_PYR_LK: ";
                    timer.freshTimer();
                    //Frame curFrameTmp = Frame(time_cur, image_cur, image_cur_distort, &lastFrame, &camera_params, vImuMeas, keypoint_number, threshold_of_predict_new_keypoint); //Frame(curFrame);
                    Frame curFrameTmp(curFrame);
                    GyroPredictMatcher gyroPredictMatcherTmp(lastFrame, curFrameTmp, imuCalib, biasg, cv::Mat(),
                                                             GyroPredictMatcher::OPENCV_OPTICAL_FLOW_PYR_LK, saveFolderPath,
                                                             half_patch_size);
                    gyroPredictMatcherTmp.TrackFeatures();
                    setFrameWithoutGeometryValid(curFrameTmp, gyroPredictMatcherTmp);
                    gyroPredictMatcherTmp.GeometryValidation();
                    gyroPredictMatcherTmp.SetBackToFrame(curFrameTmp);

                    if(compare_with_SuperGlue){
                        curFrameTmp.SetPredictKeyPointsAndMask();
                        int drawFlowType = 1;   // 0: draw flows on the current frame; 1: draw match line across two frames
                        bool bDrawPatch = false, bDrawMistracks = true, bDrawGyroPredictPosition = false;
                        curFrameTmp.Display("Image-only KLT", drawFlowType, bDrawPatch, bDrawMistracks, bDrawGyroPredictPosition);
                    }else {
                        curFrameTmp.DetectKeyPoints();
                        int drawFlowType = 0;   // 0: draw flows on the current frame; 1: draw match line across two frames
                        bool bDrawPatch = true, bDrawMistracks = true, bDrawGyroPredictPosition = true;
                        curFrameTmp.Display("Image-only KLT", drawFlowType, bDrawPatch, bDrawMistracks, bDrawGyroPredictPosition);
                    }
                }


                /////////////////////////////////////////////////
                /*// for patch_size_comparison
                if(4)   // for OPENCV_OPTICAL_FLOW_PYR_LK
                {
                    //LOG(INFO) << BLUE"--- OPENCV_OPTICAL_FLOW_PYR_LK: ";
                    timer.freshTimer();
                    Frame curFrameTmp = Frame(time_cur, image_cur, image_cur_distort, &lastFrame, &camera_params, vImuMeas, keypoint_number, threshold_of_predict_new_keypoint); //Frame(curFrame);
                    GyroPredictMatcher gyroPredictMatcherTmp(lastFrame, curFrameTmp, imuCalib, biasg, cv::Mat(),
                                                             GyroPredictMatcher::OPENCV_OPTICAL_FLOW_PYR_LK, saveFolderPath+"/HalfPatchSize_0.5/",
                                                             half_patch_size * 0.5);

                    gyroPredictMatcherTmp.TrackFeatures(); setFrameWithoutGeometryValid(curFrameTmp, gyroPredictMatcherTmp);
                    gyroPredictMatcherTmp.GeometryValidation();
                    gyroPredictMatcherTmp.SetBackToFrame(curFrameTmp);

                    curFrameTmp.DetectKeyPoints();
                    curFrameTmp.Display("Image-only KLT (0.5*half_patch_size)");  // OPENCV_OPTICAL_FLOW_PYR_LK
                }

                if(4)   // for OPENCV_OPTICAL_FLOW_PYR_LK
                {
                    //LOG(INFO) << BLUE"--- OPENCV_OPTICAL_FLOW_PYR_LK: ";
                    timer.freshTimer();
                    Frame curFrameTmp = Frame(time_cur, image_cur, image_cur_distort, &lastFrame, &camera_params, vImuMeas, keypoint_number, threshold_of_predict_new_keypoint); //Frame(curFrame);
                    GyroPredictMatcher gyroPredictMatcherTmp(lastFrame, curFrameTmp, imuCalib, biasg, cv::Mat(),
                                                             GyroPredictMatcher::OPENCV_OPTICAL_FLOW_PYR_LK, saveFolderPath+"/HalfPatchSize_1.5/",
                                                             half_patch_size * 1.5);

                    gyroPredictMatcherTmp.TrackFeatures(); setFrameWithoutGeometryValid(curFrameTmp, gyroPredictMatcherTmp);
                    gyroPredictMatcherTmp.GeometryValidation();
                    gyroPredictMatcherTmp.SetBackToFrame(curFrameTmp);

                    curFrameTmp.DetectKeyPoints();
                    curFrameTmp.Display("Image-only KLT (1.5*half_patch_size)");  // OPENCV_OPTICAL_FLOW_PYR_LK
                }

                if(4)   // for OPENCV_OPTICAL_FLOW_PYR_LK
                {
                    //LOG(INFO) << BLUE"--- OPENCV_OPTICAL_FLOW_PYR_LK: ";
                    timer.freshTimer();
                    Frame curFrameTmp = Frame(time_cur, image_cur, image_cur_distort, &lastFrame, &camera_params, vImuMeas, keypoint_number, threshold_of_predict_new_keypoint); //Frame(curFrame);
                    GyroPredictMatcher gyroPredictMatcherTmp(lastFrame, curFrameTmp, imuCalib, biasg, cv::Mat(),
                                                             GyroPredictMatcher::OPENCV_OPTICAL_FLOW_PYR_LK, saveFolderPath+"/HalfPatchSize_2.0/",
                                                             half_patch_size * 2.0);

                    gyroPredictMatcherTmp.TrackFeatures(); setFrameWithoutGeometryValid(curFrameTmp, gyroPredictMatcherTmp);
                    gyroPredictMatcherTmp.GeometryValidation();
                    gyroPredictMatcherTmp.SetBackToFrame(curFrameTmp);

                    curFrameTmp.DetectKeyPoints();
                    curFrameTmp.Display("Image-only KLT (2.0*half_patch_size)");  // OPENCV_OPTICAL_FLOW_PYR_LK
                }

                if(4)   // for OPENCV_OPTICAL_FLOW_PYR_LK
                {
                    //LOG(INFO) << BLUE"--- OPENCV_OPTICAL_FLOW_PYR_LK: ";
                    timer.freshTimer();
                    Frame curFrameTmp = Frame(time_cur, image_cur, image_cur_distort, &lastFrame, &camera_params, vImuMeas, keypoint_number, threshold_of_predict_new_keypoint); //Frame(curFrame);
                    GyroPredictMatcher gyroPredictMatcherTmp(lastFrame, curFrameTmp, imuCalib, biasg, cv::Mat(),
                                                             GyroPredictMatcher::OPENCV_OPTICAL_FLOW_PYR_LK, saveFolderPath+"/HalfPatchSize_2.5/",
                                                             half_patch_size * 2.5);

                    gyroPredictMatcherTmp.TrackFeatures(); setFrameWithoutGeometryValid(curFrameTmp, gyroPredictMatcherTmp);
                    gyroPredictMatcherTmp.GeometryValidation();
                    gyroPredictMatcherTmp.SetBackToFrame(curFrameTmp);

                    curFrameTmp.DetectKeyPoints();
                    curFrameTmp.Display("Image-only KLT (2.5*half_patch_size)");  // OPENCV_OPTICAL_FLOW_PYR_LK
                }
                */
                /////////////////////////////////////////////////

                /// ours
                GyroPredictMatcher gyroPredictMatcher(lastFrame, curFrame, imuCalib, biasg, cv::Mat(),
                                                      GyroPredictMatcher::GYRO_PREDICT_WITH_OPTICAL_FLOW_REFINED_CONSIDER_ILLUMINATION_DEFORMATION,
                                                      saveFolderPath, half_patch_size, 1);

                double t_instant_construct = timer.runTime_s(); timer.freshTimer();

                n_predict = gyroPredictMatcher.TrackFeatures(); setFrameWithoutGeometryValid(curFrame, gyroPredictMatcher);
                gyroPredictMatcher.GeometryValidation();
                gyroPredictMatcher.SetBackToFrame(curFrame);
                double t_track_features = timer.runTime_s(); timer.freshTimer();

                if(compare_with_SuperGlue){ /// temporal, for compare with SuperGLue

                    double t_detect_features = timer.runTime_s(); timer.freshTimer();
                    curFrame.SetPredictKeyPointsAndMask();
                    int drawFlowType = 1;   // 0: draw flows on the current frame; 1: draw match line across two frames
                    bool bDrawPatch = false, bDrawMistracks = true, bDrawGyroPredictPosition = false;
                    curFrame.Display("GA KLT - Ours", drawFlowType, bDrawPatch, bDrawMistracks, bDrawGyroPredictPosition);

                    std::string::size_type p0 = rosbag_file.rfind("/");
                    std::string::size_type p1 = rosbag_file.find(".bag");
                    std::string seq = rosbag_file.substr(p0, p1-p0);
                    char *curDir = getcwd(NULL, 0);

                    int idx = findTimeCorrespondenIndex(curFrame.mTimeStamp);
                    std::string path = std::string(curDir) + "/../scripts/data/SuperGluePretrainedNetwork/detected_points/"
                            + seq + "/" + vpTimeCorrespondens[idx].second + ".txt";
                    curFrame.Reset();
                    curFrame.LoadKeypointFromSuperPoint(path);
                    /// end temporal
                }else{/// Default
                    curFrame.DetectKeyPoints();   // Default
                    double t_detect_features = timer.runTime_s(); timer.freshTimer();
                    int drawFlowType = 0;   // 0: draw flows on the current frame; 1: draw match line across two frames
                    bool bDrawPatch = true, bDrawMistracks = true, bDrawGyroPredictPosition = true;
                    curFrame.Display("Gyro-Aided KLT Feature Tracking", drawFlowType, bDrawPatch, bDrawMistracks, bDrawGyroPredictPosition);
                }



                /// TODO: estimate time offset
                if(bEstimateTD && lastLastFrame.mnId > 0){
                    double dTimeDelay = timeDelayEstimation(lastLastFrame, lastFrame, curFrame);
                    // estimated_time_delay += dTimeDelay;

                    LOG(INFO) << "TimeStamp: " << std::to_string(curFrame.mTimeStamp)  << ", dt: " << dTimeDelay
                              << ", estimated_time_delay: " << estimated_time_delay;

                }

                /*// tmp, to be deleted
                int cnt = 0; double sum = 0;
                int cnt2 = 0; double sum2 = 0;
                for(size_t i = 0, iend = mvPtGyroPredict_tmp.size(); i < iend; i++){
                    cv::Point2f pt_gyro_tmp = mvPtGyroPredict_tmp[i];
                    if(pt_gyro_tmp.x != 0 && pt_gyro_tmp.y != 0){
                        cv::Point2f pt_gt = curFrame.mvPtPredict[i];
                        cv::Point2f ept = pt_gyro_tmp - pt_gt;
                        sum += std::sqrt(ept.x * ept.x + ept.y * ept.y);
                        cnt ++;
                    }

                    cv::Point2f pt_gyro_cur = curFrame.mvPtGyroPredict[i];
                    if(pt_gyro_cur.x != 0 && pt_gyro_cur.y != 0){
                        cv::Point2f pt_gt = curFrame.mvPtPredict[i];
                        cv::Point2f ept = pt_gyro_cur - pt_gt;
                        sum2 += std::sqrt(ept.x * ept.x + ept.y * ept.y);
                        cnt2 ++;
                    }

                }
                LOG(INFO) << "T: " << std::to_string(curFrame.mTimeStamp) << ", (homography) Gyro Pred. Err. : " << sum / cnt << ", proposed: " << sum2 / cnt2;
                // end tmp
                */

                // test ORB feature detect and match for comparison
                if(test_orb_feature)
                {
                    ORBDetectAndDespMatcher matcher(lastFrame, curFrame, pORBextractorLeft, pORBextractorRight, saveFolderPath);
                    matcher.FindFeatureMatches();
                    matcher.PoseEstimation2d2d();
                    matcher.Display();
                }

            }

//            LOG(INFO) << "frame dt: " << curFrame.mTimeStamp - lastFrame.mTimeStamp << ", IMU dt: " << vImuMeas.back().t - vImuMeas.front().t
//                      << ", curFrame.t: " << std::to_string(curFrame.mTimeStamp) << ", imu back.t: " << std::to_string( vImuMeas.back().t)
//                         << ", imu front.t: " << std::to_string( vImuMeas.front().t)
//                      << ", dt: " << curFrame.mTimeStamp - vImuMeas.back().t;



            // update states
            {
                lastLastFrame = Frame(lastFrame);
                lastFrame = Frame(curFrame);
                time_prev = time_cur;
                vImuMeas.clear();
            }
        }


    }
}

void setFrameWithoutGeometryValid(Frame& frame, GyroPredictMatcher& matcher)
{
    curFrameWithoutGeometryValid = Frame(frame);
    matcher.SetBackToFrame(curFrameWithoutGeometryValid);
    frame.SetCurFrameWithoutGeometryValid(&curFrameWithoutGeometryValid);
}


void visualizationTimer(const ros::TimerEvent& event)
{
//    int key = cv::waitKey(1);
//    if(key == 's'){
//        step_mode = true;
//        LOG(INFO) << "Enable step mode, please press null space button to step-continue, press 'q' to normal mode.";
//    }
//    else if(key == 'q'){
//        step_mode = false;
//        LOG(INFO) << "Normal mode.";
//    }
//    else if(key == 32)  // null space key
//    {
//        //LOG(INFO) << "Next iteration";
//    }

}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "gyro_predict_and_feature_detect_node");
    ros::NodeHandle nh("~");
    ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);

    nh.getParam("rosBag", rosbag_file);
    nh.getParam("manuallyAddTimeDelay", MANUALLY_ADD_TIME_DELAY);
    nh.getParam("output_file", output_file);

    std::cout << "rosbag_file: " << rosbag_file << std::endl;
    std::cout << "manuallyAddTimeDelay: " << MANUALLY_ADD_TIME_DELAY << std::endl;
    std::cout << "output_file: " << output_file << std::endl;

    cout << "running gyro_predict_and_feature_detect_node ..." << endl;

    for (int i = 0; i < argc; i++) {
        std::cout << "i: " << i << ", argv[i]: " << argv[i] << std::endl;
    }

    if(argc != 2)
    {
        cerr << endl << "[Usage]: rosrun gyro_predict_matcher gyro_predict_and_feature_detect_node path_to_settings" << endl;
        cerr << endl << "If you set 'readFromRosBag' flag in setting file, then set the bag path to 'rosBag'. The software will automatically read messages in the bag file." << endl;
        cerr << endl << "Else, you should publish the messages through imuTopic and imageTopic." << endl;

        ros::shutdown();
        return 1;
    }

    loadConfigureFile(argv[1]);
    cout << "Tbc: " << imuCalib.Tbc << endl;

    // create folder for store the processing results
    const time_t t = time(NULL);
    struct tm* current_time = localtime(&t);
    std::stringstream s;
    s << current_time->tm_year + 1900 << "-" << current_time->tm_mon + 1 << "-" << current_time->tm_mday << "-"
      << current_time->tm_hour << ":" << current_time->tm_min << ":" << current_time->tm_sec;
    char *path = getcwd(NULL, 0);
    saveFolderPath = path + output_file;
    std::cout << "saveFolderPath: " << saveFolderPath << std::endl;

    ros::Rate r(1000);    // 1000
    ros::Timer process_timer = nh.createTimer(ros::Duration(0.005), sensorProcessTimer);
//    ros::Timer visualization_timer = nh.createTimer(ros::Duration(0.01), visualizationTimer);


    // initial ORBextractor
    if(test_orb_feature){
        int nFeatures = keypoint_number;
        float fScaleFactor = 1.2;
        int nLevels = 8;
        int fIniThFAST = 20;
        int fMinThFAST = 7;
        pORBextractorLeft = new ORB_SLAM2::ORBextractor(nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);
        pORBextractorRight = new ORB_SLAM2::ORBextractor(nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);
    }

    // Load time correspondences
    if(compare_with_SuperGlue){
        std::string::size_type p0 = rosbag_file.rfind("/");
        std::string::size_type p1 = rosbag_file.find(".bag");
        std::string seq = rosbag_file.substr(p0, p1-p0);
        char *curDir = getcwd(NULL, 0);
        std::string path = std::string(curDir) + "/../scripts/data/SuperGluePretrainedNetwork/detected_points/"
                + seq + "/corresponds.txt";

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

    ////////////////////////////////////////////////////////////
    if(read_from_rosbag)
    {
        // test: read from rosbag
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
                                 "and then press null space button to step-continue or press 'q' button to normal mode.";
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

    std::cout << endl << "Waiting to subscribe messages from topics ..." << endl;
    ros::Subscriber sub_imu = nh.subscribe(imu_topic, 2000, imuCallback);
    ros::Subscriber sub_img0 = nh.subscribe(image_topic, 200, imageCallback);

    ros::spin();
    return 0;
}


