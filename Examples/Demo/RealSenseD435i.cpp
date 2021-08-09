/**
* This file is added by xiefei2929@126.com
*/

//#include <ros/ros.h>
//#include <rosbag/bag.h>
//#include <rosbag/view.h>
//#include <cv_bridge/cv_bridge.h>
//#include <message_filters/subscriber.h>
//#include <std_msgs/Header.h>
//#include <sensor_msgs/Imu.h>
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

//#include "Thirdparty/glog/include/glog/logging.h"
#include "glog/logging.h"

#include "frame.h"
#include "gyro_aided_tracker.h"
#include "ORBDetectAndDespMatcher.h"
#include "ORBextractor.h"

#include "common.h"

float MANUALLY_ADD_TIME_DELAY = 0;

double time_cur = 0;
double time_prev = 0;

cv::Mat image_cur, image_cur_distort;
Frame curFrame, lastFrame;
Frame curFrameWithoutGeometryValid;  // used to display temporal tracked featrues
vector<IMU::Point> vImuMeas;        // IMU measurements from previous image to current image

std::string saveFolderPath;

bool step_mode = false;
bool do_rectify = false;    // D435i sequence has been rectified

string dataset;
string datasetDir;

bool test_orb_detect_and_desp_matcher = false;
ORB_SLAM2::ORBextractor *pORBextractorLeft, *pORBextractorRight;

std::vector<std::pair<double, std::string>> vpTimeCorrespondens;


void setFrameWithoutGeometryValid(Frame& frame, GyroAidedTracker& matcher)
{
    curFrameWithoutGeometryValid = Frame(frame);
    matcher.SetBackToFrame(curFrameWithoutGeometryValid);
    frame.SetCurFrameWithoutGeometryValid(&curFrameWithoutGeometryValid);
}


bool getNextFrame()
{
    static string ImageCsvFile;
    static ifstream fin_image;
    static bool b_open_image = false;
    string strImage;
    string line;
    if(dataset == "PKUSZ_RealSenseD435i_sequence"){
        if(!b_open_image){
            ImageCsvFile = datasetDir + "/image_file_list.txt";
            fin_image.open(ImageCsvFile, ios::in);
            b_open_image = true;
        }

        if(getline(fin_image, line)){
            strImage = datasetDir + line;
            image_cur_distort = cv::imread(strImage);
            std::string::size_type pos1 = line.rfind("/"), pos2 = line.rfind(".png");
            time_cur = std::stol(line.substr(pos1+1, pos2-pos1-1)) * 1e-9;
            return true;
        }

    }

    return false;
}

//bool getIMUMeasurements(double t1, double t2, vector<IMU::Point> &vImus)
bool getNextIMU(IMU::Point &imu)
{
    static string IMUCsvFile;
    static ifstream fin_imu;
    static bool b_open_imu = false;
    string line;

    if(dataset == "PKUSZ_RealSenseD435i_sequence"){
        if(!b_open_imu){
            IMUCsvFile = datasetDir + "/imu.txt";
            fin_imu.open(IMUCsvFile);
            b_open_imu = true;
        }

        int id_ts = 0;
        int id_ax = 1, id_ay = 2, id_az = 3;
        int id_wx = 4, id_wy = 5, id_wz = 6;

        if(getline(fin_imu, line)){
            istringstream sin(line);
            double t, a[3], w[3];
            string str_time;
            sin >> str_time >> a[0] >> a[1] >> a[2] >> w[0] >> w[1] >> w[2];
            t = std::stol(str_time) * 1e-9;
            //printf("t1: %s, a: [%f %f %f], b: [%f %f %f]\n", std::to_string(t).c_str(), a[0], a[1], a[2], w[0], w[1], w[2]);
            imu.a.x = a[0]; imu.a.y = a[1]; imu.a.z = a[2];
            imu.w.x = w[0]; imu.w.y = w[1]; imu.w.z = w[2];
            imu.t = t;
            return true;
        }else {
            return false;
        }

    }
    else {
        return false;
    }


    return false;
}

int main(int argc, char **argv)
{
    std::cout << "Demo RealSenseD435i" << std::endl;

    if(argc != 2) {
        cerr << endl << "[Usage]: ./RealSenseD435i RealSenseD435i.yaml" << endl;
        return 1;
    }

    loadConfigureFile(argv[1]);
    cout << "Tbc: " << imuCalib.Tbc << endl;

    cv::FileStorage fSettings(argv[1], cv::FileStorage::READ);
    dataset = string(fSettings["dataset"]);
    datasetDir = string(fSettings["datasetDir"]);
    string output_file = string(fSettings["outputFile"]);
    char *path = getcwd(NULL, 0);
    saveFolderPath = path + output_file;

    cout << "dataset: " << dataset << endl;
    cout << "datasetDir: " << datasetDir << endl;
    cout << "output_file: " << output_file << endl;
    std::cout << "saveFolderPath: " << saveFolderPath << std::endl;

    detectedKeypointsFile = path + detectedKeypointsFile;
    if(loadDetectedKeypoints){ // if Load keypoints from file. Default: not execute
        std::string path = detectedKeypointsFile + "/corresponds.txt";
        std::ifstream fin(path.c_str());
        if(!fin.is_open()){
            LOG(ERROR) << RED"open file failed. file: " << path << RESET;
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

    // Initial ORBextractor
    int nFeatures = keypoint_number;
    float fScaleFactor = 1.2;
    int nLevels = 1;    // 8
    int fIniThFAST = 20;
    int fMinThFAST = 7;
    pORBextractorLeft = new ORB_SLAM2::ORBextractor(nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);
    if(test_orb_detect_and_desp_matcher){
        pORBextractorRight = new ORB_SLAM2::ORBextractor(nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);
    }

    // Load and process sequence
    IMU::Point last_imu; getNextIMU(last_imu);
    bool valid_imu = true;
    Timer timer;
    while(getNextFrame()){
        timer.freshTimer();
        if(do_rectify)  // Default: not execute since D435i sequence provided in \data\ folder has been rectified
            cv::remap(image_cur_distort, image_cur, pCameraParams->M1, pCameraParams->M2, cv::INTER_LINEAR);
        else {
            image_cur = image_cur_distort.clone();
        }

        // Load IMU measurements
        if(time_prev != 0){
            while(last_imu.t < time_prev - MANUALLY_ADD_TIME_DELAY && getNextIMU(last_imu))
                continue;

            vImuMeas.clear();
            while(last_imu.t < time_cur - MANUALLY_ADD_TIME_DELAY && valid_imu){
                vImuMeas.push_back(last_imu);
                valid_imu = getNextIMU(last_imu);
            }
        }

        // Feature tracking
        curFrame = Frame(time_cur, image_cur, image_cur_distort, &lastFrame, pCameraParams, pORBextractorLeft, vImuMeas, keypoint_number, threshold_of_predict_new_keypoint);
        if(lastFrame.mGray.empty()){    // if lastFrame is empty, then detect keypoints
            if(!loadDetectedKeypoints){ // Default: detect keypoints using ORBextractor
                curFrame.DetectKeyPoints(pORBextractorLeft);
            }else{  // else: Load keypoints from file. Default: not execute
                int idx = findTimeCorrespondenIndex(vpTimeCorrespondens, curFrame.mTimeStamp);
                if (idx < 0){
                    LOG(ERROR) << "Can not find detected keypoints !!! Please check the setting of parameter - 'DetectedKeypointsFile'. curFrame.T: " << std::to_string(curFrame.mTimeStamp);
                    continue;
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
        // end Feature tracking

        // update states
        {
            lastFrame = Frame(curFrame);
            time_prev = time_cur;
            vImuMeas.clear();
        }

        // sleep
        double t_total_us = timer.runTime_us();
        double t_sleep = pCameraParams->dt * 1e6 - t_total_us;
        if(t_sleep > 0)
            usleep(t_sleep);

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
            }
        }

    }

    return 0;
}


