#ifndef FEATURE_DETECT_MATCHER_H
#define FEATURE_DETECT_MATCHER_H

#include <thread>
#include <cstdlib>

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include "../Thirdparty/glog/include/glog/logging.h"
#include "frame.h"

#include "ORBextractor.h"

using namespace std;
//using namespace cv;

class ORBDetectAndDespMatcher{
public:
    ORBDetectAndDespMatcher(const Frame& pFrameRef, const Frame& pFrameCur,
                            ORB_SLAM2::ORBextractor* pORBextractorLeft,
                            ORB_SLAM2::ORBextractor* pORBextractorRight,
                            std::string saveFolderPath);

    void ExtractORB(int flag, const cv::Mat& im);
    void FindFeatureMatches();

    // transformation from figure 1 to figure 2
    void PoseEstimation2d2d();
    void Display();

    void SaveMsgToFile(std::string filename, std::string &msg);

private:
    cv::Point2d Pixel2Cam(const cv::Point2d& p){
        // x= (px -cx)/fx, y = (py-cy) / fy
        return cv::Point2d((p.x - mcx) * mfx_inv, (p.y - mcy) * mfy_inv);
    }


protected:
    double mTimeStamp;
    double mTimeStampRef;
    const cv::Mat &mImgGrayRef;
    const cv::Mat &mImgGrayCur;

    //GeometricCamera* mpCamera;
    cv::Mat mK;         // The camera intrinsic parameters
    cv::Mat mKRKinv;
    float mfx, mfy, mcx, mcy, mfx_inv, mfy_inv;

    cv::Mat mDistCoef;  // The camera distortion coeffections
    float mk1, mk2, mp1, mp2, mk3;

    std::vector<cv::KeyPoint> mvKeyPointsCur;
    std::vector<cv::KeyPoint> mvKeyPointsRef;
    cv::Mat mDescriptorCur, mDescriptorRef;
    std::vector<cv::DMatch> mvDMatches;
    std::vector<cv::DMatch> mvDMatchesByDescriptorMatch;

    cv::Mat mR, mt;
    cv::Mat mF, mE, mH;

    ORB_SLAM2::ORBextractor* mpORBextractorLeft;
    ORB_SLAM2::ORBextractor* mpORBextractorRight;

    std::string mSaveFolderPath;
};



#endif // FEATURE_DETECT_MATCHER_H
