#ifndef FRAME_H
#define FRAME_H

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include "../src/imu_types.h"

class Frame
{
public:
    Frame();
    Frame(const Frame& frame);
    Frame(double &t, cv::Mat &im, cv::Mat &im_dist, Frame* pLastFrame, CameraParams *pCameraParams,
          std::vector<IMU::Point> &vImu, int keypointNumber = 512, double th = 1.0);

    void DetectKeyPoints();

    // read features from file. the feature is detected by SuperPoint (Paper - "SuperPoint: Self-supervised interest point detection and description")
    void LoadDetectedKeypointFromFile(std::string path);

    void UndistortPoints(std::vector<cv::Point2f> &corners);

    //void Display(std::string winname);
    void Display(std::string winname, int drawFlowType, bool bDrawPatch, bool bDrawMistracks, bool bDrawGyroPredictPosition=true);

    void SetCurFrameWithoutGeometryValid(Frame* pframe) {curFrameWithoutGeometryValid = pframe;}

    void SetPredictKeyPointsAndMask();

    void Reset();

public:
    static long unsigned int nNextId;

    long unsigned int mnId;
    int mN;     // KeyPoints number
    double mThresholdOfPredictNewKeyPoint;
    double mTimeStamp;
    cv::Mat mGray;          // rectified
    cv::Mat mGrayDistort;   // original distorted image, just used for display
    Frame *mpLastFrame;
    Frame *curFrameWithoutGeometryValid;

//    cv::Mat mRbc;
    cv::Mat mRcl;
    CameraParams *mpCameraParams;
    float mfx, mfy, mcx, mcy, mfx_inv, mfy_inv;

    std::vector<IMU::Point> mvImuFromLastFrame;
    std::vector<cv::KeyPoint> mvKeys;
    std::vector<cv::KeyPoint> mvKeysUn;
    std::vector<cv::KeyPoint> mvKeysNormal;
    std::vector<cv::Point2f> mvFlowVelocityInNormalPlane;   // Note: the flow velocity is calculated only
                                                            // when the corresponding feature is successfully
                                                            // tracked in its next frame.
    cv::Mat mMask;
    std::vector<int> mvPtIndexInLastFrame;  // The index of the corresponding features in the reference frame.
                                            // Note: if >= 0, the ndex of the corresponding features
                                            //       else if < 0, the keypoints are new detected.

    std::vector<cv::Point2f> mvPtPredict;   // Pixels predicted from reference frame. (Distorted)
    std::vector<cv::Point2f> mvPtPredictUn; // Pixels predicted from reference frame. (Undistorted)
    std::vector<cv::Point2f> mvPtGyroPredictUn; // Pixels predicted from reference frame.
    std::vector<uchar> mvStatus;            // States. 1: valid predict; 0: unvalid predict
                                            // Note: the vector size is equal to the keypoint number of reference frame
    std::vector<float> mvNcc;

    std::vector<std::vector<cv::Point2f>> mvvFlowsPredictCorners;
};

#endif // FRAME_H
