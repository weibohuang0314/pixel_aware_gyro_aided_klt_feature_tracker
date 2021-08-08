
#ifndef GYROPREDICTMATCHER_H
#define GYROPREDICTMATCHER_H

#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
//#include <Eigen/Core>
#include <Eigen/Dense>
#include <iostream>
#include <string>
#include <chrono>
#include <stdio.h>
#include <sys/time.h>

//#include "CameraModels/GeometricCamera.h"
//#include "CameraModels/Pinhole.h"
//#include "CameraModels/KannalaBrandt8.h"

#include "../Thirdparty/glog/include/glog/logging.h"

#include "utils.h"
#include "imu_types.h"
#include "frame.h"

using namespace std;
using namespace cv;


class GyroPredictMatcher
{

public:
    enum eType{
        OPENCV_OPTICAL_FLOW_PYR_LK = 0,
        GYRO_PREDICT = 1,
        GYRO_PREDICT_WITH_OPTICAL_FLOW_REFINED = 2,
        GYRO_PREDICT_WITH_OPTICAL_FLOW_REFINED_CONSIDER_ILLUMINATION = 3,
        GYRO_PREDICT_WITH_OPTICAL_FLOW_REFINED_CONSIDER_ILLUMINATION_DEFORMATION = 4,
        GYRO_PREDICT_WITH_OPTICAL_FLOW_REFINED_CONSIDER_ILLUMINATION_DEFORMATION_REGULAR = 6,
        IMAGE_ONLY_OPTICAL_FLOW_CONSIDER_ILLUMINATION = 5
    };

    struct sMatch{
        int queryIdx;   //!< query descriptor index
        int trainIdx;   //!< train descriptor index
        float distance; //!< train image index
        float ncc;
        int level;      //!< level == 1, search region: 1.0 * mRadiusForFindNearNeighbor
                        //!< level == 2, search region: 2.0 * mRadiusForFindNearNeighbor
                        //!< level == 3, search region: 3.0 * mRadiusForFindNearNeighbor
        sMatch() {
            queryIdx = -1;
            trainIdx = -1;
            distance = -1;
            ncc = 0.0f;
            level = 0;
        }

        sMatch(int queryIdx_, int trainIdx_, float distance_, float ncc_=0.0f, int level_=0) {
            queryIdx = queryIdx_;
            trainIdx = trainIdx_;
            distance = distance_;
            ncc = ncc_;
            level = level_;
        }

        // less is better
        bool operator<(const sMatch &m) const{
            return distance < m.distance;
        }

        void operator=(const sMatch &m) {
            queryIdx = m.queryIdx;
            trainIdx = m.trainIdx;
            distance = m.distance;
            ncc = m.ncc;
            level = m.level;
        }
    };

public:
//    GyroPredictMatcher();
    GyroPredictMatcher(double t, double t_ref, const cv::Mat &imgGrayRef_, const cv::Mat &imgGrayCur_,
                       const std::vector<cv::KeyPoint> &vKeysRef_, const std::vector<cv::KeyPoint> &vKeysCur_,
                       const std::vector<cv::KeyPoint> &vKeysUnRef_, const std::vector<cv::KeyPoint> &vKeysUnCur_,
                       const std::vector<IMU::Point> &vImuFromLastFrame, const cv::Point3f &bias_,
                       cv::Mat K_, cv::Mat DistCoef_, const cv::Mat &normalizeTable_,
                       eType type_ = OPENCV_OPTICAL_FLOW_PYR_LK,
                       std::string saveFolderPath="",
                       int halfPatchSize_ = 5,
                       int predictMethod_ = 0);

    GyroPredictMatcher(const Frame& pFrameRef, const Frame& pFrameCur,
                       const IMU::Calib& imuCalib,
                       const cv::Point3f &biasg_,
                       const cv::Mat &normalizeTable,
                       eType type_ = OPENCV_OPTICAL_FLOW_PYR_LK,
                       std::string saveFolderPath="",
                       int halfPatchSize_ = 5,
                       int predictMethod_ = 0);

    void Initialize();

    void SetRegularizationPenalty(bool flag) {mbRegularizationPenalty = flag;}

    void SetBackToFrame(Frame& pFrame);

    // Search matches between keypoints in current frame and reference frame, using optical flow tracking (KLT)
    int SearchByOpencvKLT();

    // Search matches between keyoints in current frame and reference frame, using gyroscope integration
    // Find minimum distance
    int SearchByGyroPredict();

    int TrackFeatures();
    int GeometryValidation();

    void SetRcl(const cv::Mat Rcl_);
    cv::Mat GetRcl() {return mRcl.clone();};

    void SetType(eType type_){mType = type_;}

    void Display(bool isKeyFrame = false);

    void IntegrateGyroMeasurements();

    static const float TH_NCC_HIGH;
    static const float TH_NCC_LOW;
    static const float TH_RATIO;

private:
    void GyroPredictOnePixel(cv::Point2f &pt_ref, cv::Point2f &pt_predict, cv::Point2f &pt_predict_distort, cv::Point2f &flow);

    // Predict features using gyroscope integrated rotation (Rcl), do not use depth and translation
    int GyroPredictFeatures();
    //void GyroPredictFeaturesParallel(const cv::Range& range);

    // Predict features using gyroscope integrated rotation (Rcl) and further optimize by find the best matched patch
    int GyroPredictFeaturesAndOpticalFlowRefined();

    // Find the nearest and the second nearest ORB features points to the predicted point.
    void FindAndSortNearNeighbor(const cv::Range& range, int level);

    // Match reference keypoints and current keypoints using the results of FindAndSortNearNeighbor()
    void MatchFeatures(std::vector<sMatch> &vMatches, const vector<vector<sMatch>> &vvNearNeighbors);

    cv::Mat IntegrateOneGyroMeasurement(cv::Point3f &gyro, double dt);

    float CheckHomography(cv::Mat &H21, float &score, std::vector<bool> &vbMatchesInliers, std::vector<cv::Point2f> &vPts1, std::vector<cv::Point2f> &vPts2, float sigma);
    float CheckFundamental(cv::Mat &F21, float &score, std::vector<bool> &vbMatchesInliers, std::vector<cv::Point2f> &vPts1, std::vector<cv::Point2f> &vPts2, float sigma);

    /*
    bool ReconstructF(std::vector<bool> &vbMatchesInliers, cv::Mat &F21, cv::Mat &K,
                      cv::Mat &R21, cv::Mat &t21, vector<cv::Point3f> &vP3D, vector<bool> &vbTriangulated, float minParallax, int minTriangulated);
    bool ReconstructH(std::vector<bool> &vbMatchesInliers, cv::Mat &H21, cv::Mat &K,
                      cv::Mat &R21, cv::Mat &t21, vector<cv::Point3f> &vP3D, vector<bool> &vbTriangulated, float minParallax, int minTriangulated);

    int CheckRT(const cv::Mat &R, const cv::Mat &t, const vector<cv::Point2f> &vPts1, const vector<cv::Point2f> &vPts2,
                           vector<bool> &vbMatchesInliers,
                           const cv::Mat &K, vector<cv::Point3f> &vP3D, float th2, vector<bool> &vbGood, float &parallax);

    void Triangulate(const cv::Point2f &pt1, const cv::Point2f &pt2, const cv::Mat &P1, const cv::Mat &P2, cv::Mat &x3D);
    */

    void SaveMsgToFile(std::string filename, std::string &msg);

public:
    std::string mSaveFolderPath;

    double mTimeStamp;
    double mTimeStampRef;
    const cv::Mat &mImgGrayRef;
    const cv::Mat &mImgGrayCur;
    const std::vector<cv::KeyPoint> &mvKeysRef;      // Keypoints in original reference image
    const std::vector<cv::KeyPoint> &mvKeysRefUn;    // Undistorted keypoint of reference image. Used for Gyro. predict
    const std::vector<cv::KeyPoint> &mvKeysCur;      // Keypoints in original current image
    const std::vector<cv::KeyPoint> &mvKeysCurUn;    // Undistorted keypoint of current image. Used for Gyro. predict

    const std::vector<IMU::Point> &mvImuFromLastFrame;
    const cv::Point3f &mBias;

    std::vector<cv::Point2f> mvPtPredict;   // Pixels predicted through gyro integration. (Distorted)
    std::vector<cv::Point2f> mvPtPredictUn; // Pixels predicted through gyro integration. (Undistorted)
    std::vector<cv::Point2f> mvPtGyroPredict; // Pixels predicted from reference frame. (Distorted)
    std::vector<cv::Point2f> mvPtGyroPredictUn; // Pixels predicted from reference frame. (Undistorted)

    std::vector<std::vector<cv::Point2f>> mvvPtPredictCorners;      // for corners of the predicted pixels. (Distorted)
    std::vector<std::vector<cv::Point2f>> mvvPtPredictCornersUn;    // for corners of the predicted pixels. (Undistorted)
    std::vector<std::vector<cv::Point2f>> mvvFlowsPredictCorners;

    std::vector<uchar> mvStatus;            // States. 1: matched; 0: unmatched
    std::vector<float> mvError;             // Record errors
    std::vector<double> mvDisparities;      // Disparities
    std::vector<sMatch> mvMatches;          // queryIdx: index in reference detected keys;
                                            // trainIdx: index in current detected keys
    vector<vector<sMatch>> mvvNearNeighbors;     // neighbors of each predicted points

    // States for patch matched
    int mHalfPatchSize;
    float mRadiusForFindNearNeighbor;

    std::vector<cv::Point2f> mvPatchCorners; // Patch vector of four corners.
    cv::Mat mMatPatchCorners;   // mMatPatchCorners is used for predicting the affine deformation matrix
    std::vector<cv::Mat> mvAffineDeformationMatrix; // The affine deformation matrix of predicted by gyro-aided. A = [1 + dxx, dxy; dyx, 1 + dyy]

    std::vector<cv::Point2f> mvPtPredictAfterPatchMatched;   // Pixels predicted by patch match. (Distorted)
    std::vector<cv::Point2f> mvPtPredictAfterPatchMatchedUn; // Pixels predicted by patch match. (Un-Distorted)
    std::vector<uchar> mvStatusAfterPatchMatched;            // States. 1: matched; 0: unmatched
    std::vector<double> mvPixelErrorsOfPatchMatched;
    std::vector<double> mvDistanceBetweenPredictedAndPatchMatched;
    std::vector<float> mvNccAfterPatchMatched;

    std::vector<cv::Point2f> mvFlowsPredictUn;  // Flows of gyro. predict
    std::vector<cv::Point2f> mvFlowsErrorUn;  // Flows between the gyro. predict pixel and the detected features

    bool mbNCC;  // if true, use ncc to find the nearest matches; else, use distance

    float mTimeCost = 0;
    float mTimeCostGyroPredict = 0;
    float mTimeCostOptFlow = 0;
    float mTimeCostOptFlowResultFilterOut = 0;
    float mTimeCostGeometryValidation = 0;
    float mTImeCostTotalFeatureTrack = 0;

    float mTimeFeaturePredict = 0;

    float mTimeFindNearest = 0;
    float mTimeFilterOut = 0;


    cv::Mat mRbc;
    IMU::Calib *mpIMUCalib;

    cv::Mat mRcl;       // The relative camera rotation from reference frame to current frame
    float mr11, mr12, mr13, mr21, mr22, mr23, mr31, mr32, mr33;

    //GeometricCamera* mpCamera;
    cv::Mat mK;         // The camera intrinsic parameters
    cv::Mat mKRKinv;
    float mfx, mfy, mcx, mcy, mfx_inv, mfy_inv;

    cv::Mat mDistCoef;  // The camera distortion coeffections
    float mk1, mk2, mp1, mp2, mk3;

    int mWidth;
    int mHeight;
    int mN;
    const cv::Mat &mNormalizeTable;
    eType mType;

    bool mbHasGyroPredictInitial = true;
    bool mbConsiderIllumination = true;
    bool mbConsiderAffineDeformation = false;
    bool mbRegularizationPenalty = false;   // true; // true also performs well

    int mPreditcMethod = 0;
};


#endif // GYROPREDICTMATCHER_H


