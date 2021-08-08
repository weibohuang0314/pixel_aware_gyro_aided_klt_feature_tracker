#ifndef PATCHMATCH_H
#define PATCHMATCH_H

#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
//#include <Eigen/Core>
#include <Eigen/Dense>
#include <iostream>
#include <string>
#include <chrono>
#include <functional>
//#include "Thirdparty/glog/include/glog/logging.h"

//#include "GyroPredictMatcher.h"

using namespace std;
using namespace cv;


class GyroPredictMatcher;

class PatchMatch
{
public:
    PatchMatch(GyroPredictMatcher* pMatcher_,
               int halfPatchSize_, int iterations_, int pyramids_,
               bool bHasGyroPredictInitial_, bool bInverse_,
               bool bConsiderIllumination_, bool bConsiderAffineDeformation_,
               bool bRegularizationPenalty_ = true,
               bool bCalculateNCC_ = false);

    void CreatePyramids();

    // Multi level optical flow tracking
    void OpticalFlowMultiLevel();

    //void OpticalFlowCommonMethod_onePixel(const int i);

    // Optical flow considering the illumination change.
    void OpticalFlowConsideringIlluminationChange_onePixel(const int i, const bool bRegularizationPenalty);

    void ThreadComputeErrorAndJocabianForOnePoint(int index, cv::Point2f pt, float dx, float dy,
                                                  float x, float y, float wx, float wy, float db, float dg,
                                                  vector<Eigen::Vector4d> &vJ, vector<float> &vE, vector<bool> &vFlag);

    void SetMatcher();

    // Get a gray scale value from reference image (bi-linear interpolated)
    inline float GetPixelValue(const cv::Mat &img, float x, float y) const;

//    void UnDistortPoints();

    void DistortPoints();

    // Zero-Normalized cross correlation
    float NCC(int halfPathSize, const cv::Mat &ref, const cv::Mat &cur, const cv::Point2f &pt_ref, const cv::Point2f &pt_cur, const cv::Mat &warp_mat);


private:
    GyroPredictMatcher* mpMatcher;
    int mN;
    int mHalfPatchSize;
    int mIterations;
    int mPyramids;
    bool mbHasGyroPredictInitial;
    bool mbInverse;
    bool mbConsiderIllumination;
    bool mbConsiderAffineDeformation;
    bool mbCalculateNCC;

    // parameters for multi level

    double mPyramidScale;
    int mLevel;
    double mWinSizeInv;

    // parameters for regularization penalty term
    bool mbRegularizationPenalty;
    float mLambda;          // a factor that controls the overall strength of the penalty
    float mAlpha;          // a coefficient for distance
    int mMaxDistance;    // the max distance between patch matched position and gyro predict position (constant, unit: pixels)
    float mInvLogMaxDist;

    std::vector<float> mvScales;
    std::vector<bool> mvSuccess;
    std::vector<double> mvPixelErrorsOfPatchMatched;   // pixel errors of patched matched
    std::vector<float> mvNcc;

    std::vector<uchar> mvGyroPredictStatus;
    std::vector<cv::Mat> mvImgPyr1, mvImgPyr2;  // image pyramids
    std::vector<cv::Point2f> mvPtPyr1Un, mvPtPyr2, mvPtPyr2Un;


};



#endif // PATCHMATCH_H
