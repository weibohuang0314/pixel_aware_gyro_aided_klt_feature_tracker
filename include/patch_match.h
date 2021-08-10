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

#ifndef PATCHMATCH_H
#define PATCHMATCH_H

#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <eigen3/Eigen/Dense>
#include <iostream>
#include <string>
#include <chrono>
#include <functional>

using namespace std;
using namespace cv;

class GyroAidedTracker;

class PatchMatch
{
public:
    PatchMatch(GyroAidedTracker* pMatcher_,
               int halfPatchSize_, int iterations_, int pyramids_,
               bool bHasGyroPredictInitial_, bool bInverse_,
               bool bConsiderIllumination_, bool bConsiderAffineDeformation_,
               bool bRegularizationPenalty_ = true,
               bool bCalculateNCC_ = false);

    void CreatePyramids();

    // Multi level optical flow tracking
    void OpticalFlowMultiLevel();

    // Optical flow considering the illumination change.
    void OpticalFlowConsideringIlluminationChange_onePixel(const int i,
                                                           const bool bConsiderIllumination,
                                                           const bool bConsiderAffineDeformation,
                                                           const bool bRegularizationPenalty);
    void SetMatcher();

    // Get a gray scale value from reference image (bi-linear interpolated)
    inline float GetPixelValue(const cv::Mat &img, float x, float y) const;

    void DistortPoints();

    // Zero-Normalized cross correlation
    float NCC(int halfPathSize, const cv::Mat &ref, const cv::Mat &cur, const cv::Point2f &pt_ref, const cv::Point2f &pt_cur, const cv::Mat &warp_mat);

private:
    GyroAidedTracker* mpMatcher;
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
    float mAlpha;           // a coefficient for distance
    int mMaxDistance;       // the max distance between patch matched position and gyro predict position (constant, unit: pixels)
    float mInvLogMaxDist;

    std::vector<float> mvScales;
    std::vector<bool> mvSuccess;
    std::vector<double> mvPixelErrorsOfPatchMatched;    // pixel errors of patched matched
    std::vector<float> mvNcc;

    std::vector<uchar> mvGyroPredictStatus;
    std::vector<cv::Mat> mvImgPyr1, mvImgPyr2;          // image pyramids
    std::vector<cv::Point2f> mvPtPyr1Un, mvPtPyr2, mvPtPyr2Un;
};



#endif // PATCHMATCH_H
