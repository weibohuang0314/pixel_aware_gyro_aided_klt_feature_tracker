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
