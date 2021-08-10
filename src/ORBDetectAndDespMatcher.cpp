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

#include "ORBDetectAndDespMatcher.h"

ORBDetectAndDespMatcher::ORBDetectAndDespMatcher(const Frame& pFrameRef, const Frame& pFrameCur,
                                                 ORB_SLAM2::ORBextractor* pORBextractorLeft,
                                                 ORB_SLAM2::ORBextractor* pORBextractorRight,
                                                 std::string saveFolderPath):
    mTimeStamp(pFrameCur.mTimeStamp), mTimeStampRef(pFrameRef.mTimeStamp),
    mImgGrayRef(pFrameRef.mGray), mImgGrayCur(pFrameCur.mGray),
    mK(pFrameCur.mpCameraParams->mK), mDistCoef(pFrameCur.mpCameraParams->mDistCoef),
    mpORBextractorLeft(pORBextractorLeft), mpORBextractorRight(pORBextractorRight),
    mSaveFolderPath(saveFolderPath)
{
    mfx = mK.at<float>(0,0); mfy = mK.at<float>(1,1);
    mcx = mK.at<float>(0,2); mcy = mK.at<float>(1,2);
    mfx_inv = 1.0 / mfx; mfy_inv = 1.0 / mfy;

    mk1 = mDistCoef.at<float>(0); mk2 = mDistCoef.at<float>(1);
    mp1 = mDistCoef.at<float>(2); mp2 = mDistCoef.at<float>(3);
    mk3 = mDistCoef.total() == 5? mDistCoef.at<float>(4): 0;
}

void ORBDetectAndDespMatcher::ExtractORB(int flag, const cv::Mat& im)
{
    if(flag == 0){
        cv::Mat mask;
        (*mpORBextractorLeft)(im, mask, mvKeyPointsRef, mDescriptorRef);
    }else {
        cv::Mat mask;
        (*mpORBextractorRight)(im, mask, mvKeyPointsCur, mDescriptorCur);
    }
}

void ORBDetectAndDespMatcher::FindFeatureMatches(){
    std::thread threadLeft(&ORBDetectAndDespMatcher::ExtractORB, this, 0, mImgGrayRef);
    std::thread threadRight(&ORBDetectAndDespMatcher::ExtractORB, this, 1, mImgGrayCur);
    threadLeft.join();
    threadRight.join();

    // Step 3: match descriptor, using Hamming distance
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");
    std::vector<cv::DMatch> matches;
    matcher->match(mDescriptorRef, mDescriptorCur, matches);
    mvDMatchesByDescriptorMatch = matches;

    // Step 4: filter out matches
    double min_dist = 10000, max_dist = 0;
    // Find the maximum and minimum distance
    for (auto m: matches){
        if(m.distance < min_dist) min_dist = m.distance;
        if(m.distance > max_dist) max_dist = m.distance;
    }

    // Select matches
    double experiment_value = 30;
    double threshold = 2 * min_dist > experiment_value? 2 * min_dist: experiment_value;
    for(auto m:matches){
        if(m.distance <= threshold)
            mvDMatches.push_back(m);
    }

    // save results to files
    std::stringstream s;
    s << "RefKey Num: " << mvKeyPointsRef.size() << ", descriptorMatch Num: " << mvDMatchesByDescriptorMatch.size()
      << ", fitler-out distance: " << mvDMatches.size();

    std::string msg = "T: " + std::to_string(mTimeStamp) + ", " + s.str();
    SaveMsgToFile("ORBDetectAndDespMatch.txt", msg);
    // LOG(INFO) << msg;
}

void ORBDetectAndDespMatcher::PoseEstimation2d2d()
{
    std::vector<cv::Point2f> points1, points2;
    for(auto m:mvDMatches){
        points1.push_back(mvKeyPointsRef[m.queryIdx].pt);
        points2.push_back(mvKeyPointsCur[m.trainIdx].pt);
    }

    mF = cv::findFundamentalMat(points1, points2, cv::RANSAC);
    mE = cv::findEssentialMat(points1, points2, (mfx+mfy)/2.0, cv::Point2d(mcx, mcy), cv::RANSAC);
    mH = cv::findHomography(points1, points2, cv::RANSAC, 3);
    cv::recoverPose(mE, points1, points2, mR, mt, (mfx+mfy)/2.0, cv::Point2d(mcx, mcy));
}

void ORBDetectAndDespMatcher::SaveMsgToFile(std::string filename, std::string &msg)
{
    std::ofstream fp(mSaveFolderPath + filename, ofstream::app);
    if(!fp.is_open()) LOG(ERROR) << "cannot open: " << mSaveFolderPath + filename;
    else {
        fp << std::fixed << std::setprecision(6);
        fp << msg << std::endl;
    }
}

void ORBDetectAndDespMatcher::Display()
{
    int h = mImgGrayCur.rows, w = mImgGrayCur.cols;
    cv::Mat im_out = cv::Mat(h, 2*w, CV_8UC1, cv::Scalar(0));
    mImgGrayRef.copyTo(im_out.rowRange(0, h).colRange(0, w));
    mImgGrayCur.copyTo(im_out.rowRange(0, h).colRange(w, 2*w));
    if(im_out.channels() < 3) //this should be always true
        cv::cvtColor(im_out, im_out, CV_GRAY2BGR);

    std::set<int> sPtIndexInRefFrame;
    for (uint i = 0; i < mvDMatches.size(); i++) {
        cv::DMatch m = mvDMatches[i];
        cv::Point2f pt_ref = mvKeyPointsRef[m.queryIdx].pt;
        cv::Point2f pt_cur = mvKeyPointsCur[m.trainIdx].pt + cv::Point2f(w, 0);
        cv::circle(im_out, pt_ref, 2, cv::Scalar(0, 255, 0), -1);
        cv::circle(im_out, pt_cur, 2, cv::Scalar(0, 255, 0), -1);

        std::srand(i);
        int r = std::rand() % 256, g = std::rand() % 256, b = std::rand() % 256;
        cv::line(im_out, pt_ref, pt_cur, cv::Scalar(b,g,r), 1);

        sPtIndexInRefFrame.insert(m.queryIdx);
    }

    // draw lost tracked features
    for(int i = 0; i < mvKeyPointsRef.size(); i++){
        if(sPtIndexInRefFrame.find(i) != sPtIndexInRefFrame.end())
            continue;

        cv::Point2f pt_ref = mvKeyPointsRef[i].pt;
        cv::circle(im_out, pt_ref, 2, cv::Scalar(0, 0, 255), -1);
    }

    cv::imshow("feature_detect_matcher", im_out);
    cv::waitKey(1);
}
