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

#include "frame.h"
#include "../Thirdparty/glog/include/glog/logging.h"
#include "utils.h"
#include <iostream>

long unsigned int Frame::nNextId = 0;

Frame::Frame()
{
    mnId = 0;
}

Frame::Frame(const Frame& frame):
    mnId(frame.mnId),
    mN(frame.mN),
    mTimeStamp(frame.mTimeStamp),
    mGray(frame.mGray.clone()), mGrayDistort(frame.mGrayDistort.clone()),
    mpLastFrame(frame.mpLastFrame),
    mRcl(frame.mRcl.clone()),
    mpCameraParams(frame.mpCameraParams),
    mvImuFromLastFrame(frame.mvImuFromLastFrame),
    mvKeys(frame.mvKeys), mvKeysUn(frame.mvKeysUn),
    mvKeysNormal(frame.mvKeysNormal),
    mvFlowVelocityInNormalPlane(frame.mvFlowVelocityInNormalPlane),
    mvPtIndexInLastFrame(frame.mvPtIndexInLastFrame),
    mfx(frame.mfx), mfy(frame.mfy), mcx(frame.mcx), mcy(frame.mcy),
    mfx_inv(frame.mfx_inv), mfy_inv(frame.mfy_inv),
    mThresholdOfPredictNewKeyPoint(frame.mThresholdOfPredictNewKeyPoint)
{
    mMask = cv::Mat::ones(mGray.rows, mGray.cols, CV_8UC1);
    mRcl = cv::Mat();
    curFrameWithoutGeometryValid = nullptr;

}

Frame::Frame(double &t, cv::Mat &im, cv::Mat &im_dist, Frame* pLastFrame, CameraParams *pCameraParams,
             ORB_SLAM2::ORBextractor* pORBextractor,
             std::vector<IMU::Point> &vImu, int keypointNumber, double th):
    mTimeStamp(t), mpLastFrame(pLastFrame),
    mpCameraParams(pCameraParams),
    mvImuFromLastFrame(vImu),
    mN(keypointNumber), mGrayDistort(im_dist)
{
    mnId = nNextId ++;

    mfx = mpCameraParams->mK.at<float>(0,0); mfy = mpCameraParams->mK.at<float>(1,1);
    mcx = mpCameraParams->mK.at<float>(0,2); mcy = mpCameraParams->mK.at<float>(1,2);
    mfx_inv = 1.0 / mfx; mfy_inv = 1.0 / mfy;

    mvKeys.reserve(mN);
    mvKeysUn.reserve(mN);
    mvKeysNormal.reserve(mN);
    mvFlowVelocityInNormalPlane.resize(mN);

    mvPtIndexInLastFrame.reserve(mN);

    mThresholdOfPredictNewKeyPoint = mN * th;

    if(im.channels() == 3)
        cv::cvtColor(im, mGray, CV_RGB2GRAY);
    else if(im.channels() == 4)
        cv::cvtColor(im, mGray, CV_RGBA2GRAY);
    else {
        im.copyTo(mGray);
    }

    mMask = cv::Mat::ones(im.rows, im.cols, CV_8UC1); // cv::Mat(im.rows, im.cols, CV_8UC1);

    mRcl = cv::Mat();

    curFrameWithoutGeometryValid = nullptr;
}

void Frame::Reset()
{
    mvKeys.clear();
    mvKeysUn.clear();
    mvKeysNormal.clear();
    mvvFlowsPredictCorners.clear();
    mvFlowVelocityInNormalPlane.clear();

    mMask = cv::Mat::ones(mGray.rows, mGray.cols, CV_8UC1);

    mvPtIndexInLastFrame.clear();
    mvPtPredict.clear();
    mvPtPredictUn.clear();
    mvPtGyroPredictUn.clear();
    mvStatus.clear();
    mvNcc.clear();

}

void Frame::SetPredictKeyPointsAndMask()
{
    int half_path_size = 7;
    int cnt = 0;
    cv::Mat roi = cv::Mat::zeros(half_path_size*2, half_path_size*2, CV_8UC1);
    for (size_t i = 0, iend = mvStatus.size(); i < iend; ++i)
    {
        if(!mvStatus[i])
            continue;

        cv::Point2f pt_pred = mvPtPredict[i];
        cv::Point2f pt_pred_un = mvPtPredictUn[i];
        cv::Point2f pt_pred_normal;
        pt_pred_normal.x = (pt_pred_un.x - mcx) * mfx_inv;
        pt_pred_normal.y = (pt_pred_un.y - mcy) * mfy_inv;

        // set predicted points to mvKeys and mvKeysUn
        mvKeys.push_back(cv::KeyPoint(pt_pred, half_path_size));
        mvKeysUn.push_back(cv::KeyPoint(pt_pred_un, half_path_size));
        mvKeysNormal.push_back(cv::KeyPoint(pt_pred_normal, half_path_size));
        mvPtIndexInLastFrame.push_back(i);

        // When the feature is tracked from lastFrame, we calculate
        // its flow velocity in normalized plane for last frame.
        cv::KeyPoint pt_normal_lastF = mpLastFrame->mvKeysNormal[i];
        cv::Point2f dpt_normal = pt_pred_normal - pt_normal_lastF.pt;
        cv::Point2f dv = dpt_normal / (mTimeStamp - mpLastFrame->mTimeStamp);

        mpLastFrame->mvFlowVelocityInNormalPlane[cnt] = dv;

        cnt ++;

        // set mask
        int _x = std::min(std::max(0, int(pt_pred_un.x) - half_path_size), mpCameraParams->width - roi.cols);
        int _y = std::min(std::max(0, int(pt_pred_un.y) - half_path_size), mpCameraParams->height - roi.rows);
        cv::Rect roi_rect = cv::Rect(_x, _y, roi.cols, roi.rows);
        roi.copyTo(mMask(roi_rect));
    }
}

// Default. detect new feature when the predicted features is less than a threshold.
void Frame::DetectKeyPoints(ORB_SLAM2::ORBextractor* pORBextractor)
{
    SetPredictKeyPointsAndMask();
    int num_predicted = mvKeysUn.size();

    static bool reach_max_feature_flag = false; // ensure to detect max feature

    // We detected new featrues only when the predicted features is less than a threshold.
    if(num_predicted < mThresholdOfPredictNewKeyPoint || !reach_max_feature_flag)
    {
        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();

        int n_new = mN - num_predicted;
        if (n_new <= 0) return;

        std::vector<cv::Point2f> corners_un;
        if(pORBextractor){  // use ORBextractor
            std::vector<cv::KeyPoint> keypoints;
            pORBextractor->DetectFeatures(mGray, mMask, keypoints);

            for(auto key:keypoints){
                if(corners_un.size() < n_new)
                    corners_un.push_back(key.pt);
            }
        }else{  // use cv::goodFeaturesToTrack
            int block_size = 3;
            double min_distance = 20;
            double quality_level = 0.005;
            cv::goodFeaturesToTrack(mGray, corners_un, n_new, quality_level, min_distance, mMask, block_size, true, 0.04);
        }

        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();    // start timer
        double timeOfDetectNewFeatures = 1000 * std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();

        for (auto pt: corners_un) {
            mvKeysUn.push_back(cv::KeyPoint(pt, 0));
            mvPtIndexInLastFrame.push_back(-1);
            float x_normal = (pt.x - mcx) * mfx_inv;
            float y_normal = (pt.y - mcy) * mfy_inv;
            mvKeysNormal.push_back(cv::KeyPoint(x_normal, y_normal, 1));
        }

        // Distort points
        std::vector<cv::Point2f> corners_dist;
        DistortVecPoints(corners_un, corners_dist, mpCameraParams->mK, mpCameraParams->mDistCoef);
        for(size_t i = 0, iend = corners_dist.size(); i < iend; i++){
            mvKeys.push_back(cv::KeyPoint(corners_dist[i], 0));
        }

        std::chrono::steady_clock::time_point t3 = std::chrono::steady_clock::now();    // start timer
        double timeOfDistortPoints = 1000 * std::chrono::duration_cast<std::chrono::duration<double> >(t3 - t2).count();

//        LOG(INFO) << "detect new keypoint: num_predicted: " << num_predicted << ", n_new: " << n_new
//                  << ", corners_un.size(): " << corners_un.size()
//                  << ", mvKeysUn.size(): " << mvKeysUn.size();
//        LOG(INFO) << "timeOfDetectNewFeatures: " << timeOfDetectNewFeatures
//                  << ", timeOfDistortPoints: " << timeOfDistortPoints;

        reach_max_feature_flag = mvKeysUn.size() == mN;
    }

    mN = mvKeysUn.size();
}


// Load features from file. the feature is detected by SuperPoint (Paper - "SuperPoint: Self-supervised interest point detection and description")
void Frame::LoadDetectedKeypointFromFile(std::string path)
{
    // read features detect by SuperPoint
    std::vector<cv::Point2f> vNewPts; vNewPts.reserve(mN);
    std::ifstream fin(path.c_str());
    if(!fin.is_open()){
        std::cout << "open file failed. file: " << path << std::endl;
        return;
    }else {
        std::string line;
        while(getline(fin, line)){
            std::istringstream sin(line);
            std::vector<double> data;
            std::string field;
            while (getline(sin, field, ',')) {
                data.push_back(std::atof(field.c_str()));
            }
            vNewPts.push_back(cv::Point2f(data[1], data[2]));
        }
    }
    fin.close();

    // add new feature when the predicted features is less than a threshold.
    int num_predicted = mvKeysUn.size();
    static bool reach_max_feature_flag = false; // ensure to detect max feature

    int n_new = mN - num_predicted;
    if(n_new > 0 && (num_predicted < mThresholdOfPredictNewKeyPoint || !reach_max_feature_flag))
    {
        std::vector<cv::Point2f> corners_un;
        for(auto pt: vNewPts){
            if(mMask.at<uchar>(int(pt.y), int(pt.x)) == 0)
                continue;

            corners_un.push_back(pt);
            mvKeysUn.push_back(cv::KeyPoint(pt, 0));
            mvPtIndexInLastFrame.push_back(-1);
            float x_normal = (pt.x - mcx) * mfx_inv;
            float y_normal = (pt.y - mcy) * mfy_inv;
            mvKeysNormal.push_back(cv::KeyPoint(x_normal, y_normal, 1));

            n_new --;
            if (n_new <= 0)
                break;
        }

        // Distort points
        std::vector<cv::Point2f> corners_dist;
        DistortVecPoints(corners_un, corners_dist, mpCameraParams->mK, mpCameraParams->mDistCoef);
        for(size_t i = 0, iend = corners_dist.size(); i < iend; i++){
            mvKeys.push_back(cv::KeyPoint(corners_dist[i], 0));
        }

        reach_max_feature_flag = false;
        if(mvKeysUn.size() >= mN)
            reach_max_feature_flag = true;
    }

    mN = mvKeysUn.size();
}

// un-use
void Frame::UndistortPoints(std::vector<cv::Point2f> &corners)
{
    // Fill matrix with points
    cv::Mat mat(corners.size(), 2, CV_32F);
    for(int i = 0; i < corners.size(); i++){
        mat.at<float>(i,0) = corners[i].x;
        mat.at<float>(i,1) = corners[i].y;
    }

    // Undistort points
    try {
        mat = mat.reshape(2);
        cv::undistortPoints(mat, mat, mpCameraParams->mK, mpCameraParams->mDistCoef, cv::Mat(), mpCameraParams->mK);
        mat = mat.reshape(1);
    } catch (cv::Exception& e) {
        LOG(INFO) << e.err;
        LOG(INFO) << RED"corners.size(): " << corners.size() << RESET;
    }

    // Fill undistorted keypoint vector
    for (size_t i = 0; i < corners.size(); i++){
        mvKeysUn.push_back(cv::KeyPoint(mat.at<float>(i,0), mat.at<float>(i,1), 1));
        cv::Point2f pt_pred_normal;
        pt_pred_normal.x = (mat.at<float>(i,0) - mcx) * mfx_inv;
        pt_pred_normal.y = (mat.at<float>(i,1) - mcy) * mfy_inv;
        mvKeysNormal.push_back(cv::KeyPoint(pt_pred_normal, 1));
    }
}


/**
 * @brief Frame::Display
 * @param winname
 * @param drawFlowType      0: draw flows on the current frame; 1: draw match line across two frames
 * @param bDrawPatch        true: draw square patches on the reference frame and deformed patches on the current frame
 * @param bDrawMistracks    true: draw mistacks
 * @param bDrawGyroPredictPosition  true: draw gyro predict positions
 */
void Frame::Display(std::string winname, int drawFlowType, bool bDrawPatch, bool bDrawMistracks, bool bDrawGyroPredictPosition)
{
    cv::Scalar COLOR_BLUE(255, 0, 0);
    cv::Scalar COLOR_GREEN(0, 255, 0);
    cv::Scalar COLOR_RED(0, 0, 255);
    cv::Scalar COLOR_WHITE(255, 255, 255);
    cv::Scalar COLOR_YELLOW(0, 255, 255);

    int margin = 10;
    int h = mpCameraParams->height, w = mpCameraParams->width;
    cv::Mat im_out = cv::Mat(h, 2 * w + margin, CV_8UC1, cv::Scalar(255));
    mpLastFrame->mGray.copyTo(im_out.rowRange(0, h).colRange(0, w));
    mGray.copyTo(im_out.rowRange(0, h).colRange(w+margin, 2*w+margin));

    if(im_out.channels() < 3) //this should be always true
        cv::cvtColor(im_out, im_out, CV_GRAY2BGR);

    int cnt_new_detected = 0;
    std::set<int> sPtIndexInLastFrame;
    for(size_t i = 0, iend = mvKeysUn.size(); i < iend; i++){
        if(mvKeysUn[i].size > 1){ // this keypoint is tracked by gyro prediction and patch matched.

            sPtIndexInLastFrame.insert(mvPtIndexInLastFrame[i]); // record good tracks after geometric validation

            // draw patches
            if(bDrawPatch && mvvFlowsPredictCorners[mvPtIndexInLastFrame[i]].size() > 0){

                // on current frame
                cv::Point2f pt_cur = mvKeysUn[i].pt + cv::Point2f(w + margin,0);
                cv::Point2f pt_tl = mvvFlowsPredictCorners[mvPtIndexInLastFrame[i]][0] + pt_cur;
                cv::Point2f pt_tr = mvvFlowsPredictCorners[mvPtIndexInLastFrame[i]][1] + pt_cur;
                cv::Point2f pt_bl = mvvFlowsPredictCorners[mvPtIndexInLastFrame[i]][2] + pt_cur;
                cv::Point2f pt_br = mvvFlowsPredictCorners[mvPtIndexInLastFrame[i]][3] + pt_cur;

                cv::line(im_out, pt_tl, pt_tr, COLOR_GREEN, 1, cv::LINE_AA);
                cv::line(im_out, pt_tr, pt_br, COLOR_GREEN, 1, cv::LINE_AA);
                cv::line(im_out, pt_bl, pt_br, COLOR_GREEN, 1, cv::LINE_AA);
                cv::line(im_out, pt_tl, pt_bl, COLOR_GREEN, 1, cv::LINE_AA);

                // on reference frame
                cv::Point2f lr = pt_tr - pt_tl; int w = int(sqrt(lr.x * lr.x + lr.y * lr.y)/2.0 + 0.5);
                cv::Point2f pt_ref = mpLastFrame->mvKeysUn[mvPtIndexInLastFrame[i]].pt;
                pt_tl = cv::Point2f(-w, -w) + pt_ref;
                pt_tr = cv::Point2f(w, -w) + pt_ref;
                pt_bl = cv::Point2f(-w, w) + pt_ref;
                pt_br = cv::Point2f(w, w) + pt_ref;
                cv::line(im_out, pt_tl, pt_tr, COLOR_GREEN, 1, cv::LINE_AA);
                cv::line(im_out, pt_tr, pt_br, COLOR_GREEN, 1, cv::LINE_AA);
                cv::line(im_out, pt_bl, pt_br, COLOR_GREEN, 1, cv::LINE_AA);
                cv::line(im_out, pt_tl, pt_bl, COLOR_GREEN, 1, cv::LINE_AA);
            }
        }
        else {  // new predict feature
            cnt_new_detected ++;
        }
    }

    // tracked results
    int circle_radius = 2;
    int circle_thickness = -1;
    int line_thickness = 1;
    int cnt_total_tracks = 0, cnt_good_tracks = 0, cnt_bad_tracks = 0;

    // draw the gyro predicd and patch-matched features that do not filter out by geometry validation
    for(size_t i = 0, iend = curFrameWithoutGeometryValid->mvStatus.size(); i < iend; i++){
        cv::Point2f pt_cur = curFrameWithoutGeometryValid->mvPtPredictUn[i] + cv::Point2f(w+margin,0);
        cv::Point2f pt_ref = mpLastFrame->mvKeysUn[i].pt;

        if(!curFrameWithoutGeometryValid->mvStatus[i]){ // Loss-tracked, red circle in reference frame
            cv::circle(im_out, pt_ref, circle_radius, COLOR_BLUE, circle_thickness);
            continue;
        }

        cnt_total_tracks ++;

        if(bDrawMistracks && sPtIndexInLastFrame.find(i) == sPtIndexInLastFrame.end()){   // Mistracked, red circle, red line
            cnt_bad_tracks ++;

            cv::circle(im_out, pt_cur, circle_radius, COLOR_RED, circle_thickness); // Bad track, red circle in reference frame
            cv::circle(im_out, pt_ref, circle_radius, COLOR_RED, circle_thickness); // Bad track, red circle in current frame

            // draw flows
            if (drawFlowType == 0)      // flow on the current frame
                cv::line(im_out, pt_ref + cv::Point2f(w+margin, 0), pt_cur, COLOR_RED, line_thickness, cv::LINE_AA);
            else if (drawFlowType == 1) // line across two frames
                cv::line(im_out, pt_ref, pt_cur, COLOR_RED, line_thickness, cv::LINE_AA);
        }else {
            cnt_good_tracks ++;

            cv::circle(im_out, pt_ref, circle_radius, COLOR_GREEN, circle_thickness);   // Good track, green circle in reference frame
            cv::circle(im_out, pt_cur, circle_radius, COLOR_GREEN, circle_thickness);   // Good track, green circle in current frame

            // draw flows
            if (drawFlowType == 0)      // flow on current frame
                cv::line(im_out, pt_ref + cv::Point2f(w+margin, 0), pt_cur, COLOR_WHITE, line_thickness, cv::LINE_AA);
            else if (drawFlowType == 1) // line across two frames
                cv::line(im_out, pt_ref, pt_cur, COLOR_WHITE, line_thickness, cv::LINE_AA);
        }

        // draw gyro predict position
        if(bDrawGyroPredictPosition && !curFrameWithoutGeometryValid->mvPtGyroPredictUn.empty()){
             cv::Point2f pt_gyro = curFrameWithoutGeometryValid->mvPtGyroPredictUn[i] + cv::Point2f(w + margin,0);
             if(pt_gyro.x != 0 && pt_gyro.y != 0){
                 cv::circle(im_out, pt_gyro, circle_radius, COLOR_YELLOW, 1); // yellow circle
             }
        }
    }


    std::stringstream s;
    int ref_kp_num = mvStatus.size();

    s << std::fixed << std::setprecision(4)
      << "Id: " << mnId
      << ", T: " << std::to_string(mTimeStamp)
      << ", RGT: " << 100.0 * cnt_good_tracks / ref_kp_num << "%"
      << ", RGP: " << 100.0 * cnt_good_tracks / cnt_total_tracks << "%"
         ;

    cv::Size textSize = cv::getTextSize(s.str(),cv::FONT_HERSHEY_PLAIN,1,1,0);
    cv::Mat imText = cv::Mat(im_out.rows + textSize.height + 10, im_out.cols, im_out.type());
    im_out.copyTo(imText.rowRange(0, im_out.rows).colRange(0, im_out.cols));
    imText.rowRange(im_out.rows, imText.rows) = cv::Mat::zeros(textSize.height + 10, im_out.cols, im_out.type());

    cv::Scalar txt_color_fg(255, 255, 255);
    cv::Scalar txt_color_bg(0, 0, 0);

    cv::putText(imText, s.str(), cv::Point(5, imText.rows-5), cv::FONT_HERSHEY_PLAIN, 1, txt_color_fg, 1.0, cv::LINE_AA);

    std::vector<std::string> vTxt;
    if(!mvPtGyroPredictUn.empty()){
        vTxt.push_back("Gyro-Aided KLT Feature Tracking");
    }
    else {  // for OPENCV_OPTICAL_FLOW_PYR_LK
        vTxt.push_back("Image-only KLT Feature Tracking");
    }

    vTxt.push_back("ID: " + std::to_string(mnId));
    vTxt.push_back("Keypoints: " + std::to_string(mpLastFrame->mvKeys.size()));
    vTxt.push_back("Tracks: " + std::to_string(cnt_total_tracks)
                   + " (G: " + std::to_string(cnt_good_tracks) + " | B: " + std::to_string(cnt_bad_tracks) + ")" );
    std::stringstream s_rgt; s_rgt << std::fixed << std::setprecision(2) << "RGT: " << 100.0 * cnt_good_tracks / ref_kp_num << "%";
    std::stringstream s_rgp; s_rgp << std::fixed << std::setprecision(2) << "RGP: " << 100.0 * cnt_good_tracks / cnt_total_tracks << "%";
    vTxt.push_back(s_rgt.str() + ", " + s_rgp.str());

    double sc = std::min(mGray.rows/640.0, 2.0);    // scale factor for consistent visualization across scales
    int Ht = int(sc * 30);
    for (size_t i = 0, iend = vTxt.size(); i < iend; i++) {
        cv::putText(imText, vTxt[i], cv::Point(int(8*sc) + w + margin, Ht*(i+1)), cv::FONT_HERSHEY_DUPLEX, 1.0*sc, txt_color_bg, 2.0, cv::LINE_AA);
        cv::putText(imText, vTxt[i], cv::Point(int(8*sc) + w + margin, Ht*(i+1)), cv::FONT_HERSHEY_DUPLEX, 1.0*sc, txt_color_fg, 1.0, cv::LINE_AA);
    }

    cv::imshow(winname, imText);
    cv::waitKey(1);
}
