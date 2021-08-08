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
//        mvKeys(std::vector<cv::KeyPoint>(frame.mvKeys.begin(), frame.mvKeys.end())),
//        mvKeysUn(std::vector<cv::KeyPoint>(frame.mvKeysUn.begin(), frame.mvKeysUn.end()))
{
    mMask = cv::Mat::ones(mGray.rows, mGray.cols, CV_8UC1);
    mRcl = cv::Mat();
    curFrameWithoutGeometryValid = nullptr;

}

Frame::Frame(double &t, cv::Mat &im, cv::Mat &im_dist, Frame* pLastFrame, CameraParams *pCameraParams,
      std::vector<IMU::Point> &vImu, int keypointNumber, double th):
    mTimeStamp(t), mpLastFrame(pLastFrame),
    mpCameraParams(pCameraParams),
    mvImuFromLastFrame(vImu),
    mN(keypointNumber), mGrayDistort(im_dist)
//        mvImuFromLastFrame(std::vector<IMU::Point>(vImu.begin(), vImu.end()))
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

//    DetectKeyPoints();
    //LOG(INFO) << "t: " << std::to_string(mTimeStamp) << ", mvKeysUn.size(): " << mvKeysUn.size();
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

//     LOG(INFO) << "after SetPredictKeyPointsAndMask, mvKeys.size(): " << mvKeys.size();
}

// Default. detect new feature when the predicted features is less than a threshold.
void Frame::DetectKeyPoints()
{
    SetPredictKeyPointsAndMask();
    int num_predicted = mvKeys.size();

    static bool reach_max_feature_flag = false; // ensure to detect max feature
    static int cnt = 0;

    // We detected new featrues only when the predicted features is less than a threshold.
    if(num_predicted < mThresholdOfPredictNewKeyPoint || !reach_max_feature_flag)
    {
        int block_size = 3;
        double min_distance = 20;       // klt_imu_tracker: 20
        double quality_level = 0.005;    // klt_imu_tracker: 0.005
        int n_new = mN - num_predicted;                 // TODO: dynamic adjust the n_new according to the tracked results
        std::vector<cv::Point2f> corners_un;
        if (n_new > 0)
            cv::goodFeaturesToTrack(mGray, corners_un, n_new, quality_level, min_distance, mMask, block_size, true, 0.04);

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


        reach_max_feature_flag = false;
        if(mvKeysUn.size() == mN)
            reach_max_feature_flag = true;
        cnt = 0;
    }

    mN = mvKeysUn.size();
    //LOG(INFO) << "after DetectKeyPoints, mvKeys.size(): " << mvKeys.size() << ", cnt: " << cnt++;
}

// temp: read features from file. the feature is detected by SuperPoint (Paper - "SuperPoint: Self-supervised interest point detection and description")
void Frame::LoadDetectedKeypointFromFile(std::string path)
{
    // predict from gyro-aided tracking
    //SetPredictKeyPointsAndMask();

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
                //std::cout << field << std::endl;
            }
            vNewPts.push_back(cv::Point2f(data[1], data[2]));
            //std::cout << line << std::endl;
        }
    }
    fin.close();

    // add new feature when the predicted features is less than a threshold.
    int num_predicted = mvKeys.size();
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

//// display on rectified iamge
//void Frame::Display(std::string winname)
//{
//    if(curFrameWithoutGeometryValid == nullptr){    // display lastFrame and currentFrame
//        int h = mpCameraParams->height, w = mpCameraParams->width;
//        cv::Mat im_out = cv::Mat(h, 2 * w, CV_8UC1, cv::Scalar(0));
//        mpLastFrame->mGray.copyTo(im_out.rowRange(0, h).colRange(0, w));
//        mGray.copyTo(im_out.rowRange(0, h).colRange(w, 2*w));

//        if(im_out.channels() < 3) //this should be always true
//            cv::cvtColor(im_out, im_out, CV_GRAY2BGR);

//        int cnt_tracked = 0, cnt_new_detected = 0;
//        std::set<int> sPtIndexInLastFrame;
//        for(size_t i = 0, iend = mvKeys.size(); i < iend; i++){
//            cv::Point2f pt_cur = mvKeys[i].pt + cv::Point2f(w,0);
//            if(mvKeys[i].size > 1){ // this keypoint is tracked by gyro prediction and patch matched.
//                cnt_tracked ++;

//                sPtIndexInLastFrame.insert(mvPtIndexInLastFrame[i]);
//                cv::Point2f pt_ref = mpLastFrame->mvKeys[mvPtIndexInLastFrame[i]].pt + cv::Point2f(w,0);
//                cv::line(im_out, pt_ref, pt_cur, cv::Scalar(255,255,255), 1);

//                //if (!mvNcc.empty() && mvNcc[mvPtIndexInLastFrame[i]] < 0.2){
//                //    // negetive correlation
//                //    cv::circle(im_out, pt_cur, 3, cv::Scalar(255, 0, 0), -1);   // blue, the features tracked from last frame
//                //    cv::circle(im_out, mpLastFrame->mvKeys[mvPtIndexInLastFrame[i]].pt, 3, cv::Scalar(255, 0, 0), -1);  // corresponding features in last frame
//                //} else {
//                    cv::circle(im_out, pt_cur, 2, cv::Scalar(0, 255, 0), -1);   // green, the features tracked from last frame
//                    cv::circle(im_out, mpLastFrame->mvKeys[mvPtIndexInLastFrame[i]].pt, 2, cv::Scalar(0, 255, 0), -1);  // corresponding features in last frame
//                //}

//                // for gyro predicted points
//                if(!mvPtGyroPredictUn.empty()){
//                    cv::Point2f pt_gyro = mvPtGyroPredictUn[i] + cv::Point2f(w,0);
//                    cv::circle(im_out, pt_gyro, 2, cv::Scalar(0, 255, 255), 1);
//                }

//            }
//            else {  // new predict feature
//                cnt_new_detected ++;
//                cv::circle(im_out, pt_cur, 2, cv::Scalar(255, 0, 0), -1);   // blue, new detected feature
//            }
//        }

//        // draw the features in the lastframe that unsuccessful tracked in current frame
//        for(int i = 0, iend = mpLastFrame->mvKeys.size(); i < iend; i++){
//            if(sPtIndexInLastFrame.find(i) != sPtIndexInLastFrame.end())
//                continue;
//            cv::Point2f pt_ref = mpLastFrame->mvKeys[i].pt;
//            cv::circle(im_out, pt_ref, 2, cv::Scalar(0, 0, 255), -1);   // red
//        }

//        std::stringstream s;
//        s << std::fixed << std::setprecision(4)
//          << "Id: " << mnId
//          << ", T: " << std::to_string(mTimeStamp)
//          << ", RefF kp num: " << mvStatus.size()
//          << ", CurF kp num: " << cnt_tracked + cnt_new_detected
//          << " (tracked: " << cnt_tracked << ", new det.: " << cnt_new_detected
//          << ")" << ", track rate: " << 100.0 * cnt_tracked / mvStatus.size() << "%";

//        cv::Size textSize = cv::getTextSize(s.str(),cv::FONT_HERSHEY_PLAIN,1,1,0);
//        cv::Mat imText = cv::Mat(im_out.rows + textSize.height + 10, im_out.cols, im_out.type());
//        im_out.copyTo(imText.rowRange(0, im_out.rows).colRange(0, im_out.cols));
//        imText.rowRange(im_out.rows, imText.rows) = cv::Mat::zeros(textSize.height + 10, im_out.cols, im_out.type());
//        cv::putText(imText, s.str(), cv::Point(5, imText.rows-5), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(255,255,255),1.0);

//        cv::putText(imText, "reference frame", cv::Point(5, 15), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(255,255,255),1.0);
//        cv::putText(imText, "current frame with geometry validation", cv::Point(5 + w, 15), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(255,255,255),1.0);

//        cv::imshow(winname, imText);
//        cv::waitKey(1);
//    }
//    else { // display lastFrame, currentFrame without geometry validation, and currentFrame
//        int h = mpCameraParams->height, w = mpCameraParams->width;
//        cv::Mat im_out = cv::Mat(h, 3 * w, CV_8UC1, cv::Scalar(0));
//        mpLastFrame->mGray.copyTo(im_out.rowRange(0, h).colRange(0, w));
//        curFrameWithoutGeometryValid->mGray.copyTo(im_out.rowRange(0, h).colRange(w, 2*w));
//        mGray.copyTo(im_out.rowRange(0, h).colRange(2*w, 3*w));

//        if(im_out.channels() < 3) //this should be always true
//            cv::cvtColor(im_out, im_out, CV_GRAY2BGR);

//        int cnt_tracked = 0, cnt_new_detected = 0;
//        std::set<int> sPtIndexInLastFrame;
//        for(size_t i = 0, iend = mvKeysUn.size(); i < iend; i++){
//            cv::Point2f pt_cur = mvKeysUn[i].pt + cv::Point2f(2*w,0);
//            if(mvKeysUn[i].size > 1){ // this keypoint is tracked by gyro prediction and patch matched.
//                cnt_tracked ++;

//                sPtIndexInLastFrame.insert(mvPtIndexInLastFrame[i]);
//                cv::Point2f pt_ref = mpLastFrame->mvKeysUn[mvPtIndexInLastFrame[i]].pt + cv::Point2f(2*w,0);
//                cv::line(im_out, pt_ref, pt_cur, cv::Scalar(255,255,255), 1);

//                //if (!mvNcc.empty() && mvNcc[mvPtIndexInLastFrame[i]] < 0.2){
//                //    // negetive correlation
//                //    cv::circle(im_out, pt_cur, 3, cv::Scalar(255, 0, 0), -1);   // blue, the features tracked from last frame
//                //    cv::circle(im_out, mpLastFrame->mvKeysUn[mvPtIndexInLastFrame[i]].pt, 3, cv::Scalar(255, 0, 0), -1);  // corresponding features in last frame
//                //} else {
//                cv::circle(im_out, pt_cur, 2, cv::Scalar(0, 255, 0), -1);   // green, the features tracked from last frame
//                cv::circle(im_out, mpLastFrame->mvKeysUn[mvPtIndexInLastFrame[i]].pt, 2, cv::Scalar(0, 255, 0), -1);  // corresponding features in last frame
//                //}

//                // Draw the affine deformated rectangles for the matched keypoints.
//                if(mvvFlowsPredictCorners[mvPtIndexInLastFrame[i]].size() > 0){
//                    // on current frame
//                    cv::Point2f pt_tl = mvvFlowsPredictCorners[mvPtIndexInLastFrame[i]][0] + pt_cur;
//                    cv::Point2f pt_tr = mvvFlowsPredictCorners[mvPtIndexInLastFrame[i]][1] + pt_cur;
//                    cv::Point2f pt_bl = mvvFlowsPredictCorners[mvPtIndexInLastFrame[i]][2] + pt_cur;
//                    cv::Point2f pt_br = mvvFlowsPredictCorners[mvPtIndexInLastFrame[i]][3] + pt_cur;

//                    cv::Scalar scalar = cv::Scalar(0,255,0);
//                    cv::line(im_out, pt_tl, pt_tr, scalar, 1);
//                    cv::line(im_out, pt_tr, pt_br, scalar, 1);
//                    cv::line(im_out, pt_bl, pt_br, scalar, 1);
//                    cv::line(im_out, pt_tl, pt_bl, scalar, 1);

//                    // on reference frame
//                    cv::Point2f lr = pt_tr - pt_tl; int w = int(sqrt(lr.x * lr.x + lr.y * lr.y)/2.0 + 0.5);
//                    pt_tl = cv::Point2f(-w, -w) + mpLastFrame->mvKeysUn[mvPtIndexInLastFrame[i]].pt;
//                    pt_tr = cv::Point2f(w, -w) + mpLastFrame->mvKeysUn[mvPtIndexInLastFrame[i]].pt;
//                    pt_bl = cv::Point2f(-w, w) + mpLastFrame->mvKeysUn[mvPtIndexInLastFrame[i]].pt;
//                    pt_br = cv::Point2f(w, w) + mpLastFrame->mvKeysUn[mvPtIndexInLastFrame[i]].pt;
//                    cv::line(im_out, pt_tl, pt_tr, scalar, 1);
//                    cv::line(im_out, pt_tr, pt_br, scalar, 1);
//                    cv::line(im_out, pt_bl, pt_br, scalar, 1);
//                    cv::line(im_out, pt_tl, pt_bl, scalar, 1);
//                }

//            }
//            else {  // new predict feature
//                cnt_new_detected ++;
//                cv::circle(im_out, pt_cur, 2, cv::Scalar(255, 0, 0), -1);   // blue, new detected feature
//            }
//        }

//        // draw the lost tracks in the lastframe
//        for(int i = 0, iend = mpLastFrame->mvKeysUn.size(); i < iend; i++){
//            if(sPtIndexInLastFrame.find(i) != sPtIndexInLastFrame.end())
//                continue;
//            cv::Point2f pt_ref = mpLastFrame->mvKeysUn[i].pt;
//            cv::circle(im_out, pt_ref, 2, cv::Scalar(0, 0, 255), -1);   // red
//        }

//        // middle results
//        int cnt_gyro_predict = 0;
//        int cnt_total_tracks = 0, cnt_bad_tracks = 0;
//        //double error_gyro_predict_patch_match = 0;
//        // for curFrameWithoutGeometryValid
//        // draw the gyro predicd and patch-matched features that do not filter out by geometry validation
//        for(size_t i = 0, iend = curFrameWithoutGeometryValid->mvStatus.size(); i < iend; i++){
//            if(!curFrameWithoutGeometryValid->mvStatus[i])
//                continue;

//            cv::Point2f pt_cur = curFrameWithoutGeometryValid->mvPtPredictUn[i] + cv::Point2f(w,0);
//            cv::Point2f pt_ref = mpLastFrame->mvKeysUn[i].pt + cv::Point2f(w,0);
//            cv::line(im_out, pt_ref, pt_cur, cv::Scalar(255,255,255), 1);

//            cv::circle(im_out, pt_cur, 2, cv::Scalar(0, 255, 0), -1);   // green, the features tracked from last frame
//            cnt_total_tracks ++;

//            if(sPtIndexInLastFrame.find(i) == sPtIndexInLastFrame.end()){
//                cv::circle(im_out, pt_cur, 5, cv::Scalar(0, 0, 255), 1);   // red, the points that would be filter-out
//                cnt_bad_tracks ++;
//            }
//        }

//        if(!curFrameWithoutGeometryValid->mvPtGyroPredictUn.empty()){
//            for(size_t i = 0, iend = curFrameWithoutGeometryValid->mvPtGyroPredictUn.size(); i < iend; i++){
//                cv::Point2f pt_gyro = curFrameWithoutGeometryValid->mvPtGyroPredictUn[i] + cv::Point2f(w,0);
//                if(pt_gyro.x != 0 && pt_gyro.y != 0){
//                    cv::circle(im_out, pt_gyro, 2, cv::Scalar(0, 255, 255), 1); // yellow circle
//                    cnt_gyro_predict ++;

////                    cv::Point2f pt_patch_match = curFrameWithoutGeometryValid->mvPtPredict[i] + cv::Point2f(w,0);
////                    cv::Point2f ept = pt_gyro - pt_patch_match;
////                    error_gyro_predict_patch_match += std::sqrt(ept.x * ept.x + ept.y * ept.y);
//                }

//            }
//        }

//        std::stringstream s;
//        s << std::fixed << std::setprecision(4)
//          << "Id: " << mnId
//          << ", T: " << std::to_string(mTimeStamp)
//          << ", RefF kp Num: " << mvStatus.size()
//          << ", CurF kp Num: " << cnt_tracked + cnt_new_detected
//          << " (tracked: " << cnt_tracked << ", new det.: " << cnt_new_detected
//          << ")" << ", Track Rate: " << 100.0 * cnt_tracked / mvStatus.size() << "%"
//          //<< ", Avg Err. Gyro Pred. Patch Match: " << error_gyro_predict_patch_match / cnt_gyro_predict
//             ;

//        cv::Size textSize = cv::getTextSize(s.str(),cv::FONT_HERSHEY_PLAIN,1,1,0);
//        cv::Mat imText = cv::Mat(im_out.rows + textSize.height + 10, im_out.cols, im_out.type());
//        im_out.copyTo(imText.rowRange(0, im_out.rows).colRange(0, im_out.cols));
//        imText.rowRange(im_out.rows, imText.rows) = cv::Mat::zeros(textSize.height + 10, im_out.cols, im_out.type());

//        cv::Scalar txt_color_fg(255, 255, 255);
//        cv::Scalar txt_color_bg(0, 0, 0);

//        cv::putText(imText, s.str(), cv::Point(5, imText.rows-5), cv::FONT_HERSHEY_PLAIN, 1, txt_color_fg, 1.0, cv::LINE_AA);

//        cv::putText(imText, "reference frame", cv::Point(5, 15), cv::FONT_HERSHEY_PLAIN, 1, txt_color_bg, 2.0, cv::LINE_AA);
//        cv::putText(imText, "reference frame", cv::Point(5, 15), cv::FONT_HERSHEY_PLAIN, 1, txt_color_fg, 1.0, cv::LINE_AA);
//        if(!mvPtGyroPredictUn.empty()){
//            std::string str = "cur. frame: yellow circle - gyro predicted features (" + std::to_string(cnt_gyro_predict) + ")";
//            cv::putText(imText, str, cv::Point(5 + w, 15), cv::FONT_HERSHEY_PLAIN, 1, txt_color_bg, 2.0, cv::LINE_AA);
//            cv::putText(imText, str, cv::Point(5 + w, 15), cv::FONT_HERSHEY_PLAIN, 1, txt_color_fg, 1.0, cv::LINE_AA);

//            str = "             green dot - PatchMatch features (" + std::to_string(cnt_total_tracks) + ")";
//            cv::putText(imText, str, cv::Point(5 + w, 35), cv::FONT_HERSHEY_PLAIN, 1, txt_color_bg, 2.0, cv::LINE_AA);
//            cv::putText(imText, str, cv::Point(5 + w, 35), cv::FONT_HERSHEY_PLAIN, 1, txt_color_fg, 1.0, cv::LINE_AA);

//            str = "             red circle - features to be filter out by geo. valid. (" + std::to_string(cnt_bad_tracks) + ")";
//            cv::putText(imText, str, cv::Point(5 + w, 55), cv::FONT_HERSHEY_PLAIN, 1, txt_color_bg, 2.0, cv::LINE_AA);
//            cv::putText(imText, str, cv::Point(5 + w, 55), cv::FONT_HERSHEY_PLAIN, 1, txt_color_fg, 1.0, cv::LINE_AA);
//        }
//        else {
//            // for OPENCV_OPTICAL_FLOW_PYR_LK
//            std::string str = "cur. frame: green dot - openCV optical flow predicted features (" + std::to_string(cnt_total_tracks) + ")";
//            cv::putText(imText, str, cv::Point(5 + w, 15), cv::FONT_HERSHEY_PLAIN, 1, txt_color_bg, 2.0, cv::LINE_AA);
//            cv::putText(imText, str, cv::Point(5 + w, 15), cv::FONT_HERSHEY_PLAIN, 1, txt_color_fg, 1.0, cv::LINE_AA);

//            str = "             red circle - features to be filter out by geo. valid. (" + std::to_string(cnt_bad_tracks) + ")";
//            cv::putText(imText, str, cv::Point(5 + w, 35), cv::FONT_HERSHEY_PLAIN, 1, txt_color_bg, 2.0, cv::LINE_AA);
//            cv::putText(imText, str, cv::Point(5 + w, 35), cv::FONT_HERSHEY_PLAIN, 1, txt_color_fg, 1.0, cv::LINE_AA);
//        }

//        std::string str = "cur. frame: green dot - successful tracked features (" + std::to_string(cnt_tracked) + ")";
//        cv::putText(imText, str, cv::Point(5 + 2 * w, 15), cv::FONT_HERSHEY_PLAIN, 1, txt_color_bg, 2.0, cv::LINE_AA);
//        cv::putText(imText, str, cv::Point(5 + 2 * w, 15), cv::FONT_HERSHEY_PLAIN, 1, txt_color_fg, 1.0, cv::LINE_AA);

//        str = "             blue dot - new feature detection (" + std::to_string(cnt_new_detected) + ")";
//        cv::putText(imText, str, cv::Point(5 + 2 * w, 35), cv::FONT_HERSHEY_PLAIN, 1, txt_color_bg, 2.0, cv::LINE_AA);
//        cv::putText(imText, str, cv::Point(5 + 2 * w, 35), cv::FONT_HERSHEY_PLAIN, 1, txt_color_fg, 1.0, cv::LINE_AA);

//        cv::imshow(winname, imText);
//        cv::waitKey(1);

//    }
//}


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

        if(!curFrameWithoutGeometryValid->mvStatus[i]){ // Loss-tracked, red circle
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
//    s << std::fixed << std::setprecision(4)
//      << "Id: " << mnId
//      << ", T: " << std::to_string(mTimeStamp)
//      << ", RefF kp Num: " << mvStatus.size()
//      << ", CurF kp Num: " << cnt_good_tracks + cnt_new_detected
//      << " (tracked: " << cnt_good_tracks << ", new det.: " << cnt_new_detected
//      << ")" << ", Track Rate: " << 100.0 * cnt_good_tracks / mvStatus.size() << "%"
//         ;
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
    // vTxt.push_back("T: " + std::to_string(mTimeStamp));
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
