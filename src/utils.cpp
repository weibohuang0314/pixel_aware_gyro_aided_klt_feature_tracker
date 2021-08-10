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

#include "utils.h"

void DistortOnePoint(cv::Point2f& pt, cv::Point2f& pt_dist, cv::Mat& K, cv::Mat& DistCoef)
{
    float mfx = K.at<float>(0,0); float mfy = K.at<float>(1,1);
    float mcx = K.at<float>(0,2); float mcy = K.at<float>(1,2);
    float mfx_inv = 1.0 / mfx; float mfy_inv = 1.0 / mfy;

    float K1 = DistCoef.at<float>(0); float K2 = DistCoef.at<float>(1);
    float mp1 = DistCoef.at<float>(2); float mp2 = DistCoef.at<float>(3);
    float K3 = DistCoef.total() == 5? DistCoef.at<float>(4): 0;

    float x = (pt.x - mcx) * mfx_inv;
    float y = (pt.y - mcy) * mfy_inv;

    float r2 = x * x + y * y;
    float r4 = r2 * r2;
    float r6 = r4 * r2;
    float x_distort = x * (1 + K1 * r2 + K2 * r4 + K3 * r6) + 2 * mp1 * x * y + mp2 * (r2 + 2 * x * x);
    float y_distort = y * (1 + K1 * r2 + K2 * r4 + K3 * r6) + mp1 * (r2 + 2 * y * y) + 2 * mp2 * x * y;
    float u_distort = mfx * x_distort + mcx;
    float v_distort = mfy * y_distort + mcy;

    pt_dist = cv::Point2f(u_distort, v_distort); // Distorted
}

void DistortVecPoints(std::vector<cv::Point2f>& vpts, std::vector<cv::Point2f>& vpts_dist, cv::Mat& K, cv::Mat& DistCoef)
{
    float mfx = K.at<float>(0,0); float mfy = K.at<float>(1,1);
    float mcx = K.at<float>(0,2); float mcy = K.at<float>(1,2);
    float mfx_inv = 1.0 / mfx; float mfy_inv = 1.0 / mfy;

    float K1 = DistCoef.at<float>(0); float K2 = DistCoef.at<float>(1);
    float mp1 = DistCoef.at<float>(2); float mp2 = DistCoef.at<float>(3);
    float K3 = DistCoef.total() == 5? DistCoef.at<float>(4): 0;

    vpts_dist.resize(vpts.size());
    for(size_t i = 0, iend = vpts.size(); i<iend; i++){
        cv::Point2f pt = vpts[i];

        float x = (pt.x - mcx) * mfx_inv;
        float y = (pt.y - mcy) * mfy_inv;

        float r2 = x * x + y * y;
        float r4 = r2 * r2;
        float r6 = r4 * r2;
        float x_distort = x * (1 + K1 * r2 + K2 * r4 + K3 * r6) + 2 * mp1 * x * y + mp2 * (r2 + 2 * x * x);
        float y_distort = y * (1 + K1 * r2 + K2 * r4 + K3 * r6) + mp1 * (r2 + 2 * y * y) + 2 * mp2 * x * y;
        float u_distort = mfx * x_distort + mcx;
        float v_distort = mfy * y_distort + mcy;

        vpts_dist[i] = cv::Point2f(u_distort, v_distort); // Distorte
    }
}

void UndistortVecPoints(std::vector<cv::Point2f>& vpts_dist, std::vector<cv::Point2f>& vpts_undist, cv::Mat& K, cv::Mat& DistCoef)
{
    int N = vpts_dist.size();
    cv::Mat mat(N, 2, CV_32F);
    for(int i = 0; i < N; i++){
        mat.at<float>(i,0) = vpts_dist[i].x;
        mat.at<float>(i,1) = vpts_dist[i].y;
    }
    mat = mat.reshape(2);
    cv::undistortPoints(mat, mat, K, DistCoef, cv::Mat(), K);
    mat = mat.reshape(1);

    vpts_undist.resize(N);
    for(size_t i = 0; i < N; i++){
        vpts_undist[i] = cv::Point2f(mat.at<float>(i,0), mat.at<float>(i,1));
    }
}

/**
 * Zero-Normalized cross correlation
 * @brief NCC
 * @param vValuesRef[in] Pre-obtained pixel values of the patch around the reference keypoint.
 * @param mean_ref[in]   Pre-calibrated mean value of the patch around the reference keypoint.
 * @param cur[in]        Gray image of current frame. cv::Mat
 * @param pt_cur[in]     Keypoint in current frame.
 * @param warp_mat[in]   The 2x2 affine deformation matrix.
 * @return score. [-1, 1]: -1: uncorrelation. 1: correlation
 *
 * NCC(A, B) =           \Sigma_{i,j}(A(i,j) - \bar{A}(i,j)) \cdot (B(i,j) - \bar{B}(i,j))
 *            ----------------------------------------------------------------------------------------
 *             sqrt( \Sigma_{i,j}(A(i,j) - \bar{A}(i,j))^2 \cdot \Sigma_{i,j}(B(i,j) - \bar{B}(i,j))^2 )
 */
float NCC(int halfPathSize,
          std::vector<float> &vValuesRef,
          const float mean_ref,
          const cv::Mat &cur,
          const cv::Point2f &pt_cur,
          const cv::Mat &warp_mat)
{
    // First: calculate mean value
    float mean_cur = 0.0f;
    std::vector<float> vValuesCur;
    for (int x = -halfPathSize; x <= halfPathSize; x++)
        for (int y = -halfPathSize; y <= halfPathSize; y++) {
            // consider the affine deformation matrix
            float value_cur = 0;
            if (warp_mat.empty())
                value_cur = GetPixelValue(cur, pt_cur.x + x, pt_cur.y + y);
            else {
                float wx = warp_mat.at<float>(0,0) * x + warp_mat.at<float>(0,1) * y;
                float wy = warp_mat.at<float>(1,0) * x + warp_mat.at<float>(1,1) * y;
                value_cur = GetPixelValue(cur, pt_cur.x + wx, pt_cur.y + wy);
            }
            mean_cur += value_cur;
            vValuesCur.push_back(value_cur);
        }

    mean_cur /= vValuesCur.size();

    // Second: calculate Zero-mean NCC
    float numerator = 0, demoniator1 = 0, demoniator2 = 0;
    for (int i = 0; i < vValuesRef.size(); i++) {
        float v_ref_dot = vValuesRef[i] - mean_ref;
        float v_cur_dot = vValuesCur[i] - mean_cur;
        numerator += (v_ref_dot * v_cur_dot);
        demoniator1 += v_ref_dot * v_ref_dot;
        demoniator2 += v_cur_dot * v_cur_dot;
    }

    return numerator / std::sqrt(demoniator1 * demoniator2 + 1e-10);    // avoid denominator == 0
}


/**
 * Zero-Normalized cross correlation
 * @brief NCC
 * @param ref
 * @param cur
 * @param pt_ref
 * @param pt_cur
 * @param warp_mat: 2x2 affine deformation matrix
 * @return score, [-1, 1]
 *
 * NCC(A, B) =           \Sigma_{i,j}(A(i,j) - \bar{A}(i,j)) \cdot (B(i,j) - \bar{B}(i,j))
 *            ----------------------------------------------------------------------------------------
 *             sqrt( \Sigma_{i,j}(A(i,j) - \bar{A}(i,j))^2 \cdot \Sigma_{i,j}(B(i,j) - \bar{B}(i,j))^2 )
 *
 */
float NCC(int halfPathSize, const cv::Mat &ref, const cv::Mat &cur, const cv::Point2f &pt_ref, const cv::Point2f &pt_cur, const cv::Mat &warp_mat)
{
    // First: calculate mean value
    float mean_ref = 0.0f, mean_cur = 0.0f;
    std::vector<float> vValuesRef, vValuesCur;
    for (int x = -halfPathSize; x <= halfPathSize; x++)
        for (int y = -halfPathSize; y <= halfPathSize; y++) {
            float value_ref = GetPixelValue(ref, pt_ref.x + x, pt_ref.y + y);
            mean_ref += value_ref;
            vValuesRef.push_back(value_ref);

            // consider the affine deformation matrix
            float value_cur = 0;
            if (warp_mat.empty())
                value_cur = GetPixelValue(cur, pt_cur.x + x, pt_cur.y + y);
            else {
                float wx = warp_mat.at<float>(0,0) * x + warp_mat.at<float>(0,1) * y;
                float wy = warp_mat.at<float>(1,0) * x + warp_mat.at<float>(1,1) * y;
                value_cur = GetPixelValue(cur, pt_cur.x + wx, pt_cur.y + wy);
            }
            mean_cur += value_cur;
            vValuesCur.push_back(value_cur);
        }

    mean_ref /= vValuesRef.size();
    mean_cur /= vValuesCur.size();

    // Second: calculate Zero-mean NCC
    float numerator = 0, demoniator1 = 0, demoniator2 = 0;
    for (int i = 0; i < vValuesRef.size(); i++) {
        numerator += ((vValuesRef[i] - mean_ref) * (vValuesCur[i] - mean_cur));
        demoniator1 += (vValuesRef[i] - mean_ref) * (vValuesRef[i] - mean_ref);
        demoniator2 += (vValuesCur[i] - mean_cur) * (vValuesCur[i] - mean_cur);
    }

    return numerator / std::sqrt(demoniator1 * demoniator2 + 1e-10);    // avoid denominator == 0
}
