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

#ifndef UTILS_H
#define UTILS_H

#include "iostream"
#include "vector"
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>

// get a gray scale value from reference image (bi-linear interpolated)
inline float GetPixelValue(const cv::Mat &img, float x, float y)
{
    // boundary check
    if (x < 0) x = 0;
    if (y < 0) y = 0;
    if (x > img.cols) x = img.cols - 1;
    if (y > img.rows) y = img.rows - 1;

    uchar *data = &img.data[int(y) * img.step + int(x)];
    float xx = x - std::floor(x);
    float yy = y - std::floor(y);
    float pixel = (1 - yy) * (1 - xx) * data[0] + (1 - yy) * xx * data[1]
            + yy  * (1 - xx) * data[img.step]  + yy * xx * data[img.step + 1];
    return pixel;
}

void DistortOnePoint(cv::Point2f& pt, cv::Point2f& pt_dist, cv::Mat& K, cv::Mat& DistCoef);
void DistortVecPoints(std::vector<cv::Point2f>& vpts, std::vector<cv::Point2f>& vpts_dist, cv::Mat& K, cv::Mat& DistCoef);
void UndistortVecPoints(std::vector<cv::Point2f>& vpts_dist, std::vector<cv::Point2f>& vpts_undist, cv::Mat& K, cv::Mat& DistCoef);


// Zero-Normalized cross correlation
float NCC(int halfPathSize, std::vector<float> &vValuesRef, const float mean_ref, const cv::Mat &cur, const cv::Point2f &pt_cur, const cv::Mat &warp_mat);

// Zero-Normalized cross correlation
float NCC(int halfPathSize, const cv::Mat &ref, const cv::Mat &cur, const cv::Point2f &pt_ref, const cv::Point2f &pt_cur, const cv::Mat &warp_mat);

//the following are UBUNTU/LINUX ONLY terminal color
#define RESET "\033[0m"
#define BLACK "\033[30m" /* Black */
#define RED "\033[31m" /* Red */
#define GREEN "\033[32m" /* Green */
#define YELLOW "\033[33m" /* Yellow */
#define BLUE "\033[34m" /* Blue */
#define MAGENTA "\033[35m" /* Magenta*/
#define CYAN "\033[36m" /* Cyan */
#define WHITE "\033[37m" /* White */
#define BOLDBLACK "\033[1m\033[30m" /* Bold Black */
#define BOLDRED "\033[1m\033[31m" /* Bold Red */
#define BOLDGREEN "\033[1m\033[32m" /* Bold Green */
#define BOLDYELLOW "\033[1m\033[33m" /* Bold Yellow */
#define BOLDBLUE "\033[1m\033[34m" /* Bold Blue */
#define BOLDMAGENTA "\033[1m\033[35m" /* Bold Magenta */
#define BOLDCYAN "\033[1m\033[36m" /* Bold Cyan */
#define BOLDWHITE "\033[1m\033[37m" /* Bold White */

/*
 * example:
 *		 Timer time; // time recording
 *		 ....
 *		 printf("time: %d us\n", time.runTime());

*/
#include <stdio.h>
#include <sys/time.h>
class Timer
{
    public:

        struct timeval start, end;
        Timer()
        {
            gettimeofday( &start, NULL );
        }

        void freshTimer()
        {
            gettimeofday( &start, NULL );
        }

        int runTime()
        {
            gettimeofday( &end, NULL );
            return 1000000 * ( end.tv_sec - start.tv_sec ) + end.tv_usec -start.tv_usec;
        }
        double runTime_us()
        {
            return runTime()/1.0;
        }
        double runTime_ms()
        {
            return runTime() / 1000.0;
        }
        double runTime_s()
        {
            return runTime() / 1000000.0;
        }
};


#endif // UTILS_H
