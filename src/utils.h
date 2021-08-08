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

void DistortOnePoint(cv::Point2f& pt, cv::Point2f& pt_dist, cv::Mat& mK, cv::Mat& mDistCoef);
void DistortVecPoints(std::vector<cv::Point2f>& vpts, std::vector<cv::Point2f>& vpts_dist, cv::Mat& mK, cv::Mat& mDistCoef);

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
#define MAGENTA "\033[35m" /* Magenta 品红*/
#define CYAN "\033[36m" /* Cyan 青色 */
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
 * 在构造函数中记录当前的系统时间，在析构函数中输出当前的系统时间与之前的差，精度是us
 * 使用方法：在需要计时的程序段之前构造类对象，在程序段之后获取时间
 * example:
 *		 Timer time; //开始计时
 *		 ....
 *		 printf("time: %d us\n", time.runTime()); //显示时间

*/
// 计时
#include <stdio.h>
#include <sys/time.h>
class Timer
{
    public:

        struct timeval start, end;
        Timer() // 构造函数，开始记录时间
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
