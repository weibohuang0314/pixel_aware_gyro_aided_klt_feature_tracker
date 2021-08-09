#include "patch_match.h"
#include "gyro_aided_tracker.h"
#include <thread>
#include <pthread.h>
#include <omp.h>
#include "utils.h"

//namespace ORB_SLAM3{

typedef Eigen::Matrix<double, 5, 1> Vector5d;
typedef Eigen::Matrix<double, 5, 5> Matrix5d;

PatchMatch::PatchMatch(GyroAidedTracker* pMatcher_,
                       int halfPatchSize_, int iterations_, int pyramids_,
                       bool bHasGyroPredictInitial_, bool bInverse_,
                       bool bConsiderIllumination_, bool bConsiderAffineDeformation_,
                       bool bRegularizationPenalty_,
                       bool bCalculateNCC_):
    mpMatcher(pMatcher_),
    mN(pMatcher_->mvKeysRef.size()),
    mHalfPatchSize(halfPatchSize_), mIterations(iterations_), mPyramids(pyramids_),
    mbHasGyroPredictInitial(bHasGyroPredictInitial_), mbInverse(bInverse_),
    mbConsiderIllumination(bConsiderIllumination_), mbConsiderAffineDeformation(bConsiderAffineDeformation_),
    mbRegularizationPenalty(bRegularizationPenalty_),
    mbCalculateNCC(bCalculateNCC_)
{
    // parameters for regularization penalty term
    mLambda = 1.0f;
    mAlpha = 0.5f;
    mMaxDistance = 25;
    mInvLogMaxDist = 1.0 / (std::log(mAlpha * mMaxDistance + 1));


    // parameters for multi level
    mPyramidScale = 0.5f;
    mLevel = 0;

    mWinSizeInv = 1.0f / (2.0f * mHalfPatchSize + 1.0f) / (2.0f * mHalfPatchSize + 1.0f);
    mvGyroPredictStatus = std::vector<uchar>(pMatcher_->mvStatus.begin(), pMatcher_->mvStatus.end());
}

void PatchMatch::CreatePyramids(){
    for (int i = 0; i < mPyramids; i++) {
        if (i == 0) {
            mvImgPyr1.push_back(mpMatcher->mImgGrayRef);
            mvImgPyr2.push_back(mpMatcher->mImgGrayCur);
            mvScales.push_back(1.0f);
        } else {
            cv::Mat img1_pyr, img2_pyr;
            cv::resize(mvImgPyr1[i-1], img1_pyr, cv::Size(mvImgPyr1[i-1].cols * mPyramidScale, mvImgPyr1[i-1].rows * mPyramidScale));
            cv::resize(mvImgPyr2[i-1], img2_pyr, cv::Size(mvImgPyr2[i-1].cols * mPyramidScale, mvImgPyr2[i-1].rows * mPyramidScale));
            mvImgPyr1.push_back(img1_pyr);
            mvImgPyr2.push_back(img2_pyr);
            mvScales.push_back(mvScales[i-1] * mPyramidScale);
        }
    }
}

// Multi level optical flow tracking
void PatchMatch::OpticalFlowMultiLevel(){
    CreatePyramids();

    // Set initial points for top pyramid
    for (size_t i = 0; i < mN; i++) {
        mvPtPyr1Un.push_back(mpMatcher->mvKeysRefUn[i].pt);
        if (mbHasGyroPredictInitial)
            mvPtPyr2Un.push_back(mpMatcher->mvPtPredictUn[i]);
        else {
            mvPtPyr2Un.push_back(mpMatcher->mvKeysRefUn[i].pt);
        }
    }

    // coarse-to-fine LK tracking in pyramids
    mvSuccess.clear(); mvSuccess.resize(mN);
    mvPixelErrorsOfPatchMatched.clear(); mvPixelErrorsOfPatchMatched.resize(mN);
    mvNcc.clear(); mvNcc.resize(mN);

    //const float pyramidScale_inv = 1.0 / mPyramidScale;
    for (int level = mPyramids - 1; level >= 0; level --) {
        mLevel = level;
        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();

        // use opencv parallel_for_ function
        cv::parallel_for_(cv::Range(0,mN), [&](const cv::Range& range){
            for (auto i = range.start; i < range.end; i++)
                OpticalFlowConsideringIlluminationChange_onePixel(i,
                                                                  mbConsiderIllumination,
                                                                  mbConsiderAffineDeformation,
                                                                  mbRegularizationPenalty);
        });

        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
        double t = std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
    }

    // Distort
    DistortPoints();

    // Set mpMatcher
    SetMatcher();

    // Display
    /*
    cv::Mat img2_multi;
    cv::cvtColor(mvImgPyr1[0], img2_multi, CV_GRAY2BGR);
    for (int i = 0; i < mvPtPyr2Un.size(); i ++) {
        if (mvSuccess[i])
        {
            cv::circle(img2_multi, mpMatcher->mvPtPredict[i], 2, cv::Scalar(255, 0, 0), -1);    // blue, predict
            cv::circle(img2_multi, mvPtPyr2Un[i], 2, cv::Scalar(0, 140, 255), -1);  //DarkOrange, optical flow

            cv::line(img2_multi, mvPtPyr1Un[i], mpMatcher->mvPtPredict[i], cv::Scalar(255, 255, 255), 1); // white: reference to predict
            cv::line(img2_multi, mpMatcher->mvPtPredict[i], mvPtPyr2Un[i], cv::Scalar(0, 0, 255), 1); // red: predict to optical flow
        }
    }
    if (mpMatcher->mType == GyroAidedTracker::GYRO_PREDICT_WITH_OPTICAL_FLOW_REFINED)
        cv::imshow("tracked multi level", img2_multi);
    else if (mpMatcher->mType == GyroAidedTracker::GYRO_PREDICT_WITH_OPTICAL_FLOW_REFINED_CONSIDER_ILLUMINATION)
        cv::imshow("tracked multi level (considering illumination change)", img2_multi);
    cv::waitKey(1);
    */

}



/**
 * Optical flow considering the illumination change.
 * Assumption: (1 + g(x, y)) * I1(x, y, t) = I2(x + u, y + v, t + 1) + b(x, y)
 * Note: If bConsiderAffineDeformation == true, then we consider the affine
 *       deformation of the patch. The affine deformation matrix A modeled as:
 *              A = [1 + dxx, dxy;
 *                  dyx, 1 + dyy].
 *
 *       If mbRegularizationPenalty == true, then we also consider the regularization penalty term.
 *       The regularization error term is:
 *               P(x) = lambda * ln(alpha * d + 1) / ln(alpha * d_max + 1),
 *       where $d$ is the distance between patch matched position and gyro predict position,
 *       $alpha$ is a coefficient factor for distance, $d_max$ is the max distance,
 *       $lambda$ is a factor that controls the overall strength of the penalty.
 *
 *       Let dx = u - u_gyro, dy = v - v_gyro, then d = sqrt(dx * dx + dy * dy), we have
 *       dP(x) / dx = dP/dd * dd/dx
 *
 *
 * @brief OpticalFlowConsideringIlluminationChange_onePixel
 */
void PatchMatch::OpticalFlowConsideringIlluminationChange_onePixel(
        const int i,
        const bool bConsiderIllumination,
        const bool bConsiderAffineDeformation,
        const bool bRegularizationPenalty)
{
    if (!mvGyroPredictStatus[i])
        return;

    // Use distorted points to perform the patch match on raw image
    cv::Point2f pt = mvPtPyr1Un[i] * mvScales[mLevel]; // (float)(1./(1<<mLevel));
    cv::Point2f nextPt;
    if (mLevel == mPyramids - 1){
        nextPt = mvPtPyr2Un[i] * mvScales[mLevel]; // initial for points on the top level
    }else {
        nextPt = mvPtPyr2Un[i] * 1.0f / mPyramidScale; //2.0f;    // initial for points on the next level
    }

    // dx, dy need to be estimated.
    float dx = nextPt.x - pt.x;
    float dy = nextPt.y - pt.y;

    // dg, db also need to be estimated.
    float dg = 0.0f;
    float db = 0.0f;

    float cost = 0.0f, lastCost = 0.0f;
    bool succ = true;   // indicate if this point succeeded

    // calculate the warp patch (i.e., affine deformation patch)
    cv::Mat warp_patch(cv::Size(mHalfPatchSize*2+1, mHalfPatchSize*2+1), CV_32FC2);
    if (bConsiderAffineDeformation) {
        cv::Mat A = mpMatcher->mvAffineDeformationMatrix[i];
        assert(A.empty() == false);
        for (int x = - mHalfPatchSize; x <= mHalfPatchSize; x ++) {
            for (int y = - mHalfPatchSize; y <= mHalfPatchSize; y++) {
                float wx = A.at<float>(0,0) * x + A.at<float>(0,1) * y;
                float wy = A.at<float>(1,0) * x + A.at<float>(1,1) * y;
                warp_patch.at<Vec2f>(x + mHalfPatchSize, y + mHalfPatchSize)[0] = wx;
                warp_patch.at<Vec2f>(x + mHalfPatchSize, y + mHalfPatchSize)[1] = wy;
            }
        }
    }

    // Gauss-Newton iterations
    Eigen::Matrix4d H = Eigen::Matrix4d::Zero();    // hessian for patch match
    Eigen::Vector4d b = Eigen::Vector4d::Zero();    // bias
    Eigen::Vector4d J;                              // jacobian for patch match
    for(int iter = 0; iter < mIterations; iter++) {
        if (mbInverse == false) {
            H = Eigen::Matrix4d::Zero();
            b = Eigen::Vector4d::Zero();
        } else{
            // only reset b. not support yet
            b = Eigen::Vector4d::Zero();
        }

        // try to compute cost and jacobian in multi-thread
        int N_index = (2 * mHalfPatchSize + 1) * (2 * mHalfPatchSize + 1);
        vector<Eigen::Vector4d> vJ; vJ.resize(N_index);
        vector<float> vE; vE.resize(N_index);
        vector<bool> vFlag(N_index, false);
        int index = 0;

        vector<cv::Point2f> vPt_x_y; vPt_x_y.resize(N_index);
        vector<cv::Point2f> vPt_wx_wy; vPt_wx_wy.resize(N_index);
        for (int y = - mHalfPatchSize; y <= mHalfPatchSize; y ++) {
            for (int x = - mHalfPatchSize; x <= mHalfPatchSize; x++) {
                float wx = x, wy = y;
                if (bConsiderAffineDeformation) {
                    wx = warp_patch.at<Vec2f>(x + mHalfPatchSize, y + mHalfPatchSize)[0];
                    wy = warp_patch.at<Vec2f>(x + mHalfPatchSize, y + mHalfPatchSize)[1];
                }

                vPt_x_y[index].x = x; vPt_x_y[index].y = y;
                vPt_wx_wy[index].x = wx; vPt_wx_wy[index].y = wy;

                index ++;
            }
        }

        for (auto i = 0; i < N_index; i++){
            int x = vPt_x_y[i].x, y = vPt_x_y[i].y;
            float wx = vPt_wx_wy[i].x, wy = vPt_wx_wy[i].y;

            float error = GetPixelValue(mvImgPyr2[mLevel], pt.x + dx + wx, pt.y + dy + wy) + db
                    - (1.0f + dg) * GetPixelValue(mvImgPyr1[mLevel], pt.x + x, pt.y + y);

            // Jacobian
            Eigen::Vector4d J;
            if (mbInverse == false) {
                // For each new estimation, we calculate a new Jacobian
                float Ix = 0.5 * (GetPixelValue(mvImgPyr2[mLevel], pt.x + dx + wx + 1, pt.y + dy + wy)
                                  - GetPixelValue(mvImgPyr2[mLevel], pt.x + dx + wx - 1, pt.y + dy + wy));
                float Iy = 0.5 * (GetPixelValue(mvImgPyr2[mLevel], pt.x + dx + wx, pt.y + dy + wy + 1)
                                  - GetPixelValue(mvImgPyr2[mLevel], pt.x + dx + wx, pt.y + dy + wy - 1));
                float de_dg = - GetPixelValue(mvImgPyr1[mLevel], pt.x, pt.y);
                J = Eigen::Vector4d(Ix, Iy, de_dg, 1);
            } else if (iter == 0)  {
                // In inverse mode, J keeps same for all iterations.
                // Note this J does not change when dx, dy is updated, so we can store it
                // and only compute error.
                float Ix = 0.5 * (GetPixelValue(mvImgPyr1[mLevel], pt.x + x + 1, pt.y + y)
                                  - GetPixelValue(mvImgPyr1[mLevel], pt.x + x - 1, pt.y + y));
                float Iy = 0.5 * (GetPixelValue(mvImgPyr1[mLevel], pt.x + x, pt.y + y + 1)
                                  - GetPixelValue(mvImgPyr1[mLevel], pt.x + x, pt.y + y - 1));
                float de_dg = - GetPixelValue(mvImgPyr1[mLevel], pt.x, pt.y);
                J = Eigen::Vector4d(Ix, Iy, de_dg, 1);
            }

            vJ[i] = J;
            vE[i] = error;
            vFlag[i] = true;
        }


        cost = 0;
        for (size_t i = 0; i < N_index; i ++) {
            // Jacobian
            if (mbInverse == false) {
                J = vJ[i];
            }else if(iter == 0){
                J = vJ[i];
            }

            // compute H, b and set cost
            b += -J * vE[i];
            cost += vE[i] * vE[i];
            if (mbInverse == false || iter == 0) {
                H += J * J.transpose();
            }

        }

        // for gyro regularization penalty term
        if(bRegularizationPenalty)
        {
            double d = std::sqrt(dx * dx + dy * dy);
            double e_penalty = mLambda * mInvLogMaxDist * std::log(mAlpha * d + 1);

            double JdePenalty_dx = mLambda * mInvLogMaxDist * mAlpha / (mAlpha * d + 1) * (dx / d);
            double JdePenalty_dy = mLambda * mInvLogMaxDist * mAlpha / (mAlpha * d + 1) * (dy / d);
            double de_dlambda = e_penalty / mLambda;
            Eigen::Vector4d JPenalty(JdePenalty_dx, JdePenalty_dy, 0, 0);
            H += JPenalty * JPenalty.transpose();
            b += JPenalty * e_penalty;
            cost += e_penalty * e_penalty;
        }


        // Compute update
        // Eigen::Vector4d update = H.ldlt().solve(b); // solve: A * x = b
        Eigen::Vector4d update = H.llt().solve(b);

        // Check termination condition
        if (std::isnan(update[0])) {
            // sometimes occured when we have a black or white patch and H is irreversible
            succ = false;
            break;
        }

        if (iter > 0 && cost > lastCost)
            break;

        // Update dx, dy
        dx += update[0];
        dy += update[1];
        if (bConsiderIllumination){
            dg += update[2];
            db += update[3];
        }

        lastCost = cost;
        succ = true;

        // Check converge
        if (update.norm() < 1e-2)
            break; // converge

    } // end for: iter \in [0, iterations)

    mvPtPyr2Un[i] = pt + cv::Point2f(dx, dy);

    if (mLevel == 0){
        mvSuccess[i] = succ;
        mvPixelErrorsOfPatchMatched[i] = std::sqrt(lastCost * mWinSizeInv);
    }

    // calculate zero-normilized cross correlation
    if(mbCalculateNCC){
        if (bConsiderAffineDeformation){
            mvNcc[i] = NCC(mHalfPatchSize, mpMatcher->mImgGrayRef, mpMatcher->mImgGrayCur,
                           mvPtPyr1Un[i],  mvPtPyr2Un[i], mpMatcher->mvAffineDeformationMatrix[i]);
        }else {
            mvNcc[i] = NCC(mHalfPatchSize, mpMatcher->mImgGrayRef, mpMatcher->mImgGrayCur,
                           mvPtPyr1Un[i],  mvPtPyr2Un[i], cv::Mat());
        }
    }else {
        mvNcc[i] = 1;
    }
}

// Set mpMatcher
void PatchMatch::SetMatcher()
{
    mpMatcher->mvPtPredictAfterPatchMatched.resize(mN);
    mpMatcher->mvPtPredictAfterPatchMatchedUn.resize(mN);
    mpMatcher->mvStatusAfterPatchMatched.resize(mN);
    mpMatcher->mvPixelErrorsOfPatchMatched.resize(mN);
    mpMatcher->mvDistanceBetweenPredictedAndPatchMatched.resize(mN);
    mpMatcher->mvNccAfterPatchMatched.resize(mN);
    for (size_t i = 0; i < mN; i ++){
        mpMatcher->mvPtPredictAfterPatchMatched[i] = mvPtPyr2[i];
        mpMatcher->mvPtPredictAfterPatchMatchedUn[i] = mvPtPyr2Un[i];
        mpMatcher->mvStatusAfterPatchMatched[i] = mvSuccess[i];
        mpMatcher->mvPixelErrorsOfPatchMatched[i] = mvPixelErrorsOfPatchMatched[i];

        cv::Point2f pt_dist = mpMatcher->mvPtPredictUn[i] - mvPtPyr2Un[i];
        mpMatcher->mvDistanceBetweenPredictedAndPatchMatched[i] = std::sqrt(pt_dist.x * pt_dist.x + pt_dist.y * pt_dist.y);
        mpMatcher->mvNccAfterPatchMatched[i] = mvNcc[i];
    }
}

// Get a gray scale value from reference image (bi-linear interpolated)
inline float PatchMatch::GetPixelValue(const cv::Mat &img, float x, float y) const
{
    // boundary check
    if (x < 0) x = 0;
    if (y < 0) y = 0;
    if (x >= img.cols) x = img.cols - 1;
    if (y >= img.rows) y = img.rows - 1;

    uchar *data = &img.data[int(y) * img.step + int(x)];
    float xx = x - std::floor(x), yy = y - std::floor(y);
    float a = 1.0f - xx, b = 1.0f - yy;
    float pixel = b * (a * data[0] + xx * data[1])
            + yy  * (a * data[img.step]  + xx * data[img.step + 1]);

    return pixel;
}


void PatchMatch::DistortPoints(){
    if (mpMatcher->mDistCoef.at<float>(0) == 0.0)
        mvPtPyr2 = mvPtPyr2Un;
    else {
        mvPtPyr2.resize(mN);
        DistortVecPoints(mvPtPyr2Un, mvPtPyr2, mpMatcher->mK, mpMatcher->mDistCoef);
    }
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
float PatchMatch::NCC(int halfPathSize, const cv::Mat &ref, const cv::Mat &cur, const cv::Point2f &pt_ref, const cv::Point2f &pt_cur, const cv::Mat &warp_mat)
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
