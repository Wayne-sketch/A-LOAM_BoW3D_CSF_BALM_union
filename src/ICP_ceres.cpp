#include <iostream>
#include <ceres/ceres.h>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/core/eigen.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/SVD>
#include <ceres/ceres.h>
#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>
#include <g2o/types/sba/types_six_dof_expmap.h>
#include <chrono>
#include "rotation.h"
using namespace std;
using namespace cv;
using namespace Eigen;
struct ICPCeres
{
    ICPCeres ( Point3f uvw,Point3f xyz ) : _uvw(uvw),_xyz(xyz) {}
    // 残差的计算
    template <typename T>
    bool operator() (
            const T* const camera,     // 模型参数，有4维
            T* residual ) const     // 残差
    {
        T p[3];
        T point[3];
        point[0]=T(_xyz.x);
        point[1]=T(_xyz.y);
        point[2]=T(_xyz.z);
        AngleAxisRotatePoint(camera, point, p);//计算RP
        p[0] += camera[3]; p[1] += camera[4]; p[2] += camera[5];//相机坐标2
        residual[0] = T(_uvw.x)-p[0];
        residual[1] = T(_uvw.y)-p[1];
        residual[2] = T(_uvw.z)-p[2];
        return true;
    }
    static ceres::CostFunction* Create(const Point3f uvw,const Point3f xyz) {
        return (new ceres::AutoDiffCostFunction<ICPCeres, 3, 6>(
                new ICPCeres(uvw,xyz)));
    }
    const Point3f _uvw;
    const Point3f _xyz;
};
void find_feature_matches (
        const Mat& img_1, const Mat& img_2,
        std::vector<KeyPoint>& keypoints_1,
        std::vector<KeyPoint>& keypoints_2,
        std::vector< DMatch >& matches );
 
// 像素坐标转相机归一化坐标
Point2d pixel2cam ( const Point2d& p, const Mat& K );
 
void pose_estimation_3d3d (
        const vector<Point3f>& pts1,
        const vector<Point3f>& pts2,
        Mat& R, Mat& t
);
int main ( int argc, char** argv ) {
    if (argc != 5) {
        cout << "usage: pose_estimation_3d3d img1 img2 depth1 depth2" << endl;
        return 1;
    }
    double camera[6]={0,1,2,0,0,0};
    //-- 读取图像
    Mat img_1 = imread(argv[1], CV_LOAD_IMAGE_COLOR);
    Mat img_2 = imread(argv[2], CV_LOAD_IMAGE_COLOR);
 
    vector<KeyPoint> keypoints_1, keypoints_2;
    vector<DMatch> matches;
    find_feature_matches(img_1, img_2, keypoints_1, keypoints_2, matches);
    cout << "一共找到了" << matches.size() << "组匹配点" << endl;
 
    // 建立3D点
    Mat depth1 = imread(argv[3], CV_LOAD_IMAGE_UNCHANGED);       // 深度图为16位无符号数，单通道图像
    Mat depth2 = imread(argv[4], CV_LOAD_IMAGE_UNCHANGED);       // 深度图为16位无符号数，单通道图像
    Mat K = (Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);
    vector<Point3f> pts1, pts2;
 
    for (DMatch m:matches) {
        ushort d1 = depth1.ptr<unsigned short>(int(keypoints_1[m.queryIdx].pt.y))[int(keypoints_1[m.queryIdx].pt.x)];
        ushort d2 = depth2.ptr<unsigned short>(int(keypoints_2[m.trainIdx].pt.y))[int(keypoints_2[m.trainIdx].pt.x)];
        if (d1 == 0 || d2 == 0)   // bad depth
            continue;
        Point2d p1 = pixel2cam(keypoints_1[m.queryIdx].pt, K);
        Point2d p2 = pixel2cam(keypoints_2[m.trainIdx].pt, K);
        float dd1 = float(d1) / 5000.0;
        float dd2 = float(d2) / 5000.0;
        pts1.push_back(Point3f(p1.x * dd1, p1.y * dd1, dd1));
        pts2.push_back(Point3f(p2.x * dd2, p2.y * dd2, dd2));
    }
 
    cout << "3d-3d pairs: " << pts1.size() << endl;
    Mat R, t;
    pose_estimation_3d3d(pts1, pts2, R, t);
    cout << "ICP via SVD results: " << endl;
    cout << "R = " << R << endl;
    cout << "t = " << t << endl;
    cout << "R_inv = " << R.t() << endl;
    cout << "t_inv = " << -R.t() * t << endl;
 
    cout << "calling bundle adjustment" << endl;
    // verify p1 = R*p2 + t
    for ( int i=0; i<5; i++ )
    {
        cout<<"p1 = "<<pts1[i]<<endl;
        cout<<"p2 = "<<pts2[i]<<endl;
        cout<<"(R*p2+t) = "<<
            R * (Mat_<double>(3,1)<<pts2[i].x, pts2[i].y, pts2[i].z) + t
            <<endl;
        cout<<endl;
    }
    ceres::Problem problem;
    for (int i = 0; i < pts2.size(); ++i)
    {
        ceres::CostFunction* cost_function =
                ICPCeres::Create(pts2[i],pts1[i]);
        problem.AddResidualBlock(cost_function,
                                 NULL /* squared loss */,
                                 camera);
    }
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.FullReport() << "\n";
 
    Mat R_vec = (Mat_<double>(3,1) << camera[0],camera[1],camera[2]);//数组转cv向量
    Mat R_cvest;
    Rodrigues(R_vec,R_cvest);//罗德里格斯公式，旋转向量转旋转矩阵
    cout<<"R_cvest="<<R_cvest<<endl;
    Eigen::Matrix<double,3,3> R_est;
    cv2eigen(R_cvest,R_est);//cv矩阵转eigen矩阵
    cout<<"R_est="<<R_est<<endl;
    Eigen::Vector3d t_est(camera[3],camera[4],camera[5]);
    cout<<"t_est="<<t_est<<endl;
    Eigen::Isometry3d T(R_est);//构造变换矩阵与输出
    T.pretranslate(t_est);
    cout<<T.matrix()<<endl;
 
    return 0;
}
void find_feature_matches ( const Mat& img_1, const Mat& img_2,
                            std::vector<KeyPoint>& keypoints_1,
                            std::vector<KeyPoint>& keypoints_2,
                            std::vector< DMatch >& matches )
{
    //-- 初始化
    Mat descriptors_1, descriptors_2;
    // used in OpenCV3
    Ptr<FeatureDetector> detector = ORB::create();
    Ptr<DescriptorExtractor> descriptor = ORB::create();
    // use this if you are in OpenCV2
    // Ptr<FeatureDetector> detector = FeatureDetector::create ( "ORB" );
    // Ptr<DescriptorExtractor> descriptor = DescriptorExtractor::create ( "ORB" );
    Ptr<DescriptorMatcher> matcher  = DescriptorMatcher::create("BruteForce-Hamming");
    //-- 第一步:检测 Oriented FAST 角点位置
    detector->detect ( img_1,keypoints_1 );
    detector->detect ( img_2,keypoints_2 );
 
    //-- 第二步:根据角点位置计算 BRIEF 描述子
    descriptor->compute ( img_1, keypoints_1, descriptors_1 );
    descriptor->compute ( img_2, keypoints_2, descriptors_2 );
 
    //-- 第三步:对两幅图像中的BRIEF描述子进行匹配，使用 Hamming 距离
    vector<DMatch> match;
    // BFMatcher matcher ( NORM_HAMMING );
    matcher->match ( descriptors_1, descriptors_2, match );
 
    //-- 第四步:匹配点对筛选
    double min_dist=10000, max_dist=0;
 
    //找出所有匹配之间的最小距离和最大距离, 即是最相似的和最不相似的两组点之间的距离
    for ( int i = 0; i < descriptors_1.rows; i++ )
    {
        double dist = match[i].distance;
        if ( dist < min_dist ) min_dist = dist;
        if ( dist > max_dist ) max_dist = dist;
    }
 
    printf ( "-- Max dist : %f \n", max_dist );
    printf ( "-- Min dist : %f \n", min_dist );
 
    //当描述子之间的距离大于两倍的最小距离时,即认为匹配有误.但有时候最小距离会非常小,设置一个经验值30作为下限.
    for ( int i = 0; i < descriptors_1.rows; i++ )
    {
        if ( match[i].distance <= max ( 2*min_dist, 30.0 ) )
        {
            matches.push_back ( match[i] );
        }
    }
}
 
Point2d pixel2cam ( const Point2d& p, const Mat& K )
{
    return Point2d
            (
                    ( p.x - K.at<double> ( 0,2 ) ) / K.at<double> ( 0,0 ),
                    ( p.y - K.at<double> ( 1,2 ) ) / K.at<double> ( 1,1 )
            );
}
 
void pose_estimation_3d3d (
        const vector<Point3f>& pts1,
        const vector<Point3f>& pts2,
        Mat& R, Mat& t
)
{
    Point3f p1, p2;     // center of mass
    int N = pts1.size();
    for ( int i=0; i<N; i++ )
    {
        p1 += pts1[i];
        p2 += pts2[i];
    }
    p1 = Point3f( Vec3f(p1) /  N);
    p2 = Point3f( Vec3f(p2) / N);
    vector<Point3f>     q1 ( N ), q2 ( N ); // remove the center
    for ( int i=0; i<N; i++ )
    {
        q1[i] = pts1[i] - p1;
        q2[i] = pts2[i] - p2;
    }
 
    // compute q1*q2^T
    Eigen::Matrix3d W = Eigen::Matrix3d::Zero();
    for ( int i=0; i<N; i++ )
    {
        W += Eigen::Vector3d ( q1[i].x, q1[i].y, q1[i].z ) * Eigen::Vector3d ( q2[i].x, q2[i].y, q2[i].z ).transpose();
    }
    cout<<"W="<<W<<endl;
 
    // SVD on W
    Eigen::JacobiSVD<Eigen::Matrix3d> svd ( W, Eigen::ComputeFullU|Eigen::ComputeFullV );
    Eigen::Matrix3d U = svd.matrixU();
    Eigen::Matrix3d V = svd.matrixV();
    cout<<"U="<<U<<endl;
    cout<<"V="<<V<<endl;
 
    Eigen::Matrix3d R_ = U* ( V.transpose() );
    Eigen::Vector3d t_ = Eigen::Vector3d ( p1.x, p1.y, p1.z ) - R_ * Eigen::Vector3d ( p2.x, p2.y, p2.z );
 
    // convert to cv::Mat
    R = ( Mat_<double> ( 3,3 ) <<
                               R_ ( 0,0 ), R_ ( 0,1 ), R_ ( 0,2 ),
            R_ ( 1,0 ), R_ ( 1,1 ), R_ ( 1,2 ),
            R_ ( 2,0 ), R_ ( 2,1 ), R_ ( 2,2 )
    );
    t = ( Mat_<double> ( 3,1 ) << t_ ( 0,0 ), t_ ( 1,0 ), t_ ( 2,0 ) );
}