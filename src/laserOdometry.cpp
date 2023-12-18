// This is an advanced implementation of the algorithm described in the following paper:
//   J. Zhang and S. Singh. LOAM: Lidar Odometry and Mapping in Real-time.
//     Robotics: Science and Systems Conference (RSS). Berkeley, CA, July 2014. 

// Modifier: Tong Qin               qintonguav@gmail.com
// 	         Shaozu Cao 		    saozu.cao@connect.ust.hk


// Copyright 2013, Ji Zhang, Carnegie Mellon University
// Further contributions copyright (c) 2016, Southwest Research Institute
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from this
//    software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#include <cmath>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <geometry_msgs/PoseStamped.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl_conversions/pcl_conversions.h>
#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>
#include <eigen3/Eigen/Dense>
#include <mutex>
#include <queue>
#include "aloam_velodyne/common.h"
#include "aloam_velodyne/tic_toc.h"
#include "lidarFactor.hpp"
#include<fstream>
#include <iostream>
#include <sstream>

//LinK3D
#include <eigen3/Eigen/Dense>
#include <pcl/filters/extract_indices.h>
#include <sstream>
#include <iomanip>
#include "BoW3D/LinK3D_Extractor.h"
#include "BoW3D/BoW3D.h"
//智能指针头文件
#include <memory>
//link3d 关键点帧间ICP匹配
#include "ICP_ceres/ICP_ceres.h"

using namespace BoW3D;
//Parameters of LinK3D
//雷达扫描线数
int nScans = 64; //Number of LiDAR scan lines
//雷达扫描周期
float scanPeriod_LinK3D = 0.1; 
//最小点云距离范围，距离原点小于该阈值的点将被删除
float minimumRange = 0.1;
//判断区域内某点和聚类点均值距离，以及在x，y轴上的距离
float distanceTh = 0.4;
//描述子匹配所需的最低分数阈值 ，描述子匹配分数低于此分数的两个关键点不匹配
int matchTh = 6;
//Parameters of BoW3D
//比率阈值。
float thr = 3.5;
//频率阈值。
int thf = 5;
//每帧添加或检索的特征数量。
int num_add_retrieve_features = 5;
pcl::PointCloud<pcl::PointXYZ>::Ptr plaserCloudIn_LinK3D(new pcl::PointCloud<pcl::PointXYZ>); //LinK3D 当前帧点云
//帧间ICP匹配的位姿变换
Eigen::Matrix3d RelativeR;
Eigen::Vector3d Relativet;


using namespace std;
// ofstream os_pose;
#define DISTORTION 0
int corner_correspondence = 0, plane_correspondence = 0;
//编译时求值
constexpr double SCAN_PERIOD = 0.1;
constexpr double DISTANCE_SQ_THRESHOLD = 25;
constexpr double NEARBY_SCAN = 2.5;

int skipFrameNum = 5;
bool systemInited = false;

double timeCornerPointsSharp = 0;
double timeCornerPointsLessSharp = 0;
double timeSurfPointsFlat = 0;
double timeSurfPointsLessFlat = 0;
double timeLaserCloudFullRes = 0;

// //link3d
// double timeLaserCloudLink3d_odom = 0;

// 创建KD-Tree对象
pcl::KdTreeFLANN<pcl::PointXYZI>::Ptr kdtreeCornerLast(new pcl::KdTreeFLANN<pcl::PointXYZI>());
pcl::KdTreeFLANN<pcl::PointXYZI>::Ptr kdtreeSurfLast(new pcl::KdTreeFLANN<pcl::PointXYZI>());

// 接受配准端发布的点云
pcl::PointCloud<PointType>::Ptr cornerPointsSharp(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr cornerPointsLessSharp(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr surfPointsFlat(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr surfPointsLessFlat(new pcl::PointCloud<PointType>());

pcl::PointCloud<PointType>::Ptr laserCloudCornerLast(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr laserCloudSurfLast(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr laserCloudFullRes(new pcl::PointCloud<PointType>());
// //link3d
// pcl::PointCloud<PointType>::Ptr laserCloudLink3d(new pcl::PointCloud<PointType>());

int laserCloudCornerLastNum = 0;
int laserCloudSurfLastNum = 0;

// Transformation from current frame to world frame
// 全局变量，被不断更新
Eigen::Quaterniond q_w_curr(1, 0, 0, 0);
Eigen::Vector3d t_w_curr(0, 0, 0);

// 帧间位姿：q_curr_last(x, y, z, w), t_curr_last
double para_q[4] = {0, 0, 0, 1};
double para_t[3] = {0, 0, 0};
// 内存映射，相当于引用
Eigen::Map<Eigen::Quaterniond> q_last_curr(para_q);
Eigen::Map<Eigen::Vector3d> t_last_curr(para_t);

std::queue<sensor_msgs::PointCloud2ConstPtr> cornerSharpBuf;
std::queue<sensor_msgs::PointCloud2ConstPtr> cornerLessSharpBuf;
std::queue<sensor_msgs::PointCloud2ConstPtr> surfFlatBuf;
std::queue<sensor_msgs::PointCloud2ConstPtr> surfLessFlatBuf;
std::queue<sensor_msgs::PointCloud2ConstPtr> fullPointsBuf;

// //link3d
// std::queue<sensor_msgs::PointCloud2ConstPtr> odom_link3dPointsBuf;

std::mutex mBuf;


/**
 * @brief 除去距离lidar过近的点
 * 
 * @tparam PointT 
 * @param[in] cloud_in 输入点云
 * @param[out] cloud_out 输出点云
 * @param[in] thres 距离阈值
 */
template <typename PointT>
void removeClosedPointCloud(const pcl::PointCloud<PointT> &cloud_in,
                              pcl::PointCloud<PointT> &cloud_out, float thres)
{
    // 入参和出参的地址一样
    if (&cloud_in != &cloud_out)
    {
        cloud_out.header = cloud_in.header;
        cloud_out.points.resize(cloud_in.points.size());
    }

    size_t j = 0;
 
    for (size_t i = 0; i < cloud_in.points.size(); ++i)
    {
        // 如果 x² + y² +　z² < th ，则跳过
        if (cloud_in.points[i].x * cloud_in.points[i].x + cloud_in.points[i].y * cloud_in.points[i].y + cloud_in.points[i].z * cloud_in.points[i].z < thres * thres)
            continue;
        cloud_out.points[j] = cloud_in.points[i];
        j++;
    }

    // 如果有点被去除了
    if (j != cloud_in.points.size())
    {
        // 不在声明额外的变量，节约空间
        cloud_out.points.resize(j);
    }
    // ROS中的点云高度为点云的线数，点云的宽度为该线数对应的点云个数
    // 经过remove操作之后没有办法保证每个每根线上都有点云，所以操作完后的点云为无序的点云，高度为1，宽度为所有点云的个数
    cloud_out.height = 1;
    // 有效点云个数
    cloud_out.width = static_cast<uint32_t>(j);
    cloud_out.is_dense = true;
}

/*
计算当前位置到起始位置的坐标变换，将当前的点转换到起始点位置，
即将100毫秒内的一帧的点云，统一到一个时间点的坐标系上
P_start = T_curr2start * P_curr
如何获取 T_curr2start:
1.如果有高频里程计，可以方便的获取每个点相对于起始扫描的位姿
2.如果有imu，可以方便的求出每个点对起始点的旋转
3.如果没有其他里程计，可以使用匀速运动模型，使用上一个帧间里程计的结果作为当前两帧之间的
运动，同时假设当前帧也是匀速运动，也可以估计出每个点相对于起始时刻的位姿

使用激光点云的强度构建图像，根据两帧数据图像之间的位移估计地面车辆的线速度和角速度
有了速度之后，再根据两帧之间的时间间隔，乘以速度，就可以得到两帧的位移增量和速度增量
在加上上一时刻的后验位姿就可以得到当前

k-1 到 k 帧 和 k到k+1帧的运动是一至的,用k-1到k帧的位姿变换当做k到k+1帧的位姿变换, 可以求到k到k+1帧的每个点的位姿变换
*/

// 输入当前点的地址，当前点无法修改
// 把所有的点补偿到起始时刻，输出的是一个带有强度信息的三维点
void TransformToStart(PointType const *const pi, PointType *const po)
{
    // interpolation ratio
    double s;
    // 默认为0
    if (DISTORTION)
    {
        // s = pi->timestamp / SCAN_PERIOD;
        // 当前点相对与起始时刻的时间差(小数部分)/100ms = 比例
        s = (pi->intensity - int(pi->intensity)) / SCAN_PERIOD;
    }
    else
    {
        // 由于kitti数据集上的lidar已经做过运动补偿，因此这里就不做具体补偿了DISTORTION = false
        s = 1.0;// s = 1s说明全部补偿到点云结束的时刻
    }

    /*
    s = 1 ->  T_curr2start =  T_end2start;
    做一个匀速模型假设,即上一帧的位姿变换,就是这帧的位姿变换
    以此来求出输入点坐标系到起始时刻点坐标系的位姿变换,通过上面求的时间占比s，这里把位姿变换 分解成了 旋转 + 平移 的方式
    由于四元数是一个超复数,不是简单的乘法,求它的占比用的 Eigen的slerp函数(球面线性插值)
    上一帧到当前帧的位姿变化量 为当前帧结束点的位姿到起始点的位姿的变化量： q_last_curr =  q_curr2last = R_end2start
    R_curr2start  = R_end2start(100ms) * 差值比例为s
    */
    //q_last_curr是当前帧初始时刻到上一帧初始时刻的位姿变换，根据匀速模型假设，等于当前帧结束点到初始点的位姿变换
    //s这个比例越大，插值结果越接近q_last_curr
    Eigen::Quaterniond q_point_last = Eigen::Quaterniond::Identity().slerp(s, q_last_curr);
    // 平移增量 * s
    Eigen::Vector3d t_point_last = s * t_last_curr;
    Eigen::Vector3d point(pi->x, pi->y, pi->z);
    
    // 当前帧点云转换到上一帧坐标系后的点：P_last = T_curr2last * P_curr
    Eigen::Vector3d un_point = q_point_last * point + t_point_last;

    po->x = un_point.x();
    po->y = un_point.y();
    po->z = un_point.z();
    po->intensity = pi->intensity;
}

// transform all lidar points to the start of the next frame
// 把所有的点补偿到结束时刻
void TransformToEnd(PointType const *const pi, PointType *const po)
{
    // undistort point first
    pcl::PointXYZI un_point_tmp;
    // 把所有的点补偿到起始时刻
    TransformToStart(pi, &un_point_tmp);
    // 保存去畸变后的点 P_start
    Eigen::Vector3d un_point(un_point_tmp.x, un_point_tmp.y, un_point_tmp.z);
    /*
    把所有的点补偿到结束时刻:
    p_start = P_end * R_end2start + t_end2start 
    P_end = (p_start - t_end2start) * R_end2start¯¹
    */
    Eigen::Vector3d point_end = q_last_curr.inverse() * (un_point - t_last_curr);

    po->x = point_end.x();
    po->y = point_end.y();
    po->z = point_end.z();

    //Remove distortion time info
    po->intensity = int(pi->intensity);
}

//都是把点存入buffer
void laserCloudSharpHandler(const sensor_msgs::PointCloud2ConstPtr &cornerPointsSharp2)
{
    mBuf.lock();
    cornerSharpBuf.push(cornerPointsSharp2);
    mBuf.unlock();
}

void laserCloudLessSharpHandler(const sensor_msgs::PointCloud2ConstPtr &cornerPointsLessSharp2)
{
    mBuf.lock();
    cornerLessSharpBuf.push(cornerPointsLessSharp2);
    mBuf.unlock();
}

void laserCloudFlatHandler(const sensor_msgs::PointCloud2ConstPtr &surfPointsFlat2)
{
    mBuf.lock();
    surfFlatBuf.push(surfPointsFlat2);
    mBuf.unlock();
}

void laserCloudLessFlatHandler(const sensor_msgs::PointCloud2ConstPtr &surfPointsLessFlat2)
{
    mBuf.lock();
    surfLessFlatBuf.push(surfPointsLessFlat2);
    mBuf.unlock();
}

//receive all point cloud
void laserCloudFullResHandler(const sensor_msgs::PointCloud2ConstPtr &laserCloudFullRes2)
{
    mBuf.lock();
    fullPointsBuf.push(laserCloudFullRes2);
    mBuf.unlock();
}

// //link3d
// void laserCloudLink3d_odomHandler(const sensor_msgs::PointCloud2ConstPtr &laserCloudLink3d_odom)
// {
//     mBuf.lock();
//     odom_link3dPointsBuf.push(laserCloudLink3d_odom);
//     mBuf.unlock();
// }

int main(int argc, char **argv)
{

    ros::init(argc, argv, "laserOdometry");
    ros::NodeHandle nh;
    //好像是保存轨迹用的？？
	// os_pose.open("/home/wb/ALOAM_Noted_WS/wb/pose/f2f_Poses.txt", std::fstream::out);

    // 1
    nh.param<int>("mapping_skip_frame", skipFrameNum, 2);
    // 10hz
    printf("Mapping %d Hz \n", 10 / skipFrameNum);
    // ------------------------ 输入 ---------------------
    // 订阅极大边线点，并传入回调函数中处理，消息队列的长度为100
    ros::Subscriber subCornerPointsSharp = nh.subscribe<sensor_msgs::PointCloud2>("/laser_cloud_sharp", 100, laserCloudSharpHandler);
    // 订阅次极大边线点
    ros::Subscriber subCornerPointsLessSharp = nh.subscribe<sensor_msgs::PointCloud2>("/laser_cloud_less_sharp", 100, laserCloudLessSharpHandler);
    // 订阅极小平面点
    ros::Subscriber subSurfPointsFlat = nh.subscribe<sensor_msgs::PointCloud2>("/laser_cloud_flat", 100, laserCloudFlatHandler);
    // 订阅次极小平面点
    ros::Subscriber subSurfPointsLessFlat = nh.subscribe<sensor_msgs::PointCloud2>("/laser_cloud_less_flat", 100, laserCloudLessFlatHandler);
    // 当前点云 去除nan点后未作任何处理
    ros::Subscriber subLaserCloudFullRes = nh.subscribe<sensor_msgs::PointCloud2>("/velodyne_cloud_2", 100, laserCloudFullResHandler);

    // //link3d
    // ros::Subscriber subLaserCloudLink3d_odom = nh.subscribe<sensor_msgs::PointCloud2>("/laser_cloud_LinK3D", 100, laserCloudLink3d_odomHandler);

    // ------------------------ 输出 ---------------------
    // 发布给mapping模块
    ros::Publisher pubLaserCloudCornerLast = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_corner_last", 100);
    ros::Publisher pubLaserCloudSurfLast = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_surf_last", 100);
    ros::Publisher pubLaserCloudFullRes = nh.advertise<sensor_msgs::PointCloud2>("/velodyne_cloud_3", 100);

    //LinK3D 聚类关键点发布
    ros::Publisher pubLaserCloud_LinK3D = nh.advertise<sensor_msgs::PointCloud2>("/odomLink3d_cloud", 100);
    //保存上一帧的 link3d Frame用于帧间匹配
    Frame* pPreviousFrame = nullptr;
    //存当前点云中的聚类后的关键点
    pcl::PointCloud<pcl::PointXYZI>::Ptr AggregationKeypoints_LinK3D(new pcl::PointCloud<pcl::PointXYZI>);
    // //link3d
    // ros::Publisher pubLaserCloudLink3d_odom = nh.advertise<sensor_msgs::PointCloud2>("/odomLink3d_cloud", 100);

    // 发布里程计数据(位姿轨迹)给后端，后端接收 当前帧到初始帧的位姿
    ros::Publisher pubLaserOdometry = nh.advertise<nav_msgs::Odometry>("/laser_odom_to_init", 100);
    // 发布前端里程计的高频低精 位姿轨迹
    ros::Publisher pubLaserPath = nh.advertise<nav_msgs::Path>("/laser_odom_path", 100);
    nav_msgs::Path laserPath;


    int frameCount = 0;
    // 高频低精 100hz ， 最好处理一帧需要10毫秒
    ros::Rate rate(100);
    
    // 不断循环直到ctrl c
    while (ros::ok())
    {
        /*
        ros::spin() 和 ros::spinOnce() 区别及详解:
            其实看函数名也能理解个差不多，一个是一直调用回调函数；
            ros::spin() 在调用后不会再返回，也就是你的主程序到这儿就不往下执行了。
            另一个是只调用一次回调函数，如果还想再调用，就需要加上循环了，后者在调用后还可以继续执行之后的程序

        Subscriber每接收到一个消息(一帧)，就会触发一次spinOnce，其滴调用回调函数
        如果spinOnce下面的代码没有执行完，就不会触发下一次回调，就不会丢帧

        对于有些传输特别快的消息，尤其需要注意合理控制消息池大小和ros::spinOnce()执行频率; 
        比如消息送达频率为10Hz, ros::spinOnce()的调用回调函数频率为5Hz，那么消息池的大小就一定要大于2，才能保证数据不丢失，无延迟。
        */
        // 每调用一次回调函数就执行以下代码一次
        ros::spinOnce();// 如果改为ros::spin则下面的代码就不会被执行

        // 确保订阅的五个消息队列(五种点云)都有，有一个队列为空都不行
        if (!cornerSharpBuf.empty() && !cornerLessSharpBuf.empty() &&
            !surfFlatBuf.empty() && !surfLessFlatBuf.empty() && !fullPointsBuf.empty())
        {
            // 取出每一个队列最早的时间戳
            timeCornerPointsSharp = cornerSharpBuf.front()->header.stamp.toSec();
            timeCornerPointsLessSharp = cornerLessSharpBuf.front()->header.stamp.toSec();
            timeSurfPointsFlat = surfFlatBuf.front()->header.stamp.toSec();
            timeSurfPointsLessFlat = surfLessFlatBuf.front()->header.stamp.toSec();
            timeLaserCloudFullRes = fullPointsBuf.front()->header.stamp.toSec();

            // //link3d
            // timeLaserCloudLink3d_odom = odom_link3dPointsBuf.front()->header.stamp.toSec();

            // 判断是否是同一帧，同一帧的时间戳相同
            if (timeCornerPointsSharp != timeLaserCloudFullRes ||
                timeCornerPointsLessSharp != timeLaserCloudFullRes ||
                timeSurfPointsFlat != timeLaserCloudFullRes ||
                timeSurfPointsLessFlat != timeLaserCloudFullRes)
            {
                printf("unsync messeage!");
                ROS_BREAK();
            }

            mBuf.lock();
            // 每次进来前，都要记得清除
            cornerPointsSharp->clear();
            pcl::fromROSMsg(*cornerSharpBuf.front(), *cornerPointsSharp);
            // 用完即丢
            cornerSharpBuf.pop();
            // 清除上一帧
            cornerPointsLessSharp->clear();
            pcl::fromROSMsg(*cornerLessSharpBuf.front(), *cornerPointsLessSharp);
            cornerLessSharpBuf.pop();

            surfPointsFlat->clear();
            pcl::fromROSMsg(*surfFlatBuf.front(), *surfPointsFlat);
            surfFlatBuf.pop();

            surfPointsLessFlat->clear();
            pcl::fromROSMsg(*surfLessFlatBuf.front(), *surfPointsLessFlat);
            surfLessFlatBuf.pop();

            laserCloudFullRes->clear();
            pcl::fromROSMsg(*fullPointsBuf.front(), *laserCloudFullRes);
            //link3d 把当前帧点云单独存一个用作link3d处理
            pcl::fromROSMsg(*fullPointsBuf.front(), *plaserCloudIn_LinK3D);
            //测试
            // cout << plaserCloudIn_LinK3D->points[0].x;
            fullPointsBuf.pop();
            // //link3d
            // laserCloudLink3d->clear();
            // pcl::fromROSMsg(*odom_link3dPointsBuf.front(), *laserCloudLink3d);
            // odom_link3dPointsBuf.pop();
            //这个锁很关键
            mBuf.unlock();

            TicToc t_whole;
            // initializing，第一帧，这个条件语句后面会把当前帧点云存到上一帧点云，并创建KDtree
            if (!systemInited)
            {
                // link3d 去除近距离点云
                removeClosedPointCloud(*plaserCloudIn_LinK3D, *plaserCloudIn_LinK3D, 0.1);
                // link3d 提取器
                BoW3D::LinK3D_Extractor* pLinK3dExtractor(new BoW3D::LinK3D_Extractor(nScans, scanPeriod_LinK3D, minimumRange, distanceTh, matchTh)); 
                //第一帧直接赋值给pPreviousFrame
                pPreviousFrame = new Frame(pLinK3dExtractor, plaserCloudIn_LinK3D);
                systemInited = true;
                std::cout << "Initialization finished \n";
            }
            else// 第二帧之后进来
            {
                // ----------------------- 雷达里程计 -------------------------
                // 当前帧中，极大边线点的个数
                int cornerPointsSharpNum = cornerPointsSharp->points.size();
                // 当前帧中，极小平面点的个数
                int surfPointsFlatNum = surfPointsFlat->points.size();
                // 计算优化的时间
                TicToc t_opt;
                // 把当前帧明显的角点与面点，与上一帧所有的角点与面点进行比较，建立约束
                // 点到线以及点到面的非线性优化，迭代2次（选择当前优化位姿的特征点匹配，并优化位姿（4次迭代），然后重新选择特征点匹配并优化）

                //link3d 这里需要开始进行修改，利用link3d提取的关键点（质心）进行帧间ICP匹配，仅优化位姿
                // LinK3D
                removeClosedPointCloud(*plaserCloudIn_LinK3D, *plaserCloudIn_LinK3D, 0.1);
                //在这里植入LinK3D，把接收到的点云数据用LinK3D提取边缘点和描述子，发布关键点数据，打印输出描述子
                //LinK3D提取器
                BoW3D::LinK3D_Extractor* pLinK3dExtractor(new BoW3D::LinK3D_Extractor(nScans, scanPeriod_LinK3D, minimumRange, distanceTh, matchTh)); 
                //创建点云帧,该函数中利用LinK3D仿函数执行了提取边缘点，聚类，计算描述子的操作
                Frame* pCurrentFrame_LinK3D(new Frame(pLinK3dExtractor, plaserCloudIn_LinK3D));
                //此时pCurrentFrame_LinK3D这个类指针中包含了边缘点，聚类，描述子的信息
        //测试 输出关键点数量和第一个关键点信息 正常输出 
        // cout << "------------------------" << endl << "关键点数量:" << pCurrentFrame_LinK3D->mvAggregationKeypoints.size();
        // cout << "第一个关键点信息x坐标" << pCurrentFrame_LinK3D->mvAggregationKeypoints[0].x;
                //存当前点云中的聚类后的关键点
                AggregationKeypoints_LinK3D->points.insert(AggregationKeypoints_LinK3D->points.end(), pCurrentFrame_LinK3D->mvAggregationKeypoints.begin(), pCurrentFrame_LinK3D->mvAggregationKeypoints.end());
        //测试 输出点云中信息 也能正常输出
        // cout << "------------------------" << endl << "关键点数量:" << AggregationKeypoints_LinK3D->points.size();
        // cout << "第一个关键点信息x坐标" << AggregationKeypoints_LinK3D->points[0].x;
                // 2.对描述子进行匹配 3.使用匹配对进行帧间icp配准 pPreviousFrame是上一个link3d Frame帧 pCurrentFrame_LinK3D是当前link3d Frame帧
                // 获取上一帧和当前帧之间的匹配索引
                 vector<pair<int, int>> vMatchedIndex;  
                pLinK3dExtractor->match(pCurrentFrame_LinK3D->mvAggregationKeypoints, pPreviousFrame->mvAggregationKeypoints, pCurrentFrame_LinK3D->mDescriptors, pPreviousFrame->mDescriptors, vMatchedIndex);
                //仿照BoW3D函数写一个帧间ICP匹配函数求出R,t
                int returnValue = 0;
                // 进行帧间ICP匹配 求当前帧到上一帧的位姿变换
                // 这里求的R t是当前帧点云到上一帧点云的位姿变换
                returnValue = pose_estimation_3d3d(pCurrentFrame_LinK3D, pPreviousFrame, vMatchedIndex, RelativeR, Relativet, pLinK3dExtractor);
                //至此获得了当前帧点云到上一帧点云的位姿变换

                //当前帧Frame用完以后，赋值给上一帧Frame,赋值前先把要丢掉的帧内存释放
                //这里Frame里有成员指针，析构函数里delete成员指针
                delete pPreviousFrame;
                pPreviousFrame = pCurrentFrame_LinK3D;
                //LinK3D 植入结束

                // for (size_t opti_counter = 0; opti_counter < 2; ++opti_counter)
                // {
                //     corner_correspondence = 0;
                //     plane_correspondence = 0;

                //     //ceres::LossFunction *loss_function = NULL;
                //     // 定义一下ceres的核函数，使用Huber核函数来减少外点的影响，即去除outliers，残差小于0.1则不用核函数
                //     ceres::LossFunction *loss_function = new ceres::HuberLoss(0.1);
                //     // H△x = g --> △x ，x + △x
                //     // 由于旋转不满足一般意义的加法，因此这里使用ceres自带的local param
                //     ceres::LocalParameterization *q_parameterization = new ceres::EigenQuaternionParameterization();
                    
                //     ceres::Problem::Options problem_options;
                //     // 实例化求解最优化问题
                //     ceres::Problem problem(problem_options);
                //     // para_q和para_t是数组，ceres只会对double数据进行处理，运算的速度比较快
                //     // 添加待优化变量q(姿态)，到参数块，维度为4；
                //     problem.AddParameterBlock(para_q, 4, q_parameterization);
                //     // 添加待优化变量t(平移)，到参数块，维度为3；
                //     problem.AddParameterBlock(para_t, 3);


                //     // ------------------------------ 寻找约束 ------------------------------
                //     // 去运动畸变后的角点
                //     pcl::PointXYZI pointSel;
                //     // 找到的点的ID
                //     std::vector<int> pointSearchInd;
                //     // 当前点 到 最近点的距离
                //     std::vector<float> pointSearchSqDis;
                //     // 计算寻找关联点的时间
                //     TicToc t_data;
                //     // find correspondence for corner features
                //     /*
                //     基于最近邻原理建立corner特征点（边线点）之间的关联，每一个极大边线点去上一帧的次极大边线
                //     点中找匹配；采用边线点匹配方法:假如在第k+1帧中发现了边线点i，通过KD-tree查询在第k帧中的最
                //     近邻点j，查询j的附近扫描线上的最近邻点l，j与l相连形成一条直线l-j，让点i与这条直线的距离最短。
                //     构建一个非线性优化问题：以点i与直线lj的距离为代价函数，以位姿变换T(四元数表示旋转+位移t)为优化变量。

                //     把点云补偿到起始时刻，把过去0.1s收集到的点云，统一到一个时间点上
                //     P_start = T_curr2start * p_curr
                //     当前点相对于起始时刻的位姿：T_curr2start的获取方式：
                //         1.轮速计
                //         2.imu：当前点相对于起始时刻的姿态，平移的可信度不高，所以有GNSS
                //         3.匀速运动模型：使用上一个帧间里程计的结果，当前作为两帧之间的运动
                //     */
                //     // 寻找角点约束，遍历当前帧曲率大的角点 先进行线点的匹配
                //     for (int i = 0; i < cornerPointsSharpNum; ++i)
                //     {
                //         // 为了进行ICP，将当前帧点云转换到上一帧坐标系下，统一坐标系
                //         TransformToStart(&(cornerPointsSharp->points[i]), &pointSel);

                //         /*
                //         brief Search for k-nearest neighbors for the given query point.  //搜素给定点的K近邻。
                //             参数1： 给定的查询点。
                //             参数2： 要搜索的近邻点的数量
                //             参数3： 输出的k个近邻点索引（上一帧构建kdtree后的近邻点的id）
                //             参数4： 输出查询点到邻近点的平方距离
                //             返回值：返回找到的近邻点的数量
                //         在上一帧所有角点(弱角点)构成的kdtree中寻找距离当前帧最近的一个点   因为前面有初始化的判断 所有 第二帧肯定有上一帧
                //         */
                //         kdtreeCornerLast->nearestKSearch(pointSel, 1, pointSearchInd, pointSearchSqDis);

                //         int closestPointInd = -1, minPointInd2 = -1;
                //         // 帧间的距离不能过大，因为是匀速运动模型
                //         if (pointSearchSqDis[0] < DISTANCE_SQ_THRESHOLD)
                //         {
                //             closestPointInd = pointSearchInd[0];

                //             // 整数部分：线束id，线束信息藏在intensity的整数部分
                //             int closestPointScanID = int(laserCloudCornerLast->points[closestPointInd].intensity);
                //             double minPointSqDis2 = DISTANCE_SQ_THRESHOLD;

                //             // search in the direction of increasing scan line
                //             // 两点构成一线，所以要寻找第二个角点
                //             // 在刚刚角点id上下分别继续寻找，目的是寻找最近的角点，由于其按照线束排列，所以就是向上寻找
                //             for (int j = closestPointInd + 1; j < (int)laserCloudCornerLast->points.size(); ++j)
                //             {
                //                 // if in the same scan line, continue
                //                 // 第二个点与第一个最近点不在同一根线束的
                //                 if (int(laserCloudCornerLast->points[j].intensity) <= closestPointScanID)
                //                     continue;

                //                 // if not in nearby scans, end the loop
                //                 // 要求找到的线束距离第一个点的线束不能太远，如果当前的点不满足，后面的点更不满足，因为该容器中的点云根据线束从小到大排序
                //                 if (int(laserCloudCornerLast->points[j].intensity) > (closestPointScanID + NEARBY_SCAN))
                //                     break;

                //                 // 计算当前找到的最近点之间的距离，上一帧的点到当前帧的点的距离
                //                 double pointSqDis = (laserCloudCornerLast->points[j].x - pointSel.x) *
                //                                         (laserCloudCornerLast->points[j].x - pointSel.x) +
                //                                     (laserCloudCornerLast->points[j].y - pointSel.y) *
                //                                         (laserCloudCornerLast->points[j].y - pointSel.y) +
                //                                     (laserCloudCornerLast->points[j].z - pointSel.z) *
                //                                         (laserCloudCornerLast->points[j].z - pointSel.z);
                                
                //                 // 寻找距离最小的角点及其索引
                //                 if (pointSqDis < minPointSqDis2)
                //                 {
                //                     // find nearer point
                //                     minPointSqDis2 = pointSqDis;
                //                     minPointInd2 = j;
                //                 }
                //             }
                //             // 向下寻找
                //             // search in the direction of decreasing scan line
                //             for (int j = closestPointInd - 1; j >= 0; --j)
                //             {
                //                 // if in the same scan line, continue
                //                 if (int(laserCloudCornerLast->points[j].intensity) >= closestPointScanID)
                //                     continue;

                //                 // if not in nearby scans, end the loop
                //                 if (int(laserCloudCornerLast->points[j].intensity) < (closestPointScanID - NEARBY_SCAN))
                //                     break;

                //                 double pointSqDis = (laserCloudCornerLast->points[j].x - pointSel.x) *
                //                                         (laserCloudCornerLast->points[j].x - pointSel.x) +
                //                                     (laserCloudCornerLast->points[j].y - pointSel.y) *
                //                                         (laserCloudCornerLast->points[j].y - pointSel.y) +
                //                                     (laserCloudCornerLast->points[j].z - pointSel.z) *
                //                                         (laserCloudCornerLast->points[j].z - pointSel.z);

                //                 if (pointSqDis < minPointSqDis2)
                //                 {
                //                     // find nearer point，第二个最近点
                //                     minPointSqDis2 = pointSqDis;
                //                     minPointInd2 = j;
                //                 }
                //             }
                //         }

                //         // 如果这个点是有效的角点
                //         // 如果特征点i的两个最近邻点j和m都有效，构建非线性优化问题
                //         if (minPointInd2 >= 0) // both closestPointInd and minPointInd2 is valid
                //         {
                //             // 取出当前点和上一帧的两个角点a、b，a、b不能在同一个scan下
                //             Eigen::Vector3d curr_point(cornerPointsSharp->points[i].x,
                //                                        cornerPointsSharp->points[i].y,
                //                                        cornerPointsSharp->points[i].z);
                //             // 最近的点a
                //             Eigen::Vector3d last_point_a(laserCloudCornerLast->points[closestPointInd].x,
                //                                          laserCloudCornerLast->points[closestPointInd].y,
                //                                          laserCloudCornerLast->points[closestPointInd].z);
                //             // 次近点b
                //             Eigen::Vector3d last_point_b(laserCloudCornerLast->points[minPointInd2].x,
                //                                          laserCloudCornerLast->points[minPointInd2].y,
                //                                          laserCloudCornerLast->points[minPointInd2].z);

                //             double s;
                //             if (DISTORTION)
                //                 // Δt/T
                //                 s = (cornerPointsSharp->points[i].intensity - int(cornerPointsSharp->points[i].intensity)) / SCAN_PERIOD;
                //             else
                //                 s = 1.0;

                //             // notice: 构建点到线的约束，残差
                //             ceres::CostFunction *cost_function = LidarEdgeFactor::Create(curr_point, last_point_a, last_point_b, s);
                //             // 添加残差到约束问题，残差项、损失函数、具体待优化变量(初值)
                //             problem.AddResidualBlock(cost_function, loss_function, para_q, para_t);
                //             corner_correspondence++;
                //         }
                //     }

                //     /*
                //     notice: 下面采用平面点匹配方法：假如在第k+1帧中发现了平面点i，通过KD-tree查询在第k帧（上一帧）中
                //     的最近邻点j，查询j的附近扫描线上的最近邻点l和同一条扫描线的最近邻点m，这三点确定一个平面，
                //     让点i与这个平面的距离最短；
                //     构建一个非线性优化问题：以点i与平面lmj的距离为代价函数，以位姿变换T(四元数表示旋转+t)为优化变量。

                //     找面的第一个点与找线的第一个点一致
                //     第二个点与第一个点在一条scan上，第三个点与其他两个点不能在同一条线上
                //             l
                //            .        
                //           / \     .
                //     -----.---.-----
                //          j   m
                //     */
                //     // find correspondence for plane features
                //     for (int i = 0; i < surfPointsFlatNum; ++i)
                //     {
                //         TransformToStart(&(surfPointsFlat->points[i]), &pointSel);
                //         // 先寻找上一帧距离这个面点最近的面点
                //         kdtreeSurfLast->nearestKSearch(pointSel, 1, pointSearchInd, pointSearchSqDis);

                //         int closestPointInd = -1, minPointInd2 = -1, minPointInd3 = -1;
                //         // 距离必须小于给定阈值
                //         if (pointSearchSqDis[0] < DISTANCE_SQ_THRESHOLD)
                //         {
                //             // 取出找到的上一帧面点的索引
                //             closestPointInd = pointSearchInd[0];

                //             // get closest point's scan ID
                //             // 取出最近的面点在上一帧的第几根scan上面
                //             int closestPointScanID = int(laserCloudSurfLast->points[closestPointInd].intensity);
                //             double minPointSqDis2 = DISTANCE_SQ_THRESHOLD, minPointSqDis3 = DISTANCE_SQ_THRESHOLD;

                //             // search in the direction of increasing scan line
                //             // 额外再寻找 两个点，要求一个点和最近点同一个scan，另一个是不同scan，先升序遍历搜索点寻找这些点
                //             for (int j = closestPointInd + 1; j < (int)laserCloudSurfLast->points.size(); ++j)
                //             {
                //                 // if not in nearby scans, end the loop
                //                 // 不能和当前找到的上一帧面点线束距离太远
                //                 if (int(laserCloudSurfLast->points[j].intensity) > (closestPointScanID + NEARBY_SCAN))
                //                     break;
                //                 // 计算和当前帧该点距离
                //                 double pointSqDis = (laserCloudSurfLast->points[j].x - pointSel.x) *
                //                                         (laserCloudSurfLast->points[j].x - pointSel.x) +
                //                                     (laserCloudSurfLast->points[j].y - pointSel.y) *
                //                                         (laserCloudSurfLast->points[j].y - pointSel.y) +
                //                                     (laserCloudSurfLast->points[j].z - pointSel.z) *
                //                                         (laserCloudSurfLast->points[j].z - pointSel.z);

                //                 // if in the same or lower scan line
                //                 // 如果是同一根scan且距离最近
                //                 if (int(laserCloudSurfLast->points[j].intensity) <= closestPointScanID && pointSqDis < minPointSqDis2)
                //                 {
                //                     // 第二个点
                //                     minPointSqDis2 = pointSqDis;
                //                     minPointInd2 = j;
                //                 }
                //                 // if in the higher scan line
                //                 // 如果是其他线束点
                //                 else if (int(laserCloudSurfLast->points[j].intensity) > closestPointScanID && pointSqDis < minPointSqDis3)
                //                 {
                //                     // 第三个点
                //                     minPointSqDis3 = pointSqDis;
                //                     minPointInd3 = j;
                //                 }
                //             }

                //             // search in the direction of decreasing scan line
                //             // 同样的方式，按照降序方向寻找这两个点
                //             for (int j = closestPointInd - 1; j >= 0; --j)
                //             {
                //                 // if not in nearby scans, end the loop
                //                 if (int(laserCloudSurfLast->points[j].intensity) < (closestPointScanID - NEARBY_SCAN))
                //                     break;

                //                 double pointSqDis = (laserCloudSurfLast->points[j].x - pointSel.x) *
                //                                         (laserCloudSurfLast->points[j].x - pointSel.x) +
                //                                     (laserCloudSurfLast->points[j].y - pointSel.y) *
                //                                         (laserCloudSurfLast->points[j].y - pointSel.y) +
                //                                     (laserCloudSurfLast->points[j].z - pointSel.z) *
                //                                         (laserCloudSurfLast->points[j].z - pointSel.z);

                //                 // if in the same or higher scan line
                //                 // 如果是同一scan上的点，并且求出的距离比前面的更小
                //                 if (int(laserCloudSurfLast->points[j].intensity) >= closestPointScanID && pointSqDis < minPointSqDis2)
                //                 {
                //                     minPointSqDis2 = pointSqDis;
                //                     minPointInd2 = j;
                //                 }
                //                 else if (int(laserCloudSurfLast->points[j].intensity) < closestPointScanID && pointSqDis < minPointSqDis3)
                //                 {
                //                     // find nearer point
                //                     minPointSqDis3 = pointSqDis;
                //                     minPointInd3 = j;
                //                 }
                //             }

                //             // 如果另外找到的两个点是有效点，就取出他们的3d坐标
                //             if (minPointInd2 >= 0 && minPointInd3 >= 0)
                //             {

                //                 Eigen::Vector3d curr_point(surfPointsFlat->points[i].x,
                //                                             surfPointsFlat->points[i].y,
                //                                             surfPointsFlat->points[i].z);
                //                 Eigen::Vector3d last_point_a(laserCloudSurfLast->points[closestPointInd].x,
                //                                                 laserCloudSurfLast->points[closestPointInd].y,
                //                                                 laserCloudSurfLast->points[closestPointInd].z);
                //                 Eigen::Vector3d last_point_b(laserCloudSurfLast->points[minPointInd2].x,
                //                                                 laserCloudSurfLast->points[minPointInd2].y,
                //                                                 laserCloudSurfLast->points[minPointInd2].z);
                //                 Eigen::Vector3d last_point_c(laserCloudSurfLast->points[minPointInd3].x,
                //                                                 laserCloudSurfLast->points[minPointInd3].y,
                //                                                 laserCloudSurfLast->points[minPointInd3].z);

                //                 double s;
                //                 if (DISTORTION)//去运动畸变，这里没有做，kitii数据已经做了
                //                     s = (surfPointsFlat->points[i].intensity - int(surfPointsFlat->points[i].intensity)) / SCAN_PERIOD;
                //                 else
                //                     s = 1.0;

                //                 // 构建点到面的约束，构建cere的非线性优化问题。
                //                 ceres::CostFunction *cost_function = LidarPlaneFactor::Create(curr_point, last_point_a, last_point_b, last_point_c, s);
                //                 problem.AddResidualBlock(cost_function, loss_function, para_q, para_t);
                //                 plane_correspondence++;
                //             }
                //         }
                //     }// 面点约束

                //     // 输出寻找关联点消耗的时间
                //     //printf("coner_correspondance %d, plane_correspondence %d \n", corner_correspondence, plane_correspondence);
                //     // 如果总的线约束和面约束太少，就打印一下 
                //     printf("data association time %f ms \n", t_data.toc());
                    
                //     // 如果总的约束小于10，就可能有问题
                //     if ((corner_correspondence + plane_correspondence) < 10)
                //     {
                //         printf("less correspondence! *************************************************\n");
                //     }

                //     // 调用ceres求解器求解 ，设定求解器类型，最大迭代次数，不输出过程信息，优化报告存入summary
                //     TicToc t_solver;
                //     ceres::Solver::Options options;
                //     // QR分解
                //     options.linear_solver_type = ceres::DENSE_QR;
                //     // 前端需要保证实时，不能迭代太久
                //     options.max_num_iterations = 4;
                //     options.minimizer_progress_to_stdout = false;
                //     ceres::Solver::Summary summary;
                //     ceres::Solve(options, &problem, &summary);
                //     printf("solver time %f ms \n", t_solver.toc());
                // }

                // 经过两次 LM仅位姿优化 消耗的时间
                // printf("optimization twice time %f \n", t_opt.toc());
                // notice: 更新帧间匹配的结果，得到lidar odom位姿
                /*
                               T_curr2last
                T_w_curr o-----o-----o-----o-----o
                         w     1     2     3     4
                            T_last2w
                */

                // 以下两个全局变量(t_w_curr、q_w_curr，当前坐标系到世界坐标系的坐标变换)被不断更新
                // t_last_curr、q_last_curr，是被LM优化过的帧间位姿，这里的w_curr 实际上是 w_last，即上一帧
                // T_curr2last * T_last2w = T_last2w
    			// os_pose << q_last_curr.x() << " " << q_last_curr.y()<< " "  << q_last_curr.z() << " " << q_last_curr.w() << " " << t_last_curr.x() << " " << t_last_curr.y() << " " << t_last_curr.z() << endl;
                // t_w_curr = t_w_curr + q_w_curr * t_last_curr;
                // q_w_curr = q_w_curr * q_last_curr;

                //这里改为用link3d关键点优化的帧间位姿赋值
                //帧间匹配的位姿转换成四元数/
                q_last_curr = Eigen::Quaterniond(RelativeR);
                t_last_curr = Relativet;
                //更新当前帧到世界系变换
                t_w_curr = t_w_curr + q_w_curr * t_last_curr;
                q_w_curr = q_w_curr * q_last_curr;
            }

            TicToc t_pub;//计算发布运行时间
            // 发布lidar里程计结果
            // publish odometry
            // 创建nav_msgs::Odometry消息类型，把信息导入，并发布
            nav_msgs::Odometry laserOdometry;
            laserOdometry.header.frame_id = "camera_init";
            laserOdometry.child_frame_id = "/laser_odom";
            laserOdometry.header.stamp = ros::Time().fromSec(timeSurfPointsLessFlat);
            // 以四元数和平移向量发出去，第一帧的时候发布 T = [I|0]
            laserOdometry.pose.pose.orientation.x = q_w_curr.x();
            laserOdometry.pose.pose.orientation.y = q_w_curr.y();
            laserOdometry.pose.pose.orientation.z = q_w_curr.z();
            laserOdometry.pose.pose.orientation.w = q_w_curr.w();
            laserOdometry.pose.pose.position.x = t_w_curr.x();
            laserOdometry.pose.pose.position.y = t_w_curr.y();
            laserOdometry.pose.pose.position.z = t_w_curr.z();
            // 发布里程计数据(位姿轨迹)，后端接收
            pubLaserOdometry.publish(laserOdometry);

            geometry_msgs::PoseStamped laserPose;
            laserPose.header = laserOdometry.header;
            laserPose.pose = laserOdometry.pose.pose;
            laserPath.header.stamp = laserOdometry.header.stamp;
            // 位姿容器，即轨迹
            laserPath.poses.push_back(laserPose);
            laserPath.header.frame_id = "camera_init";
            // 发布位姿轨迹可视化
            pubLaserPath.publish(laserPath);

            // transform corner features and plane features to the scan end point
            if (0)//去畸变，没有调用
            {
                int cornerPointsLessSharpNum = cornerPointsLessSharp->points.size();
                for (int i = 0; i < cornerPointsLessSharpNum; i++)
                {
                    TransformToEnd(&cornerPointsLessSharp->points[i], &cornerPointsLessSharp->points[i]);
                }

                int surfPointsLessFlatNum = surfPointsLessFlat->points.size();
                for (int i = 0; i < surfPointsLessFlatNum; i++)
                {
                    TransformToEnd(&surfPointsLessFlat->points[i], &surfPointsLessFlat->points[i]);
                }

                int laserCloudFullResNum = laserCloudFullRes->points.size();
                for (int i = 0; i < laserCloudFullResNum; i++)
                {
                    TransformToEnd(&laserCloudFullRes->points[i], &laserCloudFullRes->points[i]);
                }
            }
            
            // 位姿估计完毕之后，当前边线点和平面点就变成了上一帧的边线点和平面点，把索引和数量都转移过去
            // curr 存一下当前帧的指针
            //利用一个临时点云指针把当前从配准端接收到的次边缘点和次平面点分别与上一帧边缘点和平面点指针互换
            //? 这里应该是为了下次匹配的时候有更多的点进行匹配好做帧间匹配吧
            pcl::PointCloud<PointType>::Ptr laserCloudTemp = cornerPointsLessSharp;
            // curr --> last 当前帧指向上一帧
            cornerPointsLessSharp = laserCloudCornerLast;
            // last -> curr
            laserCloudCornerLast = laserCloudTemp;

            // 平面点 curr
            laserCloudTemp = surfPointsLessFlat;
            // curr --> last
            surfPointsLessFlat = laserCloudSurfLast;
            // last -> curr
            laserCloudSurfLast = laserCloudTemp;

            laserCloudCornerLastNum = laserCloudCornerLast->points.size();
            laserCloudSurfLastNum = laserCloudSurfLast->points.size();

            // std::cout << "the size of corner last is " << laserCloudCornerLastNum << ", and the size of surf last is " << laserCloudSurfLastNum << '\n';
            // kdtree设置当前帧，用来下一帧lidar odom使用，把当前帧点云送到KD树，用来下一帧的匹配
            // 向KDTREE中传入数据，即将点云数据设置成KD-Tree结构
            kdtreeCornerLast->setInputCloud(laserCloudCornerLast);
            kdtreeSurfLast->setInputCloud(laserCloudSurfLast);

            // 控制后端节点的执行频率，降频后给后端发送，只有整除时才发布结果
            // 每隔skipFrameNum帧，往后端发送一帧，如果算力不够skipFrameNum就要增大
            if (frameCount % skipFrameNum == 0)// 0除以1，当然是商0，且余数也是0
            {
                frameCount = 0;
                //此时当前帧边缘点 平面点已经存入last指针里了
                // 原封不动发布当前角点
                sensor_msgs::PointCloud2 laserCloudCornerLast2;
                pcl::toROSMsg(*laserCloudCornerLast, laserCloudCornerLast2);
                laserCloudCornerLast2.header.stamp = ros::Time().fromSec(timeSurfPointsLessFlat);
                laserCloudCornerLast2.header.frame_id = "/camera";
                pubLaserCloudCornerLast.publish(laserCloudCornerLast2);
                
                // 原封不动发布当前平面点
                sensor_msgs::PointCloud2 laserCloudSurfLast2;
                pcl::toROSMsg(*laserCloudSurfLast, laserCloudSurfLast2);
                laserCloudSurfLast2.header.stamp = ros::Time().fromSec(timeSurfPointsLessFlat);
                laserCloudSurfLast2.header.frame_id = "/camera";
                pubLaserCloudSurfLast.publish(laserCloudSurfLast2);
                
                // 原封不动的转发当前帧点云，后端优化是低频，高精的，需要更多的点加入，约束越多鲁棒性越好
                sensor_msgs::PointCloud2 laserCloudFullRes3;
                pcl::toROSMsg(*laserCloudFullRes, laserCloudFullRes3);
                laserCloudFullRes3.header.stamp = ros::Time().fromSec(timeSurfPointsLessFlat);
                laserCloudFullRes3.header.frame_id = "/camera";
                pubLaserCloudFullRes.publish(laserCloudFullRes3);

                //link3d
                // sensor_msgs::PointCloud2 laserCloudLink3d_odom;
                // pcl::toROSMsg(*laserCloudLink3d, laserCloudLink3d_odom);
                // laserCloudLink3d_odom.header.stamp = ros::Time().fromSec(timeSurfPointsLessFlat);
                // laserCloudLink3d_odom.header.frame_id = "/camera";
                // pubLaserCloudLink3d_odom.publish(laserCloudLink3d_odom);

                //LinK3D 关键点云发布 laserCLoud要换成关键点云
                sensor_msgs::PointCloud2 laserCloud_LinK3D;
                pcl::toROSMsg(*AggregationKeypoints_LinK3D, laserCloud_LinK3D);
                laserCloud_LinK3D.header.stamp = ros::Time().fromSec(timeSurfPointsLessFlat);
                laserCloud_LinK3D.header.frame_id = "/camera";
                pubLaserCloud_LinK3D.publish(laserCloud_LinK3D);// 发布当前帧点云
                //用完清空 释放原来的对象，指向新的对象
                plaserCloudIn_LinK3D.reset(new pcl::PointCloud<pcl::PointXYZ>);
                //AggregationKeypoints_LinK3D需要手动清空
                AggregationKeypoints_LinK3D.reset(new pcl::PointCloud<pcl::PointXYZI>);
            }
            printf("publication time %f ms \n", t_pub.toc());
            printf("whole laserOdometry time %f ms \n \n", t_whole.toc());
            
            // 若处理时间，大于bag发布的频率，则来不及处理
            // notice: 里程计超过100ms，即小于10hz则有问题(超过理想范围的10倍)，算力不够 ??
            if(t_whole.toc() > 100)
                ROS_WARN("odometry process over 100ms");

            frameCount++;
        }
        // 处理一帧的时间最好小于等于 0.01s
        // 每秒执行100次，所以执行一次是0.01s；而如果执行完小于0.01s，就sleep一下，直到0.01s再继续走
        rate.sleep();
    }
    return 0;
}