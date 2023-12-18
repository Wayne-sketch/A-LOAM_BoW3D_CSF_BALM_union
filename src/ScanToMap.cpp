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

#include <math.h>
#include <vector>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <geometry_msgs/PoseStamped.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>
#include <eigen3/Eigen/Dense>
#include <ceres/ceres.h>
#include <mutex>
#include <queue>
#include <thread>
#include <iostream>
#include <string>

#include "lidarFactor.hpp"
// #include "utility/common.h"
// #include "utility/tic_toc.h"
#include "aloam_velodyne/common.h"
#include "aloam_velodyne/tic_toc.h"


#include <pcl/io/pcd_io.h>
#include <image_transport/image_transport.h>

#include <iostream>

#include<fstream>
#include <sstream>
#include "CSF/CSF.h"
using namespace std;


int frameCount = 0;

double timeLaserCloudCornerLast = 0;
double timeLaserCloudSurfLast = 0;
double timeLaserCloudFullRes = 0;
double timeLaserOdometry = 0;
double timecenter = 0;
double timeimage = 0;


int laserCloudCenWidth = 10;
int laserCloudCenHeight = 10;
int laserCloudCenDepth = 5;
const int laserCloudWidth = 21;
const int laserCloudHeight = 21;
const int laserCloudDepth = 11;


ofstream os_pose;



const int laserCloudNum = laserCloudWidth * laserCloudHeight * laserCloudDepth; //4851

// 记录submap中的有效cube的index，注意submap中cube的最大数量为 5 * 5 * 5 = 125
int laserCloudValidInd[125];
int laserCloudSurroundInd[125];

// input: from odom
pcl::PointCloud<PointType>::Ptr laserCloudCornerLast(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr laserCloudSurfLast(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr centerLast(new pcl::PointCloud<PointType>());

// ouput: all visualble cube points
pcl::PointCloud<PointType>::Ptr laserCloudSurround(new pcl::PointCloud<PointType>());

// surround points in map to build tree
pcl::PointCloud<PointType>::Ptr laserCloudCornerFromMap(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr laserCloudSurfFromMap(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr centerFromMap(new pcl::PointCloud<PointType>());


//input & output: points in one frame. local --> global
pcl::PointCloud<PointType>::Ptr laserCloudFullRes(new pcl::PointCloud<PointType>());

// points in every cube
pcl::PointCloud<PointType>::Ptr laserCloudCornerArray[laserCloudNum];



//kd-tree
pcl::KdTreeFLANN<PointType>::Ptr kdtreeCornerFromMap(new pcl::KdTreeFLANN<PointType>());
pcl::KdTreeFLANN<PointType>::Ptr kdtreeSurfFromMap(new pcl::KdTreeFLANN<PointType>());
pcl::KdTreeFLANN<PointType>::Ptr kdtreeCenterFromMap(new pcl::KdTreeFLANN<PointType>());


// 点云特征匹配时的优化变量
double parameters[7] = {0, 0, 0, 1, 0, 0, 0};

// Mapping线程估计的是frame在world坐标系下的位姿P,因为Mapping的算法耗时很有可能会超过100ms，所以
// 这个位姿P不是实时的，LOAM最终输出的实时位姿P_realtime,需要Mapping线程计算的相对低频位姿和
// Odometry线程计算的相对高频位姿做整合
// Mapping得到的位姿
Eigen::Map<Eigen::Quaterniond> q_w_curr(parameters);
Eigen::Map<Eigen::Vector3d> t_w_curr(parameters + 4);

// wmap_T_odom * odom_T_curr = wmap_T_curr;
// transformation between odom's world and map's world frame

// 这俩变量表示map中的world与Odometry模块中的world的偏差，transformation between odom's world and map's world frame
// laserOdometry模块"/laser_odom_to_init"话题发布的位姿是T_{cur}^{world}，这个world是Odometry中的world
// wmap_T_odom * odom_T_curr = wmap_T_curr;
Eigen::Quaterniond q_wmap_wodom(1, 0, 0, 0);
Eigen::Vector3d t_wmap_wodom(0, 0, 0);

// Odometry线程计算的frame在world坐标系的位姿
Eigen::Quaterniond q_wodom_curr(1, 0, 0, 0);
Eigen::Vector3d t_wodom_curr(0, 0, 0);


std::queue<sensor_msgs::PointCloud2ConstPtr> cornerLastBuf;
std::queue<sensor_msgs::PointCloud2ConstPtr> surfLastBuf;
std::queue<sensor_msgs::PointCloud2ConstPtr> fullResBuf;
std::queue<nav_msgs::Odometry::ConstPtr> odometryBuf;
std::queue<sensor_msgs::PointCloud2ConstPtr> centerBuf;
// std::queue<sensor_msgs::ImageConstPtr> imageBuf;



std::mutex mBuf;

pcl::VoxelGrid<PointType> downSizeFilterCorner;
pcl::VoxelGrid<PointType> downSizeFilterCenter;

std::vector<int> pointSearchInd;
std::vector<float> pointSearchSqDis;

PointType pointOri, pointSel;

ros::Publisher pubLaserCloudSurround, pubLaserCloudMap, pubLaserCloudFullRes, pubOdomAftMapped, pubOdomAftMappedHighFrec, pubLaserAfterMappedPath;
ros::Publisher pubOffGround;
ros::Publisher pubGround;
ros::Publisher pubEdge;

image_transport::Publisher pubImgae;


nav_msgs::Path laserAfterMappedPath;

/**
 * @brief   上一帧的增量wmap_wodom * 本帧Odometry位姿wodom_curr，旨在为本帧Mapping位姿w_curr设置一个初始值
 */
void transformAssociateToMap()
{
    q_w_curr = q_wmap_wodom * q_wodom_curr;
    t_w_curr = q_wmap_wodom * t_wodom_curr + t_wmap_wodom;
}
/**
 * @brief   用在最后，当Mapping的位姿w_curr计算完毕后，更新增量wmap_wodom，
 *          旨在为下一次执行transformAssociateToMap函数时做准备
 */
void transformUpdate()
{
    q_wmap_wodom = q_w_curr * q_wodom_curr.inverse();
    t_wmap_wodom = t_w_curr - q_wmap_wodom * t_wodom_curr;
}
/**
 * @brief      将Lidar坐标系下的点变换到map中world坐标系下
 *
 * @param      pi    原激光点
 * @param      po    变换后的激光点
 */
void pointAssociateToMap(PointType const *const pi, PointType *const po)
{
    Eigen::Vector3d point_curr(pi->x, pi->y, pi->z);
    Eigen::Vector3d point_w = q_w_curr * point_curr + t_w_curr;
    po->x = point_w.x();
    po->y = point_w.y();
    po->z = point_w.z();
    po->intensity = pi->intensity;
    //po->intensity = 1.0;
}
/**
 * @brief      将map中world坐标系下的点变换到Lidar坐标系下，这个没有用到
 *
 * @param      pi    原激光点
 * @param      po    变换后的激光点
 */
void pointAssociateTobeMapped(PointType const *const pi, PointType *const po)
{
    Eigen::Vector3d point_w(pi->x, pi->y, pi->z);
    Eigen::Vector3d point_curr = q_w_curr.inverse() * (point_w - t_w_curr);
    po->x = point_curr.x();
    po->y = point_curr.y();
    po->z = point_curr.z();
    po->intensity = pi->intensity;
}

void laserCloudCornerLastHandler(const sensor_msgs::PointCloud2ConstPtr &laserCloudCornerLast2)
{
    mBuf.lock();
    cornerLastBuf.push(laserCloudCornerLast2);
    mBuf.unlock();
}

void laserCloudSurfLastHandler(const sensor_msgs::PointCloud2ConstPtr &laserCloudSurfLast2)
{
    mBuf.lock();
    surfLastBuf.push(laserCloudSurfLast2);
    mBuf.unlock();
}

// void laserCloudFullResHandler(const sensor_msgs::PointCloud2ConstPtr &laserCloudFullRes2)
// {
// 	mBuf.lock();
// 	fullResBuf.push(laserCloudFullRes2);
// 	mBuf.unlock();
// }
// 一帧的质心
void centerHandler(const sensor_msgs::PointCloud2ConstPtr &center_msg)
{
    mBuf.lock();
    centerBuf.push(center_msg);
    mBuf.unlock();
}

// 一帧的质心图片
// void imageHandler(const sensor_msgs::ImageConstPtr &image_msg)
// {
//      mBuf.lock();
//      imageBuf.push(image_msg);
//      // cout << "image_msg->header.stamp =  "<< image_msg->header.stamp << endl;
//      mBuf.unlock();
// }

// receive odomtry
void laserOdometryHandler(const nav_msgs::Odometry::ConstPtr &laserOdometry)
{
    mBuf.lock();
    odometryBuf.push(laserOdometry);
    mBuf.unlock();
    // high frequence publish
    Eigen::Quaterniond q_wodom_curr;
    Eigen::Vector3d t_wodom_curr;
    q_wodom_curr.x() = laserOdometry->pose.pose.orientation.x;
    q_wodom_curr.y() = laserOdometry->pose.pose.orientation.y;
    q_wodom_curr.z() = laserOdometry->pose.pose.orientation.z;
    q_wodom_curr.w() = laserOdometry->pose.pose.orientation.w;
    t_wodom_curr.x() = laserOdometry->pose.pose.position.x;
    t_wodom_curr.y() = laserOdometry->pose.pose.position.y;
    t_wodom_curr.z() = laserOdometry->pose.pose.position.z;

    Eigen::Quaterniond q_w_curr = q_wmap_wodom * q_wodom_curr;
    Eigen::Vector3d t_w_curr = q_wmap_wodom * t_wodom_curr + t_wmap_wodom;

    nav_msgs::Odometry odomAftMapped;
    odomAftMapped.header.frame_id = "/camera_init";
    odomAftMapped.child_frame_id = "/aft_mapped";
    odomAftMapped.header.stamp = laserOdometry->header.stamp;
    odomAftMapped.pose.pose.orientation.x = q_w_curr.x();
    odomAftMapped.pose.pose.orientation.y = q_w_curr.y();
    odomAftMapped.pose.pose.orientation.z = q_w_curr.z();
    odomAftMapped.pose.pose.orientation.w = q_w_curr.w();
    odomAftMapped.pose.pose.position.x = t_w_curr.x();
    odomAftMapped.pose.pose.position.y = t_w_curr.y();
    odomAftMapped.pose.pose.position.z = t_w_curr.z();
    pubOdomAftMappedHighFrec.publish(odomAftMapped);
}
/**
 * @brief   进行Mapping，即帧与submap的匹配，对Odometry计算的位姿进行finetune
 */
void process()
{
    while(ros::ok())
    {
        while (!cornerLastBuf.empty() && !surfLastBuf.empty() && !odometryBuf.empty() && !centerBuf.empty() )// && !imageBuf.empty()
        {
            // laserOdometry模块对本节点的执行频率进行了控制，laserOdometry模块publish的位姿是10Hz，点云的publish频率则可以没这么高
            // 为了保证LOAM算法整体的实时性，Mapping线程每次只处理cornerLastBuf.front()及其他与之时间同步的消息
            // odometryBuf只保留一个与cornerLastBuf.front()时间同步的最新消息
            mBuf.lock();
            while (!odometryBuf.empty() && odometryBuf.front()->header.stamp.toSec() < cornerLastBuf.front()->header.stamp.toSec())
            {
                odometryBuf.pop();
            }
            if (odometryBuf.empty())
            {
                mBuf.unlock();
                break;
            }
            // surfLastBuf也只保留一个与cornerLastBuf.front()时间同步的最新消息
            while (!surfLastBuf.empty() && surfLastBuf.front()->header.stamp.toSec() < cornerLastBuf.front()->header.stamp.toSec())
            {
                surfLastBuf.pop();
            }
            if (surfLastBuf.empty())
            {
                mBuf.unlock();
                break;
            }
            // fullResBuf也如此也只保留一个与cornerLastBuf.front()时间同步的最新消息
            // while (!fullResBuf.empty() && fullResBuf.front()->header.stamp.toSec() < cornerLastBuf.front()->header.stamp.toSec())
            // {
            //     fullResBuf.pop();

            // }
            // if (fullResBuf.empty())
            // {
            // 	mBuf.unlock();
            // 	break;
            // }
            // 以cornerLastBuf为基准，把时间戳小于它的全部pop
            while (!centerBuf.empty() && centerBuf.front()->header.stamp.toSec() < cornerLastBuf.front()->header.stamp.toSec())
            {
                centerBuf.pop();
            }

            if (centerBuf.empty())
            {
                mBuf.unlock();
                break;
            }

            timeLaserCloudCornerLast = cornerLastBuf.front()->header.stamp.toSec();
            timeLaserCloudSurfLast = surfLastBuf.front()->header.stamp.toSec();
            //timeLaserCloudFullRes = fullResBuf.front()->header.stamp.toSec();
            timeLaserOdometry = odometryBuf.front()->header.stamp.toSec();
            timecenter = centerBuf.front()->header.stamp.toSec();
            // timeimage = imageBuf.front()->header.stamp.toSec();

            //  timeimage != timeLaserOdometry || timeLaserCloudFullRes != timeLaserOdometry
            if (timeLaserCloudCornerLast != timeLaserOdometry ||
                timeLaserCloudSurfLast != timeLaserOdometry ||
                timecenter != timeLaserOdometry)
            {
                printf("time corner %f surf %f odom %f center %f\n", timeLaserCloudCornerLast, timeLaserCloudSurfLast,  timeLaserOdometry, timecenter);
                //printf("unsync messeage!");
                mBuf.unlock();
                break;
            }

            laserCloudCornerLast->clear();
            pcl::fromROSMsg(*cornerLastBuf.front(), *laserCloudCornerLast);
            // 用完即弃pop()，队列先进先出
            cornerLastBuf.pop();

            laserCloudSurfLast->clear();
            pcl::fromROSMsg(*surfLastBuf.front(), *laserCloudSurfLast);
            surfLastBuf.pop();

            // laserCloudFullRes->clear();
            // pcl::fromROSMsg(*fullResBuf.front(), *laserCloudFullRes);
            // fullResBuf.pop();

            centerLast->clear();
            pcl::fromROSMsg(*centerBuf.front(), *centerLast);
            centerBuf.pop();

            q_wodom_curr.x() = odometryBuf.front()->pose.pose.orientation.x;
            q_wodom_curr.y() = odometryBuf.front()->pose.pose.orientation.y;
            q_wodom_curr.z() = odometryBuf.front()->pose.pose.orientation.z;
            q_wodom_curr.w() = odometryBuf.front()->pose.pose.orientation.w;
            t_wodom_curr.x() = odometryBuf.front()->pose.pose.position.x;
            t_wodom_curr.y() = odometryBuf.front()->pose.pose.position.y;
            t_wodom_curr.z() = odometryBuf.front()->pose.pose.position.z;
            odometryBuf.pop();
            // 为了保证LOAM算法整体的实时性，因Mapping线程耗时>51ms导致的历史缓存都会被清空
            while(!cornerLastBuf.empty())
            {
                cornerLastBuf.pop();
                printf("drop lidar frame in mapping for real time performance \n");
            }

            mBuf.unlock();

            TicToc t_whole;
            // 为本帧Mapping位姿w_curr设置一个初始值
            transformAssociateToMap();

            TicToc t_shift;
            int centerCubeI = int((t_w_curr.x() + 25.0) / 50.0) + laserCloudCenWidth;
            int centerCubeJ = int((t_w_curr.y() + 25.0) / 50.0) + laserCloudCenHeight;
            int centerCubeK = int((t_w_curr.z() + 25.0) / 50.0) + laserCloudCenDepth;
            // 由于int(-2.1)=-2是向0取整，当被求余数为负数时求余结果统一向左偏移一个单位

            if (t_w_curr.x() + 25.0 < 0)
                centerCubeI--;
            if (t_w_curr.y() + 25.0 < 0)
                centerCubeJ--;
            if (t_w_curr.z() + 25.0 < 0)
                centerCubeK--;
            // 调整取值范围:3 < centerCubeI < 18， 3 < centerCubeJ < 18, 3 < centerCubeK < 8
            // 目的是为了防止后续向四周拓展cube（图中的黄色cube就是拓展的cube）时，index（即IJK坐标）成为负数
            // 如果处于下边界，表明地图向负方向延伸的可能性比较大，则循环移位，将数组中心点向上边界调整一个单位
            while (centerCubeI < 3)
            {
                for (int j = 0; j < laserCloudHeight; j++)
                {
                    for (int k = 0; k < laserCloudDepth; k++)
                    {
                        int i = laserCloudWidth - 1;
                        pcl::PointCloud<PointType>::Ptr laserCloudCubeCornerPointer =
                                laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];

                        for (; i >= 1; i--)
                        {
                            laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
                                    laserCloudCornerArray[i - 1 + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];

                        }

                        laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
                                laserCloudCubeCornerPointer;

                        laserCloudCubeCornerPointer->clear();
                    }
                }
                centerCubeI++;
                laserCloudCenWidth++;
            }

            while (centerCubeI >= laserCloudWidth - 3)
            {
                for (int j = 0; j < laserCloudHeight; j++)
                {
                    for (int k = 0; k < laserCloudDepth; k++)
                    {
                        int i = 0;
                        pcl::PointCloud<PointType>::Ptr laserCloudCubeCornerPointer =
                                laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];


                        for (; i < laserCloudWidth - 1; i++)
                        {
                            laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
                                    laserCloudCornerArray[i + 1 + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];

                        }
                        laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
                                laserCloudCubeCornerPointer;

                        laserCloudCubeCornerPointer->clear();
                    }
                }

                centerCubeI--;
                laserCloudCenWidth--;
            }

            while (centerCubeJ < 3)
            {
                for (int i = 0; i < laserCloudWidth; i++)
                {
                    for (int k = 0; k < laserCloudDepth; k++)
                    {
                        int j = laserCloudHeight - 1;
                        pcl::PointCloud<PointType>::Ptr laserCloudCubeCornerPointer =
                                laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];



                        for (; j >= 1; j--)
                        {
                            laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
                                    laserCloudCornerArray[i + laserCloudWidth * (j - 1) + laserCloudWidth * laserCloudHeight * k];

                        }
                        laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
                                laserCloudCubeCornerPointer;


                        laserCloudCubeCornerPointer->clear();
                    }
                }

                centerCubeJ++;
                laserCloudCenHeight++;
            }

            while (centerCubeJ >= laserCloudHeight - 3)
            {
                for (int i = 0; i < laserCloudWidth; i++)
                {
                    for (int k = 0; k < laserCloudDepth; k++)
                    {
                        int j = 0;
                        pcl::PointCloud<PointType>::Ptr laserCloudCubeCornerPointer =
                                laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];


                        for (; j < laserCloudHeight - 1; j++)
                        {
                            laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
                                    laserCloudCornerArray[i + laserCloudWidth * (j + 1) + laserCloudWidth * laserCloudHeight * k];

                        }
                        laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
                                laserCloudCubeCornerPointer;


                        laserCloudCubeCornerPointer->clear();
                    }
                }

                centerCubeJ--;
                laserCloudCenHeight--;
            }

            while (centerCubeK < 3)
            {
                for (int i = 0; i < laserCloudWidth; i++)
                {
                    for (int j = 0; j < laserCloudHeight; j++)
                    {
                        int k = laserCloudDepth - 1;
                        pcl::PointCloud<PointType>::Ptr laserCloudCubeCornerPointer =
                                laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];


                        for (; k >= 1; k--)
                        {
                            laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
                                    laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * (k - 1)];

                        }
                        laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
                                laserCloudCubeCornerPointer;

                        laserCloudCubeCornerPointer->clear();
                    }
                }

                centerCubeK++;
                laserCloudCenDepth++;
            }

            while (centerCubeK >= laserCloudDepth - 3)
            {
                for (int i = 0; i < laserCloudWidth; i++)
                {
                    for (int j = 0; j < laserCloudHeight; j++)
                    {
                        int k = 0;
                        pcl::PointCloud<PointType>::Ptr laserCloudCubeCornerPointer =
                                laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];

                        for (; k < laserCloudDepth - 1; k++)
                        {
                            laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
                                    laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * (k + 1)];

                        }
                        laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
                                laserCloudCubeCornerPointer;

                        laserCloudCubeCornerPointer->clear();
                    }
                }
                centerCubeK--;
                laserCloudCenDepth--;
            }

            int laserCloudValidNum = 0;
            int laserCloudSurroundNum = 0;
            // 向IJ坐标轴的正负方向各拓展2个cube，K坐标轴的正负方向各拓展1个cube
            for (int i = centerCubeI - 2; i <= centerCubeI + 2; i++)
            {
                for (int j = centerCubeJ - 2; j <= centerCubeJ + 2; j++)
                {
                    for (int k = centerCubeK - 1; k <= centerCubeK + 1; k++)
                    {
                        // 如果坐标合法
                        if (i >= 0 && i < laserCloudWidth &&
                            j >= 0 && j < laserCloudHeight &&
                            k >= 0 && k < laserCloudDepth)
                        {
                            // 记录submap中的有效cube的index,（在所有cube中submap的cube的idx）
                            laserCloudValidInd[laserCloudValidNum] = i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k;
                            laserCloudValidNum++;
                            laserCloudSurroundInd[laserCloudSurroundNum] = i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k;
                            laserCloudSurroundNum++;
                        }
                    }
                }
            }
            // surround points in map to build tree
            // 将有效index的cube中的点云叠加到一起组成submap的特征点云
            laserCloudCornerFromMap->clear();


            for (int i = 0; i < laserCloudValidNum; i++)
            {
                *laserCloudCornerFromMap += *laserCloudCornerArray[laserCloudValidInd[i]];
            }
            int laserCloudCornerFromMapNum = laserCloudCornerFromMap->points.size();


            // 降采样当前帧特征点云（次极大边线点云）
            pcl::PointCloud<PointType>::Ptr laserCloudCornerStack(new pcl::PointCloud<PointType>());
            downSizeFilterCorner.setInputCloud(laserCloudCornerLast);
            downSizeFilterCorner.filter(*laserCloudCornerStack);
            int laserCloudCornerStackNum = laserCloudCornerStack->points.size();

            //printf("map prepare time %f ms\n", t_shift.toc());
            // 小局部地图中的点会不断的变多
            //printf(" corner =  %d  surf = %d center = %d\n", laserCloudCornerFromMapNum, laserCloudSurfFromMapNum, centerFromMapNum);// 3852  9097
            // 第一帧点云还没加入到cube中，此时不会满足这一条件
            if (laserCloudCornerFromMapNum > 10  )
            {
                TicToc t_opt;
                TicToc t_tree;
                kdtreeCornerFromMap->setInputCloud(laserCloudCornerFromMap);

                for (int iterCount = 0; iterCount < 2; iterCount++)
                {
                    //ceres::LossFunction *loss_function = NULL;
                    ceres::LossFunction *loss_function = new ceres::HuberLoss(0.1);
                    ceres::LocalParameterization *q_parameterization =
                            new ceres::EigenQuaternionParameterization();
                    ceres::Problem::Options problem_options;

                    ceres::Problem problem(problem_options);
                    problem.AddParameterBlock(parameters, 4, q_parameterization);
                    problem.AddParameterBlock(parameters + 4, 3);

                    TicToc t_data;
                    int corner_num = 0;
                    int center_num = 0;
                    // ------------------------------- p2edge -------------------------------------
                    for (int i = 0; i < laserCloudCornerStackNum; i++)
                    {
                        pointOri = laserCloudCornerStack->points[i];
                        //double sqrtDis = pointOri.x * pointOri.x + pointOri.y * pointOri.y + pointOri.z * pointOri.z;
                        // submap中的点云都是在(map)world坐标系下，而接收到的当前帧点云都是Lidar坐标系下
                        // 所以在搜寻最近邻点时，先用预测的Mapping位姿w_curr，将Lidar坐标系下的特征点变换到(map)world坐标系下
                        pointAssociateToMap(&pointOri, &pointSel);
                        // 在submap的corner特征点中，寻找距离当前帧corner特征点最近的5个点
                        kdtreeCornerFromMap->nearestKSearch(pointSel, 5, pointSearchInd, pointSearchSqDis);

                        if (pointSearchSqDis[4] < 1.0)
                        {
                            // 计算这个5个最近邻点的中心
                            std::vector<Eigen::Vector3d> nearCorners;
                            Eigen::Vector3d center(0, 0, 0);
                            for (int j = 0; j < 5; j++)
                            {
                                Eigen::Vector3d tmp(laserCloudCornerFromMap->points[pointSearchInd[j]].x,
                                                    laserCloudCornerFromMap->points[pointSearchInd[j]].y,
                                                    laserCloudCornerFromMap->points[pointSearchInd[j]].z);
                                center = center + tmp;
                                nearCorners.push_back(tmp);
                            }
                            center = center / 5.0;
                            // 计算这个5个最近邻点的协方差矩阵
                            Eigen::Matrix3d covMat = Eigen::Matrix3d::Zero();
                            for (int j = 0; j < 5; j++)
                            {
                                Eigen::Matrix<double, 3, 1> tmpZeroMean = nearCorners[j] - center;
                                covMat = covMat + tmpZeroMean * tmpZeroMean.transpose();
                            }

                            Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(covMat);

                            // if is indeed line feature
                            // note Eigen library sort eigenvalues in increasing order
                            // 计算协方差矩阵的特征值和特征向量，用于判断这5个点是不是呈线状分布，此为PCA的原理
                            // 如果5个点呈线状分布，最大的特征值对应的特征向量就是该线的方向矢量
                            Eigen::Vector3d unit_direction = saes.eigenvectors().col(2);
                            Eigen::Vector3d curr_point(pointOri.x, pointOri.y, pointOri.z);
                            // 如果最大的特征值 >> 其他特征值，则5个点确实呈线状分布，否则认为直线“不够直”
                            if (saes.eigenvalues()[2] > 3 * saes.eigenvalues()[1])
                            {
                                Eigen::Vector3d point_on_line = center;
                                Eigen::Vector3d point_a, point_b;
                                // 从中心点沿着方向向量向两端移动0.1m，使用两个点代替一条直线，这样计算点到直线的距离的形式就跟laserOdometry中保持一致
                                point_a = 0.1 * unit_direction + point_on_line;
                                point_b = -0.1 * unit_direction + point_on_line;
                                // 这里点到线的ICP过程就比Odometry中的要鲁棒和准确一些了（当然也就更耗时一些）
                                // 因为不是简单粗暴地搜最近的两个corner点作为target的线，而是PCA计算出最近邻的5个点的主方向作为线的方向，而且还会check直线是不是“足够直”
                                ceres::CostFunction *cost_function = LidarEdgeFactor::Create(curr_point, point_a, point_b, 1.0);
                                problem.AddResidualBlock(cost_function, loss_function, parameters, parameters + 4);
                                corner_num++;
                            }
                        }
                    }

                    //printf("mapping data assosiation time %f ms \n", t_data.toc());
                    TicToc t_solver;
                    ceres::Solver::Options options;
                    options.linear_solver_type = ceres::DENSE_QR;
                    //options.max_num_iterations = 4;
                    options.minimizer_progress_to_stdout = false;
                    options.check_gradients = false;
                    options.gradient_check_relative_precision = 1e-4;
                    ceres::Solver::Summary summary;
                    ceres::Solve(options, &problem, &summary);
                    //printf("mapping solver time %f ms \n", t_solver.toc());

                    //printf("time %f \n", timeLaserOdometry);
                    //printf("corner factor num %d surf factor num %d\n", corner_num, surf_num);
                    //printf("result q %f %f %f %f result t %f %f %f\n", parameters[3], parameters[0], parameters[1], parameters[2],
                    //	   parameters[4], parameters[5], parameters[6]);
                }
                //printf("mapping optimization time %f \n", t_opt.toc());
            }
            else
            {
                ROS_WARN("time Map corner and surf num are not enough");
            }

            // 完成ICP（迭代2次）的特征匹配后，用最后匹配计算出的优化变量w_curr，更新增量wmap_wodom，为下一次Mapping做准备
            transformUpdate();
            // 下面两个for loop的作用就是将当前帧的特征点云，逐点进行操作：转换到world坐标系并添加到对应位置的cube中
            TicToc t_add;
            // 将当前帧的（次极大边线点云，经过降采样后的）存入对应的边线点云的cube
            for (int i = 0; i < laserCloudCornerStackNum; i++)
            {
                pointAssociateToMap(&laserCloudCornerStack->points[i], &pointSel);
                int cubeI = int((pointSel.x + 25.0) / 50.0) + laserCloudCenWidth;
                int cubeJ = int((pointSel.y + 25.0) / 50.0) + laserCloudCenHeight;
                int cubeK = int((pointSel.z + 25.0) / 50.0) + laserCloudCenDepth;

                if (pointSel.x + 25.0 < 0)
                    cubeI--;
                if (pointSel.y + 25.0 < 0)
                    cubeJ--;
                if (pointSel.z + 25.0 < 0)
                    cubeK--;

                if (cubeI >= 0 && cubeI < laserCloudWidth &&
                    cubeJ >= 0 && cubeJ < laserCloudHeight &&
                    cubeK >= 0 && cubeK < laserCloudDepth)
                {
                    int cubeInd = cubeI + laserCloudWidth * cubeJ + laserCloudWidth * laserCloudHeight * cubeK;
                    laserCloudCornerArray[cubeInd]->push_back(pointSel);
                }
            }

            // 因为cube中新增加了点云，之前已经存有点云的cube的密度会发生改变，这里全部重新进行一次降采样
            TicToc t_filter;
            for (int i = 0; i < laserCloudValidNum; i++)
            {
                // 记录submap中的有效cube的index（在所有cube中submap的cube的idx）
                int ind = laserCloudValidInd[i];

                pcl::PointCloud<PointType>::Ptr tmpCorner(new pcl::PointCloud<PointType>());
                downSizeFilterCorner.setInputCloud(laserCloudCornerArray[ind]);
                downSizeFilterCorner.filter(*tmpCorner);
                laserCloudCornerArray[ind] = tmpCorner;

            }
            //printf("filter time %f ms \n", t_filter.toc());
            TicToc t_pub;

            // 
            CSF csf;
            csf.params.iterations = 600;
            csf.params.time_step = 0.95;
            csf.params.cloth_resolution = 3;
            csf.params.bSloopSmooth = false;

            csf.setPointCloud(*laserCloudSurfLast);
            // pcl::io::savePCDFileBinary(map_save_directory, *SurfFrame);

            std::vector<int>  groundIndexes, offGroundIndexes;
            // 输出的是vector<int>类型的地面点和非地面点索引
            pcl::PointCloud<pcl::PointXYZI>::Ptr groundFrame(new pcl::PointCloud<pcl::PointXYZI>);
            pcl::PointCloud<pcl::PointXYZI>::Ptr offGroundFrame(new pcl::PointCloud<pcl::PointXYZI>);
            csf.do_filtering(groundIndexes, offGroundIndexes);
            pcl::copyPointCloud(*laserCloudSurfLast, groundIndexes, *groundFrame);
            pcl::copyPointCloud(*laserCloudSurfLast, offGroundIndexes, *offGroundFrame);

            // ----------------------------------- for BA ----------------------------------------
            sensor_msgs::PointCloud2 offGround_msg;
            // centerLast、laserCloudFullRes、laserCloudSurfLast、laserCloudCornerLast
            pcl::toROSMsg(*offGroundFrame, offGround_msg);
            offGround_msg.header.stamp = ros::Time().fromSec(timeLaserOdometry);
            offGround_msg.header.frame_id = "/camera_init";
            pubOffGround.publish(offGround_msg);

            sensor_msgs::PointCloud2 ground_msg;
            pcl::toROSMsg(  *groundFrame, ground_msg);
            ground_msg.header.stamp = ros::Time().fromSec(timeLaserOdometry);
            ground_msg.header.frame_id = "/camera_init";
            pubGround.publish(ground_msg);

            sensor_msgs::PointCloud2 edge_msg;
            pcl::toROSMsg(*laserCloudCornerLast, edge_msg);
            edge_msg.header.stamp = ros::Time().fromSec(timeLaserOdometry);
            edge_msg.header.frame_id = "/camera_init";
            pubEdge.publish(edge_msg);


            //printf("mapping pub time %f ms \n", t_pub.toc());
            //printf("whole mapping time %f ms +++++\n", t_whole.toc());

            nav_msgs::Odometry odomAftMapped;
            odomAftMapped.header.frame_id = "/camera_init";
            // odomAftMapped.child_frame_id = "/aft_mapped";
            odomAftMapped.header.stamp = ros::Time().fromSec(timeLaserOdometry);
            odomAftMapped.pose.pose.orientation.x = q_w_curr.x();
            odomAftMapped.pose.pose.orientation.y = q_w_curr.y();
            odomAftMapped.pose.pose.orientation.z = q_w_curr.z();
            odomAftMapped.pose.pose.orientation.w = q_w_curr.w();
            odomAftMapped.pose.pose.position.x = t_w_curr.x();
            odomAftMapped.pose.pose.position.y = t_w_curr.y();
            odomAftMapped.pose.pose.position.z = t_w_curr.z();

            os_pose << q_w_curr.x() << " " << q_w_curr.y()<< " "  << q_w_curr.z() << " " << q_w_curr.w() << " " << t_w_curr.x() << " " << t_w_curr.y() << " " << t_w_curr.z() << endl;

            // ---------------------- 发布优化后的位姿给Rviz 和 loop模块 ------------------------------
            pubOdomAftMapped.publish(odomAftMapped);
            geometry_msgs::PoseStamped laserAfterMappedPose;
            laserAfterMappedPose.header = odomAftMapped.header;
            laserAfterMappedPose.pose = odomAftMapped.pose.pose;
            laserAfterMappedPath.header.stamp = odomAftMapped.header.stamp;
            laserAfterMappedPath.header.frame_id = "/camera_init";
            laserAfterMappedPath.poses.push_back(laserAfterMappedPose);
            pubLaserAfterMappedPath.publish(laserAfterMappedPath);

            frameCount++;
        }
        std::chrono::milliseconds dura(2);
        std::this_thread::sleep_for(dura);
    }
}


int main(int argc, char **argv)
{
    ros::init(argc, argv, "laserMapping");
    ros::NodeHandle nh;
    os_pose.open("/home/wb/SC_FASTLOAM_WS/wb/pose/poses.txt", std::fstream::out);

    float lineRes = 0;
    float planeRes = 0;
    // 线特征点云的分辨率
    nh.param<float>("mapping_line_resolution", lineRes, 0.4);
    // 面特征点云的分辨率
    nh.param<float>("mapping_plane_resolution", planeRes, 0.8);
    //printf("line resolution %f plane resolution %f \n", lineRes, planeRes);

    // 全局变量，避免计算量太大，进行下采样，体素滤波
    downSizeFilterCorner.setLeafSize(lineRes, lineRes,lineRes);
    downSizeFilterCenter.setLeafSize(lineRes,lineRes, lineRes);
    // 从laserOdometry节点订阅角点
    ros::Subscriber subLaserCloudCornerLast = nh.subscribe<sensor_msgs::PointCloud2>("/laser_cloud_corner_last", 100, laserCloudCornerLastHandler);
    // 从laserOdometry节点订阅面点
    ros::Subscriber subLaserCloudSurfLast = nh.subscribe<sensor_msgs::PointCloud2>("/laser_cloud_surf_last", 100, laserCloudSurfLastHandler);
    // 从laserOdometry节点订阅到的最新帧的位姿T_cur^w，当前帧的位姿粗估计
    ros::Subscriber subLaserOdometry = nh.subscribe<nav_msgs::Odometry>("/laser_odom_to_init", 100, laserOdometryHandler);

    // 订阅质心
    ros::Subscriber subCenter = nh.subscribe<sensor_msgs::PointCloud2>("/odomLink3d_cloud", 100, centerHandler);


    // submap所在cube中的点云，附近5帧组成的降采样子地图 for rviz
    pubLaserCloudSurround = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_surround", 100);
    // 所欲cube中的点云，所有帧组成的点云地图
    pubLaserCloudMap = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_map", 100);
    // 当前帧原始点云
    // pubLaserCloudFullRes = nh.advertise<sensor_msgs::PointCloud2>("/velodyne_cloud_registered", 100);

    // -------------------------- for BA --------------------
    pubOffGround = nh.advertise<sensor_msgs::PointCloud2>("/OffGround", 100);
    pubGround = nh.advertise<sensor_msgs::PointCloud2>("/Ground", 100);
    pubEdge = nh.advertise<sensor_msgs::PointCloud2>("/Edge", 100);
    // Mapping模块优化之后的T_{cur}^{w}，Map to Map
    pubOdomAftMapped = nh.advertise<nav_msgs::Odometry>("/aft_mapped_to_init", 100);
    // Transform Integration之后的高频位姿数据，里程计坐标系位姿转化到地图坐标系，mapping输出的1Hz位姿，odometry输出的10Hz位姿，整合成10hz作为最终结果
    pubOdomAftMappedHighFrec = nh.advertise<nav_msgs::Odometry>("/aft_mapped_to_init_high_frec", 100);
    // Mapping模块优化之后的T_{cur}^{w}构成的轨迹，scan to Map
    pubLaserAfterMappedPath = nh.advertise<nav_msgs::Path>("/aft_mapped_path", 100);
    
    // 1hz -> 10 hz
    for (int i = 0; i < laserCloudNum; i++)
    {
        // 储存后端地图的数组，元素是智能指针，在堆区开辟空间，让智能指针指向该内存
        laserCloudCornerArray[i].reset(new pcl::PointCloud<PointType>());
    }
    
    // 开辟后端建图子线程，线程入口函数process
    std::thread mapping_process{process};

    ros::spin();

    return 0;
}
