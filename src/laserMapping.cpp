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
#include <aloam_velodyne/common.h>
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
#include "aloam_velodyne/common.h"
#include "aloam_velodyne/tic_toc.h"
#include <pcl/io/pcd_io.h>
#include<fstream>
#include <sstream>

//CSF
#include "CSF/Cfg.h"
#include "CSF/CSF.h"

using namespace std;

ofstream os_pose;

int frameCount = 0;

double timeLaserCloudCornerLast = 0;
double timeLaserCloudSurfLast = 0;
double timeLaserCloudFullRes = 0;
double timeLaserOdometry = 0;
//link3d
double timeLaserCloudLink3d = 0;


int laserCloudCenWidth = 10;
int laserCloudCenHeight = 10;
int laserCloudCenDepth = 5;

// 21*21*11
//这个名字取的很奇怪
const int laserCloudWidth = 21;
const int laserCloudHeight = 21;
const int laserCloudDepth = 11;


const int laserCloudNum = laserCloudWidth * laserCloudHeight * laserCloudDepth; //4851个栅格


int laserCloudValidInd[125];
int laserCloudSurroundInd[125];

// input: from odom
pcl::PointCloud<PointType>::Ptr laserCloudCornerLast(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr laserCloudSurfLast(new pcl::PointCloud<PointType>());

// ouput: all visualble cube points
pcl::PointCloud<PointType>::Ptr laserCloudSurround(new pcl::PointCloud<PointType>());

// surround points in map to build tree
pcl::PointCloud<PointType>::Ptr laserCloudCornerFromMap(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr laserCloudSurfFromMap(new pcl::PointCloud<PointType>());

//input & output: points in one frame. local --> global
pcl::PointCloud<PointType>::Ptr laserCloudFullRes(new pcl::PointCloud<PointType>());

//link3d
pcl::PointCloud<PointType>::Ptr laserCloudLink3dPtr(new pcl::PointCloud<PointType>());


// points in every cube
pcl::PointCloud<PointType>::Ptr laserCloudCornerArray[laserCloudNum];
// pcl::PointCloud<PointType>::Ptr laserCloudSurfArray[laserCloudNum];

//kd-tree 小局部地图的kd-tree
pcl::KdTreeFLANN<PointType>::Ptr kdtreeCornerFromMap(new pcl::KdTreeFLANN<PointType>());
pcl::KdTreeFLANN<PointType>::Ptr kdtreeSurfFromMap(new pcl::KdTreeFLANN<PointType>());

double parameters[7] = {0, 0, 0, 1, 0, 0, 0};
// 实时更新 T_curr2w_map
Eigen::Map<Eigen::Quaterniond> q_w_curr(parameters);
Eigen::Map<Eigen::Vector3d> t_w_curr(parameters + 4);

// wmap_T_odom * odom_T_curr = wmap_T_curr;
// transformation between odom's world and map's world frame
Eigen::Quaterniond q_wmap_wodom(1, 0, 0, 0);
Eigen::Vector3d t_wmap_wodom(0, 0, 0);

Eigen::Quaterniond q_wodom_curr(1, 0, 0, 0);
Eigen::Vector3d t_wodom_curr(0, 0, 0);


std::queue<sensor_msgs::PointCloud2ConstPtr> cornerLastBuf;
std::queue<sensor_msgs::PointCloud2ConstPtr> surfLastBuf;
std::queue<sensor_msgs::PointCloud2ConstPtr> fullResBuf;
std::queue<nav_msgs::Odometry::ConstPtr> odometryBuf;
//link3d
std::queue<sensor_msgs::PointCloud2ConstPtr> Link3dBuf;

std::mutex mBuf;

pcl::VoxelGrid<PointType> downSizeFilterCorner;
pcl::VoxelGrid<PointType> downSizeFilterSurf;

std::vector<int> pointSearchInd;
std::vector<float> pointSearchSqDis;

PointType pointOri, pointSel;

ros::Publisher pubLaserCloudSurround, pubLaserCloudMap, pubLaserCloudFullRes, pubOdomAftMapped, pubOdomAftMappedHighFrec, pubLaserAfterMappedPath;
//link3d
ros::Publisher pubLaserCloudLink3d;
//csf groundPoints
ros::Publisher pubLaserCloudCSFGroundPoints;


// 
nav_msgs::Path laserAfterMappedPath;

// T_curr2w_map = T_odmo2map * T_curr2w_odom
void transformAssociateToMap()
{
	// T_curr2w_odom = T_last2w_odom * T_curr2last_odom
	// 一开始q_wmap_wodom是单位矩阵，odom==map
	q_w_curr = q_wmap_wodom * q_wodom_curr;
	t_w_curr = q_wmap_wodom * t_wodom_curr + t_wmap_wodom;
}

//? T_curr2w_odom * T_w2last_odom * T_last2w_map = T_curr2last_odom * T_last2w_map = 
 
// ------------------------ 不断循环 ----------------------------------

// T_odmo2map = T_curr2w_map * T_curr2w_odom¯¹ = T_curr2w_map * T_w_odom2curr
void transformUpdate()
{
	// 一开始q_wmap_wodom为单位矩阵
	// q_wodom_curr 是前端里程计发来的  q_w_curr 是Ceres优化的curr到map的旋转四元数
	// 当ceres 更新 parameters之后 会通过 eigen的map 实时更新q_w_curr和t_w_curr 
	q_wmap_wodom = q_w_curr * q_wodom_curr.inverse();
	// R_curr2w_map * R_w2curr_odom * t_curr2w_odom
	t_wmap_wodom = t_w_curr - q_wmap_wodom * t_wodom_curr;
}

void pointAssociateToMap(PointType const *const pi, PointType *const po)
{
	Eigen::Vector3d point_curr(pi->x, pi->y, pi->z);
	// p_map = P_curr * T_curr2map
	Eigen::Vector3d point_w = q_w_curr * point_curr + t_w_curr;
	po->x = point_w.x();
	po->y = point_w.y();
	po->z = point_w.z();
	po->intensity = pi->intensity;
	//po->intensity = 1.0;
}

void pointAssociateTobeMapped(PointType const *const pi, PointType *const po)
{
	Eigen::Vector3d point_w(pi->x, pi->y, pi->z);
	Eigen::Vector3d point_curr = q_w_curr.inverse() * (point_w - t_w_curr);
	po->x = point_curr.x();
	po->y = point_curr.y();
	po->z = point_curr.z();
	po->intensity = pi->intensity;
}

/*
注意:回调函数要加线程锁,因为一个回调函数就是一个子线程,主函数里还有一个子线程process,
两个线程都会访问Buf信息,避免线程冲突,在存入数据前加锁,写入后解锁
*/
// 前端向后端发布的数据有
// 上一帧的角点 发布频率10hz，后端有100ms的时间处理一帧
void laserCloudCornerLastHandler(const sensor_msgs::PointCloud2ConstPtr &laserCloudCornerLast2)
{
	mBuf.lock();
	//这里存的点云的坐标系是LiDAR系下的 在做scan2map的优化前会调用函数把点都转换到世界坐标系下去
	cornerLastBuf.push(laserCloudCornerLast2);
	mBuf.unlock();
}
// 上一帧的面点
void laserCloudSurfLastHandler(const sensor_msgs::PointCloud2ConstPtr &laserCloudSurfLast2)
{
	mBuf.lock();
	surfLastBuf.push(laserCloudSurfLast2);
	mBuf.unlock();
}
// 所有的点云
void laserCloudFullResHandler(const sensor_msgs::PointCloud2ConstPtr &laserCloudFullRes2)
{
	mBuf.lock();
	fullResBuf.push(laserCloudFullRes2);
	mBuf.unlock();
}

//link3d
void laserCloudLink3dHandler(const sensor_msgs::PointCloud2ConstPtr &laserCloudLink3d)
{
	mBuf.lock();
	Link3dBuf.push(laserCloudLink3d);
	mBuf.unlock();
}

/*
每帧都会执行：
前端里程计会定期的向后端发送位姿：T_curr2w_odom，但是在mapping中，我们需要得到的位姿是
T_curr2w_map，因此mapping模块就是要估计出odom坐标系和map坐标系之间的相对位姿T_odom2map
notice: 在建图的过程中map坐标系和odom坐标系的起点一开始都是一样的
随着时间的推移，odmo坐标系下的位姿的置信度(累积误差增大)越来越差，
map坐标系下的位姿相对于里程计坐标系下的位姿较为准确
*/
//接收前端发来的当前LiDAR系在w_odom系下的位姿
void laserOdometryHandler(const nav_msgs::Odometry::ConstPtr &laserOdometry)
{
	mBuf.lock();
	odometryBuf.push(laserOdometry);
	mBuf.unlock();

	// high frequence publish
	// T_curr2odom
	Eigen::Quaterniond q_wodom_curr;
	Eigen::Vector3d t_wodom_curr;
	// 将ros的消息转成eigen的消息
	q_wodom_curr.x() = laserOdometry->pose.pose.orientation.x;
	q_wodom_curr.y() = laserOdometry->pose.pose.orientation.y;
	q_wodom_curr.z() = laserOdometry->pose.pose.orientation.z;
	q_wodom_curr.w() = laserOdometry->pose.pose.orientation.w;
	t_wodom_curr.x() = laserOdometry->pose.pose.position.x;
	t_wodom_curr.y() = laserOdometry->pose.pose.position.y;
	t_wodom_curr.z() = laserOdometry->pose.pose.position.z;

	// map坐标系下的位姿(置信度较高)，T_curr2w_map = T_odmo2map * T_curr2w_odom
	// T_odmo2map 是后端需要估计的位姿
	Eigen::Quaterniond q_w_curr = q_wmap_wodom * q_wodom_curr;
	Eigen::Vector3d t_w_curr = q_wmap_wodom * t_wodom_curr + t_wmap_wodom; 

	nav_msgs::Odometry odomAftMapped;
	odomAftMapped.header.frame_id = "camera_init";
	odomAftMapped.child_frame_id = "/aft_mapped";
	odomAftMapped.header.stamp = laserOdometry->header.stamp;
	// eigen消息再转换成ros消息，才能发布
	odomAftMapped.pose.pose.orientation.x = q_w_curr.x();
	odomAftMapped.pose.pose.orientation.y = q_w_curr.y();
	odomAftMapped.pose.pose.orientation.z = q_w_curr.z();
	odomAftMapped.pose.pose.orientation.w = q_w_curr.w();
	odomAftMapped.pose.pose.position.x = t_w_curr.x();
	odomAftMapped.pose.pose.position.y = t_w_curr.y();
	odomAftMapped.pose.pose.position.z = t_w_curr.z();
	// 发布map坐标系下的位姿 这个是经过scan2map匹配优化后的精位姿
	pubOdomAftMappedHighFrec.publish(odomAftMapped);
}


// 后端建图和位姿优化，前端发布的频率为10hz，所以后端处理一帧的时间最好在0.1s内
void process()
{
	while(ros::ok())
	{
		// 确保buf里面都有值
		while (!cornerLastBuf.empty() && !surfLastBuf.empty() && !fullResBuf.empty() && !odometryBuf.empty() && !Link3dBuf.empty())
		{
			mBuf.lock();
			// 每次新的数据进来后做数据对齐：把之前残留的数据清空
			// 以cornerLastBuf为基准，把时间戳小于它的全部pop
			//? 为什么以conerLastBuf为基准，如果cornerLastBuf的最早时间基准比起其他的提前呢？实际情况是什么样子的？这个要看发送端是什么样子的，
			while (!odometryBuf.empty() && odometryBuf.front()->header.stamp.toSec() < cornerLastBuf.front()->header.stamp.toSec())
			{
				odometryBuf.pop();
			}
			
			if (odometryBuf.empty())
			{
				//线程锁放开,可以继续存数据
				mBuf.unlock();
				break;
			}
			// 以cornerLastBuf为基准，把时间戳小于它的全部pop 
			while (!surfLastBuf.empty() && surfLastBuf.front()->header.stamp.toSec() < cornerLastBuf.front()->header.stamp.toSec())
			{
				surfLastBuf.pop();
			}

			
			if (surfLastBuf.empty())
			{
				mBuf.unlock();
				break;
			}
			// 以cornerLastBuf为基准，把时间戳小于它的全部pop 
			while (!fullResBuf.empty() && fullResBuf.front()->header.stamp.toSec() < cornerLastBuf.front()->header.stamp.toSec())
			{
				fullResBuf.pop();
			}

			if (fullResBuf.empty())
			{
				mBuf.unlock();
				break;
			}

			//link3d
			while (!Link3dBuf.empty() && Link3dBuf.front()->header.stamp.toSec() < cornerLastBuf.front()->header.stamp.toSec())
			{
				Link3dBuf.pop();
			}

			if (Link3dBuf.empty())
			{
				mBuf.unlock();
				break;
			}

			// 点云数据时间同步后，取出它们的时间戳，一定是相等于 时间戳都是double类型的数据
			timeLaserCloudCornerLast = cornerLastBuf.front()->header.stamp.toSec();
			timeLaserCloudSurfLast = surfLastBuf.front()->header.stamp.toSec();
			timeLaserCloudFullRes = fullResBuf.front()->header.stamp.toSec();

			timeLaserOdometry = odometryBuf.front()->header.stamp.toSec();
			//link3d
			timeLaserCloudLink3d = Link3dBuf.front()->header.stamp.toSec();

			// 如果时间未同步
			if (timeLaserCloudCornerLast != timeLaserOdometry ||
				timeLaserCloudSurfLast != timeLaserOdometry ||
				timeLaserCloudFullRes != timeLaserOdometry ||
				timeLaserCloudLink3d != timeLaserOdometry)
			{
				printf("time corner %f surf %f full %f odom %f \n", timeLaserCloudCornerLast, timeLaserCloudSurfLast, timeLaserCloudFullRes, timeLaserOdometry);
				printf("unsync messeage!");
				mBuf.unlock();
				break;
			}

			// 为了对点云进行操作，把ros的消息转换为pcl的消息 把最新的一帧的ros消息转化为pcl消息，把buf里的该帧消息pop掉
			laserCloudCornerLast->clear();
			pcl::fromROSMsg(*cornerLastBuf.front(), *laserCloudCornerLast);
			cornerLastBuf.pop();

			laserCloudSurfLast->clear();
			pcl::fromROSMsg(*surfLastBuf.front(), *laserCloudSurfLast);
			surfLastBuf.pop();

			laserCloudFullRes->clear();
			pcl::fromROSMsg(*fullResBuf.front(), *laserCloudFullRes);
			fullResBuf.pop();
			
			//link3d
			laserCloudLink3dPtr->clear();
			pcl::fromROSMsg(*Link3dBuf.front(), *laserCloudLink3dPtr);
			Link3dBuf.pop();

			// 为了对前端位姿进行操作把ros的数据格式转换为eigen的数据格式 位姿取出来
			q_wodom_curr.x() = odometryBuf.front()->pose.pose.orientation.x;
			q_wodom_curr.y() = odometryBuf.front()->pose.pose.orientation.y;
			q_wodom_curr.z() = odometryBuf.front()->pose.pose.orientation.z;
			q_wodom_curr.w() = odometryBuf.front()->pose.pose.orientation.w;
			t_wodom_curr.x() = odometryBuf.front()->pose.pose.position.x;
			t_wodom_curr.y() = odometryBuf.front()->pose.pose.position.y;
			t_wodom_curr.z() = odometryBuf.front()->pose.pose.position.z;
			odometryBuf.pop();

			/*
			假设前端每秒发送一个数据给后端的buf，但是后端处理这个数据需要10s(cpu性能不太好)
			过了10s，后端才处理1个数据，但是buf中有9个待处理的数据
			长此以往，后端的buf就会累积很多的数据，导致内存爆炸
			运行到100s之后，后端才处理10帧返回第10s的结果，无法实时
			*/
			// 处理完后全部清空，保证内存不会爆满，保证实时性能
			while(!cornerLastBuf.empty())
			{
				// 每次处理一帧，就把buf中的数据全部清空，保证实时性，这可能就是为什么前面以cornerLastBuf为基准的原因
				cornerLastBuf.pop();
				// 去掉过多的数据,仅处理能处理的数据.这样虽然有可能丢掉一部分数据,但是保障了后端的低延时和计算内存
				printf("drop lidar frame in mapping for real time performance \n");
			}

			//所有对buf的处理结束后解锁
			mBuf.unlock();
			TicToc t_whole;
			// notice: 根据前端的位姿，作为后端优化的初始化值
			// T_curr2w_map = T_odom2map(后端) * T_curr2w_odom(前端) ，因为之前会清空buf，所以以下不会每帧都执行
			transformAssociateToMap();
			TicToc t_shift;
			/*
			前端是scan2scan，先计算相邻帧间的坐标变换，在计算当前帧到第一帧的坐标变换
			后端是scan2map，把当前帧与地图进行匹配，得到更准确的位姿，就可以构建更好的地图
			由于scan2map的算法计算量远远高于scan2scan的算法，所以后端通常处于一个低频的运行频率，但是其精度高
			为了提高后端的处理速度,所以要进行地图的栅格化处理

			地图的拼接：
			地图通常是当前帧通过匹配得到在地图坐标系下的准确位姿之后拼接而成的，如果我们保留
			所有的拼接的点云，此时随着时间的推移，内存很容易吃不消，因此考虑储存离当前帧比较近的部分地图
			距离当前帧比较远的点云对于定位也没有帮助，但却占据存储空间，去掉栅格之外的点云。相当于储存一个局部地图
			
			局部地图的存储：
			基于三维栅格的方式，把局部地图分成21*21*11的栅格，每个正方体小栅格的边长是50m的正方体
			这个魔方的长度为1050米，宽度为1050米，高度为550米
			如果当前位姿远离栅格地图的覆盖范围，那地图就没有意义，所以局部地图也要随着当前
			位姿动态调整，从而保证我们可以从栅格地图中取出距离当前位姿比较近的点云来进行
			scan2map，以此获得最优位姿估计

			局部地图的动态调整：
			希望当前帧的位姿在地图局部地图的中心
			当地图逐渐累加时,栅格之外的部分就被舍弃,这样可以保证内存空间不会随着程序的运行而爆炸
			*/

			// 后端地图的本质是一个以当前位姿为中心的一个大的三维栅格地图
			// 因为ALOAM针对机械式雷达,360度扫描,所以没有考虑旋转,如果要是livox的固态雷达,则必须考虑旋转了
			//? 这个实际要怎么考虑？ 以后学习一下
			// 当前位姿，在21*21*11的三维栅格中的索引
			// 因为栅格是21*21*11,所以初始是10,10,5,为了上初始时刻,在栅格地图的中心.
			// 物理坐标转换为栅格坐标
			// -25 ~ 25是10索引
			// 这里就是对索引的一个处理
			int centerCubeI = int((t_w_curr.x() + 25.0) / 50.0) + laserCloudCenWidth;// 10
			int centerCubeJ = int((t_w_curr.y() + 25.0) / 50.0) + laserCloudCenHeight;// 10
			int centerCubeK = int((t_w_curr.z() + 25.0) / 50.0) + laserCloudCenDepth;// 5
			
			//以x为例，x在-25~25之间，centerCubeI = 10 x在-75~-25之间，centerCubeI = 9
			//由于c语言的取整是向0取整，因此-1.66取整就成了-1，但是应该是-2，因此这里自减1
			if (t_w_curr.x() + 25.0 < 0)
				centerCubeI--;
			if (t_w_curr.y() + 25.0 < 0)
				centerCubeJ--;
			if (t_w_curr.z() + 25.0 < 0)
				centerCubeK--;

			// 首先是x轴,当前帧栅格索引小于三,说明块出边界了,让整体向x方向移动
			// notice: 滑动窗口  最前面的往前移动，最后的往前移动时会空出一个位置
			while (centerCubeI < 3)
			{
				// 遍历宽 21 注意名字取得挺奇怪的，Height代表的其实是宽度
				for (int j = 0; j < laserCloudHeight; j++)
				{
					// 遍历高 11 Depth代表的是高度
					for (int k = 0; k < laserCloudDepth; k++)
					{
						// i值先取最大,从x最大值开始处理.然后取出了最右边的一片点云
						int i = laserCloudWidth - 1;
						// 从x的最大值开始递减，三维坐标转换为一维数组的一维坐标
						pcl::PointCloud<PointType>::Ptr laserCloudCubeCornerPointer =
							laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k]; 
						// pcl::PointCloud<PointType>::Ptr laserCloudCubeSurfPointer =
						// 	laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
						// 点云整体右移
						for (; i >= 1; i--)
						{
							//右移动地图角点
							laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
								laserCloudCornerArray[i - 1 + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
							//右移动地图面点
							// laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
							// 	laserCloudSurfArray[i - 1 + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
						}
						// 此时i==0，也就是这一步操作后最左边的格子赋值了之前最右边的格子
						laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
							laserCloudCubeCornerPointer;
						// laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
						// 	laserCloudCubeSurfPointer;
						// 点云清零，由于是指针操作，相当于最左边的格子清空了
						laserCloudCubeCornerPointer->clear();
						// laserCloudCubeSurfPointer->clear();
					}
				}
				// 索引右移 在下一个循环判断是不是还小于3
				centerCubeI++;
				// 地图中心索引也右移，因为后面的点计算需要 这个是全局变量
				laserCloudCenWidth++;
			}

			// 同理，如果点云抵达左边界，就整体左移动
			while (centerCubeI >= laserCloudWidth - 3)
			{
				for (int j = 0; j < laserCloudHeight; j++)
				{
					for (int k = 0; k < laserCloudDepth; k++)
					{
						int i = 0;
						pcl::PointCloud<PointType>::Ptr laserCloudCubeCornerPointer =
							laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
						// pcl::PointCloud<PointType>::Ptr laserCloudCubeSurfPointer =
						// 	laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
						// 整体左移
						for (; i < laserCloudWidth - 1; i++)
						{
							laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
								laserCloudCornerArray[i + 1 + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
							// laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
							// 	laserCloudSurfArray[i + 1 + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
						}
						laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
							laserCloudCubeCornerPointer;
						// laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
						// 	laserCloudCubeSurfPointer;
						laserCloudCubeCornerPointer->clear();
						// laserCloudCubeSurfPointer->clear();
					}
				}

				centerCubeI--;
				laserCloudCenWidth--;
			}
			
			// y和z操作同x类似
			while (centerCubeJ < 3)
			{
				for (int i = 0; i < laserCloudWidth; i++)
				{
					for (int k = 0; k < laserCloudDepth; k++)
					{
						int j = laserCloudHeight - 1;
						pcl::PointCloud<PointType>::Ptr laserCloudCubeCornerPointer =
							laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
						// pcl::PointCloud<PointType>::Ptr laserCloudCubeSurfPointer =
						// 	laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
						for (; j >= 1; j--)
						{
							laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
								laserCloudCornerArray[i + laserCloudWidth * (j - 1) + laserCloudWidth * laserCloudHeight * k];
							// laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
							// 	laserCloudSurfArray[i + laserCloudWidth * (j - 1) + laserCloudWidth * laserCloudHeight * k];
						}
						laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
							laserCloudCubeCornerPointer;
						// laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
						// 	laserCloudCubeSurfPointer;
						laserCloudCubeCornerPointer->clear();
						// laserCloudCubeSurfPointer->clear();
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
						// pcl::PointCloud<PointType>::Ptr laserCloudCubeSurfPointer =
						// 	laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
						for (; j < laserCloudHeight - 1; j++)
						{
							laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
								laserCloudCornerArray[i + laserCloudWidth * (j + 1) + laserCloudWidth * laserCloudHeight * k];
							// laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
							// 	laserCloudSurfArray[i + laserCloudWidth * (j + 1) + laserCloudWidth * laserCloudHeight * k];
						}
						laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
							laserCloudCubeCornerPointer;
						// laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
						// 	laserCloudCubeSurfPointer;
						laserCloudCubeCornerPointer->clear();
						// laserCloudCubeSurfPointer->clear();
					}
				}

				centerCubeJ--;
				laserCloudCenHeight--;
			}

			// y和z操作同x类似
			while (centerCubeK < 3)
			{
				for (int i = 0; i < laserCloudWidth; i++)
				{
					for (int j = 0; j < laserCloudHeight; j++)
					{
						int k = laserCloudDepth - 1;
						pcl::PointCloud<PointType>::Ptr laserCloudCubeCornerPointer =
							laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
						// pcl::PointCloud<PointType>::Ptr laserCloudCubeSurfPointer =
						// 	laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
						for (; k >= 1; k--)
						{
							laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
								laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * (k - 1)];
							// laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
							// 	laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * (k - 1)];
						}
						laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
							laserCloudCubeCornerPointer;
						// laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
						// 	laserCloudCubeSurfPointer;
						laserCloudCubeCornerPointer->clear();
						// laserCloudCubeSurfPointer->clear();
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
						// pcl::PointCloud<PointType>::Ptr laserCloudCubeSurfPointer =
						// 	laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
						for (; k < laserCloudDepth - 1; k++)
						{
							laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
								laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * (k + 1)];
							// laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
							// 	laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * (k + 1)];
						}
						laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
							laserCloudCubeCornerPointer;
						// laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
						// 	laserCloudCubeSurfPointer;
						laserCloudCubeCornerPointer->clear();
						// laserCloudCubeSurfPointer->clear();
					}
				}
				centerCubeK--;
				laserCloudCenDepth--;
			}

			// 以上操作维护了一个局部地图，保证当前帧不在这个局部地图的边界，
			// laserCloudCornerArray和laserCloudSurfArray中存的就是每个栅格点云的指针，
			// 这样才可以从局部地图中获取足够的约束
			//这两个变量都是记录小局部地图有效各自的数量，由于前面的处理，根本不会出现无效的格子，因为地图边缘都留出了3个格子
			int laserCloudValidNum = 0;
			int laserCloudSurroundNum = 0;
			// 距离越远，点越稀疏，在局部地图中再次选取一个更小的局部地图 
			// 以当前格子为中心，选出一定范围内的点云
			// 在x、y方向上，左右各取两个栅格，z方向，上下各取一个栅格：5*5*3
			// 250m*250m*150m 的小局部地图，用来做scan2map
			for (int i = centerCubeI - 2; i <= centerCubeI + 2; i++)
			{
				for (int j = centerCubeJ - 2; j <= centerCubeJ + 2; j++)
				{
					for (int k = centerCubeK - 1; k <= centerCubeK + 1; k++)
					{
						// 确保不越界
						if (i >= 0 && i < laserCloudWidth && j >= 0 && j < laserCloudHeight && k >= 0 && k < laserCloudDepth)
						{
							// 记录当前小局部地图中的三维栅格的索引
							laserCloudValidInd[laserCloudValidNum] = i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k;
							laserCloudValidNum++;
							laserCloudSurroundInd[laserCloudSurroundNum] = i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k;
							laserCloudSurroundNum++;
						}
					}
				}
			}
			// 线特征的小局部地图
			laserCloudCornerFromMap->clear();
			// 面特征的小局部地图
			// laserCloudSurfFromMap->clear();

			// 开始真正构建用来这一帧优化的小局部地图
			for (int i = 0; i < laserCloudValidNum; i++)
			{
				// 先取小局部地图中的三维栅格的索引，再取出栅格中的点云指针，最后解引用，再累加，就得到小局部地图 laserCloudCornerArray中是map世界系下的坐标
				*laserCloudCornerFromMap += *laserCloudCornerArray[laserCloudValidInd[i]];
				// *laserCloudSurfFromMap += *laserCloudSurfArray[laserCloudValidInd[i]];
			}
			//统计线特征小局部地图的点云个数
			int laserCloudCornerFromMapNum = laserCloudCornerFromMap->points.size();
			// int laserCloudSurfFromMapNum = laserCloudSurfFromMap->points.size();

			//对当前帧点云做下采样，减少计算量
			pcl::PointCloud<PointType>::Ptr laserCloudCornerStack(new pcl::PointCloud<PointType>());
			// 对当前帧的角点和面点进行下采样
			downSizeFilterCorner.setInputCloud(laserCloudCornerLast);
			// 保存下采样的点云 下采样后的角点点云
			downSizeFilterCorner.filter(*laserCloudCornerStack);
			int laserCloudCornerStackNum = laserCloudCornerStack->points.size();




			// ---------------------- scan2map，精匹配当前帧到小局部地图的过程 --------------------------------
			// 小局部地图中的角点和面点不能太少
			// if (laserCloudCornerFromMapNum > 10 && laserCloudSurfFromMapNum > 50)
			if (laserCloudCornerFromMapNum > 10)
			{
				TicToc t_opt;
				TicToc t_tree;
				// 对局部地图建立KDtree
				// 地图坐标系下P_map：把小局部地图的角点和面点分别用KDTree存储，便于最近邻搜索
				kdtreeCornerFromMap->setInputCloud(laserCloudCornerFromMap);
				// kdtreeSurfFromMap->setInputCloud(laserCloudSurfFromMap);
				// 构建KDtree比较耗时 建立kd-tree的时间为
				printf("build KDtree time %f ms \n", t_tree.toc());
				// 根据前端的位姿，迭代两次scan2map得到优化后的位姿
				for (int iterCount = 0; iterCount < 2; iterCount++)
				{
					//ceres::LossFunction *loss_function = NULL;
					// 建立ceres问题，和前端一样
					ceres::LossFunction *loss_function = new ceres::HuberLoss(0.1);
					//适配ceres2.1版本
					ceres::LocalParameterization *q_parameterization = new ceres::EigenQuaternionParameterization();
					//todo 适配ceres2.2版本 目前还是有bug没有解决 暂时不用 只用2.1版本的ceres
					// ceres::Manifold *q_parameterization = new ceres::EigenQuaternionManifold();
					ceres::Problem::Options problem_options;

					ceres::Problem problem(problem_options);
					// q，不符合加法操作，所以要定义LocalParameter
					problem.AddParameterBlock(parameters, 4, q_parameterization);
					// t，符合加法操作
					problem.AddParameterBlock(parameters + 4, 3);

					TicToc t_data;
					int corner_num = 0;
					// 构建角点相关的约束 构建约束用当前帧下采样过后的每个点构建
					// 遍历当前帧下采样之后的角点，即线特征 laserCloudCornerStackNum是laserCloudCornerLast下采样后得到的laserCloudCornerStack的角点点云的个数
					for (int i = 0; i < laserCloudCornerStackNum; i++)
					{
						// 取出当前帧下采样之后的边缘点 pointOri是一个全局变量，是用来取出一个个点进行处理的
						pointOri = laserCloudCornerStack->points[i];
						//double sqrtDis = pointOri.x * pointOri.x + pointOri.y * pointOri.y + pointOri.z * pointOri.z;
						
						// 把当前点根据初值，投影到地图坐标系下
						//每次大循环处理完时间戳后都更新了q_w_curr和t_w_curr
						pointAssociateToMap(&pointOri, &pointSel);
						/*
						在后端的当前帧与地图匹配的时候，我们需要从局部地图中寻找线特征和面特征的约束						
						线特征的提取：
						通过kdtree在小局部地图中寻找5个最近的点，为了判断它们是否符合线特征的特性，我们需要
						对其进行特征值分解，通常来说，当5个点都在同一条直线上时，它们的协方差矩阵只有一个主方向，也就是
						特征值是一个大特征值，以及两个小特征值，最大特征值对应的特征向量就对应着直线的方向向量
						*/

						// 在小局部地图构建的KDTree中寻找距离当前帧角点最近的5个点，其索引记录在pointSearchInd
						// map的世界坐标系下搜索
						kdtreeCornerFromMap->nearestKSearch(pointSel, 5, pointSearchInd, pointSearchSqDis);
					
						// 如果最近五个点中的最后一个点(从大到小)距离小于1
						if (pointSearchSqDis[4] < 1.0)
						{
							std::vector<Eigen::Vector3d> nearCorners;
							Eigen::Vector3d center(0, 0, 0);

							for (int j = 0; j < 5; j++)
							{
								Eigen::Vector3d tmp(laserCloudCornerFromMap->points[pointSearchInd[j]].x,
													laserCloudCornerFromMap->points[pointSearchInd[j]].y,
													laserCloudCornerFromMap->points[pointSearchInd[j]].z);
								center = center + tmp;
								// 储存五个最近点
								nearCorners.push_back(tmp);
							}
							// 五个点的均值
							center = center / 5.0;
							// 协方差矩阵 3*3
							Eigen::Matrix3d covMat = Eigen::Matrix3d::Zero();
							for (int j = 0; j < 5; j++)
							{
								// 去中心化：x-μ
								Eigen::Matrix<double, 3, 1> tmpZeroMean = nearCorners[j] - center;
								//协方差矩阵的意义是三个轴上的点的离散程度，把所有点的三个轴的坐标当做了三个随机变量，由于做了去中心化处理，均值已经变成0了
								covMat = covMat + tmpZeroMean * tmpZeroMean.transpose();
							}
							// 特征值分解 齐次线性方程组：AX = 0
							Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(covMat);

							// 特征是从小到大排序，根据特征值分解来看看是不是真正的线特征
							// 最大特征值对应的特征向量就对应着直线的方向向量
							Eigen::Vector3d unit_direction = saes.eigenvectors().col(2);
							// 当前帧下采样之后的角点
							Eigen::Vector3d curr_point(pointOri.x, pointOri.y, pointOri.z);

							// 最大特征值大于次大特征值的3倍，则认为是线特征
							if (saes.eigenvalues()[2] > 3 * saes.eigenvalues()[1])
							{
								Eigen::Vector3d point_on_line = center;
								Eigen::Vector3d point_a, point_b;

								// 根据拟合出来的线特征的方向(unit_direction)，以平均点为中心构建两个虚拟点
								// 中心点往上0.1个特征向量(主方向)，0.1是步长，步长*方向 = 距离
								point_a = 0.1 * unit_direction + point_on_line;
								// 中心点往下0.1个特征向量(主方向)
								point_b = -0.1 * unit_direction + point_on_line;
								// 参数：当前帧的角点、局部地图中最近的两个线特征
								ceres::CostFunction *cost_function = LidarEdgeFactor::Create(curr_point, point_a, point_b, 1.0);
								// 构建点到线的距离的残差和前端一致
								problem.AddResidualBlock(cost_function, loss_function, parameters, parameters + 4);
								corner_num++;
							}
						}

					}

					// int surf_num = 0;
					/*
					面特征的提取：
					同样首先通过KDtree在地图中找到最近的5个面特征，使用平面拟合的方式：
					平面方程：Ax + By + Cz + D = 0，等式两边乘以1/D，得到，Ax + By + Cz + 1 = 0
					也就是三个未知数，五个方程，写成矩阵的形式就是5*3大小的矩阵，求出解(A'B'C')后，对解进行校验
					来观察是否符合平面约束，具体是分别求出5个点到平面的距离，如果有的太远则说明平面拟合不成功
					*/
					// for (int i = 0; i < laserCloudSurfStackNum; i++)
					// {
					// 	// 取出当前帧的面点
					// 	pointOri = laserCloudSurfStack->points[i];
					// 	// double sqrtDis = pointOri.x * pointOri.x + pointOri.y * pointOri.y + pointOri.z * pointOri.z;
					// 	// P_curr_map
					// 	pointAssociateToMap(&pointOri, &pointSel);
					// 	// map坐标系下搜索
					// 	kdtreeSurfFromMap->nearestKSearch(pointSel, 5, pointSearchInd, pointSearchSqDis);
					// 	/*
					// 	AX + BY + CZ = -1
					// 	|x0 y0 z0|   |A|
					// 	|x1 y1 z1| * |B| = -I
					// 	|  ....  |   |C|

					// 	5*3 * 3*1 = 5*1
					// 	*/
					// 	Eigen::Matrix<double, 5, 3> matA0;
					// 	Eigen::Matrix<double, 5, 1> matB0 = -1 * Eigen::Matrix<double, 5, 1>::Ones();
						
					// 	if (pointSearchSqDis[4] < 1.0)// 1.0米
					// 	{
					// 		// 五个面点构建A矩阵
					// 		for (int j = 0; j < 5; j++)
					// 		{
					// 			matA0(j, 0) = laserCloudSurfFromMap->points[pointSearchInd[j]].x;
					// 			matA0(j, 1) = laserCloudSurfFromMap->points[pointSearchInd[j]].y;
					// 			matA0(j, 2) = laserCloudSurfFromMap->points[pointSearchInd[j]].z;
					// 			//printf(" pts %f %f %f ", matA0(j, 0), matA0(j, 1), matA0(j, 2));
					// 		}
					// 		// find the norm of plane
					// 		// AX = B ，平面法向量norm = (A,B,C) 
					// 		Eigen::Vector3d norm = matA0.colPivHouseholderQr().solve(matB0);
					// 		// 模的倒数 == D
					// 		double negative_OA_dot_norm = 1 / norm.norm();
					// 		// 法向量归一化，其模为1
					// 		norm.normalize();

					// 		// Here n(pa, pb, pc) is unit norm of plane
					// 		bool planeValid = true;
					// 		// 求五个面点到达平面的距离，根据求出的平面方程进行校验，看看是不是符合平面约束
					// 		for (int j = 0; j < 5; j++)
					// 		{
					// 			// if OX * n > 0.2, then plane is not fit well
					// 			/*
					// 					|Ax0 + By0 + Cz0 + D|
					// 				d = ---------------------- = |Ax0 + By0 + Cz0 + D|/1
					// 					   (A² + B² + C²)½
					// 					       1
					// 				D = ----------------
					// 					(A² + B² + C²)½
					// 			*/
					// 			if (fabs(norm(0) * laserCloudSurfFromMap->points[pointSearchInd[j]].x +
					// 					 norm(1) * laserCloudSurfFromMap->points[pointSearchInd[j]].y +
					// 					 norm(2) * laserCloudSurfFromMap->points[pointSearchInd[j]].z +
					// 					negative_OA_dot_norm) > 0.2)
					// 			{
					// 				// 点到面的距离太远，不是一个好的面点
					// 				planeValid = false;
					// 				break;
					// 			}
					// 		}
							
					// 		Eigen::Vector3d curr_point(pointOri.x, pointOri.y, pointOri.z);
					// 		// 如果平面有效，就构建点到面的约束
					// 		if (planeValid)
					// 		{
					// 			// 利用平面方程构建约束，和前端的构建形式不同
					// 			// 参数：当前帧的角点、地图点中的平面法向量、D模的倒数
					// 			ceres::CostFunction *cost_function = LidarPlaneNormFactor::Create(curr_point, norm, negative_OA_dot_norm);
					// 			problem.AddResidualBlock(cost_function, loss_function, parameters, parameters + 4);
					// 			// 面数加1
					// 			surf_num++;
					// 		}
					// 	}

					// }

					//printf("corner num %d used corner num %d \n", laserCloudCornerStackNum, corner_num);
					//printf("surf num %d used surf num %d \n", laserCloudSurfStackNum, surf_num);
					// 构建角点和面点约束的时间
					printf("mapping data assosiation time %f ms \n", t_data.toc());

					TicToc t_solver;
					ceres::Solver::Options options;
					options.linear_solver_type = ceres::DENSE_QR;
					options.max_num_iterations = 4;
					options.minimizer_progress_to_stdout = false;
					options.check_gradients = false;
					options.gradient_check_relative_precision = 1e-4;
					ceres::Solver::Summary summary;
					ceres::Solve(options, &problem, &summary);
					// 一次ceres求解的时间
					printf("mapping solver time %f ms \n", t_solver.toc());

					//printf("time %f \n", timeLaserOdometry);
					//printf("corner factor num %d surf factor num %d\n", corner_num, surf_num);
					//printf("result q %f %f %f %f result t %f %f %f\n", parameters[3], parameters[0], parameters[1], parameters[2],
					//	   parameters[4], parameters[5], parameters[6]);
				}// 两次迭代优化用的时间
				printf("scan to little local map optimization time %f \n", t_opt.toc());
			}// scan2map优化出最优位姿 即：优化后的 T_curr2map，优化后的T_curr2odom??
			else// 地图中的角点和面点太少
			{
				ROS_WARN("little local Map corner and surf num are not enough");
			}

			// ----------------------- 地图更新 ---------------------			 
			/*
			下面要做的是更新地图模块中维护的一个位姿，这个位姿就是odom到map之间的位姿变换
			前端里程计会高频的发布当前帧到里程计坐标系下的位姿给后端
			T_odom2map,就是在后端通过Ceres得到当前帧到map的位姿后，再计算odom到map的位姿，所以要更新这个位姿，为下一帧做准备
			并且在进行栅格地图位置更新处理的时候，也通过上一帧维护的T_odom2map，得到当前帧的一个初值估计。transformAssociateToMap()

			*/
			// 更新：T_odmo2map = T_curr2map(优化后的) * T_curr2odom¯¹(前端给的) 优化后的T_curr2map就是q_w_curr和t_w_curr
			transformUpdate();

			TicToc t_add;

			// 将优化后的当前帧角点加到局部地图中
			for (int i = 0; i < laserCloudCornerStackNum; i++)
			{
				//把当前帧的点映射到map世界系下
				pointAssociateToMap(&laserCloudCornerStack->points[i], &pointSel);
				// 算出这个点所在三维栅格的索引
				int cubeI = int((pointSel.x + 25.0) / 50.0) + laserCloudCenWidth;
				int cubeJ = int((pointSel.y + 25.0) / 50.0) + laserCloudCenHeight;
				int cubeK = int((pointSel.z + 25.0) / 50.0) + laserCloudCenDepth;
				
				if (pointSel.x + 25.0 < 0)
					cubeI--;
				if (pointSel.y + 25.0 < 0)
					cubeJ--;
				if (pointSel.z + 25.0 < 0)
					cubeK--;
				// 栅格在范围之内
				if (cubeI >= 0 && cubeI < laserCloudWidth &&
					cubeJ >= 0 && cubeJ < laserCloudHeight &&
					cubeK >= 0 && cubeK < laserCloudDepth)
				{

					// 将三维索引转换为一维索引
					int cubeInd = cubeI + laserCloudWidth * cubeJ + laserCloudWidth * laserCloudHeight * cubeK;
					// 将当前帧角点添加到局部地图中
					laserCloudCornerArray[cubeInd]->push_back(pointSel);
				}
			}

			//todo 原来的面点局部地图已经完全不维护，但是为了rviz可视化地面点，利用laserCloudSurfArray保存CSF滤波出的地面点发送出去，但是不用surround那个话题，新做一个话题消息
			// 将优化后的当前帧面点加到局部地图中
			// for (int i = 0; i < laserCloudSurfStackNum; i++)
			// {
			// 	pointAssociateToMap(&laserCloudSurfStack->points[i], &pointSel);

			// 	int cubeI = int((pointSel.x + 25.0) / 50.0) + laserCloudCenWidth;
			// 	int cubeJ = int((pointSel.y + 25.0) / 50.0) + laserCloudCenHeight;
			// 	int cubeK = int((pointSel.z + 25.0) / 50.0) + laserCloudCenDepth;

			// 	if (pointSel.x + 25.0 < 0)
			// 		cubeI--;
			// 	if (pointSel.y + 25.0 < 0)
			// 		cubeJ--;
			// 	if (pointSel.z + 25.0 < 0)
			// 		cubeK--;

			// 	if (cubeI >= 0 && cubeI < laserCloudWidth &&
			// 		cubeJ >= 0 && cubeJ < laserCloudHeight &&
			// 		cubeK >= 0 && cubeK < laserCloudDepth)
			// 	{
			// 		int cubeInd = cubeI + laserCloudWidth * cubeJ + laserCloudWidth * laserCloudHeight * cubeK;
			// 		// 将当前帧面点添加到局部地图中
			// 		laserCloudSurfArray[cubeInd]->push_back(pointSel);
			// 	}
			// }
			// printf("add points time %f ms\n", t_add.toc());

			TicToc t_filter;

			// 把当期帧涉及到的小局部地图(±250，±150)的栅格做一个下采样再放到大地图里
			for (int i = 0; i < laserCloudValidNum; i++)
			{
				int ind = laserCloudValidInd[i];

				pcl::PointCloud<PointType>::Ptr tmpCorner(new pcl::PointCloud<PointType>());
				downSizeFilterCorner.setInputCloud(laserCloudCornerArray[ind]);
				downSizeFilterCorner.filter(*tmpCorner);
				// 滤波后小局部地图的角点，加入大局部地图
				laserCloudCornerArray[ind] = tmpCorner;

				// pcl::PointCloud<PointType>::Ptr tmpSurf(new pcl::PointCloud<PointType>());
				// downSizeFilterSurf.setInputCloud(laserCloudSurfArray[ind]);
				// downSizeFilterSurf.filter(*tmpSurf);
				// // 滤波后小局部地图的面点，加入大局部地图
				// laserCloudSurfArray[ind] = tmpSurf;
			}
			printf("filter time %f ms \n", t_filter.toc());
			
			TicToc t_pub;
			//todo CSF对面点云滤波存下来在rviz中显示地面点 面点可能太多了，进行一次降采样
			// TicToc t_csf;
			// pcl::PointCloud<PointType>::Ptr laserCloudSurfStack(new pcl::PointCloud<PointType>());
			// downSizeFilterSurf.setInputCloud(laserCloudSurfLast);
			// // 保存下采样的点云
			// downSizeFilterSurf.filter(*laserCloudSurfStack);
			// // 当前帧下采样之后，面点个数
			// int laserCloudSurfStackNum = laserCloudSurfStack->points.size();
			// 局部地图提取的时间是
			// printf("little local map prepare time %f ms\n", t_shift.toc());
			// 局部地图中的角点数为 面点数是
			// printf("little local map corner num %d  surf num %d \n", laserCloudCornerFromMapNum, laserCloudSurfFromMapNum);

			CSF csf;
			csf.params.iterations = 600;
			csf.params.time_step = 0.95;
			csf.params.cloth_resolution = 3;
			csf.params.bSloopSmooth = false;

			csf.setPointCloud(*laserCloudSurfLast);

			std::vector<int> groundIndexes, offGroundIndexes;
			pcl::PointCloud<pcl::PointXYZI>::Ptr groundFrame(new pcl::PointCloud<pcl::PointXYZI>);
			pcl::PointCloud<pcl::PointXYZI>::Ptr groundFrame2(new pcl::PointCloud<pcl::PointXYZI>);
			pcl::PointCloud<pcl::PointXYZI>::Ptr offGroundFrame(new pcl::PointCloud<pcl::PointXYZI>);
			csf.do_filtering(groundIndexes, offGroundIndexes);
			pcl::copyPointCloud(*laserCloudSurfLast, groundIndexes, *groundFrame);
			pcl::copyPointCloud(*laserCloudSurfLast, offGroundIndexes, *offGroundFrame);
			// printf("csf time %f ms\n", t_csf.toc());

			//todo 提取出的地面点存下来 发布出去 发布前转换一下坐标
			TicToc t_pub_csf;
			for (int i = 0; i < groundFrame->points.size(); i++)
			{
				// p_map = P_curr * T_curr2map
				pointAssociateToMap(&groundFrame->points[i], &pointSel);
				groundFrame2->push_back(pointSel);
			}
			sensor_msgs::PointCloud2 CSFGroundPointsCloud;
			pcl::toROSMsg(*groundFrame2, CSFGroundPointsCloud);
			CSFGroundPointsCloud.header.stamp = ros::Time().fromSec(timeLaserOdometry);
			CSFGroundPointsCloud.header.frame_id = "camera_init";
			pubLaserCloudCSFGroundPoints.publish(CSFGroundPointsCloud);
			printf("pub csf time %f ms\n", t_pub_csf.toc());


			//publish surround map for every 5 frame
			// 每5帧发布大局部地图
			if (frameCount % 5 == 0)
			{
				laserCloudSurround->clear();
				// 把当前帧相关的小局部地图发布出去
				for (int i = 0; i < laserCloudSurroundNum; i++)
				{
					int ind = laserCloudSurroundInd[i];
					//laserCloudCornerArray中存的是map世界系下的点云
					*laserCloudSurround += *laserCloudCornerArray[ind];
					// laserCloudSurround += *laserCloudSurfArray[ind];
				}

				sensor_msgs::PointCloud2 laserCloudSurround3;
				pcl::toROSMsg(*laserCloudSurround, laserCloudSurround3);
				laserCloudSurround3.header.stamp = ros::Time().fromSec(timeLaserOdometry);
				laserCloudSurround3.header.frame_id = "camera_init";
				pubLaserCloudSurround.publish(laserCloudSurround3);
			}
			// 每20帧发布全量的局部地图
			if (frameCount % 20 == 0)
			{
				pcl::PointCloud<PointType> laserCloudMap;
				// 21*21*11 = 4851
				for (int i = 0; i < 4851; i++)
				{
					laserCloudMap += *laserCloudCornerArray[i];
					// laserCloudMap += *laserCloudSurfArray[i];
				}
				sensor_msgs::PointCloud2 laserCloudMsg;
				pcl::toROSMsg(laserCloudMap, laserCloudMsg);
				laserCloudMsg.header.stamp = ros::Time().fromSec(timeLaserOdometry);
				laserCloudMsg.header.frame_id = "camera_init";
				pubLaserCloudMap.publish(laserCloudMsg);
				pcl::io::savePCDFileASCII("/home/roma/a-loam/RSout.pcd",laserCloudMap);
			}

			int laserCloudFullResNum = laserCloudFullRes->points.size();
			for (int i = 0; i < laserCloudFullResNum; i++)
			{
				// p_map = P_curr * T_curr2map
				pointAssociateToMap(&laserCloudFullRes->points[i], &laserCloudFullRes->points[i]);
			}

			//link3d
			int laserCloudLink3dNum = laserCloudLink3dPtr->points.size();
			for (int i = 0; i < laserCloudLink3dNum; i++)
			{
				// p_map = P_curr * T_curr2map
				pointAssociateToMap(&laserCloudLink3dPtr->points[i], &laserCloudLink3dPtr->points[i]);
			}

			// 发布当前帧只去除nan点的完整点云，但是转换到map世界系下才发布的
			sensor_msgs::PointCloud2 laserCloudFullRes3;
			pcl::toROSMsg(*laserCloudFullRes, laserCloudFullRes3);
			laserCloudFullRes3.header.stamp = ros::Time().fromSec(timeLaserOdometry);
			laserCloudFullRes3.header.frame_id = "camera_init";
			pubLaserCloudFullRes.publish(laserCloudFullRes3);

			//link3d
			sensor_msgs::PointCloud2 laserCloudLink3d;
			pcl::toROSMsg(*laserCloudLink3dPtr, laserCloudLink3d);
			laserCloudLink3d.header.stamp = ros::Time().fromSec(timeLaserOdometry);
			laserCloudLink3d.header.frame_id = "camera_init";
			pubLaserCloudLink3d.publish(laserCloudLink3d);

			printf("mapping pub time %f ms \n", t_pub.toc());

			printf("whole mapping time %f ms +++++\n", t_whole.toc());

			nav_msgs::Odometry odomAftMapped;
			odomAftMapped.header.frame_id = "camera_init";
			odomAftMapped.child_frame_id = "/aft_mapped";
			odomAftMapped.header.stamp = ros::Time().fromSec(timeLaserOdometry);
			odomAftMapped.pose.pose.orientation.x = q_w_curr.x();
			odomAftMapped.pose.pose.orientation.y = q_w_curr.y();
			odomAftMapped.pose.pose.orientation.z = q_w_curr.z();
			odomAftMapped.pose.pose.orientation.w = q_w_curr.w();
			odomAftMapped.pose.pose.position.x = t_w_curr.x();
			odomAftMapped.pose.pose.position.y = t_w_curr.y();
			odomAftMapped.pose.pose.position.z = t_w_curr.z();

			os_pose << q_w_curr.x() << " " << q_w_curr.y()<< " "  << q_w_curr.z() << " " << q_w_curr.w() << " " << t_w_curr.x() << " " << t_w_curr.y() << " " << t_w_curr.z() << endl;

			// 发布当前帧位姿
			pubOdomAftMapped.publish(odomAftMapped);

			geometry_msgs::PoseStamped laserAfterMappedPose;
			laserAfterMappedPose.header = odomAftMapped.header;
			laserAfterMappedPose.pose = odomAftMapped.pose.pose;
			laserAfterMappedPath.header.stamp = odomAftMapped.header.stamp;
			laserAfterMappedPath.header.frame_id = "camera_init";
			laserAfterMappedPath.poses.push_back(laserAfterMappedPose);
			
			// 经过scan to Map 精估计优化后的当前帧轨迹
			pubLaserAfterMappedPath.publish(laserAfterMappedPath);

			static tf::TransformBroadcaster br;
			tf::Transform transform;
			tf::Quaternion q;
			// T_curr2map
			transform.setOrigin(tf::Vector3(t_w_curr(0),
											t_w_curr(1),
											t_w_curr(2)));
			q.setW(q_w_curr.w());
			q.setX(q_w_curr.x());
			q.setY(q_w_curr.y());
			q.setZ(q_w_curr.z());
			transform.setRotation(q);
			// 发布tf
			br.sendTransform(tf::StampedTransform(transform, odomAftMapped.header.stamp, "camera_init", "/aft_mapped"));

			frameCount++;
		}
		std::chrono::milliseconds dura(2);
		// 避免占用cpu的内存过多，休息一下，让cpu资源可以释放出来
        std::this_thread::sleep_for(dura);
	}
}
/*
ALOAM方法实现了低的漂移,并且计算的复杂度低,实时性很好.并且不需要高精度的lidar和惯导
这个方法的核心思想就是把SLAM问题进行了拆分,通过两个算法来进行.一个是执行高频率的前端里程计
但是低精度的运动估计（定位）,另一个算法在比定位低一个数量级的频率执行后端建图（建图和校正里程计）
*/
int main(int argc, char **argv)
{

	ros::init(argc, argv, "laserMapping");
	ros::NodeHandle nh;

	os_pose.open("/home/roma/a-loam_union_ws/pose/Park_Poses.txt", std::fstream::out);

	float lineRes = 0;
	float planeRes = 0;
	// 线特征点云的分辨率
	nh.param<float>("mapping_line_resolution", lineRes, 0.4);
	// 面特征点云的分辨率
	nh.param<float>("mapping_plane_resolution", planeRes, 0.8);
	printf("line resolution %f plane resolution %f \n", lineRes, planeRes);
	// 全局变量，避免计算量太大，进行下采样，体素滤波
	downSizeFilterCorner.setLeafSize(lineRes, lineRes,lineRes);
	downSizeFilterSurf.setLeafSize(planeRes, planeRes, planeRes);
	// 订阅里程计次边缘点
	ros::Subscriber subLaserCloudCornerLast = nh.subscribe<sensor_msgs::PointCloud2>("/laser_cloud_corner_last", 100, laserCloudCornerLastHandler);
	// 订阅里程计次面点
	ros::Subscriber subLaserCloudSurfLast = nh.subscribe<sensor_msgs::PointCloud2>("/laser_cloud_surf_last", 100, laserCloudSurfLastHandler);
	// 订阅前端里程计发布的位姿
	ros::Subscriber subLaserOdometry = nh.subscribe<nav_msgs::Odometry>("/laser_odom_to_init", 100, laserOdometryHandler);
	// 订阅完整点云
	ros::Subscriber subLaserCloudFullRes = nh.subscribe<sensor_msgs::PointCloud2>("/velodyne_cloud_3", 100, laserCloudFullResHandler);

	//link3d
	//订阅odom发送的Link3D点云
	ros::Subscriber subLaserCloudLink3d = nh.subscribe<sensor_msgs::PointCloud2>("/odomLink3d_cloud", 100, laserCloudLink3dHandler);


	/*
	订阅四种消息
		当前帧全部点云(经过一次降采样)
		上一帧的边线点集合
		上一帧的平面点集合
		当前帧的位姿粗估计(帧间匹配)
	发布六种消息
		附近5帧组成的降采样子地图 for rviz
		所有帧组成的点云地图
		经过Map to Map精估计优化后当前帧位姿精估计
		当前帧原始点云（从velodyne_cloud_3订阅来的点云未经其他处理）
		里程计坐标系位姿转化到地图坐标系，mapping输出的1Hz位姿，odometry输出的10Hz位姿，整合成10hz作为最终结果
		经过scan to Map 精估计优化后的当前帧平移
	*/
	pubLaserCloudSurround = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_surround", 100);

	pubLaserCloudMap = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_map", 100);

	pubLaserCloudFullRes = nh.advertise<sensor_msgs::PointCloud2>("/velodyne_cloud_registered", 100);

	//link3d
	pubLaserCloudLink3d = nh.advertise<sensor_msgs::PointCloud2>("/link3dCloud", 100);

	pubOdomAftMapped = nh.advertise<nav_msgs::Odometry>("/aft_mapped_to_init", 100);

	pubOdomAftMappedHighFrec = nh.advertise<nav_msgs::Odometry>("/aft_mapped_to_init_high_frec", 100);

	pubLaserAfterMappedPath = nh.advertise<nav_msgs::Path>("/aft_mapped_path", 100);

	//CSF Ground Points
	pubLaserCloudCSFGroundPoints = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_CSF_ground_points", 100);

	//laserCloudNum是栅格数量
	for (int i = 0; i < laserCloudNum; i++)
	{
		// 储存后端地图的数组，元素是智能指针，在堆区开辟空间，让智能指针指向该内存 每个元素都是一个栅格，一共laserCloudNum个栅格
		laserCloudCornerArray[i].reset(new pcl::PointCloud<PointType>());
		// laserCloudSurfArray[i].reset(new pcl::PointCloud<PointType>());
	}

	// 开辟后端建图子线程，线程入口函数process
	std::thread mapping_process{process};
	ros::spin();

	return 0;
}
