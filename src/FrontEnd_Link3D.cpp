/**
 * @file FrontEnd_Link3D.cpp
 * @author ctx (cuitongxin201024@163.com)
 * @brief 前端里程计的实现
 * 1、接收原始LiDAR点云数据，进行点云预处理
 * 
 * 
 * 
 * @version 0.1
 * @date 2023-12-18
 * 
 * @copyright Copyright (c) 2023
 * 
 */

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

void extractEdgePoint(const sensor_msgs::PointCloud2ConstPtr &laserCloudMsg, ScanEdgePoints &edgePoints)
{
    //函数会被循环调用，静态变量只初始化一次
    static int frameCount = 0;
    vector<int> scanStartInd(nScans, 0);
    vector<int> scanEndInd(nScans, 0);
    //每个元素都是一个线束上的点云
    vector<pcl::PointCloud<pcl::PointXYZI>> laserCloudScans(nScans);
    int cloudSize = 0;
    int count = 0;
    float startOri;
    float endOri;
    if(nScans == 80)
    {
        pcl::fromROSMsg(*laserCloudMsg, laserCloudInRS);
        removeClosedPointCloud(laserCloudInRS, laserCloudInRS, minimumRange);
        
        cloudSize = laserCloudInRS.points.size();

        count = cloudSize;
        pcl::PointXYZI point;
        // 遍历当前帧的所有点
        for (int i = 0; i < cloudSize; i++)
        {
            point.x = laserCloudInRS.points[i].x;
            point.y = laserCloudInRS.points[i].y;
            point.z = laserCloudInRS.points[i].z;
            // ω 弧度转角度：180 / M_PI
            int scanID = laserCloudInRS.points[i].ring;
            float ori = -atan2(point.y, point.x) ;
            // TODO:Link3D水平角作为强度
            point.intensity = ori;
            // TODO: 这里的强度值设置为线束id + 点的时间间隔
            // point.intensity = scanID + scanPeriod * relTime;// LOAM
            laserCloudScans[scanID].points.push_back(point);
        }
    }
    else
    {
        pcl::fromROSMsg(*laserCloudMsg, laserCloudInVD);
        // 拷贝
        vector<int> indices;
        //去nan点
        pcl::removeNaNFromPointCloud(laserCloudInVD, laserCloudInVD, indices);
        //去近距离点
        removeClosedPointCloud(laserCloudInVD, laserCloudInVD, minimumRange);
        //统计点云总数
        cloudSize = laserCloudInVD.points.size();
        // ω0
        startOri = -atan2(laserCloudInVD.points[0].y, laserCloudInVD.points[0].x);
        // ωn
        endOri = -atan2(laserCloudInVD.points[cloudSize - 1].y, laserCloudInVD.points[cloudSize - 1].x) + 2 * M_PI;

        if (endOri - startOri > 3 * M_PI)
        {
            endOri -= 2 * M_PI;
        }
        else if (endOri - startOri < M_PI)
        {
            endOri += 2 * M_PI;
        }

        bool halfPassed = false;
        count = cloudSize;
        pcl::PointXYZI point;

        // 遍历当前帧的所有点
        for (int i = 0; i < cloudSize; i++)
        {
            point.x = laserCloudInVD.points[i].x;
            point.y = laserCloudInVD.points[i].y;
            point.z = laserCloudInVD.points[i].z;
            // ω 弧度转角度：180 / M_PI
            float angle = atan(point.z / sqrt(point.x * point.x + point.y * point.y)) * 180 / M_PI;
            int scanID = 0;

            if (nScans == 16)
            {
                scanID = int((angle + 15) / 2 + 0.5);
                if (scanID > (nScans - 1) || scanID < 0)
                {
                    count--;
                    continue;
                }
            }
            else if (nScans == 32)
            {
                scanID = int((angle + 92.0/3.0) * 3.0 / 4.0);
                if (scanID > (nScans - 1) || scanID < 0)
                {
                    count--;
                    continue;
                }
            }
            else if (nScans == 64)
            {
                if (angle >= -8.83)
                    scanID = int((2 - angle) * 3.0 + 0.5);
                else
                    scanID = nScans / 2 + int((-8.83 - angle) * 2.0 + 0.5);
                if (angle > 2 || angle < -24.33 || scanID > 50 || scanID < 0)
                {
                    count--;
                    continue;
                }
            }
            else
            {
                printf("wrong scan number\n");
            }
            // α的弧度 [-pi,+pi] atan2(x)函数返回以弧度为单位的角度，当前一个点的水平角度
            float ori = -atan2(point.y, point.x) ;

            if (!halfPassed)
            {
                if (ori < startOri - M_PI / 2)
                {
                    ori += 2 * M_PI;
                }
                else if (ori > startOri + M_PI * 3 / 2)
                {
                    ori -= 2 * M_PI;
                }

                if (ori - startOri > M_PI)
                {
                    halfPassed = true;
                }
            }
            else
            {
                ori += 2 * M_PI;
                if (ori < endOri - M_PI * 3 / 2)
                {
                    ori += 2 * M_PI;
                }
                else if (ori > endOri + M_PI / 2)
                {
                    ori -= 2 * M_PI;
                }
            }
            // TODO:Link3D水平角作为强度 notice
            point.intensity = ori;
            // TODO: 这里的强度值设置为线束id + 点的时间间隔
            // point.intensity = scanID + scanPeriod * relTime;// LOAM
            laserCloudScans[scanID].points.push_back(point);
        }

    }

    // ------------------------------------------------------------------------------------
    size_t scanSize = laserCloudScans.size();
    //
    edgePoints.resize(scanSize);
    cloudSize = count;
    std::vector<float> allCurv;
    // 遍历所有线束
    for(int i = 0; i < nScans; i++)
    {
        int laserCloudScansSize = laserCloudScans[i].size();
        // 当前线束的点的个数不能太少
        if(laserCloudScansSize >= 15)
        {
            // 遍历当前线束上的点
            for(int j = 0; j < laserCloudScansSize; j++)
            {

                if( j >= 5 && j < laserCloudScansSize - 5 )
                {
                    float diffX = laserCloudScans[i].points[j - 5].x + laserCloudScans[i].points[j - 4].x
                                  + laserCloudScans[i].points[j - 3].x + laserCloudScans[i].points[j - 2].x
                                  + laserCloudScans[i].points[j - 1].x - 10 * laserCloudScans[i].points[j].x
                                  + laserCloudScans[i].points[j + 1].x + laserCloudScans[i].points[j + 2].x
                                  + laserCloudScans[i].points[j + 3].x + laserCloudScans[i].points[j + 4].x
                                  + laserCloudScans[i].points[j + 5].x;
                    float diffY = laserCloudScans[i].points[j - 5].y + laserCloudScans[i].points[j - 4].y
                                  + laserCloudScans[i].points[j - 3].y + laserCloudScans[i].points[j - 2].y
                                  + laserCloudScans[i].points[j - 1].y - 10 * laserCloudScans[i].points[j].y
                                  + laserCloudScans[i].points[j + 1].y + laserCloudScans[i].points[j + 2].y
                                  + laserCloudScans[i].points[j + 3].y + laserCloudScans[i].points[j + 4].y
                                  + laserCloudScans[i].points[j + 5].y;
                    float diffZ = laserCloudScans[i].points[j - 5].z + laserCloudScans[i].points[j - 4].z
                                  + laserCloudScans[i].points[j - 3].z + laserCloudScans[i].points[j - 2].z
                                  + laserCloudScans[i].points[j - 1].z - 10 * laserCloudScans[i].points[j].z
                                  + laserCloudScans[i].points[j + 1].z + laserCloudScans[i].points[j + 2].z
                                  + laserCloudScans[i].points[j + 3].z + laserCloudScans[i].points[j + 4].z
                                  + laserCloudScans[i].points[j + 5].z;
                    // 曲率
                    float curv = diffX * diffX + diffY * diffY + diffZ * diffZ;
                    allCurv.push_back(curv);
                    // 曲率大的点
                    if(curv > 10 && curv < 20000)
                    {
                        float ori = laserCloudScans[i].points[j].intensity;
                        float relTime;
                        relTime  = nScans == 80 ? 0.0 : relTime = (ori - startOri) / (endOri - startOri);

                        PointXYZSCA tmpPt;
                        tmpPt.x = laserCloudScans[i].points[j].x;
                        tmpPt.y = laserCloudScans[i].points[j].y;
                        tmpPt.z = laserCloudScans[i].points[j].z;
                        // ring：整数部分是scan线束的索引，小数部分是相对起始时刻的时间
                        tmpPt.scan_position = i + scanPeriod * relTime;
                        //cout << "tmpPt.scan_position = "<< int(tmpPt.scan_position) << endl;
                        tmpPt.curvature = curv;
                        tmpPt.angle = ori;
                        // 存入当前线束的角点
                        edgePoints[i].emplace_back(tmpPt);
                        // TODO: notice使用完了之后再恢复回去，因为后面要用到
                        // point.intensity = scanID + scanPeriod * relTime;// LOAM
                        laserCloudScans[i].points[j].intensity = i + scanPeriod * relTime;
                    }
                }
                else
                {
                    allCurv.push_back(-1.0);
                }
            }
        }
    }

    // ---------------------------------------- ALOAM -----------------------------------------
    for(int i = 0; i < nScans; i++)
    {
        scanStartInd[i] = laserCloud.size() + 5;
        laserCloud += laserCloudScans[i];
        scanEndInd[i] = laserCloud.size() - 6;
    }
    for (int i = 5; i < cloudSize - 5; i++)
    {
        cloudCurvature[i] = allCurv[i];
        // 储存当前计算曲率的点的ID，cloudSortInd[i] = i相当于所有点的初始自然序列，每个点得到它自己的序号(索引)
        cloudSortInd[i] = i;
        cloudNeighborPicked[i] = 0;
        cloudLabel[i] = 0;
    }

    // --------------------------  提取每一条线束上的 2 种特征 ----------------------------
    for (int i = 0; i < nScans; i++)
    {
        // 去当前线去头去尾后少于6个点，说明无法分成6个扇区，跳过
        if( scanEndInd[i] - scanStartInd[i] < 6)
            continue;
        // 用来存储不太平整的点
        pcl::PointCloud<pcl::PointXYZI>::Ptr surfPointsLessFlatScan(new pcl::PointCloud<pcl::PointXYZI>);
        // 将每条scan平均分成6等份，为了使特征点均匀分布，将一个scan分成6个扇区
        for (int j = 0; j < 6; j++)
        {
            // 每一个等份的起始标志位
            int sp = scanStartInd[i] + (scanEndInd[i] - scanStartInd[i]) * j / 6;
            // 每一个等份的终止标志位
            int ep = scanStartInd[i] + (scanEndInd[i] - scanStartInd[i]) * (j + 1) / 6 - 1;
            // 对每一个等份中的点，根据曲率的大小排序，曲率小的在前，大的在后面
            //std::sort(cloudCurvature+cloudSortInd + sp , cloudCurvature+cloudSortInd + ep + 1 );
            std::sort(cloudSortInd + sp, cloudSortInd + ep + 1, comp);
            // t_q_sort累计每个扇区曲率排序时间总和
            // 选取极大边线点（2个）和次极大边线点（20个）
            int largestPickedNum = 0;

            // -------------- 提取线点 -----------------
            // 遍历当前等份，因为曲率大的在后面，这里从后往前找
            for (int k = ep; k >= sp; k--)
            {
                // 排序后顺序就乱了，这个时候索引的作用就体现出来了，根据曲率排序后的点的ID
                int ind = cloudSortInd[k];
                // 判断当前点是否被选过，同时对应曲率是否大于阈值
                if (cloudNeighborPicked[ind] == 0 && cloudCurvature[ind] > 0.1)
                {
                    largestPickedNum++;
                    if (largestPickedNum <= 20)
                    {
                        // 给曲率稍微大的点打上标签
                        cloudLabel[ind] = 1;
                        cornerPointsLessSharp.push_back(laserCloud.points[ind]);
                    }
                    else// 超过20个就跳过
                    {
                        break;
                    }
                    // 当前点被选取后，Picked被置位1
                    cloudNeighborPicked[ind] = 1;
                    // 为了保证曲率大的特征点不过度集中，将当前点的左右各五个点置位1，避免后续会选择到作为特征点
                    for (int l = 1; l <= 5; l++)
                    {
                        // 一圈是1800个点，1800/360 = 5，每1°有五个点，1/5 = 0.2，每一个点的间隔为0.2°
                        // 计算当前点与相邻点的距离
                        float diffX = laserCloud.points[ind + l].x - laserCloud.points[ind + l - 1].x;
                        float diffY = laserCloud.points[ind + l].y - laserCloud.points[ind + l - 1].y;
                        float diffZ = laserCloud.points[ind + l].z - laserCloud.points[ind + l - 1].z;
                        if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05)
                        {
                            break;
                        }
                        cloudNeighborPicked[ind + l] = 1;
                    }

                    for (int l = -1; l >= -5; l--)
                    {
                        float diffX = laserCloud.points[ind + l].x - laserCloud.points[ind + l + 1].x;
                        float diffY = laserCloud.points[ind + l].y - laserCloud.points[ind + l + 1].y;
                        float diffZ = laserCloud.points[ind + l].z - laserCloud.points[ind + l + 1].z;
                        if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05)
                        {
                            break;
                        }
                        cloudNeighborPicked[ind + l] = 1;
                    }
                }
            }

            // -------------- 下面开始挑选面点，选取极小平面点（4个）---------------
            int smallestPickedNum = 0;
            // 遍历当前等份，曲率是从小往大排序
            for (int k = sp; k <= ep; k++)
            {
                int ind = cloudSortInd[k];
                // 确保这个点没有被pick且曲率小于阈值
                if (cloudNeighborPicked[ind] == 0 && cloudCurvature[ind] < 0.1)
                {
                    // -1认为是平坦的点
                    cloudLabel[ind] = -1;
                    smallestPickedNum++;
                    // 这里不区分平坦和比较平坦，因为剩下的点label默认是0，就是比较平坦
                    // 每等分只挑选四个曲率小的点
                    if (smallestPickedNum >= 4)
                    {
                        break;
                    }
                    cloudNeighborPicked[ind] = 1;
                    // 以下为均匀化
                    for (int l = 1; l <= 5; l++)
                    {
                        float diffX = laserCloud.points[ind + l].x - laserCloud.points[ind + l - 1].x;
                        float diffY = laserCloud.points[ind + l].y - laserCloud.points[ind + l - 1].y;
                        float diffZ = laserCloud.points[ind + l].z - laserCloud.points[ind + l - 1].z;
                        if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05)
                        {
                            break;
                        }
                        cloudNeighborPicked[ind + l] = 1;
                    }
                    for (int l = -1; l >= -5; l--)
                    {
                        float diffX = laserCloud.points[ind + l].x - laserCloud.points[ind + l + 1].x;
                        float diffY = laserCloud.points[ind + l].y - laserCloud.points[ind + l + 1].y;
                        float diffZ = laserCloud.points[ind + l].z - laserCloud.points[ind + l + 1].z;
                        if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05)
                        {
                            break;
                        }
                        cloudNeighborPicked[ind + l] = 1;
                    }
                }
            }

            // 遍历当前等份
            for (int k = sp; k <= ep; k++)
            {
                // 小于等于0的认为是面点
                if (cloudLabel[k] <= 0)
                {
                    surfPointsLessFlatScan->push_back(laserCloud.points[k]);
                }
            }
        }
        pcl::PointCloud<pcl::PointXYZI> surfPointsLessFlatScanDS;
        pcl::VoxelGrid<pcl::PointXYZI> downSizeFilter;
        downSizeFilter.setInputCloud(surfPointsLessFlatScan);
        downSizeFilter.setLeafSize(0.1, 0.1, 0.1);
        downSizeFilter.filter(surfPointsLessFlatScanDS);
        surfPointsLessFlat += surfPointsLessFlatScanDS;
    }

    frameCount++;
    laserCloudInVD.clear();
    laserCloudInRS.clear();

}

//LiDAR点云回调函数
void laserCloudHandler(const sensor_msgs::PointCloud2ConstPtr &laserCloudMsg)
{
    clock_t start, end;
    double time;
    start = clock();

    //取当前原始点云帧的时间戳
    ros::Time timestamp = laserCloudMsg->header.stamp;
    // 边缘点
    ScanEdgePoints edgePoints;
    // 1. 提取当前帧的边缘点，根据线束储存边缘点
    extractEdgePoint(laserCloudMsg, edgePoints);

    pcl::PointCloud<pcl::PointXYZ> clusters_Cloud;
    ScanEdgePoints sectorAreaCloud;
    // 2.1 输入边缘点，输出3D扇形区域点，根据扇区储存边缘点
    divideArea(edgePoints, sectorAreaCloud);
    ScanEdgePoints clusters;

    // 2.2 输入扇形区域点，输出聚合点 ，大容器：所有簇，小容器：一簇的所有点
    getCluster(sectorAreaCloud, clusters);

    // 2.3 计算所有簇的质心
    getMeanKeyPoint(clusters, keyPoints_curr);

    // 3. 创建描述子
    getDescriptors(keyPoints_curr, descriptors_curr);
    if(!keyPoints_last.empty())
    {
        vector<pair<int, int>> vMatchedIndex;
        // 4. 描述子匹配
        match(keyPoints_curr, keyPoints_last,descriptors_curr, descriptors_last, vMatchedIndex);
        // cout << vMatchedIndex.data()->first << " " << vMatchedIndex.data()->second << endl;
        // cout << keyPoints_last.size() << endl;
        // 5. ICP
        Registration(keyPoints_curr, keyPoints_last, vMatchedIndex, timestamp);
        end = clock();
        time = ((double) (end - start)) / CLOCKS_PER_SEC;
        // 0.05s 0.12
        cout << "Link3D前端里程计 comsumming Time: " << time << "s" << endl;
    }


    keyPoints_last = keyPoints_curr;
    descriptors_last = descriptors_curr;


    laserCloud.clear();
    cornerPointsLessSharp.clear();
    surfPointsLessFlat.clear();

    keyPoints_curr.clear();

}

int main(int argc, char **argv)
{

    ros::init(argc, argv, "front_end_link3d");
    ros::NodeHandle nh;
    //LiDAR线束
    nh.param<int>("scan_line", nScans, 16);
    //去除近距离点阈值
    nh.param<double>("minimum_range", minimumRange, 0.5);
    nh.param<double>("FilterGroundLeaf", FilterGroundLeaf, 0.1);
    FilterGround.setLeafSize(FilterGroundLeaf,FilterGroundLeaf,FilterGroundLeaf);

    // printf("scan line number %d \n", nScans);
    // printf("minimum_range %f \n", minimumRange);
    //确保线束正确
    if(nScans != 16 && nScans != 32 && nScans != 64 && nScans != 80)
    {
        printf("only support velodyne with 16, 32 , 64 or RS80 scan line!");
        return 0;
    }
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
    ros::Publisher pubLaserCloudFullRes = nh.advertise<sensor_msgs::PointCloud2>("/velodyne_cloud_full", 100);
    // 发布里程计数据(位姿轨迹)给后端，后端接收 当前帧到初始帧的位姿
    ros::Publisher pubLaserOdometry = nh.advertise<nav_msgs::Odometry>("/laser_odom_to_init", 100);
    // 发布前端里程计的高频低精 位姿轨迹
    ros::Publisher pubLaserPath = nh.advertise<nav_msgs::Path>("/laser_odom_path", 100);
    //LinK3D 聚类关键点发布
    ros::Publisher pubLaserCloud_LinK3D = nh.advertise<sensor_msgs::PointCloud2>("/odomLink3d_cloud", 100);
    //订阅LiDAR点云 本节点处理函数均在此回调函数中
    ros::Subscriber subLaserCloud = nh.subscribe<sensor_msgs::PointCloud2>("/velodyne_points", 100, laserCloudHandler);
    
    ros::spin();
    return 0;
}


   //保存上一帧的 link3d Frame用于帧间匹配
    Frame* pPreviousFrame = nullptr;
    //存当前点云中的聚类后的关键点
    pcl::PointCloud<pcl::PointXYZI>::Ptr AggregationKeypoints_LinK3D(new pcl::PointCloud<pcl::PointXYZI>);
    // //link3d
    // ros::Publisher pubLaserCloudLink3d_odom = nh.advertise<sensor_msgs::PointCloud2>("/odomLink3d_cloud", 100);

   

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
            fullPointsBuf.pop();

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