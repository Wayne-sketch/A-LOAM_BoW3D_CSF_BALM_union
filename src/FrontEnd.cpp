//
// Created by wb on 2023/4/12.
//

#include <ros/ros.h>
#include <string>

#include <pcl/filters/voxel_grid.h>
#include <pcl/point_types.h>
#include <eigen3/Eigen/Dense>
#include <pcl/filters/extract_indices.h>
#include <ceres/ceres.h>

#include <sstream>
#include <iomanip>

#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <opencv2/highgui.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/core/core.hpp>
#include <ceres/ceres.h>
#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>

// #include <cv.hpp>
#include <opencv2/imgproc/types_c.h>

#include <opencv2/core.hpp>
#include <chrono>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <nav_msgs/Path.h>
#include <nav_msgs/Odometry.h>
#include "aloam_velodyne/rotation.h"



using namespace std;
using namespace cv;
using namespace std::chrono;

int nScans = 0;
float scanPeriod = 0.1;
double minimumRange = 0.3;
float distanceTh = 0.4;
int matchTh = 6;

float cloudCurvature[400000];
int cloudSortInd[400000];
int cloudNeighborPicked[400000];
int cloudLabel[400000];
bool comp (int i,int j) { return (cloudCurvature[i]<cloudCurvature[j]); }

struct VelodynePointXYZIRT
{
    PCL_ADD_POINT4D
    PCL_ADD_INTENSITY;
    float ring;
    // △t
    float time;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;

// 注册Velodyne点云结构
POINT_CLOUD_REGISTER_POINT_STRUCT (VelodynePointXYZIRT,
                                   (float, x, x)(float, y, y)(float, z, z)
                                           (float, intensity, intensity)
                                           (float, ring, ring)(float, time, time))
struct PointXYZSCA
{
    PCL_ADD_POINT4D;
    float scan_position;
    float curvature;
    float angle;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
}EIGEN_ALIGN16;

POINT_CLOUD_REGISTER_POINT_STRUCT(PointXYZSCA,
                                  (float, x, x)(float, y, y)(float, z, z)(float, scan_position, scan_position)(float, curvature, curvature)(float, angle, angle))
typedef vector<vector<PointXYZSCA>> ScanEdgePoints;
#define distXY(a) sqrt(a.x * a.x + a.y * a.y)


// 点云容器
pcl::PointCloud<pcl::PointXYZI> laserCloud;// 一帧原始点云
pcl::PointCloud<pcl::PointXYZI> cornerPointsLessSharp;// 次极大边线点
pcl::PointCloud<pcl::PointXYZI> surfPointsLessFlat;// 次极小平面
pcl::PointCloud<pcl::PointXYZ> laserCloudInVD;
pcl::PointCloud<VelodynePointXYZIRT> laserCloudInRS;




cv::Mat  _LastImage;
std::vector<cv::Mat> _curr_images;
std::vector< std::pair<cv::Point2f, pcl::PointXYZI> > _LastProj;
std::map<int, cv::Point2f> _2DMatch1, _2DMatch2;

ros::Publisher pubLaserCloudCornerLast;
ros::Publisher pubLaserCloudSurfLast;
ros::Publisher pubLaserCloudFullRes;
ros::Publisher pubLaserOdometry;
ros::Publisher pubLaserPath;
image_transport::Publisher pubImage;
nav_msgs::Path laserPath;
ros::Publisher pubCenter;


// 全局变量，被不断更新
Eigen::Quaterniond q_w_curr(1, 0, 0, 0);
Eigen::Vector3d t_w_curr(0, 0, 0);
Eigen::Quaterniond q_last_curr;
Eigen::Vector3d t_last_curr;

pcl::VoxelGrid<pcl::PointXYZI> FilterGround;
double FilterGroundLeaf;

pcl::PointCloud<pcl::PointXYZI> keyPoints_curr;
pcl::PointCloud<pcl::PointXYZI> keyPoints_last;

cv::Mat descriptors_curr;
cv::Mat descriptors_last;

#define distPt2Pt(a, b) sqrt((a.x - b.x)*(a.x - b.x) + (a.y - b.y)*(a.y - b.y) + (a.z - b.z)*(a.z - b.z))

struct ICPCeres
{
    // 构造函数
    ICPCeres ( cv::Point3f uvw, cv::Point3f xyz ) : _uvw(uvw),_xyz(xyz) {}
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
        p[0] += camera[3]; p[1] += camera[4]; p[2] += camera[5];
        residual[0] = T(_uvw.x)-p[0];
        residual[1] = T(_uvw.y)-p[1];
        residual[2] = T(_uvw.z)-p[2];
        return true;
    }
    static ceres::CostFunction* Create(const cv::Point3f uvw,const cv::Point3f xyz)
    {
        return (new ceres::AutoDiffCostFunction<ICPCeres, 3, 6>(
                new ICPCeres(uvw,xyz)));
    }
    const cv::Point3f _uvw;
    const cv::Point3f _xyz;
};

/**
 * @brief 去除点云近点
 * 
 * @tparam PointT 
 * @param cloud_in 
 * @param cloud_out 
 */
template <typename PointT>
void removeClosedPointCloud(const pcl::PointCloud<PointT> &cloud_in, pcl::PointCloud<PointT> &cloud_out)
{
    if (&cloud_in != &cloud_out)
    {
        cloud_out.header = cloud_in.header;
        cloud_out.points.resize(cloud_in.points.size());
    }
    size_t j = 0;
    for (size_t i = 0; i < cloud_in.points.size(); ++i)
    {
        if (cloud_in.points[i].x * cloud_in.points[i].x
            + cloud_in.points[i].y * cloud_in.points[i].y
            + cloud_in.points[i].z * cloud_in.points[i].z
            < minimumRange * minimumRange)
        {
            continue;
        }

        cloud_out.points[j] = cloud_in.points[i];
        j++;
    }
    if (j != cloud_in.points.size())
    {
        cloud_out.points.resize(j);
    }
    cloud_out.height = 1;
    cloud_out.width = static_cast<uint32_t>(j);
    cloud_out.is_dense = true;
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
        removeClosedPointCloud(laserCloudInRS, laserCloudInRS);
        
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
        removeClosedPointCloud(laserCloudInVD, laserCloudInVD);
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


// Roughly divide the areas to save time for clustering.
void divideArea(ScanEdgePoints &scanCloud, ScanEdgePoints &sectorAreaCloud)
{
    // The horizontal plane is divided into 120 sector area centered on LiDAR coordinate.
    sectorAreaCloud.resize(120);
    // 线束
    int numScansPt = scanCloud.size();
    if(numScansPt == 0)
    {
        return;
    }
    // 遍历所有边缘点线束
    for(int i = 0; i < numScansPt; i++)
    {
        // 当前一根线上的边缘点数
        int numAScanPt = scanCloud[i].size();
        // 遍历当前线束的所有边缘点
        for(int j = 0; j < numAScanPt; j++)
        {
            int areaID = 0;
            // 当前点的角度
            float angle = scanCloud[i][j].angle;
            if(angle > 0 && angle < 2 * M_PI)
            {
                areaID = std::floor((angle / (2 * M_PI)) * 120);
            }
            else if(angle > 2 * M_PI)
            {
                areaID = std::floor(((angle - 2 * M_PI) / (2 * M_PI)) * 120);
            }
            else if(angle < 0)
            {
                areaID = std::floor(((angle + 2 * M_PI) / (2 * M_PI)) * 120);
            }
            // 当前扇形区域存放的点
            sectorAreaCloud[areaID].push_back(scanCloud[i][j]);
        }
    }
}

float computeClusterMean(vector<PointXYZSCA> &cluster)
{
    float distSum = 0;
    int numPt = cluster.size();
    for(int i = 0; i < numPt; i++)
    {
        // 绝对距离的和
        distSum += distXY(cluster[i]);
    }
    // 平均距离
    return (distSum/numPt);
}

void computeXYMean(vector<PointXYZSCA> &cluster, std::pair<float, float> &xyMeans)
{
    int numPt = cluster.size();
    float xSum = 0;
    float ySum = 0;

    for(int i = 0; i < numPt; i++)
    {
        xSum += cluster[i].x;
        ySum += cluster[i].y;
    }

    float xMean = xSum/numPt;
    float yMean = ySum/numPt;
    xyMeans = std::make_pair(xMean, yMean);
}

/*
Kmeans:
    1.运算前规定K值作为聚类个数
    2.k个聚类中心需要从所有的数据集合中随机选取
    3.当聚类中心确定后，每当有一个样本被分配给聚类中心时，各个样本距离哪一个聚类中心近，就划分到那个聚类中心所属的集合
    4.一共k个集合，重新计算每个集合的聚类中心
*/
void getCluster(const ScanEdgePoints &sectorAreaCloud, ScanEdgePoints &clusters)
{
    int scanNumTh = ceil(nScans / 6);
    int ptNumTh = ceil(1.5 * scanNumTh);

    ScanEdgePoints tmpclusters;
    PointXYZSCA curvPt;
    // 初始化一个值为curvPt的容器
    vector<PointXYZSCA> dummy(1, curvPt);
    // 扇形区域个数
    int numArea = sectorAreaCloud.size();

    // Cluster for each sector area
    // 遍历所有扇形区域
    for(int i = 0; i < numArea; i++)
    {
        // 扇形区域的点数要大于6
        if(sectorAreaCloud[i].size() < 6)
            continue;
        int numPt = sectorAreaCloud[i].size();
        // 二维容器
        ScanEdgePoints curAreaCluster(1, dummy);

        // 当前扇形的第0个点
        curAreaCluster[0][0] = sectorAreaCloud[i][0];
        // 遍历当前扇形区域的所有点（除第0个点外）
        for(int j = 1; j < numPt; j++)
        {
            // 当前扇形经过聚类后点的
            // 1 2 3 3 3 3 4 5 6 7 8 9 9 10 11 12 13 14 14 15 15 16 17 18 18 18 19 19 20 21 22
            int numCluster = curAreaCluster.size();
            /*
            Kmeans:
                1.运算前规定K值作为聚类个数
                2.k个聚类中心需要从所有的数据集合中随机选取
                3.当聚类中心确定后，每当有一个样本被分配给聚类中心时，各个样本距离哪一个聚类中心近，就划分到那个聚类中心所属的集合
                4.一共k个集合，重新计算每个集合的聚类中心
            */
            // 遍历所有簇，计算中心
            for(int k = 0; k < numCluster; k++)
            {
                // 聚类：不断计算到当前簇的中心点，输入当前扇形，一开始只有一个点，随着后面输入的点越多，均值在不断的变化
                float mean = computeClusterMean(curAreaCluster[k]);// 绝对平均距离
                std::pair<float, float> xyMean;
                computeXYMean(curAreaCluster[k], xyMean);// x、y方向上的平均距离
                // 当前扇形中的一个角点
                PointXYZSCA tmpPt = sectorAreaCloud[i][j];
                // 如果当前点距离中点比较近，则认为是一簇
                if(abs(distXY(tmpPt) - mean) < distanceTh && abs(xyMean.first - tmpPt.x) < distanceTh && abs(xyMean.second - tmpPt.y) < distanceTh)
                {
                    // 加入到同一簇
                    curAreaCluster[k].emplace_back(tmpPt);
                    break;
                }
                else if(abs(distXY(tmpPt) - mean) >= distanceTh && k == numCluster-1)
                {
                    curAreaCluster.emplace_back(dummy);
                    curAreaCluster[numCluster][0] = tmpPt;
                }
                else// 不是满足同一簇条件则跳过
                {
                    continue;
                }
            }
        }
        int numCluster = curAreaCluster.size();
        // 遍历所有簇
        for(int j = 0; j < numCluster; j++)
        {
            int numPt = curAreaCluster[j].size();
            // 一簇中的点不能太少
            if(numPt < ptNumTh)
            {
                continue;
            }
            tmpclusters.emplace_back(curAreaCluster[j]);
        }
    }// end for
    int numCluster = tmpclusters.size();

    vector<bool> toBeMerge(numCluster, false);
    multimap<int, int> mToBeMergeInd;
    set<int> sNeedMergeInd;

    // Merge the neighbor clusters.合并
    for(int i = 0; i < numCluster; i++)
    {
        if(toBeMerge[i])
        {
            continue;
        }
        // 当前簇的中心点
        float means1 = computeClusterMean(tmpclusters[i]);
        std::pair<float, float> xyMeans1;
        // 当前簇x、y方向均值
        computeXYMean(tmpclusters[i], xyMeans1);
        // 遍历相邻簇
        for(int j = 1; j < numCluster; j++)
        {
            if(toBeMerge[j])
            {
                continue;
            }
            // 相邻簇的中心
            float means2 = computeClusterMean(tmpclusters[j]);
            std::pair<float, float> xyMeans2;
            computeXYMean(tmpclusters[j], xyMeans2);
            // 如果两个簇太靠近的话
            if(abs(means1 - means2) < 2*distanceTh
               && abs(xyMeans1.first - xyMeans2.first) < 2*distanceTh
               && abs(xyMeans1.second - xyMeans2.second) < 2*distanceTh)
            {
                // 第i簇和第j簇要被合并
                mToBeMergeInd.insert(std::make_pair(i, j));
                sNeedMergeInd.insert(i);
                toBeMerge[i] = true;
                toBeMerge[j] = true;
            }
        }
    }

    if(sNeedMergeInd.empty())// 如果没有要被合并的
    {
        for(int i = 0; i < numCluster; i++)
        {
            clusters.emplace_back(tmpclusters[i]);
        }
    }
    else
    {
        for(int i = 0; i < numCluster; i++)
        {
            // 先保存没有被合并的
            if(toBeMerge[i] == false)
            {
                clusters.emplace_back(tmpclusters[i]);
            }
        }

        for(auto setIt = sNeedMergeInd.begin(); setIt != sNeedMergeInd.end(); ++setIt)
        {
            // 需要合并簇的索引
            int needMergeInd = *setIt;
            auto entries = mToBeMergeInd.count(needMergeInd);
            auto iter = mToBeMergeInd.find(needMergeInd);
            vector<int> vInd;

            while(entries)
            {
                int ind = iter->second;
                vInd.emplace_back(ind);
                ++iter;
                --entries;
            }

            clusters.emplace_back(tmpclusters[needMergeInd]);
            size_t numCluster = clusters.size();

            for(size_t j = 0; j < vInd.size(); j++)
            {
                for(size_t ptNum = 0; ptNum < tmpclusters[vInd[j]].size(); ptNum++)
                {
                    clusters[numCluster - 1].emplace_back(tmpclusters[vInd[j]][ptNum]);
                }
            }
        }
    }
}// 聚合

// 获取每一簇的质心点
void getMeanKeyPoint(const ScanEdgePoints &clusters, pcl::PointCloud<pcl::PointXYZI>& keyPoints)
{
    int scanNumTh = ceil(nScans / 6);
    int ptNumTh = ceil(1.5 * scanNumTh);

    int count = 0;
    int numCluster = clusters.size();
    pcl::PointCloud<pcl::PointXYZI> tmpKeyPoints;
    // <距离, 索引> 每个关键字在map中只能出现一次，map的排序默认按照key从小到大进行排序
    map<float, int> distanceOrder;
    // 遍历每一簇
    for(int i = 0; i < numCluster; i++)
    {
        int ptCnt = clusters[i].size();
        if(ptCnt < ptNumTh)
        {
            continue;
        }
        vector<PointXYZSCA> tmpCluster;
        set<int> scans;
        float x = 0, y = 0, z = 0, intensity = 0;
        //
        for(int ptNum = 0; ptNum < ptCnt; ptNum++)
        {
            // 当前簇的当前点
            PointXYZSCA pt = clusters[i][ptNum];
            int scan = int(pt.scan_position);
            scans.insert(scan);

            x += pt.x;
            y += pt.y;
            z += pt.z;
            intensity += pt.scan_position;
        }

        if(scans.size() < (size_t)scanNumTh)
        {
            continue;
        }

        pcl::PointXYZI pt;
        //当前簇的mean
        pt.x = x/ptCnt;
        pt.y = y/ptCnt;
        pt.z = z/ptCnt;
        pt.intensity = intensity/ptCnt;
        // 当前簇均值点到中心的距离
        float distance = pt.x * pt.x + pt.y * pt.y + pt.z * pt.z;

        auto iter = distanceOrder.find(distance);
        if(iter != distanceOrder.end())// 找到距离相同的就跳过
        {
            // 找到距离相同的就跳过
            continue;
        }
        // 距离表生成
        distanceOrder[distance] = count;
        count++;
        // 储存当前簇的质心点
        tmpKeyPoints.push_back(pt);
    }

    for(auto iter = distanceOrder.begin(); iter != distanceOrder.end(); iter++)
    {
        int index = (*iter).second;
        // 取出当前簇的均值点
        pcl::PointXYZI tmpPt = tmpKeyPoints[index];
        keyPoints.push_back(tmpPt);
    }

}




// ---------------------------------- 发布给mapping ----------------------------------------
void PointCloudToMapping(ros::Time& timestamp_ros)
{
    sensor_msgs::PointCloud2 laserCloudCornerLast2;
    pcl::toROSMsg(cornerPointsLessSharp, laserCloudCornerLast2);
    laserCloudCornerLast2.header.stamp = timestamp_ros;
    laserCloudCornerLast2.header.frame_id = "/camera_init";
    pubLaserCloudCornerLast.publish(laserCloudCornerLast2);

    // 原封不动发布当前平面点
    sensor_msgs::PointCloud2 laserCloudSurfLast2;
    pcl::toROSMsg(surfPointsLessFlat, laserCloudSurfLast2);
    laserCloudSurfLast2.header.stamp = timestamp_ros;
    laserCloudSurfLast2.header.frame_id = "/camera_init";
    pubLaserCloudSurfLast.publish(laserCloudSurfLast2);

    // 原封不动的转发当前帧点云，后端优化是低频，高精的，需要更多的点加入，约束越多鲁棒性越好
    sensor_msgs::PointCloud2 laserCloudFullRes3;
    pcl::toROSMsg(laserCloud, laserCloudFullRes3);
    laserCloudFullRes3.header.stamp = timestamp_ros;
    laserCloudFullRes3.header.frame_id = "/camera_init";

    pubLaserCloudFullRes.publish(laserCloudFullRes3);

    // ------------------------ pub img -----------------
//    sensor_msgs::ImagePtr img_msg = cv_bridge::CvImage(std_msgs::Header(),"mono8",_curr_images.back() ).toImageMsg();
//    img_msg->header.stamp = timestamp_ros;
//    img_msg->header.frame_id = "/camera_init";
//    pubImage.publish(img_msg);

//    _curr_images.clear();
    laserCloud.clear();
    cornerPointsLessSharp.clear();
    surfPointsLessFlat.clear();
}



float fRound(float in)
{
    float f;
    int temp = std::round(in * 10);
    f = temp/10.0;

    return f;
}

void getDescriptors(pcl::PointCloud<pcl::PointXYZI> &keyPoints,
                                      cv::Mat &descriptors)
{
    if(keyPoints.empty())
    {
        return;
    }

    int ptSize = keyPoints.size();

    descriptors = cv::Mat::zeros(ptSize, 180, CV_32FC1);

    vector<vector<float>> distanceTab;
    vector<float> oneRowDis(ptSize, 0);
    distanceTab.resize(ptSize, oneRowDis);

    vector<vector<Eigen::Vector2f>> directionTab;
    Eigen::Vector2f direct(0, 0);
    vector<Eigen::Vector2f> oneRowDirect(ptSize, direct);
    directionTab.resize(ptSize, oneRowDirect);

    //Build distance and direction tables for fast descriptor generation.
    for(size_t i = 0; i < keyPoints.size(); i++)
    {
        for(size_t j = i+1; j < keyPoints.size(); j++)
        {
            float dist = distPt2Pt(keyPoints[i], keyPoints[j]);
            distanceTab[i][j] = fRound(dist);
            distanceTab[j][i] = distanceTab[i][j];

            Eigen::Vector2f tmpDirection;
            tmpDirection(0, 0) = keyPoints[j].x - keyPoints[i].x;
            tmpDirection(1, 0) = keyPoints[j].y - keyPoints[i].y;
            directionTab[i][j] = tmpDirection;
            directionTab[j][i] = -tmpDirection;
        }
    }

    for(size_t i = 0; i < keyPoints.size(); i++)
    {
        vector<float> tempRow(distanceTab[i]);
        std::sort(tempRow.begin(), tempRow.end());
        int Index[3];

        //Get the closest three keypoints of current keypoint.
        for(int k = 0; k < 3; k++)
        {
            vector<float>::iterator it1 = find(distanceTab[i].begin(), distanceTab[i].end(), tempRow[k+1]);
            if(it1 == distanceTab[i].end())
            {
                continue;
            }
            else
            {
                Index[k] = std::distance(distanceTab[i].begin(), it1);
            }
        }

        //Generate the descriptor for each closest keypoint.
        //The final descriptor is based on the priority of the three closest keypoint.
        for(int indNum = 0; indNum < 3; indNum++)
        {
            int index = Index[indNum];
            Eigen::Vector2f mainDirection;
            mainDirection = directionTab[i][index];

            vector<vector<float>> areaDis(180);
            areaDis[0].emplace_back(distanceTab[i][index]);

            for(size_t j = 0; j < keyPoints.size(); j++)
            {
                if(j == i || (int)j == index)
                {
                    continue;
                }

                Eigen::Vector2f otherDirection = directionTab[i][j];
                Eigen::Matrix2f matrixDirect;
                matrixDirect << mainDirection(0, 0), mainDirection(1, 0), otherDirection(0, 0), otherDirection(1, 0);
                float deter = matrixDirect.determinant();

                int areaNum = 0;
                double cosAng = (double)mainDirection.dot(otherDirection) / (double)(mainDirection.norm() * otherDirection.norm());
                if(abs(cosAng) - 1 > 0)
                {
                    continue;
                }

                float angle = acos(cosAng) * 180 / M_PI;

                if(angle < 0 || angle > 180)
                {
                    continue;
                }

                if(deter > 0)
                {
                    areaNum = ceil((angle - 1) / 2);
                }
                else
                {
                    if(angle - 2 < 0)
                    {
                        areaNum = 0;
                    }
                    else
                    {
                        angle = 360 - angle;
                        areaNum = ceil((angle - 1) / 2);
                    }
                }

                if(areaNum != 0)
                {
                    areaDis[areaNum].emplace_back(distanceTab[i][j]);
                }
            }

            float *descriptor = descriptors.ptr<float>(i);

            for(int areaNum = 0; areaNum < 180; areaNum++)
            {
                if(areaDis[areaNum].size() == 0)
                {
                    continue;
                }
                else
                {
                    std::sort(areaDis[areaNum].begin(), areaDis[areaNum].end());

                    if(descriptor[areaNum] == 0)
                    {
                        descriptor[areaNum] = areaDis[areaNum][0];
                    }
                }
            }
        }
    }
}

void match(
        pcl::PointCloud<pcl::PointXYZI> &curAggregationKeyPt,
        pcl::PointCloud<pcl::PointXYZI> &toBeMatchedKeyPt,
        cv::Mat &curDescriptors,
        cv::Mat &toBeMatchedDescriptors,
        vector<pair<int, int>> &vMatchedIndex)
{
    int curKeypointNum = curAggregationKeyPt.size();
    int toBeMatchedKeyPtNum = toBeMatchedKeyPt.size();

    multimap<int, int> matchedIndexScore;
    multimap<int, int> mMatchedIndex;
    set<int> sIndex;

    for(int i = 0; i < curKeypointNum; i++)
    {
        std::pair<int, int> highestIndexScore(0, 0);
        float* pDes1 = curDescriptors.ptr<float>(i);

        for(int j = 0; j < toBeMatchedKeyPtNum; j++)
        {
            int sameDimScore = 0;
            float* pDes2 = toBeMatchedDescriptors.ptr<float>(j);

            for(int bitNum = 0; bitNum < 180; bitNum++)
            {
                if(pDes1[bitNum] != 0 && pDes2[bitNum] != 0 && abs(pDes1[bitNum] - pDes2[bitNum]) <= 0.2)
                {
                    sameDimScore += 1;
                }

                if(bitNum > 90 && sameDimScore < 3){
                    break;
                }
            }

            if(sameDimScore > highestIndexScore.second)
            {
                highestIndexScore.first = j;
                highestIndexScore.second = sameDimScore;
            }
        }

        //Used for removing the repeated matches.
        matchedIndexScore.insert(std::make_pair(i, highestIndexScore.second)); //Record i and its corresponding score.
        mMatchedIndex.insert(std::make_pair(highestIndexScore.first, i)); //Record the corresponding match between j and i.
        sIndex.insert(highestIndexScore.first); //Record the index that may be repeated matches.
    }

    //Remove the repeated matches.
    for(set<int>::iterator setIt = sIndex.begin(); setIt != sIndex.end(); ++setIt)
    {
        int indexJ = *setIt;
        auto entries = mMatchedIndex.count(indexJ);
        if(entries == 1)
        {
            auto iterI = mMatchedIndex.find(indexJ);
            auto iterScore = matchedIndexScore.find(iterI->second);
            if(iterScore->second >= matchTh)
            {
                vMatchedIndex.emplace_back(std::make_pair(iterI->second, indexJ));
            }
        }
        else
        {
            auto iter1 = mMatchedIndex.find(indexJ);
            int highestScore = 0;
            int highestScoreIndex = -1;

            while(entries)
            {
                int indexI = iter1->second;
                auto iterScore = matchedIndexScore.find(indexI);
                if(iterScore->second > highestScore){
                    highestScore = iterScore->second;
                    highestScoreIndex = indexI;
                }
                ++iter1;
                --entries;
            }

            if(highestScore >= matchTh)
            {
                vMatchedIndex.emplace_back(std::make_pair(highestScoreIndex, indexJ));
            }
        }
    }
}

void Registration(pcl::PointCloud<pcl::PointXYZI> &keyPoints_curr, pcl::PointCloud<pcl::PointXYZI> &keyPoints_last,
        vector<pair<int, int>> &vMatchedIndex, ros::Time &timestamp_ros)
{

    vector<cv::Point3f> pts1, pts2;
    for(int i = 0; i < vMatchedIndex.size(); i++)
    {
        cv::Point3f point_curr, point_last;
        point_curr.x = keyPoints_curr[vMatchedIndex[i].first].x;
        point_curr.y = keyPoints_curr[vMatchedIndex[i].first].y;
        point_curr.z = keyPoints_curr[vMatchedIndex[i].first].z;

        point_last.x = keyPoints_last[vMatchedIndex[i].second].x;
        point_last.y = keyPoints_last[vMatchedIndex[i].second].y;
        point_last.z = keyPoints_last[vMatchedIndex[i].second].z;
        pts2.push_back(point_curr);
        pts1.push_back(point_last);

    }
    
    // -------------------------------------------------------------------------------
    // clock_t start, end;
    // double time;
    // start = clock();
    static double T_curr2last[6] = {0,0,0,0,0,0};

    //cout << "T_curr2last = " << T_curr2last[0]<< T_curr2last[1]<<T_curr2last[2]<< T_curr2last[3] <<T_curr2last[4] << endl;
    ceres::Problem problem;
    for (int i = 0; i < pts1.size(); ++i)
    {
        ceres::CostFunction* cost_function =
                ICPCeres::Create(pts2[i], pts1[i]);
        // 剔除外点
        ceres::LossFunction *loss_function = new ceres::HuberLoss(0.1);
        problem.AddResidualBlock(cost_function,
                                 loss_function,
                                 T_curr2last);
    }

    ceres::Solver::Options options;
    options.max_num_iterations = 4;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.minimizer_progress_to_stdout = false;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    Mat R_vec = (Mat_<double>(3,1) << T_curr2last[0], T_curr2last[1], T_curr2last[2]);
    Mat R_cvest;
    // 罗德里格斯公式，旋转向量转旋转矩阵
    cv::Rodrigues(R_vec, R_cvest);
    Eigen::Matrix<double,3,3> R_est;
    cv::cv2eigen(R_cvest, R_est);
    Eigen::Quaterniond q(R_est.inverse());
    q.normalize();
    q_last_curr = q;
    //cout << "q = \n" << q.x() << " " << q.y() << " " << q.z() << " " << q.w()<< endl;
    //cout << -T_curr2last[3] << " " <<  -T_curr2last[4] << " " << -T_curr2last[5] << endl;
    //cout<<"R_est="<<R_est<<endl;
    Eigen::Vector3d t_est(T_curr2last[3], T_curr2last[4], T_curr2last[5]);
    t_last_curr = -t_est;
    //cout<<"t_est="<<t_est<<endl;
    Eigen::Isometry3d T(R_est);//构造变换矩阵与输出
    T.pretranslate(t_est);
    //cout << "T = \n" << T.matrix().inverse()<<endl;
    t_w_curr = t_w_curr + q_w_curr * t_last_curr;
    q_w_curr = q_w_curr * q_last_curr;

    // publish odometry
    nav_msgs::Odometry laserOdometry;
    laserOdometry.header.frame_id = "/camera_init";
    laserOdometry.child_frame_id = "/laser_odom";
    laserOdometry.header.stamp = timestamp_ros;
    laserOdometry.pose.pose.orientation.x = q_w_curr.x();
    laserOdometry.pose.pose.orientation.y = q_w_curr.y();
    laserOdometry.pose.pose.orientation.z = q_w_curr.z();
    laserOdometry.pose.pose.orientation.w = q_w_curr.w();
    laserOdometry.pose.pose.position.x = t_w_curr.x();
    laserOdometry.pose.pose.position.y = t_w_curr.y();
    laserOdometry.pose.pose.position.z = t_w_curr.z();
    pubLaserOdometry.publish(laserOdometry);
    geometry_msgs::PoseStamped laserPose;
    laserPose.header = laserOdometry.header;
    laserPose.pose = laserOdometry.pose.pose;
    laserPath.header.stamp = laserOdometry.header.stamp;
    laserPath.poses.push_back(laserPose);
    laserPath.header.frame_id = "/camera_init";
    pubLaserPath.publish(laserPath);

    sensor_msgs::PointCloud2 center_msg;
    pcl::toROSMsg(keyPoints_curr, center_msg);
    center_msg.header.stamp = timestamp_ros;
    center_msg.header.frame_id = "/camera_init";
    pubCenter.publish(center_msg);

    PointCloudToMapping(timestamp_ros);
}


void laserCloudHandler(const sensor_msgs::PointCloud2ConstPtr &laserCloudMsg)
{
    clock_t start, end;
    double time;
    start = clock();
    static bool init = false;

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

int main(int argc, char** argv)
{
    ros::init(argc, argv, "front_end");
    ros::NodeHandle nh;
    nh.param<int>("scan_line", nScans, 16);
    nh.param<double>("minimum_range", minimumRange, 0.5);
    nh.param<double>("FilterGroundLeaf", FilterGroundLeaf, 0.1);
    FilterGround.setLeafSize(FilterGroundLeaf,FilterGroundLeaf,FilterGroundLeaf);


    printf("scan line number %d \n", nScans);
    printf("minimum_range %f \n", minimumRange);

    if(nScans != 16 && nScans != 32 && nScans != 64 && nScans != 80)
    {
        printf("only support velodyne with 16, 32 , 64 or RS80 scan line!");
        return 0;
    }
    image_transport::ImageTransport it(nh);
    pubImage = it.advertise("/image_centroid", 100);
    pubLaserCloudFullRes = nh.advertise<sensor_msgs::PointCloud2>("/velodyne_cloud_full", 100);
    pubLaserCloudCornerLast = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_corner_last", 100);
    pubLaserCloudSurfLast = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_surf_last", 100);
    // 发布里程计数据(位姿轨迹)给后端，后端接收
    pubLaserOdometry = nh.advertise<nav_msgs::Odometry>("/laser_odom_to_init", 100);
    // 发布前端里程计的高频低精 位姿轨迹
    pubLaserPath = nh.advertise<nav_msgs::Path>("/laser_odom_path", 100);
    pubCenter = nh.advertise<sensor_msgs::PointCloud2>("/center", 100);
    ros::Subscriber subLaserCloud = nh.subscribe<sensor_msgs::PointCloud2>("/velodyne_points", 100, laserCloudHandler);
    ros::spin();

    return 0;
}

