#pragma once

#include <vector>
#include <map>
#include <set>

#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/extract_indices.h>

#include <eigen3/Eigen/Dense>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <math.h>

using namespace std;

//定义了一个新的点云类型，包含了四维点
struct PointXYZSCA
{
    PCL_ADD_POINT4D;  //宏扩展点云数据结构
    float scan_position; //应该是整数部分是点对应的扫描线数，小数部分是相对该扫描线起始点的相对时间
    float curvature; //LOAM所谓的曲率
    float angle; //相对于同一扫描线上的起始点的水平面上的角度
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW; //确保内存对齐
}EIGEN_ALIGN16; //确保内存对齐

//将新的点云类型注册到PCL中
POINT_CLOUD_REGISTER_POINT_STRUCT(PointXYZSCA, 
    (float, x, x)(float, y, y)(float, z, z)(float, scan_position, scan_position)(float, curvature, curvature)(float, angle, angle))

//存储边缘点的二维容器，点云类型是自定义类型
typedef vector<vector<PointXYZSCA>> ScanEdgePoints;

namespace BoW3D
{
    //注意是宏定义
    //所有边缘点的点云
    #define EdgePointCloud pcl::PointCloud<PointXYZSCA>
    //计算点a在水平面上投影到原点距离
    #define distXY(a) sqrt(a.x * a.x + a.y * a.y)
    //计算三维点a到原点距离
    #define distOri2Pt(a) sqrt(a.x * a.x + a.y * a.y + a.z * a.z)
    //计算两个三维点距离
    #define distPt2Pt(a, b) sqrt((a.x - b.x)*(a.x - b.x) + (a.y - b.y)*(a.y - b.y) + (a.z - b.z)*(a.z - b.z))
    
    using std::atan2;
    using std::cos;
    using std::sin;
   
    class Frame;

    class LinK3D_Extractor
    {
        public:
            LinK3D_Extractor(int nScans_, float scanPeriod_, float minimumRange_, float distanceTh_, int matchTh_);

            ~LinK3D_Extractor(){}

            bool comp (int i, int j) 
            { 
                return cloudCurvature[i] < cloudCurvature[j]; 
            }

            void removeClosedPointCloud(const pcl::PointCloud<pcl::PointXYZ> &cloud_in,
                                        pcl::PointCloud<pcl::PointXYZ> &cloud_out);      

            void extractEdgePoint(pcl::PointCloud<pcl::PointXYZ>::Ptr pLaserCloudIn, ScanEdgePoints &edgePoints);

            void divideArea(ScanEdgePoints &scanEdgePoints, ScanEdgePoints &sectorAreaCloud);

            float computeClusterMean(vector<PointXYZSCA> &cluster);

            void computeXYMean(vector<PointXYZSCA> &cluster, pair<float, float> &xyMeans);

            void getCluster(const ScanEdgePoints &sectorAreaCloud, ScanEdgePoints &clusters);

            void computeDirection(pcl::PointXYZI ptFrom, 
                                  pcl::PointXYZI ptTo, 
                                  Eigen::Vector2f &direction);

            vector<pcl::PointXYZI> getMeanKeyPoint(const ScanEdgePoints &clusters, 
                                                   ScanEdgePoints &validCluster);
            
            float fRound(float in);
                        
            void getDescriptors(const vector<pcl::PointXYZI> &keyPoints, cv::Mat &descriptors);
            
            void match(vector<pcl::PointXYZI> &curAggregationKeyPt, 
                       vector<pcl::PointXYZI> &toBeMatchedKeyPt,
                       cv::Mat &curDescriptors, 
                       cv::Mat &toBeMatchedDescriptors, 
                       vector<pair<int, int>> &vMatchedIndex);

            void filterLowCurv(ScanEdgePoints &clusters, ScanEdgePoints &filtered);

            void findEdgeKeypointMatch(ScanEdgePoints &filtered1, 
                                       ScanEdgePoints &filtered2, 
                                       vector<pair<int, int>> &vMatched, 
                                       vector<pair<PointXYZSCA, PointXYZSCA>> &matchPoints);
            
            void operator()(pcl::PointCloud<pcl::PointXYZ>::Ptr pLaserCloudIn, 
                            vector<pcl::PointXYZI> &keyPoints, 
                            cv::Mat &descriptors, 
                            ScanEdgePoints &validCluster);

        private:
            int nScans; //扫描线束总数
            float scanPeriod; //一圈扫描的周期
            float minimumRange; //最小点云距离范围，距离原点小于该阈值的点将被删除

            float distanceTh; //判断区域内某点和聚类点均值距离，以及在x，y轴上的距离
            int matchTh;     //描述子匹配所需的最低分数阈值 ，描述子匹配分数低于此分数的两个关键点不匹配     
            int scanNumTh;   //阈值？看cpp
            int ptNumTh;     //点数阈值？看cpp

            float cloudCurvature[400000]; //存储点云中点的曲率
    };
}
