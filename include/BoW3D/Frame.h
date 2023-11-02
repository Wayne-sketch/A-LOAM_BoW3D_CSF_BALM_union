/*
 * @Author: ctx cuitongxin201024@163.com
 * @Date: 2023-10-30 13:35:41
 * @LastEditors: ctx cuitongxin201024@163.com
 * @LastEditTime: 2023-10-31 14:21:29
 * @FilePath: \BoW3D\include\Frame.h
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 */
#pragma once

#include <iostream>
#include <vector>

#include <pcl/point_types.h>
#include <pcl/conversions.h>
#include <eigen3/Eigen/Dense>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

#include "LinK3D_Extractor.h"


using namespace std;
using namespace Eigen;

namespace BoW3D
{
    class LinK3D_Extractor;

    class Frame
    {
        public:
            Frame();
            
            //要用的构造函数
            Frame(LinK3D_Extractor* pLink3dExtractor, pcl::PointCloud<pcl::PointXYZ>::Ptr pLaserCloudIn);
            
            ~Frame(){};
                        
        public: 
            //静态全局变量
            static long unsigned int nNextId;

            long unsigned int mnId;
            
            //Link3D提取器
            LinK3D_Extractor* mpLink3dExtractor;

            //边缘点
            ScanEdgePoints mClusterEdgeKeypoints;

            //聚类后提取的关键点
            std::vector<pcl::PointXYZI> mvAggregationKeypoints;

            //描述子
            cv::Mat mDescriptors;                
    };

}

