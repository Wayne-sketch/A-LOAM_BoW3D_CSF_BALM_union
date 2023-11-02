/*
 * @Author: huge zhuhu00@foxmail.com
 * @Date: 2023-11-01 15:38:26
 * @LastEditors: huge zhuhu00@foxmail.com
 * @LastEditTime: 2023-11-01 15:51:14
 * @FilePath: /a-loam/src/A-LOAM/src/Frame.cpp
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 */
#include "BoW3D/Frame.h"
#include <thread>

namespace BoW3D
{
    //静态（全局？）变量要在这里初始化
    long unsigned int Frame::nNextId = 0;
   
    Frame::Frame(LinK3D_Extractor* pLink3dExtractor, pcl::PointCloud<pcl::PointXYZ>::Ptr pLaserCloudIn):mpLink3dExtractor(pLink3dExtractor)
    {
        mnId = nNextId++; 

        (*mpLink3dExtractor)(pLaserCloudIn, mvAggregationKeypoints, mDescriptors, mClusterEdgeKeypoints);
    }

                
}