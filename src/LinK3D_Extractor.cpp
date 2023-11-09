#include "BoW3D/LinK3D_Extractor.h"

namespace BoW3D
{
    //构造函数 给所有私有成员变量赋值
    LinK3D_Extractor::LinK3D_Extractor(
            int nScans_, 
            float scanPeriod_, 
            float minimumRange_, 
            float distanceTh_,           
            int matchTh_):
            nScans(nScans_), 
            scanPeriod(scanPeriod_), 
            minimumRange(minimumRange_),   
            distanceTh(distanceTh_),          
            matchTh(matchTh_)
            {
                scanNumTh = ceil(nScans / 6); //向下取整 意义？
                ptNumTh = ceil(1.5 * scanNumTh); //向下取整 意义？               
            }

    /**
     * @brief 移除近处点云
     * 
     * @param[in] cloud_in 输入点云 
     * @param[out] cloud_out 输出点云
     */
    void LinK3D_Extractor::removeClosedPointCloud(
            const pcl::PointCloud<pcl::PointXYZ> &cloud_in,
            pcl::PointCloud<pcl::PointXYZ> &cloud_out)
    {
        //把输入点云的头部信息给输出点云，头部信息包含点云序列号seq，时间戳stamp，坐标系ID frame_id
        if (&cloud_in != &cloud_out)
        {
            cloud_out.header = cloud_in.header;
            cloud_out.points.resize(cloud_in.points.size());
        }

        size_t j = 0;
        //遍历输入点云
        for (size_t i = 0; i < cloud_in.points.size(); ++i)
        {
            //计算点到原点距离，距离过小跳过，不给到输出点云
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

        //如果有点被删除，输出点云分配正好的内存大小
        if (j != cloud_in.points.size())
        {
            cloud_out.points.resize(j);
        }

        //一般height代表激光束或扫描线的数量，如果点云无序，通常设置为1
        cloud_out.height = 1;
        //一般width代表每行扫描线有多少点，点云无序通常为点云中点的总数
        cloud_out.width = static_cast<uint32_t>(j);
        //如果点云数据中没有缺失的点，点云就是密集的true，点坐标为NaN就是有缺失
        cloud_out.is_dense = true;
    }

    /**
     * @brief 从点云中提取边缘点
     * 
     * @param[in] pLaserCloudIn 输入点云指针
     * @param[out] edgePoints 提取到的边缘点，类型是存储边缘点的二维容器
     */
    void LinK3D_Extractor::extractEdgePoint(
            pcl::PointCloud<pcl::PointXYZ>::Ptr pLaserCloudIn, 
            ScanEdgePoints &edgePoints)
    {   
        //起始和终止索引
        vector<int> scanStartInd(nScans, 0);
        vector<int> scanEndInd(nScans, 0);

        //输入点云赋值
        pcl::PointCloud<pcl::PointXYZ> laserCloudIn;
        laserCloudIn = *pLaserCloudIn;
        //存储有效点在输入点云中的索引
        vector<int> indices;

        //删除NaN和inf点
        pcl::removeNaNFromPointCloud(laserCloudIn, laserCloudIn, indices);
        //删除近距离的点
        removeClosedPointCloud(laserCloudIn, laserCloudIn);

        //点云规模
        int cloudSize = laserCloudIn.points.size();
        // 一帧点云中起始点角度（弧度单位） 这里也加了负号？？？？ atan2返回-PI~PI 加负号相当于把原来的点云坐标系y轴方向调换了建了一个新的坐标系？？LOAM里说雷达是顺时针坐标系，应该就是左手坐标系的意思，我们计算用的是右手坐标系，所以要把y轴换方向？但是点坐标没换啊？
        float startOri = -atan2(laserCloudIn.points[0].y, laserCloudIn.points[0].x);
        //下面的处理是保证角度差在一个合理的范围内 还没搞懂？？？？？？？？？
        //一帧点云中结束点角度（弧度单位） 先把2PI加上，由于atan2返回的是(-PI~PI]，结束角度可能小于起始角度，加2PI确保结束角度大于起始角度，坐标系是逆时针
        float endOri = -atan2(laserCloudIn.points[cloudSize - 1].y, laserCloudIn.points[cloudSize - 1].x) + 2 * M_PI;
    
        //这段逻辑确保endOri和startOri角度差在PI~3PI之间？
        //已经加了2PI了
        if (endOri - startOri > 3 * M_PI)
        {
            endOri -= 2 * M_PI;
        }
        else if (endOri - startOri < M_PI)
        {
            endOri += 2 * M_PI;
        }
        
        bool halfPassed = false;
        //点云规模
        int count = cloudSize;
        //单个处理的点
        pcl::PointXYZI point;
        //按扫描线束分别存储点云
        vector<pcl::PointCloud<pcl::PointXYZI>> laserCloudScans(nScans);
        
        for (int i = 0; i < cloudSize; i++)
        {
            //取点
            point.x = laserCloudIn.points[i].x;
            point.y = laserCloudIn.points[i].y;
            point.z = laserCloudIn.points[i].z;
            //atan返回(-PI/2,PI/2) 算点的角度，进而算线束
            float angle = atan(point.z / sqrt(point.x * point.x + point.y * point.y)) * 180 / M_PI;
            int scanID = 0;

            //计算有效范围内的点数 count 计算所属scanID
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
            
            //水平角度
            float ori = -atan2(point.y, point.x);
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
            //存当前点的水平角度
            point.intensity = ori;
            //存点
            laserCloudScans[scanID].points.push_back(point);            
        }
        //扫描线数
        size_t scanSize = laserCloudScans.size();
        //edgePoints是二维容器，第一维是扫描线数
        edgePoints.resize(scanSize);
        //原来存输入点云的点数，改存scanID有效范围内的点数
        cloudSize = count;
                
        for(int i = 0; i < nScans; i++)
        {
            int laserCloudScansSize = laserCloudScans[i].size();
            //一条扫描线上点数过少，不处理
            if(laserCloudScansSize >= 15)
            {
                //算曲率
                for(int j = 5; j < laserCloudScansSize - 5; j++)
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

                    float curv = diffX * diffX + diffY * diffY + diffZ * diffZ;
                    if(curv > 10 && curv < 20000)
                    {
                        //前面intensity存的就是点的水平面上的角度
                        float ori = laserCloudScans[i].points[j].intensity;
                        //相对起始点的“时间”，其实是水平角度比例，做运动补偿用
                        float relTime = (ori - startOri) / (endOri - startOri);

                        //自定义的点云数据
                        PointXYZSCA tmpPt;
                        tmpPt.x = laserCloudScans[i].points[j].x;
                        tmpPt.y = laserCloudScans[i].points[j].y;
                        tmpPt.z = laserCloudScans[i].points[j].z;
                        tmpPt.scan_position = i + scanPeriod * relTime;
                        tmpPt.curvature = curv;
                        tmpPt.angle = ori; 
                        edgePoints[i].emplace_back(tmpPt);
                    }
                }
            }
        }            
    }    

    //Roughly divide the areas to save time for clustering.
    /**
     * @brief 粗分区域
     * 
     * @param[in] scanCloud 存储边缘点的二维容器，第一维代表扫描线数
     * @param[out] sectorAreaCloud 存储边缘点的二维容器，第一维代表所属区域，一共120个区域
     */
    void LinK3D_Extractor::divideArea(ScanEdgePoints &scanCloud, ScanEdgePoints &sectorAreaCloud)
    {
        //分120区域
        sectorAreaCloud.resize(120); //The horizontal plane is divided into 120 sector area centered on LiDAR coordinate.
        //扫描线数
        int numScansPt = scanCloud.size();
        if(numScansPt == 0)
        {
            return;
        }
        //按扫描线数遍历    
        for(int i = 0; i < numScansPt; i++) 
        {   
            //遍历一条扫描线上的点
            int numAScanPt = scanCloud[i].size();
            for(int j = 0; j < numAScanPt; j++)
            {               
                //计算所属区域  
                int areaID = 0;
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
                //存点
                sectorAreaCloud[areaID].push_back(scanCloud[i][j]);
            }
        }
    }

    /**
     * @brief 计算聚类点投影到水平面后的中心点到原点的平均距离
     * 
     * @param[in] cluster 三维聚类点
     * @return float 平均距离
     */
    float LinK3D_Extractor::computeClusterMean(vector<PointXYZSCA> &cluster)
    {        
        float distSum = 0;
        int numPt = cluster.size();

        for(int i = 0; i < numPt; i++)
        {
            distSum += distXY(cluster[i]);
        }

        return (distSum/numPt);
    }

    /**
     * @brief 分别计算三维聚类点投影到水平面后X Y坐标轴上的平均值
     * 
     * @param[in] cluster 三维聚类点
     * @param[out] xyMeans XY坐标轴上的平均值 
     */
    void LinK3D_Extractor::computeXYMean(vector<PointXYZSCA> &cluster, std::pair<float, float> &xyMeans)
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

    /**
     * @brief 在按120个区域存储的边缘点上聚类
     * 
     * @param[in] sectorAreaCloud 按区域存储的边缘点
     * @param[out] clusters 二维边缘点容器
     */
    void LinK3D_Extractor::getCluster(const ScanEdgePoints &sectorAreaCloud, ScanEdgePoints &clusters)
    {    
        ScanEdgePoints tmpclusters;
        PointXYZSCA curvPt;
        vector<PointXYZSCA> dummy(1, curvPt); 

        int numArea = sectorAreaCloud.size();

        //Cluster for each sector area.
        //遍历所有区域
        for(int i = 0; i < numArea; i++)
        {
            //区域内点数小于6跳过
            if(sectorAreaCloud[i].size() < 6)
                continue;

            int numPt = sectorAreaCloud[i].size();        
            ScanEdgePoints curAreaCluster(1, dummy);
            curAreaCluster[0][0] = sectorAreaCloud[i][0];

            //遍历区域内的所有点
            for(int j = 1; j < numPt; j++)
            {
                int numCluster = curAreaCluster.size();

                for(int k = 0; k < numCluster; k++)
                {
                    //mean：计算聚类点投影到水平面后到原点的平均距离
                    float mean = computeClusterMean(curAreaCluster[k]);
                    std::pair<float, float> xyMean;
                    //xyMean：聚类点投影到水平面后，分别在x，y轴上到原点的平均距离
                    computeXYMean(curAreaCluster[k], xyMean);
                    //取当前区域内一个点
                    PointXYZSCA tmpPt = sectorAreaCloud[i][j];
                    //如果当前点距离聚类中心点，分别在x，y轴上距离中心店距离小于阈值，存到当前聚类中              
                    if(abs(distXY(tmpPt) - mean) < distanceTh 
                        && abs(xyMean.first - tmpPt.x) < distanceTh 
                        && abs(xyMean.second - tmpPt.y) < distanceTh)
                    {
                        curAreaCluster[k].emplace_back(tmpPt);
                        break;
                    }
                    //反之，存一个空的，当前点存到后面？？
                    else if(abs(distXY(tmpPt) - mean) >= distanceTh && k == numCluster-1)
                    {
                        curAreaCluster.emplace_back(dummy);
                        curAreaCluster[numCluster][0] = tmpPt;
                    }
                    else
                    { 
                        continue; 
                    }                    
                }
            }

            //遍历每个区域
            int numCluster = curAreaCluster.size();
            for(int j = 0; j < numCluster; j++)
            {
                int numPt = curAreaCluster[j].size();
                //区域内点数过少跳过
                if(numPt < ptNumTh)
                {
                    continue;
                }
                //区域内点数足够，存到二维容器中
                tmpclusters.emplace_back(curAreaCluster[j]);
            }
        }

        int numCluster = tmpclusters.size();
        
        vector<bool> toBeMerge(numCluster, false);
        //键-多值映射容器：相同的键可以出现多次
        multimap<int, int> mToBeMergeInd;
        set<int> sNeedMergeInd;

        //Merge the neighbor clusters.
        //遍历所有区域
        for(int i = 0; i < numCluster; i++)
        {
            if(toBeMerge[i]){
                continue;
            }
            //当前区域聚类点投影到水平面后到原点的平均距离
            float means1 = computeClusterMean(tmpclusters[i]);
            std::pair<float, float> xyMeans1;
            //x，y轴上的平均距离
            computeXYMean(tmpclusters[i], xyMeans1);
            //再次遍历所有区域，同样的操作
            for(int j = 1; j < numCluster; j++)
            {
                if(toBeMerge[j])
                {
                    continue;
                }

                float means2 = computeClusterMean(tmpclusters[j]);
                std::pair<float, float> xyMeans2;
                computeXYMean(tmpclusters[j], xyMeans2);

                //现在有了第i区域和第j区域的信息，计算各种平均距离的差值
                if(abs(means1 - means2) < 2*distanceTh 
                    && abs(xyMeans1.first - xyMeans2.first) < 2*distanceTh 
                    && abs(xyMeans1.second - xyMeans2.second) < 2*distanceTh)
                {
                    //距离过近，就要合并掉
                    mToBeMergeInd.insert(std::make_pair(i, j));
                    sNeedMergeInd.insert(i);
                    toBeMerge[i] = true;
                    toBeMerge[j] = true;
                }
            }

        }
        //如果没有需要合并的，就全放进最终的聚类容器
        if(sNeedMergeInd.empty())
        {
            for(int i = 0; i < numCluster; i++)
            {
                clusters.emplace_back(tmpclusters[i]);
            }
        }
        else
        {
            //遍历所有的cluster
            for(int i = 0; i < numCluster; i++)
            {
                // 如果当前 cluster 不需要被合并，则加入 clusters 中
                if(toBeMerge[i] == false)
                {
                    clusters.emplace_back(tmpclusters[i]);
                }
            }
            
            // 遍历需要被合并的索引集合 sNeedMergeInd
            for(auto setIt = sNeedMergeInd.begin(); setIt != sNeedMergeInd.end(); ++setIt)
            {
                int needMergeInd = *setIt;
                // 从 mToBeMergeInd 中找到需要合并的 cluster 的索引
                auto entries = mToBeMergeInd.count(needMergeInd);
                auto iter = mToBeMergeInd.find(needMergeInd);
                vector<int> vInd;

                // 将需要合并的 cluster 的所有索引加入到 vInd 中
                while(entries)
                {
                    int ind = iter->second;
                    vInd.emplace_back(ind);
                    ++iter;
                    --entries;
                }

                // 将需要合并的 cluster 加入到 clusters 中
                clusters.emplace_back(tmpclusters[needMergeInd]);
                size_t numCluster = clusters.size();

                // 将所有需要合并的 cluster 的点加入到 clusters 中
                for(size_t j = 0; j < vInd.size(); j++)
                {
                    for(size_t ptNum = 0; ptNum < tmpclusters[vInd[j]].size(); ptNum++)
                    {
                        clusters[numCluster - 1].emplace_back(tmpclusters[vInd[j]][ptNum]);
                    }
                }
            }
        }       
    }

    /**
     * @brief 计算两点方向向量，没有归一化
     * 
     * @param[in] ptFrom 起始点
     * @param[in] ptTo 结束点
     * @param[out] direction 方向向量
     */
    void LinK3D_Extractor::computeDirection(pcl::PointXYZI ptFrom, pcl::PointXYZI ptTo, Eigen::Vector2f &direction)
    {
        direction(0, 0) = ptTo.x - ptFrom.x;
        direction(1, 0) = ptTo.y - ptFrom.y;
    }

    /**
     * @brief 从输入的 ScanEdgePoints 类型的聚类（clusters）中提取关键点，条件是聚类中点的数量大于 ptNumTh，
     * 并且在扫描线上的数量大于 scanNumTh。该函数还会返回提取的关键点的列表，并将符合条件的聚类存储在 validCluster 中
     * 
     * @param[in] clusters 聚类
     * @param[out] validCluster 有效聚类
     * @return vector<pcl::PointXYZI> 关键点
     */
    vector<pcl::PointXYZI> LinK3D_Extractor::getMeanKeyPoint(const ScanEdgePoints &clusters, ScanEdgePoints &validCluster)
    {        
        int count = 0;
        int numCluster = clusters.size();
        vector<pcl::PointXYZI> keyPoints;
        vector<pcl::PointXYZI> tmpKeyPoints;
        ScanEdgePoints tmpEdgePoints;
        map<float, int> distanceOrder;

        //遍历所有聚类
        for(int i = 0; i < numCluster; i++)
        {   
            //当前聚类点数
            int ptCnt = clusters[i].size();  
            //如果聚类中的点数小于设定的阈值 ptNumTh，则忽略该聚类。    
            if(ptCnt < ptNumTh)
            {
                continue;
            }

            //计算聚类中所有点的坐标和强度值的平均值，并判断该聚类在多少个不同的扫描线上
            vector<PointXYZSCA> tmpCluster;
            set<int> scans;
            float x = 0, y = 0, z = 0, intensity = 0;
            for(int ptNum = 0; ptNum < ptCnt; ptNum++)
            {
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
            //如果聚类在足够多的扫描线上（大于等于 scanNumTh）
            //表示该聚类的平均点，并计算该点的平方距离（distance）作为键。
            pcl::PointXYZI pt;
            pt.x = x/ptCnt;
            pt.y = y/ptCnt;
            pt.z = z/ptCnt;
            pt.intensity = intensity/ptCnt;

            float distance = pt.x * pt.x + pt.y * pt.y + pt.z * pt.z;
            //检查该距离是否已经存在于 distanceOrder 中，如果存在，则跳过该聚类，否则将该距离和计数器（count）
            //的映射添加到 distanceOrder 中，然后将平均点和原始聚类加入到 tmpKeyPoints 和 tmpEdgePoints 中。
            auto iter = distanceOrder.find(distance);
            if(iter != distanceOrder.end())
            {
                continue;
            }

            distanceOrder[distance] = count; 
            count++;
            
            tmpKeyPoints.emplace_back(pt);
            tmpEdgePoints.emplace_back(clusters[i]);            
        }

        //遍历排序后的 distanceOrder，将排序后的关键点（tmpKeyPoints）加入到 keyPoints 中，
        //对应的有效聚类（tmpEdgePoints）加入到 validCluster 中。
        for(auto iter = distanceOrder.begin(); iter != distanceOrder.end(); iter++)
        {
            int index = (*iter).second;
            pcl::PointXYZI tmpPt = tmpKeyPoints[index];
            
            keyPoints.emplace_back(tmpPt);
            validCluster.emplace_back(tmpEdgePoints[index]);
        }
                
        return keyPoints;
    }

    //将输入的浮点数保留一位小数，采用四舍五入的方式
    float LinK3D_Extractor::fRound(float in)
    {
        float f;
        int temp = std::round(in * 10);
        f = temp/10.0;
        
        return f;
    }

    /**
     * @brief 从输入的关键点（keyPoints）计算描述符（descriptors）。该函数使用了一个特殊的算法，
     * 通过计算最近的三个关键点之间的相对方向和距离，生成一个长度为180的描述符，用于描述每个关键点周围的特征。
     * 
     * ！！！关于生成描述子的细节分析
     * 找到三个最近的关键点，分别作为主方向，开始一圈算下来，找每个区域内最近的关键点
     * 但是都会对齐到最近的关键点作为主方向，也就是以areaDis[0]为第一个区域，然后按照论文中的优先级给描述子赋值
     * 
     * @param[in] keyPoints 关键点
     * @param[out] descriptors 描述子
     */
    void LinK3D_Extractor::getDescriptors(const vector<pcl::PointXYZI> &keyPoints, 
                                          cv::Mat &descriptors)
    {
        //如果输入的关键点为空，直接返回。
        if(keyPoints.empty())
        {
            return;
        }

        //初始化描述符矩阵 descriptors，它的行数为关键点的数量，列数为180，每个元素初始值为0
        int ptSize = keyPoints.size();

        descriptors = cv::Mat::zeros(ptSize, 180, CV_32FC1); 
        //创建两个二维向量表，distanceTab 用于存储关键点之间的距离，directionTab 用于存储关键点之间的相对方向。
        vector<vector<float>> distanceTab;
        vector<float> oneRowDis(ptSize, 0);
        distanceTab.resize(ptSize, oneRowDis);

        vector<vector<Eigen::Vector2f>> directionTab;
        Eigen::Vector2f direct(0, 0);
        vector<Eigen::Vector2f> oneRowDirect(ptSize, direct);
        directionTab.resize(ptSize, oneRowDirect);

        //Build distance and direction tables for fast descriptor generation.
        //遍历关键点，计算每两个关键点之间的距离和相对方向，并将结果存储在 distanceTab 和 directionTab 中。
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

        //计算描述符
        for(size_t i = 0; i < keyPoints.size(); i++)
        {
            //距离i关键点的距离表
            vector<float> tempRow(distanceTab[i]);
            //距离从小到大排序
            std::sort(tempRow.begin(), tempRow.end());
            int Index[3];
           
            //Get the closest three keypoints of current keypoint.
            // 获取当前关键点的最近的三个关键点的索引
            for(int k = 0; k < 3; k++)
            {              
                //从k+1开始是因为每个距离表中都有自身和自身的距离 为0，排序后会在最前面  
                vector<float>::iterator it1 = find(distanceTab[i].begin(), distanceTab[i].end(), tempRow[k+1]); 
                if(it1 == distanceTab[i].end())
                {
                    continue;
                }
                else
                {
                    //计算两个迭代器之间的元素个数
                    Index[k] = std::distance(distanceTab[i].begin(), it1);
                }
            }

            //Generate the descriptor for each closest keypoint. 
            //The final descriptor is based on the priority of the three closest keypoint.
            //计算描述符的180个区域
            for(int indNum = 0; indNum < 3; indNum++)
            {
                int index = Index[indNum]; // 当前最近的关键点的索引
                Eigen::Vector2f mainDirection;
                mainDirection = directionTab[i][index]; // 主方向，即当前关键点指向最近关键点的方向
                
                vector<vector<float>> areaDis(180); // 存储180个区域的距离
                areaDis[0].emplace_back(distanceTab[i][index]); // 将最近的关键点的距离存入第0个区域

                // 遍历所有关键点，计算它们相对于当前关键点的方向和距离       
                for(size_t j = 0; j < keyPoints.size(); j++)
                {
                    if(j == i || (int)j == index)
                    {
                        continue; // 跳过当前关键点和最近关键点
                    }
                    
                    Eigen::Vector2f otherDirection = directionTab[i][j]; // 当前关键点指向其他关键点的方向
                
                    Eigen::Matrix2f matrixDirect;
                    matrixDirect << mainDirection(0, 0), mainDirection(1, 0), otherDirection(0, 0), otherDirection(1, 0);
                    float deter = matrixDirect.determinant(); // 计算两个向量构成的矩阵的行列式，用于判断方向

                    int areaNum = 0;
                    double cosAng = (double)mainDirection.dot(otherDirection) / (double)(mainDirection.norm() * otherDirection.norm());                                 
                    // 计算两个向量的余弦值，用于计算角度
                    if(abs(cosAng) - 1 > 0)
                    {   
                        continue;
                    }
                    //acos返回[0,PI]弧度值
                    float angle = acos(cosAng) * 180 / M_PI; // 计算夹角的角度值
                    
                    if(angle < 0 || angle > 180)
                    {
                        continue; // 角度不在0到180度范围内，跳过
                    }
                    
                    //这段代码的目的是将一个给定的角度映射到 [0, 180) 范围内的180个2度
                    //的角度区间中的一个。它考虑了向量的相对方向，确保了角度的区间映射在180度内 GPT，可能不对！！！！
                    if(deter > 0)
                    {
                        areaNum = ceil((angle - 1) / 2); // 确定当前角度在哪个区域，通过向上取整将角度划分为区域                     
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
                        areaDis[areaNum].emplace_back(distanceTab[i][j]);// 将当前关键点到该点的距离存入areaDis
                    }
                }
                
                // 遍历180个区域，将最短距离存入描述符中
                float *descriptor = descriptors.ptr<float>(i);     // 获取当前关键点的描述符指针                           

                for(int areaNum = 0; areaNum < 180; areaNum++) 
                {
                    if(areaDis[areaNum].size() == 0)
                    {
                        continue;
                    }
                    else
                    {
                        std::sort(areaDis[areaNum].begin(), areaDis[areaNum].end()); // 将距离值排序

                        if(descriptor[areaNum] == 0)
                        {
                            descriptor[areaNum] = areaDis[areaNum][0]; // 将最短距离存入描述符中
                        }                        
                    }
                }                
            }            
        }
    }

    /**
     * @brief 实现了一个特征匹配的函数 match。该函数接受两组特征点（curAggregationKeyPt 和 toBeMatchedKeyPt）、
     * 对应的特征描述子（curDescriptors 和 toBeMatchedDescriptors）以及一个空的匹配结果容器 vMatchedIndex。
     * 函数的目标是找到两组特征点中相互匹配的特征点索引，并将匹配结果存储在 vMatchedIndex 中。
     * 
     * @param curAggregationKeyPt 
     * @param toBeMatchedKeyPt 
     * @param curDescriptors 
     * @param toBeMatchedDescriptors 
     * @param vMatchedIndex 
     */
    void LinK3D_Extractor::match(
            vector<pcl::PointXYZI> &curAggregationKeyPt, 
            vector<pcl::PointXYZI> &toBeMatchedKeyPt,
            cv::Mat &curDescriptors, 
            cv::Mat &toBeMatchedDescriptors, 
            vector<pair<int, int>> &vMatchedIndex)
    {        
        // 获取两组特征点的数量
        int curKeypointNum = curAggregationKeyPt.size();
        int toBeMatchedKeyPtNum = toBeMatchedKeyPt.size();
        // 记录每个当前特征点与待匹配特征点之间的匹配分数。其中，键为当前特征点的索引，值为匹配分数
        multimap<int, int> matchedIndexScore;      
        // 记录匹配的特征点索引。其中，键为待匹配特征点的索引，值为当前特征点的索引
        multimap<int, int> mMatchedIndex;
        // 记录可能存在重复匹配的待匹配特征点的索引。
        set<int> sIndex;
        // 外层循环遍历当前特征点
        for(int i = 0; i < curKeypointNum; i++)
        {
            std::pair<int, int> highestIndexScore(0, 0);
            //当前特征点描述子
            float* pDes1 = curDescriptors.ptr<float>(i);
            // 内层循环遍历待匹配特征点
            for(int j = 0; j < toBeMatchedKeyPtNum; j++)
            {
                int sameDimScore = 0;
                float* pDes2 = toBeMatchedDescriptors.ptr<float>(j); 
                //计算两个特征描述子之间的匹配分数。匹配分数的计算基于描述子中对应维度上的差值。
                //如果差值在阈值范围内，就认为这个维度上匹配成功，该维度的匹配分数加1。
                for(int bitNum = 0; bitNum < 180; bitNum++)
                {                    
                    if(pDes1[bitNum] != 0 && pDes2[bitNum] != 0 && abs(pDes1[bitNum] - pDes2[bitNum]) <= 0.2){
                        sameDimScore += 1;
                    }
                    // /如果某个待匹配特征点在超过90度的范围内的匹配分数小于3，就跳出内层循环。
                    //这个条件意味着如果待匹配特征点的描述子在超过90度的范围内都没有足够多的匹配维度，就不再尝试匹配。
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
            //选择匹配分数最高的待匹配特征点，将当前特征点和它的匹配特征点的索引存储到 mMatchedIndex 和 sIndex 中。
            //Used for removing the repeated matches.
            matchedIndexScore.insert(std::make_pair(i, highestIndexScore.second)); //Record i and its corresponding score.
            mMatchedIndex.insert(std::make_pair(highestIndexScore.first, i)); //Record the corresponding match between j and i.
            //这是个集合
            sIndex.insert(highestIndexScore.first); //Record the index that may be repeated matches.
        }

        //Remove the repeated matches.
        // 遍历 sIndex 中的待匹配特征点索引
        for(set<int>::iterator setIt = sIndex.begin(); setIt != sIndex.end(); ++setIt)
        {
            int indexJ = *setIt;
            //mMatchedIndex存的是<分数最高的待匹配特征点索引,当前特征点索引> 这里不好描述，参考即可
            auto entries = mMatchedIndex.count(indexJ);
            //如果某个待匹配特征点只有一个匹配项，直接将匹配结果存储到 vMatchedIndex 中
            if(entries == 1)
            {
                //通过j找到迭代器
                auto iterI = mMatchedIndex.find(indexJ);
                //迭代器second就是i，通过i找到了最高分数的迭代器
                auto iterScore = matchedIndexScore.find(iterI->second);
                //分数最低也要高于阈值
                if(iterScore->second >= matchTh)
                {        
                    //分数和j存到最终匹配结果里            
                    vMatchedIndex.emplace_back(std::make_pair(iterI->second, indexJ));
                }           
            }
            //如果某个待匹配特征点有多个匹配项，选择匹配分数最高的匹配项，将匹配结果存储到 vMatchedIndex 中
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

    //Remove the edge keypoints with low curvature for further edge keypoints matching.
    /**
     * @brief 从输入的边缘点云簇（clusters）中剔除曲率较低的点，筛选出具有高曲率的点组成的新的边缘点云簇（filtered）。
     * 
     * @param[in] clusters 输入边缘点云簇
     * @param[out] filtered 具有高曲率的点组成的新的边缘点云簇
     */
    void LinK3D_Extractor::filterLowCurv(ScanEdgePoints &clusters, ScanEdgePoints &filtered)
    {
        // 获取输入边缘点云簇的数量。
        int numCluster = clusters.size();
        // 根据输入的边缘点云簇数量，调整 filtered 的大小，使其与输入的 clusters 保持一致。
        filtered.resize(numCluster);
        //外层循环遍历每个边缘点云簇（i 是当前点云簇的索引）。
        for(int i = 0; i < numCluster; i++)
        {
            int numPt = clusters[i].size();
            ScanEdgePoints tmpCluster;
            vector<int> vScanID;
            // 内层循环遍历当前点云簇中的每个点（j 是当前点的索引）。
            for(int j = 0; j < numPt; j++)
            {
                PointXYZSCA pt = clusters[i][j];
                //获取当前点的扫描线号（scan）。
                int scan = int(pt.scan_position);
                //查找 vScanID 中是否存在当前扫描线号。
                auto it = std::find(vScanID.begin(), vScanID.end(), scan);
                //如果当前扫描线号不在 vScanID 中，将当前点加入新的临时点云簇 tmpCluster 中，并将当前扫描线号加入 vScanID。
                if(it == vScanID.end())
                {
                    vScanID.emplace_back(scan);
                    vector<PointXYZSCA> vPt(1, pt);
                    tmpCluster.emplace_back(vPt);
                }
                //如果当前扫描线号已经在 vScanID 中，将当前点加入 tmpCluster 中的相应扫描线号的子点云簇中。
                else
                {
                    int filteredInd = std::distance(vScanID.begin(), it);
                    tmpCluster[filteredInd].emplace_back(pt);
                }
            }
            // 遍历 tmpCluster 中的每个扫描线号的子点云簇
            for(size_t scanID = 0; scanID < tmpCluster.size(); scanID++)
            {
                // 如果子点云簇中只有一个点，直接将该点加入 filtered 中
                if(tmpCluster[scanID].size() == 1)
                {
                    filtered[i].emplace_back(tmpCluster[scanID][0]);
                }
                //如果子点云簇中有多个点，找出具有最大曲率的点，
                //将该点加入 filtered 中。这一步骤确保了在同一个扫描线上，只选择曲率最大的点。
                else
                {
                    float maxCurv = 0;
                    PointXYZSCA maxCurvPt;
                    for(size_t j = 0; j < tmpCluster[scanID].size(); j++)
                    {
                        if(tmpCluster[scanID][j].curvature > maxCurv)
                        {
                            maxCurv = tmpCluster[scanID][j].curvature;
                            maxCurvPt = tmpCluster[scanID][j];
                        }
                    }

                    filtered[i].emplace_back(maxCurvPt);
                }
            }  
        }
    }

    //Get the edge keypoint matches based on the matching results of aggregation keypoints.
    /**
     * @brief 该函数根据聚合关键点的匹配结果，获取边缘关键点的匹配。
     * 细节说明：根据给的子点云簇索引，把处于相同扫描线上的点匹配起来，存到输出结果中
     * 
     * @param[in] filtered1 聚合关键点云1
     * @param[in] filtered2 聚合关键点云2
     * @param[in] vMatched 匹配索引，注意是点云簇索引
     * @param[out] matchPoints 匹配出的关键点对
     */
    void LinK3D_Extractor::findEdgeKeypointMatch(
            ScanEdgePoints &filtered1, 
            ScanEdgePoints &filtered2, 
            vector<std::pair<int, int>> &vMatched, 
            vector<std::pair<PointXYZSCA, PointXYZSCA>> &matchPoints)
    {
        int numMatched = vMatched.size();// 获取匹配关系数目。
        // 外层循环遍历每个点云簇匹配对（i 是当前匹配对的索引）。
        for(int i = 0; i < numMatched; i++)
        {
            //从 vMatched 中获取当前匹配对的索引 matchedInd，该索引对应的是 filtered1 和 filtered2 中的边缘点云簇。
            pair<int, int> matchedInd = vMatched[i];
            // 获取当前匹配对在 filtered1 和 filtered2 中的点数。
            int numPt1 = filtered1[matchedInd.first].size();
            int numPt2 = filtered2[matchedInd.second].size();
            //创建两个映射（mScanID_Index1 和 mScanID_Index2）用于存储扫描线号与点云索引的关系
            map<int, int> mScanID_Index1;
            map<int, int> mScanID_Index2;
            //遍历第i个(上层循环)匹配子点云簇对中的第一个子点云簇：
            for(int i = 0; i < numPt1; i++)
            {
                int scanID1 = int(filtered1[matchedInd.first][i].scan_position);
                pair<int, int> scanID_Ind(scanID1, i);
                mScanID_Index1.insert(scanID_Ind);
            }

            for(int i = 0; i < numPt2; i++)
            {
                int scanID2 = int(filtered2[matchedInd.second][i].scan_position);
                pair<int, int> scanID_Ind(scanID2, i);
                mScanID_Index2.insert(scanID_Ind);
            }

            //遍历 mScanID_Index1 中的每个扫描线号及对应的点云索引：
            for(auto it1 = mScanID_Index1.begin(); it1 != mScanID_Index1.end(); it1++)
            {
                //获取当前扫描线号（scanID1）。
                int scanID1 = (*it1).first;
                //在 mScanID_Index2 中查找是否存在相同的扫描线号。
                auto it2 = mScanID_Index2.find(scanID1);
                if(it2 == mScanID_Index2.end()){
                    continue;
                }
                //如果存在相同的扫描线号，说明在两个边缘点云簇中找到了匹配点。
                else
                {
                    //临时变量没用 应该是bug
                    vector<PointXYZSCA> tmpMatchPt;
                    PointXYZSCA pt1 = filtered1[matchedInd.first][(*it1).second];
                    PointXYZSCA pt2 = filtered2[matchedInd.second][(*it2).second];
                    
                    pair<PointXYZSCA, PointXYZSCA> matchPt(pt1, pt2);
                    //将匹配点对加入 matchPoints 中，最终存储了所有的边缘关键点匹配结果。
                    matchPoints.emplace_back(matchPt);
                }
            }
        }
    }

    /**
     * @brief 提供输入点云，提取边缘点，聚类，计算描述子
     * 
     * @param pLaserCloudIn 
     * @param keyPoints 
     * @param descriptors 
     * @param validCluster 
     */
    void LinK3D_Extractor::operator()(pcl::PointCloud<pcl::PointXYZ>::Ptr pLaserCloudIn, vector<pcl::PointXYZI> &keyPoints, cv::Mat &descriptors, ScanEdgePoints &validCluster)
    {
        ScanEdgePoints edgePoints;
        extractEdgePoint(pLaserCloudIn, edgePoints);

        ScanEdgePoints sectorAreaCloud;
        divideArea(edgePoints, sectorAreaCloud); 

        ScanEdgePoints clusters;
        getCluster(sectorAreaCloud, clusters); 
        
        vector<int> index;
        keyPoints = getMeanKeyPoint(clusters, validCluster);
          
        getDescriptors(keyPoints, descriptors); 
    }

}
