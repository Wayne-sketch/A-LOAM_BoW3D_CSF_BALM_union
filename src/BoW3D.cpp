#include "BoW3D/BoW3D.h"
#include <fstream>

using namespace std;


namespace BoW3D
{
    BoW3D::BoW3D(LinK3D_Extractor* pLinK3D_Extractor, float thr_, int thf_, int num_add_retrieve_features_): 
            mpLinK3D_Extractor(pLinK3D_Extractor), 
            thr(thr_), 
            thf(thf_), 
            num_add_retrieve_features(num_add_retrieve_features_)
    {
       N_nw_ofRatio = std::make_pair(0, 0); 
    }
    
    /**
     * @brief 该函数用于更新3D词袋模型
     * 
     * @param pCurrentFrame 
     */
    void BoW3D::update(Frame* pCurrentFrame)
    {
        // 将当前帧添加到帧序列中
        mvFrames.emplace_back(pCurrentFrame);

        // 获取当前帧的特征描述子和帧ID
        cv::Mat descriptors = pCurrentFrame->mDescriptors;
        long unsigned int frameId = pCurrentFrame->mnId;

        // 获取特征描述子的数量
        size_t numFeature = descriptors.rows;
        // 如果特征数量小于指定的添加或检索特征数量
        if(numFeature < (size_t)num_add_retrieve_features) 
        {
            // 遍历当前帧的所有特征
            for(size_t i = 0; i < numFeature; i++)
            {
                float *p = descriptors.ptr<float>(i);
                // 遍历特征的所有维度
                for(size_t j = 0; j < (size_t)descriptors.cols; j++)
                {
                    // 如果描述子的值不为0
                    if(p[j] != 0)
                    {
                         // 在3D词袋模型中查找当前描述子值和维度的词
                        unordered_map<pair<float, int>, unordered_set<pair<int, int>, pair_hash>, pair_hash>::iterator it; 

                        pair<float, int> word= make_pair(p[j], j);
                        it = this->find(word);

                        // 如果词不存在，创建新的词条
                        if(it == this->end())
                        {
                            unordered_set<pair<int,int>, pair_hash> place;
                            //frameId是当前帧的id，i代表当前帧第i的特征点
                            place.insert(make_pair(frameId, i));
                            (*this)[word] = place;

                            // 更新比率计数器
                            N_nw_ofRatio.first++;
                            N_nw_ofRatio.second++;
                        }
                        // 如果词已存在，将当前特征添加到该词的集合中
                        else
                        {
                            (*it).second.insert(make_pair(frameId, i));
                            N_nw_ofRatio.second++;
                        }

                    }
                }
            }
        }
        else // 如果特征数量大于等于指定的添加或检索特征数量
        {
            // 只遍历指定数量的特征
            for(size_t i = 0; i < (size_t)num_add_retrieve_features; i++)
            {
                float *p = descriptors.ptr<float>(i);
                // 遍历特征的所有维度
                for(size_t j = 0; j < (size_t)descriptors.cols; j++)
                {
                    // 如果特征值不为0
                    if(p[j] != 0)
                    {
                        // 在3D词袋模型中查找当前特征值和维度的词
                        unordered_map<pair<float, int>, unordered_set<pair<int, int>, pair_hash>, pair_hash>::iterator it; 

                        pair<float, int> word= make_pair(p[j], j);
                        it = this->find(word);

                        // 如果词不存在，创建新的词条
                        if(it == this->end())
                        {
                            unordered_set<pair<int,int>, pair_hash> place;
                            place.insert(make_pair(frameId, i));
                            (*this)[word] = place;

                            // 更新比率计数器
                            N_nw_ofRatio.first++;
                            N_nw_ofRatio.second++;
                        }
                        // 如果词已存在，将当前特征添加到该词的集合中
                        else
                        {
                            (*it).second.insert(make_pair(frameId, i));

                            N_nw_ofRatio.second++;
                        }
                    }
                }
            }
        }
    }
       
    /**
     * @brief 该函数用于在3D词袋模型中检索与当前帧相似的帧。
     * 该函数的主要步骤如下：
     * 1、获取当前帧的ID和特征描述子。
     * 2、遍历特征描述子的每个特征点。
     * 3、对每个特征点，计算与3D词袋模型中的词的相似性。
     * 4、根据相似性评分和阈值筛选潜在的匹配帧。
     * 5、对评分高的帧进行匹配，使用LinK3D_Extractor::match函数获取匹配索引。
     * 6、对匹配到的帧进行循环校正，判断循环校正的有效性。
     * 7、如果循环校正成功，且循环帧之间的距离在阈值范围内，返回匹配结果。
     * 
     * @param pCurrentFrame 
     * @param loopFrameId 
     * @param loopRelR 
     * @param loopRelt 
     */
    void BoW3D::retrieve(Frame* pCurrentFrame, int &loopFrameId, Eigen::Matrix3d &loopRelR, Eigen::Vector3d &loopRelt)
    {    
        // 获取当前帧的ID和特征描述子    
        int frameId = pCurrentFrame->mnId;
        cv::Mat descriptors = pCurrentFrame->mDescriptors;      
        size_t rowSize = descriptors.rows;       
        // 存储帧的评分和帧ID
        map<int, int>mScoreFrameID;
        // 如果特征数量小于指定的添加或检索特征数量
        if(rowSize < (size_t)num_add_retrieve_features) 
        {
            for(size_t i = 0; i < rowSize; i++)
            {
                unordered_map<pair<int, int>, int, pair_hash> mPlaceScore;                
                                
                float *p = descriptors.ptr<float>(i);

                int countValue = 0;

                // 遍历特征的所有维度
                for(size_t j = 0; j < (size_t)descriptors.cols; j++)
                {
                    countValue++;

                    // 如果特征值不为0
                    if(p[j] != 0)
                    {                   
                        pair<float, int> word = make_pair(p[j], j);  
                        auto wordPlacesIter = this->find(word);

                        // 如果词在3D词袋模型中找不到，继续下一个维度
                        if(wordPlacesIter == this->end())
                        {
                            continue;
                        }
                        else
                        {
                            // 计算当前词的平均特征点数量
                            double averNumInPlaceSet = N_nw_ofRatio.second / N_nw_ofRatio.first;
                            int curNumOfPlaces = (wordPlacesIter->second).size();
                            
                            // 计算当前词的特征点数量与平均数量的比值
                            double ratio = curNumOfPlaces / averNumInPlaceSet;

                            // 如果比值超过阈值，继续下一个维度
                            if(ratio > thr)
                            {
                                continue;
                            }

                            // 遍历当前词的所有特征点
                            for(auto placesIter = (wordPlacesIter->second).begin(); placesIter != (wordPlacesIter->second).end(); placesIter++)
                            {
                                //The interval between the loop and the current frame should be at least 300.
                                // 间隔应该至少为300帧
                                if(frameId - (*placesIter).first < 300) 
                                {
                                    continue;
                                }

                                auto placeNumIt = mPlaceScore.find(*placesIter);  

                                // 如果特征点不在集合中，加入；如果已经在集合中，增加计数
                                if(placeNumIt == mPlaceScore.end())
                                {                                
                                    mPlaceScore[*placesIter] = 1;
                                }
                                else
                                {
                                    mPlaceScore[*placesIter]++;                                    
                                }                                                              
                            }                       
                        }                            
                    }                    
                }

                // 根据特征点数量和阈值筛选帧
                for(auto placeScoreIter = mPlaceScore.begin(); placeScoreIter != mPlaceScore.end(); placeScoreIter++)
                {
                    if((*placeScoreIter).second > thf) 
                    {
                       mScoreFrameID[(*placeScoreIter).second] = ((*placeScoreIter).first).first;
                    }
                }                                   
            }                  
        }
        else // 如果特征数量大于等于指定的添加或检索特征数量
        {
            for(size_t i = 0; i < (size_t)num_add_retrieve_features; i++) 
            {
                unordered_map<pair<int, int>, int, pair_hash> mPlaceScore;
                
                float *p = descriptors.ptr<float>(i);

                int countValue = 0;

                // 遍历特征的所有维度
                for(size_t j = 0; j < (size_t)descriptors.cols; j++)
                {
                    countValue++;
                    // 如果特征值不为0
                    if(p[j] != 0)
                    {                   
                        pair<float, int> word = make_pair(p[j], j);    

                        auto wordPlacesIter = this->find(word);

                        // 如果词在3D词袋模型中找不到，继续下一个维度
                        if(wordPlacesIter == this->end())
                        {
                            continue;
                        }
                        else
                        {
                            // 计算当前词的平均特征点数量
                            double averNumInPlaceSet = (double) N_nw_ofRatio.second / N_nw_ofRatio.first;
                            int curNumOfPlaces = (wordPlacesIter->second).size();

                            // 计算当前词的特征点数量与平均数量的比值
                            double ratio = curNumOfPlaces / averNumInPlaceSet;
                            // 如果比值超过阈值，继续下一个维度
                            if(ratio > thr)
                            {
                                continue;
                            }

                            // 遍历当前词的所有特征点
                            for(auto placesIter = (wordPlacesIter->second).begin(); placesIter != (wordPlacesIter->second).end(); placesIter++)
                            {
                                //The interval between the loop and the current frame should be at least 300.
                                // 间隔应该至少为300帧
                                if(frameId - (*placesIter).first < 300) 
                                {
                                    continue;
                                }

                                auto placeNumIt = mPlaceScore.find(*placesIter);                    
                                
                                // 如果特征点不在集合中，加入；如果已经在集合中，增加计数
                                if(placeNumIt == mPlaceScore.end())
                                {                                
                                    mPlaceScore[*placesIter] = 1;
                                }
                                else
                                {
                                    mPlaceScore[*placesIter]++;                                    
                                }                                                              
                            }                       
                        }                            
                    }
                }

                // 根据特征点数量和阈值筛选帧
                for(auto placeScoreIter = mPlaceScore.begin(); placeScoreIter != mPlaceScore.end(); placeScoreIter++)
                {
                    if((*placeScoreIter).second > thf) 
                    {
                       mScoreFrameID[(*placeScoreIter).second] = ((*placeScoreIter).first).first;
                    }
                }                                   
            }                           
        }     

        // 如果没有匹配的帧，直接返回
        if(mScoreFrameID.size() == 0)
        {
            return;
        }

        // 遍历评分的帧集合，从最高分开始匹配
        for(auto it = mScoreFrameID.rbegin(); it != mScoreFrameID.rend(); it++)
        {          
            int loopId = (*it).second;

            // 获取待匹配的帧和当前帧之间的匹配索引
            Frame* pLoopFrame = mvFrames[loopId];
            vector<pair<int, int>> vMatchedIndex;  
            
            mpLinK3D_Extractor->match(pCurrentFrame->mvAggregationKeypoints, pLoopFrame->mvAggregationKeypoints, pCurrentFrame->mDescriptors, pLoopFrame->mDescriptors, vMatchedIndex);               

            int returnValue = 0;
            Eigen::Matrix3d loopRelativeR;
            Eigen::Vector3d loopRelativet;

            // 进行循环校正              
            returnValue = loopCorrection(pCurrentFrame, pLoopFrame, vMatchedIndex, loopRelativeR, loopRelativet);

            //The distance between the loop and the current should less than 3m.   
            // 如果校正成功，且循环帧之间的距离在阈值范围内，返回结果               
            if(returnValue != -1 && loopRelativet.norm() < 3 && loopRelativet.norm() > 0) 
            {
                loopFrameId = (*it).second;
                loopRelR = loopRelativeR;
                loopRelt = loopRelativet;                         
                
                return;
            }     
        } 
    }

    /**
     * @brief 回环校正
     * 
     * @param currentFrame 
     * @param matchedFrame 
     * @param vMatchedIndex 
     * @param R 
     * @param t 
     * @return int 
     */
    int BoW3D::loopCorrection(Frame* currentFrame, Frame* matchedFrame, vector<pair<int, int>> &vMatchedIndex, Eigen::Matrix3d &R, Eigen::Vector3d &t)
    {
        // 如果匹配的索引数量小于等于30，返回失败
        if(vMatchedIndex.size() <= 30)
        {
            return -1;
        }

        // 对当前帧和匹配帧的边缘关键点进行低曲率过滤
        ScanEdgePoints currentFiltered;
        ScanEdgePoints matchedFiltered;
        mpLinK3D_Extractor->filterLowCurv(currentFrame->mClusterEdgeKeypoints, currentFiltered);
        mpLinK3D_Extractor->filterLowCurv(matchedFrame->mClusterEdgeKeypoints, matchedFiltered);

        // 获取匹配的边缘关键点对
        vector<std::pair<PointXYZSCA, PointXYZSCA>> matchedEdgePt;
        mpLinK3D_Extractor->findEdgeKeypointMatch(currentFiltered, matchedFiltered, vMatchedIndex, matchedEdgePt);
        
        // 构建源点云和目标点云
        pcl::PointCloud<pcl::PointXYZ>::Ptr source(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::PointCloud<pcl::PointXYZ>::Ptr target(new pcl::PointCloud<pcl::PointXYZ>());

        pcl::CorrespondencesPtr corrsPtr (new pcl::Correspondences()); 

        // 将匹配的边缘关键点对添加到源点云和目标点云中，并构建对应关系
        for(int i = 0; i < (int)matchedEdgePt.size(); i++)
        {
            std::pair<PointXYZSCA, PointXYZSCA> matchPoint = matchedEdgePt[i];

            pcl::PointXYZ sourcePt(matchPoint.first.x, matchPoint.first.y, matchPoint.first.z);            
            pcl::PointXYZ targetPt(matchPoint.second.x, matchPoint.second.y, matchPoint.second.z);
            
            source->push_back(sourcePt);
            target->push_back(targetPt);

            pcl::Correspondence correspondence(i, i, 0);
            corrsPtr->push_back(correspondence);
        }

        //创建一个Correspondences对象，该对象用于存储最终的匹配关系。
        pcl::Correspondences corrs;
        //创建了一个RANSAC算法的对象。在PCL中，CorrespondenceRejectorSampleConsensus类实现了RANSAC算法，用于从给定的对应关系中识别出最优的一组内点。
        pcl::registration::CorrespondenceRejectorSampleConsensus<pcl::PointXYZ> Ransac_based_Rejection;
        //设置RANSAC算法的输入源点云和目标点云。RANSAC算法会在这两个点云之间找到对应关系。
        Ransac_based_Rejection.setInputSource(source);
        Ransac_based_Rejection.setInputTarget(target);
        //设置RANSAC算法的阈值，即用于判断点是否为内点的距离阈值。在RANSAC中，距离小于此阈值的点被认为是内点，用于拟合模型。
        double sac_threshold = 0.4;
        Ransac_based_Rejection.setInlierThreshold(sac_threshold);
        //运行RANSAC算法，将结果保存在corrs中。getRemainingCorrespondences函数将通过RANSAC算法获得的内点的索引保存在corrs中，这些内点代表了正确的匹配关系。
        Ransac_based_Rejection.getRemainingCorrespondences(*corrsPtr, corrs);

        // 如果剩余的对应关系数量小于等于100，返回失败
        if(corrs.size() <= 100)
        {
            return -1;
        }       
                
        Eigen::Vector3d p1 = Eigen::Vector3d::Zero();
        Eigen::Vector3d p2 = p1;
        int corrSize = (int)corrs.size();

        // 计算源点云和目标点云的质心
        for(int i = 0; i < corrSize; i++)
        {  
            pcl::Correspondence corr = corrs[i];         
            p1(0) += source->points[corr.index_query].x;
            p1(1) += source->points[corr.index_query].y;
            p1(2) += source->points[corr.index_query].z; 

            p2(0) += target->points[corr.index_match].x;
            p2(1) += target->points[corr.index_match].y;
            p2(2) += target->points[corr.index_match].z;
        }
        // 计算质心
        Eigen::Vector3d center1 = Eigen::Vector3d(p1(0)/corrSize, p1(1)/corrSize, p1(2)/corrSize);
        Eigen::Vector3d center2 = Eigen::Vector3d(p2(0)/corrSize, p2(1)/corrSize, p2(2)/corrSize);
       
        // 计算去质心后的点
        vector<Eigen::Vector3d> vRemoveCenterPt1, vRemoveCenterPt2; 
        for(int i = 0; i < corrSize; i++)
        {
            pcl::Correspondence corr = corrs[i];
            pcl::PointXYZ sourcePt = source->points[corr.index_query];
            pcl::PointXYZ targetPt = target->points[corr.index_match];

            Eigen::Vector3d removeCenterPt1 = Eigen::Vector3d(sourcePt.x - center1(0), sourcePt.y - center1(1), sourcePt.z - center1(2));
            Eigen::Vector3d removeCenterPt2 = Eigen::Vector3d(targetPt.x - center2(0), targetPt.y - center2(1), targetPt.z - center2(2));
        
            vRemoveCenterPt1.emplace_back(removeCenterPt1);
            vRemoveCenterPt2.emplace_back(removeCenterPt2);
        }

        Eigen::Matrix3d w = Eigen::Matrix3d::Zero();

        // 计算协方差矩阵
        for(int i = 0; i < corrSize; i++)
        {
            w += vRemoveCenterPt1[i] * vRemoveCenterPt2[i].transpose();
        }      

        Eigen::JacobiSVD<Eigen::Matrix3d> svd(w, Eigen::ComputeFullU|Eigen::ComputeFullV);
        Eigen::Matrix3d U = svd.matrixU();
        Eigen::Matrix3d V = svd.matrixV();
        
        // 计算旋转矩阵和平移向量
        R = V * U.transpose();
        t = center2 - R * center1;

        return 1;
    }

}
