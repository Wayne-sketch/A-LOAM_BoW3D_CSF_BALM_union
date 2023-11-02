/*
 * @Author: ctx cuitongxin201024@163.com
 * @Date: 2023-10-30 13:35:41
 * @LastEditors: ctx cuitongxin201024@163.com
 * @LastEditTime: 2023-10-31 14:34:15
 * @FilePath: \BoW3D\include\BoW3D.h
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 */
#pragma once

#include <iostream>
#include <vector>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <set>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

#include <pcl/registration/ia_ransac.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/sample_consensus_prerejective.h> 
#include <pcl/registration/correspondence_estimation.h>
#include <pcl/registration/correspondence_rejection_sample_consensus.h>

#include "Frame.h"
#include "LinK3D_Extractor.h"


using namespace std;

namespace BoW3D
{
    class Frame;
    class LinK3D_extractor;
    
    //通用的哈希函数
    template <typename T>
    inline void hash_combine(std::size_t &seed, const T &val) 
    {
        seed ^= std::hash<T>()(val) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    }
    
    template <typename T> 
    inline void hash_val(std::size_t &seed, const T &val) 
    {
        hash_combine(seed, val);
    }

    template <typename T1, typename T2>
    inline void hash_val(std::size_t &seed, const T1 &val1, const T2 &val2) 
    {
        hash_combine(seed, val1);
        hash_val(seed, val2);
    }

    template <typename T1, typename T2>
    inline std::size_t hash_val(const T1 &val1, const T2 &val2) 
    {
        std::size_t seed = 0;
        hash_val(seed, val1, val2);
        return seed;
    }

    struct pair_hash 
    {
        template <class T1, class T2>
        std::size_t operator()(const std::pair<T1, T2> &p) const {
            return hash_val(p.first, p.second);
        }
    };
   
   /**
    * @brief BoW3D 类继承自 std::unordered_map，其中键是 pair<float, int> 类型（用于表示维度值和维度ID的组合），
    * 值是 std::unordered_set<pair<int, int>, pair_hash> 类型（表示Frame ID和Descriptor ID的组合的集合），
    * 并且使用了自定义的哈希函数 pair_hash。
    */
    class BoW3D: public unordered_map<pair<float, int>, unordered_set<pair<int, int>, pair_hash>, pair_hash>  //Dimension value, Dimension ID; Frame ID, Descriptor ID
    {
        public:
            BoW3D(LinK3D_Extractor* pLinK3D_Extractor, float thr_, int thf_, int num_add_retrieve_features_);

            ~BoW3D(){}
            
            void update(Frame* pCurrentFrame);

            //回环校正
            int loopCorrection(Frame* currentFrame, Frame* matchedFrame, vector<pair<int, int>> &vMatchedIndex, Eigen::Matrix3d &R, Eigen::Vector3d &t);

            //用于检索匹配的3D特征点
            void retrieve(Frame* pCurrentFrame, int &loopFrameId, Eigen::Matrix3d &loopRelR, Eigen::Vector3d &loopRelt);           

        private:
            LinK3D_Extractor* mpLinK3D_Extractor;
            //用于计算论文中的比率。
            std::pair<int, int> N_nw_ofRatio; //Used to compute the ratio in our paper.          

            vector<Frame*> mvFrames;
            //比率阈值。
            float thr; //Ratio threshold in our paper.
            //频率阈值。
            int thf; //Frequency threshold in our paper.
            //每帧添加或检索的特征数量。
            int num_add_retrieve_features; //The number of added or retrieved features for each frame.
    };
}
