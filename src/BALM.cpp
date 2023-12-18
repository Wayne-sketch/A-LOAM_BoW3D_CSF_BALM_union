#include "balmclass.hpp"
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <queue>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <visualization_msgs/MarkerArray.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>
#include <geometry_msgs/PoseArray.h>

/*
自适应voxel的划分，从1m分辨率通过八叉树划分到0.125m分辨率。目的是保证一个像素内只存在一个特征。
 做法是计算格子内点的特征向量，不满足条件则划分，直到格子内只有一个特征，或者格子内的点数小于阈值。
 本文共维护了两个voxel地图，一个edge的和一个plane的。这样做的好处是寻找最近邻时速度很快。

 两种方式可以停止格子划分，1是达到了最大深度，2是格子内点的数目小于一个阈值。

把当前scan加入到map中，并构建成voxel的形式。当插入进来5帧scan后，进行map-refinement。滑窗大小是20帧相邻的帧，进行Local-BA。
本文的优点是提升了定位的精度和增强了点云一致性，但使用的是连续帧，后续可以改成只关键帧参与Local BA，或者应用在全局地图优化中

为了获得不同帧的同一边缘/平面所对应的点集合，有必要找到帧之间的特征点对应关系。为此，作者提出了自适应体素。首先，将三维空间划分为范围单位为1米的体素，
 然后计算体素内协方差矩阵的特征值，以确定体素内的点是否落在同一边缘/平面上，如果是，则保留当前体素，否则将体素划分为8个更小的体素，重复上述操作。


 如果当前voxel中的所有特征点都位于平面或者边缘上，就保存该voxel，否则就将voxel八分化，然后检查这8个小voxel，
按照前面说的条件判断是否保存还是继续往下分，直到八分化到最小（例如，每个voxel小于0.125m）

// 对滑窗中的体素进行填充：
// 构建了一个由Hash组织的自适应体素表和每个Hash条目的八叉树。更具体地说，我们首先将空间（在全局世界坐标系中）切割成体素，每个具有粗略地图分辨率的大小。
// 然后，对于  第一次激光雷达扫描，定义了世界坐标系  包含的点被分布到体素中。已填充体素  其中的点被索引到哈希表中。然后，对于每个填充体素，
// 如果所有包含的点都位于一个平面上（  点协方差矩阵的最小特征值小于  指定的阈值），我们存储平面点并计算（5）中的平面参数（n，q）及其不确定度
// ∑n,q如（8）所示；否则，当前体素将进入  八个八分之一和重复平面检查和体素切入  每个层直到达到最大层数。
// 请注意，体素具有不同的大小，每个体素都包含一个从包含的激光雷达原始点拟合的平面特征

 */
using namespace std;
double voxel_size[2] = {1.0, 1.0};// 单位m
bool viewVoxel;
pcl::PointCloud<PointType> global;

bool useEdge;
mutex mBuf;
queue<sensor_msgs::PointCloud2ConstPtr> ground_buf, corn_buf, offground_buf;
queue<nav_msgs::Odometry::ConstPtr> odom_buf;

void ground_handler(const sensor_msgs::PointCloud2ConstPtr &msg)
{
    mBuf.lock();
    ground_buf.push(msg);
    mBuf.unlock();
}

void corn_handler(const sensor_msgs::PointCloud2ConstPtr &msg)
{
    mBuf.lock();
    corn_buf.push(msg);
    mBuf.unlock();
}


void odom_handler(const nav_msgs::Odometry::ConstPtr &msg)
{
    mBuf.lock();
    odom_buf.push(msg);
    mBuf.unlock();
}

void offground_handler(const sensor_msgs::PointCloud2ConstPtr &msg)
{
    mBuf.lock();
    offground_buf.push(msg);
    mBuf.unlock();
}

// 构建八叉树的第一层：
void cut_voxel(unordered_map<VOXEL_LOC, OCTO_TREE*> &feat_map, pcl::PointCloud<PointType>::Ptr pl_feat, Eigen::Matrix3d R_p, Eigen::Vector3d t_p, int feattype, int fnum, int capacity)
{
    uint plsize = pl_feat->size();
    for(uint i=0; i<plsize; i++)
    {
        PointType &p_c = pl_feat->points[i];
        Eigen::Vector3d pvec_orig(p_c.x, p_c.y, p_c.z);// lidar坐标系下的点云pfi
        Eigen::Vector3d pvec_tran = R_p*pvec_orig + t_p;// 世界坐标系下的点云pi

        float loc_xyz[3];
        for(int j=0; j<3; j++)
        {
            loc_xyz[j] = pvec_tran[j] / voxel_size[feattype];// 体素最大尺寸为一米
            if(loc_xyz[j] < 0)
            {
                loc_xyz[j] -= 1.0;
            }
        }

        // 取整后是体素栅格的起始坐标，在该长度范围内的点云都属于该体素
        VOXEL_LOC position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1], (int64_t)loc_xyz[2]);
        // 用于在无序映射容器中查找指定键的位置。如果找到了该键，则返回指向该键的迭代器；否则返回指向容器末尾的迭代器。
        // 在当前体素地图中，找当前点对应的体素栅格，如果找到了说明之前以前有点建立过栅格了，如果没有找到说明没有建立过，那就建立一个
        auto iter = feat_map.find(position);
        if(iter != feat_map.end())// 紧密点
        {
            // 同一体素栅格中的点
            iter->second->plvec_orig[fnum]->push_back(pvec_orig);// lidar坐标系下的体素
            iter->second->plvec_tran[fnum]->push_back(pvec_tran);// 世界坐标系下的体素
            iter->second->is2opt = true;
        }
        else// 非紧密点
        {
            // 新建体素栅格
            OCTO_TREE *ot = new OCTO_TREE(feattype, capacity);// octo_state = 0
            // 填充体素栅格
            ot->plvec_orig[fnum]->push_back(pvec_orig);// lidar坐标系下的体素
            ot->plvec_tran[fnum]->push_back(pvec_tran);// 世界坐标系下的体素
            // 当前体素的中心
            ot->voxel_center[0] = (0.5+position.x) * voxel_size[feattype];
            ot->voxel_center[1] = (0.5+position.y) * voxel_size[feattype];
            ot->voxel_center[2] = (0.5+position.z) * voxel_size[feattype];
            ot->quater_length = voxel_size[feattype] / 4.0;// 1/4宽度
            feat_map[position] = ot;// 更新体素地图
        }
    }
}

void cut_voxel2(unordered_map<VOXEL_LOC, OCTO_TREE*> &feat_map, pcl::PointCloud<PointType> &pl_feat, const IMUST &x_key, int fnum)
{
    float loc_xyz[3];
    for(PointType &p_c : pl_feat.points)
    {
        Eigen::Vector3d pvec_orig(p_c.x, p_c.y, p_c.z);
        Eigen::Vector3d pvec_tran = x_key.R*pvec_orig + x_key.p;

        for(int j=0; j<3; j++)
        {
            loc_xyz[j] = pvec_tran[j] / voxel_size[0];
            if(loc_xyz[j] < 0) loc_xyz[j] -= 1.0;
        }

        VOXEL_LOC position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1], (int64_t)loc_xyz[2]);
        auto iter = feat_map.find(position);
        if(iter != feat_map.end())
        {
            if(iter->second->octo_state != 2)
            {
                iter->second->vec_orig[fnum].push_back(pvec_orig);
                iter->second->vec_tran[fnum].push_back(pvec_tran);
            }

            if(iter->second->octo_state != 1)
            {
                iter->second->sig_orig[fnum].push(pvec_orig);
                iter->second->sig_tran[fnum].push(pvec_tran);
            }
            iter->second->is2opt = true;
            iter->second->life = life_span;
            iter->second->each_num[fnum]++;
        }
        else
        {
            OCTO_TREE *ot = new OCTO_TREE();
            ot->vec_orig[fnum].push_back(pvec_orig);
            ot->vec_tran[fnum].push_back(pvec_tran);
            ot->sig_orig[fnum].push(pvec_orig);
            ot->sig_tran[fnum].push(pvec_tran);
            ot->each_num[fnum]++;

            ot->voxel_center[0] = (0.5+position.x) * voxel_size[0];
            ot->voxel_center[1] = (0.5+position.y) * voxel_size[0];
            ot->voxel_center[2] = (0.5+position.z) * voxel_size[0];
            ot->quater_length = voxel_size[0] / 4.0;
            ot->layer = 0;
            feat_map[position] = ot;
        }

    }

}


int main(int argc, char **argv)
{
    ros::init(argc, argv, "balm_back_end");
    ros::NodeHandle n;

    ros::Subscriber sub_corn = n.subscribe<sensor_msgs::PointCloud2>("/Edge", 100, corn_handler);
    ros::Subscriber sub_ground = n.subscribe<sensor_msgs::PointCloud2>("/Ground", 100, ground_handler);
    ros::Subscriber sub_offground = n.subscribe<sensor_msgs::PointCloud2>("/OffGround", 100, offground_handler);
    ros::Subscriber sub_odom = n.subscribe<nav_msgs::Odometry>("/aft_mapped_to_init", 100, odom_handler);

    ros::Publisher pub_corn = n.advertise<sensor_msgs::PointCloud2>("/map_corn", 10);
    ros::Publisher pub_ground = n.advertise<sensor_msgs::PointCloud2>("/map_ground", 10);
    ros::Publisher pub_full = n.advertise<sensor_msgs::PointCloud2>("/map_full", 10);
    ros::Publisher pub_test = n.advertise<sensor_msgs::PointCloud2>("/map_test", 10);
    ros::Publisher pub_odom = n.advertise<nav_msgs::Odometry>("/odom_rviz_last", 100);
    ros::Publisher pub_pose = n.advertise<geometry_msgs::PoseArray>("/poseArrayTopic", 10);
    ros::Publisher pubOdomAftPGO = n.advertise<nav_msgs::Odometry>("/newest_odom", 100);
    ros::Publisher pub_cute = n.advertise<sensor_msgs::PointCloud2>("/map_cute", 100);
    n.param<bool>("viewVoxel", viewVoxel, true);


    // -------------------------- for loop --------------------
    ros::Publisher pubOdomAftMapped = n.advertise<nav_msgs::Odometry>("/BALM_mapped_to_init", 100);
    ros::Publisher puboffgrounds = n.advertise<sensor_msgs::PointCloud2>("/offground_BA", 100);
    ros::Publisher pubground = n.advertise<sensor_msgs::PointCloud2>("/ground_BA", 100);
    ros::Publisher pubEdge = n.advertise<sensor_msgs::PointCloud2>("/Edge_BA", 100);

    double ground_filter_length = 0.4;
    double corn_filter_length = 0.2;

    int window_size = 20;// sliding window size
    int margi_size = 5;// margilization size
    int filter_num = 1;// for map-refine LM optimizer
    int thread_num = 4;// for map-refine LM optimizer
    int skip_num = 0;
    int pub_skip = 1;

    n.param<double>("ground_filter_length", ground_filter_length, 0.4);
    n.param<double>("corn_filter_length", corn_filter_length, 0.2);
    n.param<double>("root_ground_voxel_size", voxel_size[0], 1);
    n.param<double>("root_corn_voxel_size", voxel_size[1], 1);
    n.param<int>("skip_num", skip_num, 0);
    n.param<double>("ground_feat_eigen_limit", feat_eigen_limit[0], 9);
    n.param<double>("corn_feat_eigen_limit", feat_eigen_limit[0], 4);
    n.param<double>("ground_opt_feat_eigen_limit", feat_eigen_limit[0], 16);
    n.param<double>("corn_opt_feat_eigen_limit", feat_eigen_limit[0], 9);
    n.param<int>("pub_skip", pub_skip, 1);
    n.param<bool>("useEdge", useEdge, false);

    thread *map_refine_thread = nullptr;

    int jump_flag = skip_num;
    printf("%d\n", skip_num);
    LM_SLWD_VOXEL opt_lsv(window_size, filter_num, thread_num);

    pcl::PointCloud<PointType>::Ptr pl_corn(new pcl::PointCloud<PointType>);
    pcl::PointCloud<PointType>::Ptr pl_ground(new pcl::PointCloud<PointType>);
    pcl::PointCloud<PointType>::Ptr pl_offground(new pcl::PointCloud<PointType>);

    vector<pcl::PointCloud<PointType>::Ptr> pl_ground_buf;
    vector<pcl::PointCloud<PointType>::Ptr> pl_edge_buf;
    vector<pcl::PointCloud<PointType>::Ptr> pl_offground_buf;

    vector<Eigen::Quaterniond> q_poses;
    vector<Eigen::Vector3d> t_poses;
    Eigen::Quaterniond q_odom, q_gather_pose(1, 0, 0, 0) ,q_last(1, 0, 0, 0);
    Eigen::Vector3d t_odom, t_gather_pose(0, 0, 0), t_last(0, 0, 0);
    int plcount = 0, window_base = 0;

    // unordered_map存储元素时是没有顺序的，只是根据key的哈希值，将元素存在指定位置，所以根据key查找单个value时非常高效，平均可以在常数时间内完成
    unordered_map<VOXEL_LOC, OCTO_TREE*> ground_map, corn_map, ground_map2;
    Eigen::Matrix4d trans(Eigen::Matrix4d::Identity());
    geometry_msgs::PoseArray parray;
    parray.header.frame_id = "/camera_init";

    while(n.ok())
    {
        ros::spinOnce();
        if(corn_buf.empty() || ground_buf.empty()  || odom_buf.empty() || offground_buf.empty())
        {
            continue;
        }

        mBuf.lock();
        uint64_t time_corn = corn_buf.front()->header.stamp.toNSec();
        uint64_t time_ground = ground_buf.front()->header.stamp.toNSec();
        uint64_t time_odom = odom_buf.front()->header.stamp.toNSec();
        uint64_t time_offground = offground_buf.front()->header.stamp.toNSec();
        if(time_odom != time_corn)
        {
            time_odom < time_corn ? odom_buf.pop() : corn_buf.pop();
            mBuf.unlock();
            continue;
        }

        if(time_odom != time_ground)
        {
            time_odom < time_ground ? odom_buf.pop() : ground_buf.pop();
            mBuf.unlock();
            continue;
        }

        if(time_odom != time_offground)
        {
            time_odom < time_ground ? odom_buf.pop() : ground_buf.pop();
            mBuf.unlock();
            continue;
        }

        ros::Time ct(ground_buf.front()->header.stamp);
        pcl::PointCloud<PointType>::Ptr pl_ground_temp(new pcl::PointCloud<PointType>);
        pcl::PointCloud<PointType>::Ptr pl_edge_temp(new pcl::PointCloud<PointType>);
        pcl::PointCloud<PointType>::Ptr pl_offground_temp(new pcl::PointCloud<PointType>);

        rosmsg2ptype(*ground_buf.front(), *pl_ground);
        rosmsg2ptype(*corn_buf.front(), *pl_corn);
        rosmsg2ptype(*offground_buf.front(), *pl_offground);

        //pcl::io::savePCDFileBinary("/home/wb/FALOAMBA_WS/wb/Map/map.pcd", *pl_ground);

        *pl_ground_temp = *pl_ground;
        *pl_edge_temp = *pl_corn;
        *pl_offground_temp = *pl_offground;
        corn_buf.pop(); ground_buf.pop(); offground_buf.pop();

        q_odom.w() = odom_buf.front()->pose.pose.orientation.w;
        q_odom.x() = odom_buf.front()->pose.pose.orientation.x;
        q_odom.y() = odom_buf.front()->pose.pose.orientation.y;
        q_odom.z() = odom_buf.front()->pose.pose.orientation.z;
        t_odom.x() = odom_buf.front()->pose.pose.position.x;
        t_odom.y() = odom_buf.front()->pose.pose.position.y;
        t_odom.z() = odom_buf.front()->pose.pose.position.z;
        odom_buf.pop();
        mBuf.unlock();

        // T_curr2last = T_curr2w * T_last2w¯¹
        Eigen::Vector3d delta_t(q_last.matrix().transpose()*(t_odom-t_last));
        Eigen::Quaterniond delta_q(q_last.matrix().transpose() * q_odom.matrix());
        q_last = q_odom;
        t_last = t_odom;

        // T_curr2last * I
        t_gather_pose = t_gather_pose + q_gather_pose * delta_t;
        q_gather_pose = q_gather_pose * delta_q;
        if(jump_flag < skip_num)
        {
            jump_flag++;
            continue;
        }
        jump_flag = 0;

        if(plcount == 0)// 第一帧
        {
            // 第一帧：T_curr2w = T_curr2last
            q_poses.push_back(q_gather_pose);
            t_poses.push_back(t_gather_pose);
        }
        else// 第二帧
        {
            // T_1_2_0 * T_2_2_1 = T_2_2_0
            // T_2_2_0 * T_3_2_2 = T_3_2_0
            q_poses.push_back(q_poses[plcount-1]*q_gather_pose);
            t_poses.push_back(t_poses[plcount-1] + q_poses[plcount-1] * t_gather_pose);
        }

        parray.header.stamp = ct;
        geometry_msgs::Pose apose;
        apose.orientation.w = q_poses[plcount].w();
        apose.orientation.x = q_poses[plcount].x();
        apose.orientation.y = q_poses[plcount].y();
        apose.orientation.z = q_poses[plcount].z();
        apose.position.x = t_poses[plcount].x();
        apose.position.y = t_poses[plcount].y();
        apose.position.z = t_poses[plcount].z();

        // ---------------------------- 当前帧位姿(优化前的)--------------------------------
        nav_msgs::Odometry laser_odom;
        laser_odom.header.frame_id = "/camera_init";
        laser_odom.child_frame_id = "/aft_BA";
        laser_odom.header.stamp = ct;
        laser_odom.pose.pose.orientation.x = apose.orientation.x;
        laser_odom.pose.pose.orientation.y = apose.orientation.y;
        laser_odom.pose.pose.orientation.z = apose.orientation.z;
        laser_odom.pose.pose.orientation.w = apose.orientation.w;
        laser_odom.pose.pose.position.x = apose.position.x;
        laser_odom.pose.pose.position.y = apose.position.y;
        laser_odom.pose.pose.position.z = apose.position.z;
        pub_odom.publish(laser_odom);
        static tf::TransformBroadcaster br;
        tf::Transform transform;
        tf::Quaternion q;
        transform.setOrigin(tf::Vector3(apose.position.x, apose.position.y, apose.position.z));
        q.setW(apose.orientation.w);
        q.setX(apose.orientation.x);
        q.setY(apose.orientation.y);
        q.setZ(apose.orientation.z);
        transform.setRotation(q);
        br.sendTransform(tf::StampedTransform(transform, laser_odom.header.stamp, "/camera_init", "/aft_BA"));
        parray.poses.push_back(apose);

        // 发布优化前的位姿
        pub_pose.publish(parray);
        pl_ground_buf.push_back(pl_ground_temp);
        pl_edge_buf.push_back(pl_edge_temp);
        pl_offground_buf.push_back(pl_offground_temp);

        plcount++;
        OCTO_TREE::voxel_windowsize = plcount - window_base;
        q_gather_pose.setIdentity();
        t_gather_pose.setZero();

        down_sampling_voxel(*pl_ground, ground_filter_length);
        if(useEdge)
        {
            down_sampling_voxel(*pl_corn, corn_filter_length);
        }
        // 滑窗中的头帧
        int frame_head = plcount-1-window_base;
        // Put current feature points into root voxel node 创建根节点(论文中的方形节点)
        // ground_map和window_size是全局变量
        // 地图由哈希表和八叉树构成，哈希表管理最上层的地图结构，八叉树每个节点中存放一个平面，如果一个节点中的点不能被表示为一个特征，拆分(recut)这个节点
        cut_voxel(ground_map, pl_ground, q_poses[plcount-1].matrix(), t_poses[plcount-1], 0, frame_head, window_size);
        //后续地图的点被送到对应的节点中，在面特征稳定后删除旧的观测，只保留最新观测，如果新的观测和旧的观测冲突，删去旧估计重写估计位姿。

        if(useEdge)
        {
            cut_voxel(corn_map, pl_corn, q_poses[plcount-1].matrix(), t_poses[plcount-1], 1, frame_head, window_size);
        }


        // Points in new scan have been distributed in corresponding root node voxel
        // Then continue to cut the root voxel until right size
        for(auto iter=ground_map.begin(); iter!=ground_map.end(); ++iter)// 遍历方形节点
        {
            if(iter->second->is2opt)// Sliding window of root voxel should have points
            {
                iter->second->root_centors.clear();
                /*
                为了获得不同帧的同一边缘/平面所对应的点集合，有必要找到帧之间的特征点对应关系。为此，作者提出了自适应体素。首先，将三维空间划分为范围单位为1米的体素，
                然后计算体素内协方差矩阵的特征值，以确定体素内的点是否落在同一边缘/平面上，如果是，则保留当前体素，否则将体素划分为8个更小的体素，重复上述操作。
                */
                // 构建八叉树(论文中的圆形节点)，参数：根节点id、滑动窗口最新帧id、待更新变量:中心点
                iter->second->recut(0, frame_head, iter->second->root_centors);
                if(!iter->second->root_centors.empty())
                {
                    //pcl::io::savePCDFileASCII ("/home/wb/FALOAMGBA_WS/wb/pcd/666.pcd", iter->second->root_centors);
                }
            }
        }

        if(useEdge)
        {
            for(auto iter=corn_map.begin(); iter!=corn_map.end(); ++iter)
            {
                if(iter->second->is2opt)
                {
                    iter->second->root_centors.clear();
                    iter->second->recut(0, frame_head, iter->second->root_centors);
                }
            }
        }
        // 对于在线激光雷达里程计，新的激光雷达点云帧不断的进入并且配准，并估计了它们的姿态。然后使用估计的姿态将新点注册到全局地图中：
        // 当新点位于一个未填充的体素中时，它将构造该体素。否则，当将新点添加到现有体素时，应该更新体素中平面的参数和不确定性。

        // 以上使用20帧的边缘点和地面点构建了体素网格，并且平面与直线已经拟合好了

        // ---------------------------- 20帧后进来，大于20帧才开始优化 ----------------------------
        // Begin map refine module
        if(plcount >= window_base+window_size)// 大于20帧才开始 进行 深度优先搜索和位姿优化
        {
            for(int i=0; i<window_size; i++)
            {
                // 设置初始值
                opt_lsv.so3_poses[i].setQuaternion(q_poses[window_base + i]);
                opt_lsv.t_poses[i] = t_poses[window_base + i];
            }
            // Do not optimize first sliding window
            if(window_base != 0)
            {
                // Push voxel map into optimizer
                for(auto iter=ground_map.begin(); iter!=ground_map.end(); ++iter)// 遍历方形节点(遍历滑窗中的所有体素)
                {
                    if(iter->second->is2opt)
                    {
                        /*
                         给定一个在具有姿态先验的世界坐标系中预测的激光雷达点Pw，
                         我们首先通过它的哈希键找到它所在的根体素（具有粗糙的地图分辨率）。然后，对所有包含的子体素进行轮询，以此与点匹配。
                         具体来说，让一个子体素包含一个具有法线ni和中心qi的平面，计算点到平面的距离
                        */
                        iter->second->traversal_opt(opt_lsv);// 对当前体素进行操作
                    }
                }

                if(useEdge)
                {
                    // // Push voxel map into optimizer
                    for(auto iter=corn_map.begin(); iter!=corn_map.end(); ++iter)
                    {
                        if(iter->second->is2opt)
                        {
                            iter->second->traversal_opt(opt_lsv);// 点线匹配
                        }
                    }
                }
                // Begin iterative optimization
                opt_lsv.damping_iter();
            }

            pcl::PointCloud<PointType> pl_send;
            // 发布边缘化出去的帧
            for(int i=0; i<margi_size; i+=pub_skip)
            {
                // 优化后的位姿
                trans.block<3, 3>(0, 0) = opt_lsv.so3_poses[i].matrix();
                trans.block<3, 1>(0, 3) = opt_lsv.t_poses[i];

                pcl::PointCloud<PointType> pcloud;
                if(viewVoxel)
                {
                    // 优化后的位姿乘以其对应帧的点云
                    pcl::transformPointCloud((*pl_offground_buf[window_base + i]), pcloud, trans);
                    IMUST T;
                    T.R = trans.block<3, 3>(0, 0);
                    T.p = trans.block<3, 1>(0, 3);

                    cut_voxel2(ground_map2, *pl_ground_buf[window_base + i], T, i);
                }
                else
                {
                    pcl::transformPointCloud((*pl_ground_buf[window_base + i]+*pl_offground_buf[window_base + i]), pcloud, trans);
                }

                // margi_size帧组成的局部地图
                pl_send += pcloud;
                // cout << "opt_lsv.t_poses = " << opt_lsv.t_poses[i] << endl;
                Eigen::Quaterniond q_w_curr(opt_lsv.so3_poses[i].matrix());
                nav_msgs::Odometry odomAftMapped;
                odomAftMapped.header.frame_id = "/camera_init";
                ros::Time time = ros::Time::now();
                odomAftMapped.header.stamp = time;
                odomAftMapped.pose.pose.orientation.x = q_w_curr.x();
                odomAftMapped.pose.pose.orientation.y = q_w_curr.y();
                odomAftMapped.pose.pose.orientation.z = q_w_curr.z();
                odomAftMapped.pose.pose.orientation.w = q_w_curr.w();
                odomAftMapped.pose.pose.position.x = opt_lsv.t_poses[i].x();
                odomAftMapped.pose.pose.position.y = opt_lsv.t_poses[i].y();
                odomAftMapped.pose.pose.position.z = opt_lsv.t_poses[i].z();
                // ---------------------- 发布优化后的位姿给Rviz 和 loop模块 ------------------------------
                pubOdomAftMapped.publish(odomAftMapped);
                // ----------------------------------- for loop ----------------------------------------
                sensor_msgs::PointCloud2 ground_msg;
                pcl::toROSMsg(  *pl_ground_buf[window_base + i], ground_msg);
                ground_msg.header.stamp = time;
                ground_msg.header.frame_id = "/camera_init";
                pubground.publish(ground_msg);
                pl_ground_buf[window_base + i]->clear();

                sensor_msgs::PointCloud2 edge_msg;
                pcl::toROSMsg(*pl_edge_buf[window_base + i], edge_msg);
                edge_msg.header.stamp = time;
                edge_msg.header.frame_id = "/camera_init";
                pubEdge.publish(edge_msg);
                pl_edge_buf[window_base + i]->clear();

                sensor_msgs::PointCloud2 offground_msg;
                pcl::toROSMsg(*pl_offground_buf[window_base + i], offground_msg);
                offground_msg.header.stamp = time;
                offground_msg.header.frame_id = "/camera_init";
                puboffgrounds.publish(offground_msg);
                pl_offground_buf[window_base + i]->clear();

            }

            //global += pl_send;
            //pcl::io::savePCDFileBinary("/home/wb/FALOAMesh_WS/test_BA/test.pcd", global);
            if(viewVoxel)
            {
                for(auto iter=ground_map2.begin(); iter!=ground_map2.end() && n.ok(); iter++)
                {
                    int win_size = 20;
                    iter->second->recut2(win_size);
                    iter->second->tras_display(pl_send, win_size);
                }
                // pub_func(pl_send, pub_cute, ct);
                ground_map2.clear();
            }
            //pub_func(pl_send, pub_full, ct);


            for(int i=0; i<margi_size; i++)
            {
                pl_ground_buf[window_base + i] = nullptr;
                pl_edge_buf[window_base + i] = nullptr;
                pl_offground_buf[window_base + i] = nullptr;
            }

            for(int i=0; i<window_size; i++)
            {
                q_poses[window_base + i] = opt_lsv.so3_poses[i].unit_quaternion();
                t_poses[window_base + i] = opt_lsv.t_poses[i];
                if(0)
                {
                    trans.block<3, 3>(0, 0) = opt_lsv.so3_poses[i].matrix();
                    trans.block<3, 1>(0, 3) = opt_lsv.t_poses[i];
                    pcl::PointCloud<PointType> pcloud;
                    pcl::transformPointCloud((*pl_ground_buf[window_base + i]+*pl_offground_buf[window_base + i]), pcloud, trans);
                    pl_send += pcloud;

                }
            }

            pub_func(pl_send, pub_full, ct);

            // Publish poses
            for(int i=window_base; i<window_base+window_size; i++)
            {
                parray.poses[i].orientation.w = q_poses[i].w();
                parray.poses[i].orientation.x = q_poses[i].x();
                parray.poses[i].orientation.y = q_poses[i].y();
                parray.poses[i].orientation.z = q_poses[i].z();
                parray.poses[i].position.x = t_poses[i].x();
                parray.poses[i].position.y = t_poses[i].y();
                parray.poses[i].position.z = t_poses[i].z();
            }


            // cout << "parray.poses.size() = " << parray.poses.size() << endl;
            // Marginalization and update voxel map
            for(auto iter=ground_map.begin(); iter!=ground_map.end(); ++iter)
            {
                if(iter->second->is2opt)
                {
                    iter->second->root_centors.clear();
                    iter->second->marginalize(0, margi_size, q_poses, t_poses, window_base, iter->second->root_centors);
                }
            }
            if(useEdge)
            {
                for(auto iter=corn_map.begin(); iter!=corn_map.end(); ++iter)
                {
                    if(iter->second->is2opt)
                    {
                        iter->second->root_centors.clear();
                        iter->second->marginalize(0, margi_size, q_poses, t_poses, window_base, iter->second->root_centors);
                    }
                }
            }

            window_base += margi_size;
            opt_lsv.free_voxel();
        }

    }
}


