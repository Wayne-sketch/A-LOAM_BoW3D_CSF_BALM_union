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

using namespace std;
//在世界坐标系下构建体素的体素大小，第一个对应平面特征构建时体素大小，第二个对应线特征构建时体素大小
double voxel_size[2] = {1, 1};

mutex mBuf;
queue<sensor_msgs::PointCloud2ConstPtr> surf_buf, corn_buf, full_buf;
queue<nav_msgs::Odometry::ConstPtr> odom_buf;

void surf_handler(const sensor_msgs::PointCloud2ConstPtr &msg)
{
  mBuf.lock();
  surf_buf.push(msg);
  mBuf.unlock();
}

void corn_handler(const sensor_msgs::PointCloud2ConstPtr &msg)
{
  mBuf.lock();
  corn_buf.push(msg);
  mBuf.unlock();
}

void full_handler(const sensor_msgs::PointCloud2ConstPtr &msg)
{
  mBuf.lock();
  full_buf.push(msg);
  mBuf.unlock();
}

void odom_handler(const nav_msgs::Odometry::ConstPtr &msg)
{
  mBuf.lock();
  odom_buf.push(msg);
  mBuf.unlock();
}

/**
 * @brief 构建哈希表，键是世界系下的体素索引位置，值是八叉树节点指针
 * 
 * @param feat_map 平面特征或线特征哈希表 键是体素位置索引，值是八叉树节点指针
 * @param pl_feat 预处理后的点云，存储着每个体素内点云中心点的坐标
 * @param R_p 当前关键帧在世界坐标系/第一帧/第一个关键帧下的旋转矩阵
 * @param t_p 当前关键帧在世界坐标系/第一帧/第一个关键帧下的位移
 * @param feattype 拟合特征种类，0代表面，1代表线
 * @param fnum 滑窗中最新关键帧的索引 例如处理第一个关键帧时这个值为0，处理第二个关键帧时这个值为1
 * @param capacity 滑窗大小
 */
void cut_voxel(unordered_map<VOXEL_LOC, OCTO_TREE*> &feat_map, pcl::PointCloud<PointType>::Ptr pl_feat, Eigen::Matrix3d R_p, Eigen::Vector3d t_p, int feattype, int fnum, int capacity)
{
  //统计体素数量
  uint plsize = pl_feat->size();
  //遍历每个体素
  for(uint i=0; i<plsize; i++)
  {
    //取出中心点坐标 在当前帧坐标系下表示
    PointType &p_c = pl_feat->points[i];
    Eigen::Vector3d pvec_orig(p_c.x, p_c.y, p_c.z);
    //计算中心点在世界坐标系下的坐标
    Eigen::Vector3d pvec_tran = R_p*pvec_orig + t_p;

    float loc_xyz[3];
    for(int j=0; j<3; j++)
    {
      //计算中心点在世界坐标系下的体素索引
      loc_xyz[j] = pvec_tran[j] / voxel_size[feattype];
      if(loc_xyz[j] < 0)
      {
        loc_xyz[j] -= 1.0;
      }
    }

    //用计算出的体素索引构建哈希表的键
    VOXEL_LOC position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1], (int64_t)loc_xyz[2]);
    auto iter = feat_map.find(position);
    //如果该位置索引的体素已经构建，
    if(iter != feat_map.end())
    {
      //存储中心点在当前帧坐标系下的坐标 fnum是当前关键帧的索引
      iter->second->plvec_orig[fnum]->push_back(pvec_orig);
      //存储中心点在世界坐标系下的坐标 fnum是当前关键帧的索引
      iter->second->plvec_tran[fnum]->push_back(pvec_tran);
      //此处为体素中添加了新的点，所以设置为待拟合
      iter->second->is2opt = true;
    }
    //如果该位置索引体素还没有构建
    else
    {
      //新建八叉树节点
      OCTO_TREE *ot = new OCTO_TREE(feattype, capacity);
      //存储中心点在当前帧坐标系下的坐标 fnum是当前关键帧的索引
      ot->plvec_orig[fnum]->push_back(pvec_orig);
      //存储中心点在世界坐标系下的坐标 fnum是当前关键帧的索引
      ot->plvec_tran[fnum]->push_back(pvec_tran);

      //第一次新建八叉树节点 记录该体素中心店在世界坐标系下的位置 0.5是为了取到体素的中心点
      ot->voxel_center[0] = (0.5+position.x) * voxel_size[feattype];
      ot->voxel_center[1] = (0.5+position.y) * voxel_size[feattype];
      ot->voxel_center[2] = (0.5+position.z) * voxel_size[feattype];
      //记录体素大小的1/4长度
      ot->quater_length = voxel_size[feattype] / 4.0;
      //放入哈希表
      feat_map[position] = ot;
    }
  }
}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "balm_only_back");
  ros::NodeHandle n;

  ros::Subscriber sub_corn = n.subscribe<sensor_msgs::PointCloud2>("/corn_last", 100, corn_handler);
  ros::Subscriber sub_surf = n.subscribe<sensor_msgs::PointCloud2>("/surf_last", 100, surf_handler);
  ros::Subscriber sub_full = n.subscribe<sensor_msgs::PointCloud2>("/full_last", 100, full_handler);
  ros::Subscriber sub_odom = n.subscribe<nav_msgs::Odometry>("/aft_mapped_to_init", 100, odom_handler);

  ros::Publisher pub_corn = n.advertise<sensor_msgs::PointCloud2>("/map_corn", 10);
  ros::Publisher pub_surf = n.advertise<sensor_msgs::PointCloud2>("/map_surf", 10);
  ros::Publisher pub_full = n.advertise<sensor_msgs::PointCloud2>("/map_full", 10);
  ros::Publisher pub_test = n.advertise<sensor_msgs::PointCloud2>("/map_test", 10);
  ros::Publisher pub_odom = n.advertise<nav_msgs::Odometry>("/odom_rviz_last", 10);
  ros::Publisher pub_pose = n.advertise<geometry_msgs::PoseArray>("/poseArrayTopic", 10);

  //用边缘点云或平面点云计算体素索引时的体素大小
  double surf_filter_length = 0.4;
  double corn_filter_length = 0.2;

  //滑窗大小 LM_SLWD_VOXEL构造函数需要的参数
  int window_size = 20;
  int margi_size = 5;
  //? LM_SLWD_VOXEL构造函数需要的参数
  int filter_num = 1;
  //? LM_SLWD_VOXEL构造函数需要的参数
  int thread_num = 4;
  //每隔几帧处理一次关键帧之间的关系
  int skip_num = 0;
  int pub_skip = 1;

  //? 解释每个参数的意义
  //用边缘点云或平面点云计算体素索引时的体素大小
  n.param<double>("surf_filter_length", surf_filter_length, 0.4);
  n.param<double>("corn_filter_length", corn_filter_length, 0.2);
  n.param<double>("root_surf_voxel_size", voxel_size[0], 1);
  n.param<double>("root_corn_voxel_size", voxel_size[1], 1);
  //每隔几帧处理一次关键帧之间的关系
  n.param<int>("skip_num", skip_num, 0);
  n.param<double>("surf_feat_eigen_limit", feat_eigen_limit[0], 9);
  n.param<double>("corn_feat_eigen_limit", feat_eigen_limit[0], 4);
  n.param<double>("surf_opt_feat_eigen_limit", feat_eigen_limit[0], 16);
  n.param<double>("corn_opt_feat_eigen_limit", feat_eigen_limit[0], 9);
  n.param<int>("pub_skip", pub_skip, 1);

  //第一次进循环要处理
  int jump_flag = skip_num;
  printf("%d\n", skip_num);

  //?滑窗优化器
  LM_SLWD_VOXEL opt_lsv(window_size, filter_num, thread_num);

  //存边缘点点云
  pcl::PointCloud<PointType>::Ptr pl_corn(new pcl::PointCloud<PointType>);
  //存平面点点云
  pcl::PointCloud<PointType>::Ptr pl_surf(new pcl::PointCloud<PointType>);
  //存储关键帧的完整点云
  vector<pcl::PointCloud<PointType>::Ptr> pl_full_buf;

  //存储世界坐标系到每一关键帧的旋转和平移
  vector<Eigen::Quaterniond> q_poses;
  vector<Eigen::Vector3d> t_poses;
  Eigen::Quaterniond q_odom, q_gather_pose(1, 0, 0, 0) ,q_last(1, 0, 0, 0); 
  Eigen::Vector3d t_odom, t_gather_pose(0, 0, 0), t_last(0, 0, 0);
  int plcount = 0, window_base = 0;
  
  //平面特征哈希表 线特征哈希表
  unordered_map<VOXEL_LOC, OCTO_TREE*> surf_map, corn_map;
  //位姿变换矩阵
  Eigen::Matrix4d trans(Eigen::Matrix4d::Identity());
  geometry_msgs::PoseArray parray;
  parray.header.frame_id = "camera_init";

  while(n.ok())
  {
    //消息都存入buf里
    ros::spinOnce();
    if(corn_buf.empty() || surf_buf.empty() || full_buf.empty() || odom_buf.empty())
    {
      continue;
    }

    mBuf.lock();
    uint64_t time_corn = corn_buf.front()->header.stamp.toNSec();
    uint64_t time_surf = surf_buf.front()->header.stamp.toNSec();
    uint64_t time_full = full_buf.front()->header.stamp.toNSec();
    uint64_t time_odom = odom_buf.front()->header.stamp.toNSec();

    //?感觉这里对齐有问题
    if(time_odom != time_corn)
    {
      time_odom < time_corn ? odom_buf.pop() : corn_buf.pop();
      mBuf.unlock();
      continue;
    }
    if(time_odom != time_surf)
    {
      time_odom < time_surf ? odom_buf.pop() : surf_buf.pop();
      mBuf.unlock();
      continue;
    }
    if(time_odom != time_full)
    {
      time_odom < time_full ? odom_buf.pop() : full_buf.pop();
      mBuf.unlock();
      continue;
    }

    //对齐后取最新帧时间
    ros::Time ct(full_buf.front()->header.stamp);
    pcl::PointCloud<PointType>::Ptr pl_full(new pcl::PointCloud<PointType>);
    //三种点云取出来
    rosmsg2ptype(*surf_buf.front(), *pl_surf);
    rosmsg2ptype(*corn_buf.front(), *pl_corn);
    rosmsg2ptype(*full_buf.front(), *pl_full);
    //buf中扔掉
    corn_buf.pop(); surf_buf.pop(); full_buf.pop();
    //位姿取出来
    q_odom.w() = odom_buf.front()->pose.pose.orientation.w;
    q_odom.x() = odom_buf.front()->pose.pose.orientation.x;
    q_odom.y() = odom_buf.front()->pose.pose.orientation.y;
    q_odom.z() = odom_buf.front()->pose.pose.orientation.z;
    t_odom.x() = odom_buf.front()->pose.pose.position.x;
    t_odom.y() = odom_buf.front()->pose.pose.position.y;
    t_odom.z() = odom_buf.front()->pose.pose.position.z;
    //buf里扔掉
    odom_buf.pop();
    mBuf.unlock();
    //上一帧坐标系下的 上一帧到当前帧的位移
    Eigen::Vector3d delta_t(q_last.matrix().transpose()*(t_odom-t_last));
    //当前帧到上一帧的旋转姿态
    Eigen::Quaterniond delta_q(q_last.matrix().transpose() * q_odom.matrix());
    //赋值上一帧的旋转和位移
    q_last = q_odom; t_last = t_odom;

    //那就把这些初始化后的帧称为关键帧
    //上一关键帧坐标系下的 上一关键帧到当前帧的位移 q_gather_pose和t_gather_pose一直在被初始化
    t_gather_pose = t_gather_pose + q_gather_pose * delta_t;
    //当前帧到上一关键帧的旋转姿态
    q_gather_pose = q_gather_pose * delta_q;

    //skip_num 每几帧向下处理一次，不像下处理的时候，q_gather_pose和t_gather_pose没有被初始化
    if(jump_flag < skip_num)
    {
      jump_flag++;
      continue;
    }
    jump_flag = 0;

    //第一个关键帧的处理
    if(plcount == 0)
    {
      //存储第二个关键帧在第一个关键帧坐标系下的旋转
      q_poses.push_back(q_gather_pose);
      //存储第一个关键帧到第二个关键帧的位移 在第一个关键帧坐标系下表示
      t_poses.push_back(t_gather_pose);
    }
    else
    {
      //存储第一关键帧到当前关键帧的位移 在第一帧坐标系下表示
      //t_firstket_currkey = t_firstkey_lastkey + R_firstkey_lastkey * t_lastkey_currkey
      t_poses.push_back(t_poses[plcount-1] + q_poses[plcount-1] * t_gather_pose);
      //当前关键帧在第一个关键帧下的旋转
      //R_firstkey_currkey = R_firstkey_lastkey * R_lastkey_currkey
      q_poses.push_back(q_poses[plcount-1]*q_gather_pose);
    }

    //待发布的每个时刻的位姿
    //时间戳
    parray.header.stamp = ct;
    geometry_msgs::Pose apose;
    //当前关键帧的位姿 在世界坐标系下 第一帧坐标系就是世界坐标系
    apose.orientation.w = q_poses[plcount].w();
    apose.orientation.x = q_poses[plcount].x();
    apose.orientation.y = q_poses[plcount].y();
    apose.orientation.z = q_poses[plcount].z();
    apose.position.x = t_poses[plcount].x();
    apose.position.y = t_poses[plcount].y();
    apose.position.z = t_poses[plcount].z();
    parray.poses.push_back(apose);
    //发布
    pub_pose.publish(parray);

    //存储关键帧的完整点云
    pl_full_buf.push_back(pl_full);
    //统计关键帧的个数
    plcount++;
    //当前关键帧的数量
    OCTO_TREE::voxel_windowsize = plcount - window_base;
    //这两个变量初始化了，为了记录关键帧之间的位姿变换
    q_gather_pose.setIdentity(); t_gather_pose.setZero();

    //对关键帧对应的边缘点和平面点云处理
    down_sampling_voxel(*pl_corn, corn_filter_length);
    down_sampling_voxel(*pl_surf, surf_filter_length);

    //滑窗中最新关键帧的索引 例如处理第一个关键帧时这个值为0，处理第二个关键帧时这个值为1
    int frame_head = plcount-1-window_base;
    //面点拟合面特征在世界系构建哈希表
    cut_voxel(surf_map, pl_surf, q_poses[plcount-1].matrix(), t_poses[plcount-1], 0, frame_head, window_size);
    //边缘点拟合线特征在世界系构建哈希表
    cut_voxel(corn_map, pl_corn, q_poses[plcount-1].matrix(), t_poses[plcount-1], 1, frame_head, window_size);

    //遍历面特征哈希表 也就是遍历每个root voxel
    for(auto iter=surf_map.begin(); iter!=surf_map.end(); ++iter)
    {
      //只有添加了新点或者第一次构建的体素才需要尝试拟合
      //如果八叉树节点标记为待拟合 
      if(iter->second->is2opt)
      {
        //清空之前拟合成功的特征参数
        iter->second->root_centors.clear();
        //尝试在该八叉树节点拟合特征，如果不成功，递归调用recut拟合特征
        iter->second->recut(0, frame_head, iter->second->root_centors);
      }
    }

    //遍历线特征哈希表
    for(auto iter=corn_map.begin(); iter!=corn_map.end(); ++iter)
    {
      //只有添加了新点或者第一次构建的体素才需要尝试拟合
      if(iter->second->is2opt)
      {
        iter->second->root_centors.clear();
        iter->second->recut(0, frame_head, iter->second->root_centors);
      }
    }

    //如果当前滑窗中关键帧数量填满了滑动窗口
    if(plcount >= window_base+window_size)
    {
      //遍历滑窗中每个关键帧
      for(int i=0; i<window_size; i++)
      {
        //todo 每个关键帧在世界系（第一个关键帧下）的位姿 初始化位姿
        opt_lsv.so3_poses[i].setQuaternion(q_poses[window_base + i]);
        opt_lsv.t_poses[i] = t_poses[window_base + i];
      }

      //?什么情况下window_base不等于0
      if(window_base != 0)
      {
        //遍历哈希表中每个八叉树
        for(auto iter=surf_map.begin(); iter!=surf_map.end(); ++iter)
        {
          //如果该八叉树节点待拟合
          //? 追踪is2opt
          if(iter->second->is2opt)
          {
            iter->second->traversal_opt(opt_lsv);
          }
        }

        for(auto iter=corn_map.begin(); iter!=corn_map.end(); ++iter)
        {
          if(iter->second->is2opt)
          {
            iter->second->traversal_opt(opt_lsv);
          }
        }

        //进行滑窗优化
        opt_lsv.damping_iter();
      }

      pcl::PointCloud<PointType> pl_send;
      
      //? 发布边缘化出去的帧？
      for(int i=0; i<margi_size; i+=pub_skip)
      {
        //todo 把优化后的位姿取出来
        trans.block<3, 3>(0, 0) = opt_lsv.so3_poses[i].matrix();
        trans.block<3, 1>(0, 3) = opt_lsv.t_poses[i];

        pcl::PointCloud<PointType> pcloud;
        //把关键帧完整点云按照优化后的位姿转换到世界坐标系下
        pcl::transformPointCloud(*pl_full_buf[window_base + i], pcloud, trans);
        //累积到一起
        pl_send += pcloud;
      }
      //发布出去
      pub_func(pl_send, pub_full, ct);

      //清空被边缘化出去的帧
      for(int i=0; i<margi_size; i++)
      {
        pl_full_buf[window_base + i] = nullptr;
      }

      //更新滑窗中关键帧的位姿
      for(int i=0; i<window_size; i++)
      {
        q_poses[window_base + i] = opt_lsv.so3_poses[i].unit_quaternion();
        t_poses[window_base + i] = opt_lsv.t_poses[i];
      }

      //更新位姿后发布出去
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
      pub_pose.publish(parray);

      for(auto iter=surf_map.begin(); iter!=surf_map.end(); ++iter)
      {
        if(iter->second->is2opt)
        {
          //清空拟合成功的特征参数
          iter->second->root_centors.clear();
          //进行边缘化
          iter->second->marginalize(0, margi_size, q_poses, t_poses, window_base, iter->second->root_centors);
        }
      }

      for(auto iter=corn_map.begin(); iter!=corn_map.end(); ++iter)
      {
        if(iter->second->is2opt)
        {
          //清空拟合成功的特征参数
          iter->second->root_centors.clear();
          iter->second->marginalize(0, margi_size, q_poses, t_poses, window_base, iter->second->root_centors);
        }
      }

      //这里修改了window_base的值 边缘化掉几帧 window_base后移几帧
      window_base += margi_size;
      //清空一次
      opt_lsv.free_voxel();
    }
  }
}