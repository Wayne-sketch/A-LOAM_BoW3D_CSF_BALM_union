#ifndef BALMCLASS
#define BALMCLASS

#include <ros/ros.h>
#include <pcl/common/transforms.h>
#include <unordered_map>
// #include <opencv/cv.h>
#include <opencv2/imgproc/types_c.h>
#include "utility/myso3.hpp"
#include <thread>
#include <mutex>
#include <fstream>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>

#include "tools.hpp"
typedef std::vector<Eigen::Vector3d> PL_VEC;
typedef pcl::PointXYZINormal PointType;
#define MIN_PS 7
using namespace std;
int life_span = 1000;
// Key of hash table 哈希表的键
class VOXEL_LOC
{
public:
  int64_t x, y, z;

  VOXEL_LOC(int64_t vx=0, int64_t vy=0, int64_t vz=0): x(vx), y(vy), z(vz){}

  bool operator== (const VOXEL_LOC &other) const
  {
    return (x==other.x && y==other.y && z==other.z);
  }
};
/*
这是一个哈希函数的表达式，它将结构体 s 中的三个整数字段 x、y 和 z 进行哈希计算并进行位运算。
具体来说，它使用了三次哈希函数 hash<int64_t>() 对 x、y 和 z 进行哈希计算，然后通过位运算进行混合。
首先，将 s.x 的哈希值与 s.y 的哈希值进行异或运算，并将结果左移一位，然后再与 s.z 的哈希值进行异或运算。最后，将结果右移一位进行平衡。
这个哈希函数的目的是将三个整数字段组合生成一个唯一的哈希值，可用于哈希表、集合等数据结构的键。通过混合不同字段的哈希值，可以尽量减少冲突，提高哈希函数的性能和效果。
*/

// Hash value 哈希值
namespace std
{
  template<>
  struct hash<VOXEL_LOC>
  {
    size_t operator() (const VOXEL_LOC &s) const
    {
      using std::size_t; using std::hash;
      return ((hash<int64_t>()(s.x) ^ (hash<int64_t>()(s.y) << 1)) >> 1) ^ (hash<int64_t>()(s.z) << 1);
    }
  };
}

struct M_POINT
{
  float xyz[3];
  int count = 0;
};

// Similar with PCL voxelgrid filter
void down_sampling_voxel(pcl::PointCloud<PointType> &pl_feat, double voxel_size)
{
  if(voxel_size < 0.01)
  {
    return;
  }

  unordered_map<VOXEL_LOC, M_POINT> feat_map;
  uint plsize = pl_feat.size();

  for(uint i=0; i<plsize; i++)
  {
    PointType &p_c = pl_feat[i];
    float loc_xyz[3];
    for(int j=0; j<3; j++)
    {
        // 物理坐标转换为栅格坐标
        // 当前点被缩放一定倍数，栅格的尺寸越大，处于同一栅格中的点越多，采样后的点越稀疏
        loc_xyz[j] = p_c.data[j] / voxel_size;
      if(loc_xyz[j] < 0)
      {
        loc_xyz[j] -= 1.0;
      }
    }
    // 哈希表的键
    // 点云处于的栅格位置是点云取整后的坐标，紧密的点的三维栅格的位置相同
    VOXEL_LOC position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1], (int64_t)loc_xyz[2]);
    auto iter = feat_map.find(position);    // 当前栅格对应的指针
    if(iter != feat_map.end())// 找到了(紧密点)
    {
      iter->second.xyz[0] += p_c.x;// 将栅格中的所有点坐标相加
      iter->second.xyz[1] += p_c.y;
      iter->second.xyz[2] += p_c.z;
      iter->second.count++;
    }
    else// 没有找到(非紧密点)
    {
      M_POINT anp;
      anp.xyz[0] = p_c.x;
      anp.xyz[1] = p_c.y;
      anp.xyz[2] = p_c.z;
      anp.count = 1;
      feat_map[position] = anp;// 键值对，储存该栅格中的点云
    }
  }

  plsize = feat_map.size();// 栅格个数
  pl_feat.clear();
  pl_feat.resize(plsize);
  
  uint i = 0;
  for(auto iter=feat_map.begin(); iter!=feat_map.end(); ++iter)// 遍历栅格
  {// 栅格中的所有点坐标取均值
    pl_feat[i].x = iter->second.xyz[0]/iter->second.count;
    pl_feat[i].y = iter->second.xyz[1]/iter->second.count;
    pl_feat[i].z = iter->second.xyz[2]/iter->second.count;
    i++;
  }

}

void down_sampling_voxel(PL_VEC &pl_feat, double voxel_size)
{
  unordered_map<VOXEL_LOC, M_POINT> feat_map;
  uint plsize = pl_feat.size();

  for(uint i=0; i<plsize; i++)
  {
    Eigen::Vector3d &p_c = pl_feat[i];
    double loc_xyz[3];
    for(int j=0; j<3; j++)
    {
      loc_xyz[j] = p_c[j] / voxel_size;
      if(loc_xyz[j] < 0)
      {
        loc_xyz[j] -= 1.0;
      }
    }

    VOXEL_LOC position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1], (int64_t)loc_xyz[2]);
    auto iter = feat_map.find(position);
    if(iter != feat_map.end())
    {
      iter->second.xyz[0] += p_c[0];
      iter->second.xyz[1] += p_c[1];
      iter->second.xyz[2] += p_c[2];
      iter->second.count++;
    }
    else
    {
      M_POINT anp;
      anp.xyz[0] = p_c[0];
      anp.xyz[1] = p_c[1];
      anp.xyz[2] = p_c[2];
      anp.count = 1;
      feat_map[position] = anp;
    }

  }

  plsize = feat_map.size();
  pl_feat.resize(plsize);

  uint i = 0;
  for(auto iter=feat_map.begin(); iter!=feat_map.end(); ++iter)
  {
    pl_feat[i][0] = iter->second.xyz[0]/iter->second.count;
    pl_feat[i][1] = iter->second.xyz[1]/iter->second.count;
    pl_feat[i][2] = iter->second.xyz[2]/iter->second.count;
    i++;
  }


}

void plvec_trans_func(vector<Eigen::Vector3d> &orig, vector<Eigen::Vector3d> &tran, Eigen::Matrix3d R, Eigen::Vector3d t)
{
  uint orig_size = orig.size();
  tran.resize(orig_size);

  for(uint i=0; i<orig_size; i++)
  {
    tran[i] = R*orig[i] + t;
  }
}

template <typename T>
void pub_func(T &pl, ros::Publisher &pub, const ros::Time &current_time)
{
  pl.height = 1; pl.width = pl.size();
  sensor_msgs::PointCloud2 output;
  pcl::toROSMsg(pl, output);
  output.header.frame_id = "/camera_init";
  output.header.stamp = current_time;
  pub.publish(output);
}

// Convert PointCloud2 to PointType
void rosmsg2ptype(const sensor_msgs::PointCloud2 &pl_msg, pcl::PointCloud<PointType> &plt)
{
  pcl::PointCloud<pcl::PointXYZI> pl;
  pcl::fromROSMsg(pl_msg, pl);

  uint asize = pl.size();
  plt.resize(asize);

  for(uint i=0; i<asize; i++)
  {
    plt[i].x = pl[i].x;
    plt[i].y = pl[i].y;
    plt[i].z = pl[i].z;
    plt[i].intensity = pl[i].intensity;
  }
}

// P_fix in the paper 边缘化出去的帧
// Summation of P_fix
class SIG_VEC_CLASS
{
public:
  Eigen::Matrix3d sigma_vTv;
  Eigen::Vector3d sigma_vi;
  int sigma_size;

  SIG_VEC_CLASS()
  {
    sigma_vTv.setZero();
    sigma_vi.setZero();
    sigma_size = 0;
  }

  void tozero()
  {
    sigma_vTv.setZero();
    sigma_vi.setZero();
    sigma_size = 0;
  }

};

const double one_three = (1.0 / 3.0);
double feat_eigen_limit[2] = {3*3, 2*2};
double opt_feat_eigen_limit[2] = {4*4, 3*3};

// LM optimizer for map-refine
class LM_SLWD_VOXEL
{
public: 
  int slwd_size, filternum, thd_num, jac_leng;
  int iter_max = 20;

  double corn_less;

  vector<SO3> so3_poses, so3_poses_temp;
  vector<Eigen::Vector3d> t_poses, t_poses_temp;

  vector<int> lam_types; // 0 surf, 1 line

  vector<SIG_VEC_CLASS> sig_vecs;
  vector<vector<Eigen::Vector3d>*> plvec_voxels;
  vector<vector<int>*> slwd_nums;
  int map_refine_flag;
  mutex my_mutex;

  LM_SLWD_VOXEL(int ss, int fn, int thnum): slwd_size(ss), filternum(fn), thd_num(thnum)
  {
    so3_poses.resize(ss); t_poses.resize(ss);
    so3_poses_temp.resize(ss); t_poses_temp.resize(ss);
    jac_leng = 6*ss;
    corn_less = 0.1;
    map_refine_flag = 0;
  }

  // Used by "push_voxel"
  void downsample(vector<Eigen::Vector3d> &plvec_orig, int cur_frame,vector<Eigen::Vector3d> &plvec_voxel, vector<int> &slwd_num, int filternum2use)
  {
    uint plsize = plvec_orig.size();// 当前帧点云个数
    if(plsize <= (uint)filternum2use)// 小于等于1
    {
      for(uint i=0; i<plsize; i++)
      {
        plvec_voxel.push_back(plvec_orig[i]);
        slwd_num.push_back(cur_frame);
      }
      return;
    }

    Eigen::Vector3d center;
    // 分成filternum2use份，每一份有part个
    double part = 1.0 * plsize / filternum2use;

    for(int i=0; i<filternum2use; i++)
    {
      uint np = part*i;
      uint nn = part*(i+1);
      center.setZero();
      for(uint j=np; j<nn; j++)
      {
        center += plvec_orig[j];
      }
      // 当前帧的质心
      center = center / (nn-np);
      plvec_voxel.push_back(center);// 质心
      slwd_num.push_back(cur_frame);
    }
  }

  // Push voxel into optimizer
  void push_voxel(vector<vector<Eigen::Vector3d>*> &plvec_orig, SIG_VEC_CLASS &sig_vec, int lam_type)
  {
    int process_points_size = 0;
    for(int i=0; i<slwd_size; i++)
    {
      if(!plvec_orig[i]->empty())
      {
        process_points_size++;// 帧数
      }
    }
    
    // Only one scan
    if(process_points_size <= 1)// 小于一帧则退出
    {
      return;
    }

    int filternum2use = filternum;// 1
    if(filternum*process_points_size < MIN_PS)
    {
      filternum2use = MIN_PS / process_points_size + 1;
    }
    // 储存当前体素中所有帧点云对应的质心(lidar 坐标系下 即公式中的 pfi ，即图片中红绿蓝点的质心)
    vector<Eigen::Vector3d> *plvec_voxel = new vector<Eigen::Vector3d>();
    // Frame num in sliding window for each point in "plvec_voxel"
    vector<int> *slwd_num = new vector<int>();
    plvec_voxel->reserve(filternum2use*slwd_size);
    slwd_num->reserve(filternum2use*slwd_size);// 滑窗的帧数

    // retain one point for one scan (you can modify)
    for(int i=0; i<slwd_size; i++)// 遍历滑窗中世界坐标系体素对应的lidar坐标系体素
    {
      if(!plvec_orig[i]->empty())
      {
          // 同一个体素包含不同滑窗帧的点云
          // 输入：滑窗中当前帧的体素、滑窗中当前帧id、  输出：1.质心对应的的帧id容器 2.储存当前体素中当前帧的质心
        downsample(*plvec_orig[i], i, *plvec_voxel, *slwd_num, filternum2use);
      }
    }

    plvec_voxels.push_back(plvec_voxel); // Push a voxel into optimizer
    slwd_nums.push_back(slwd_num);
    lam_types.push_back(lam_type);
    sig_vecs.push_back(sig_vec); // history points out of sliding window
  }

  // Calculate Hessian, Jacobian, residual
  void acc_t_evaluate(vector<SO3> &so3_ps, vector<Eigen::Vector3d> &t_ps, int head, int end, Eigen::MatrixXd &Hess, Eigen::VectorXd &JacT, double &residual)
  {
    Hess.setZero(); JacT.setZero(); residual = 0;
    Eigen::MatrixXd _hess(Hess);
    Eigen::MatrixXd _jact(JacT);

    // In program, lambda_0 < lambda_1 < lambda_2
    // For plane, the residual is lambda_0
    // For line, the residual is lambda_0+lambda_1
    // We only calculate lambda_1 here
    for(int a=head; a<end; a++)// 遍历当前体素所有的质心
    {
      uint k = lam_types[a]; // 0 is surf, 1 is line
      SIG_VEC_CLASS &sig_vec = sig_vecs[a];// int &c = a 左值引用：将c作为a的别名，一个实体取了两个名字
      // pfi
      vector<Eigen::Vector3d> &plvec_voxel = *plvec_voxels[a];// 当前质心
      // Position in slidingwindow for each point in "plvec_voxel"
      vector<int> &slwd_num = *slwd_nums[a]; // 当前质心对应的帧id
      uint backnum = plvec_voxel.size();

      Eigen::Vector3d vec_tran;
      vector<Eigen::Vector3d> plvec_back(backnum);
      // derivative point to T (R, t)
      vector<Eigen::Matrix3d> point_xis(backnum);
      Eigen::Vector3d centor(Eigen::Vector3d::Zero());
      Eigen::Matrix3d covMat(Eigen::Matrix3d::Zero());

      for(uint i=0; i<backnum; i++)
      {
        vec_tran = so3_ps[slwd_num[i]].matrix() * plvec_voxel[i];
        // left multiplication instead of right muliplication in paper
        point_xis[i] = -SO3::hat(vec_tran);
        // Pi
        plvec_back[i] = vec_tran + t_ps[slwd_num[i]]; // after trans
        // q* = p_
        centor += plvec_back[i];
        // 协方差矩阵
        covMat += plvec_back[i] * plvec_back[i].transpose();
      }
      
      double N_points = backnum + sig_vec.sigma_size;
      centor += sig_vec.sigma_vi;
      covMat += sig_vec.sigma_vTv;
      // A
      covMat = covMat - centor*centor.transpose()/N_points;
      covMat = covMat / N_points;
      centor = centor / N_points;

      Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(covMat);
      // λ3(A)
      Eigen::Vector3d eigen_value = saes.eigenvalues();
      Eigen::Matrix3d U = saes.eigenvectors();
      // n* = u3
      Eigen::Vector3d u[3]; // eigenvectors
      for(int j=0; j<3; j++)
      {
        u[j] = U.block<3, 1>(0, j);
      }

      // Jacobian matrix
      Eigen::Matrix3d ukukT = u[k] * u[k].transpose();
      Eigen::Vector3d vec_Jt;
      for(uint i=0; i<backnum; i++)
      {
        // pi - q = pi - p_
        plvec_back[i] = plvec_back[i] - centor;
        vec_Jt = 2.0/N_points * ukukT * plvec_back[i];
        _jact.block<3, 1>(6*slwd_num[i]+3, 0) += vec_Jt;
        _jact.block<3, 1>(6*slwd_num[i], 0) -= point_xis[i] * vec_Jt;
      }

      // Hessian matrix
      Eigen::Matrix3d Hessian33;
      Eigen::Matrix3d C_k;
      vector<Eigen::Matrix3d> C_k_np(3);
      for(uint i=0; i<3; i++)
      {
        if(i == k)
        {
          C_k_np[i].setZero();
          continue;
        }
        Hessian33 = u[i]*u[k].transpose();
        // part of F matrix in paper
        C_k_np[i] = -1.0/N_points/(eigen_value[i]-eigen_value[k])*(Hessian33 + Hessian33.transpose());
      }

      Eigen::Matrix3d h33;
      uint rownum, colnum;
      for(uint j=0; j<backnum; j++)
      {
        for(int f=0; f<3; f++)
        {
          C_k.block<1, 3>(f, 0) = plvec_back[j].transpose() * C_k_np[f];
        }
        C_k = U * C_k;
        colnum = 6*slwd_num[j];
        // block matrix operation, half Hessian matrix
        for(uint i=j; i<backnum; i++)
        {
          Hessian33 = u[k]*(plvec_back[i]).transpose()*C_k + u[k].dot(plvec_back[i])*C_k;

          rownum = 6*slwd_num[i];
          if(i == j)
          {
            Hessian33 += (N_points-1)/N_points * ukukT;
          }
          else
          {
            Hessian33 -= 1.0/N_points * ukukT;
          }
          Hessian33 = 2.0/N_points * Hessian33; // Hessian matrix of lambda and point

          // Hessian matrix of lambda and pose
          if(rownum==colnum && i!=j)
          {
            _hess.block<3, 3>(rownum+3, colnum+3) += Hessian33 + Hessian33.transpose();

            h33 = -point_xis[i]*Hessian33;
            _hess.block<3, 3>(rownum, colnum+3) += h33;
            _hess.block<3, 3>(rownum+3, colnum) += h33.transpose();
            h33 = Hessian33*point_xis[j];
            _hess.block<3, 3>(rownum+3, colnum) += h33;
            _hess.block<3, 3>(rownum, colnum+3) += h33.transpose();
            h33 = -point_xis[i] * h33;
            _hess.block<3, 3>(rownum, colnum) += h33 + h33.transpose();
          }
          else
          {
            _hess.block<3, 3>(rownum+3, colnum+3) += Hessian33;
            h33 = Hessian33*point_xis[j];
            _hess.block<3, 3>(rownum+3, colnum) += h33;
            _hess.block<3, 3>(rownum, colnum+3) -= point_xis[i]*Hessian33;
            _hess.block<3, 3>(rownum, colnum) -= point_xis[i]*h33;
          } 
        }
      }

      if(k == 1)
      {
        // add weight for line feature
        residual += corn_less*eigen_value[k];
        Hess += corn_less*_hess; JacT += corn_less*_jact;
      }
      else
      {
        residual += eigen_value[k];
        Hess += _hess; JacT += _jact;
      }
      _hess.setZero(); _jact.setZero();
    }

    // Hessian is symmetric, copy to save time
    for(int j=0; j<jac_leng; j+=6)
    {
      for(int i=j+6; i<jac_leng; i+=6)
      {
        Hess.block<6, 6>(j, i) = Hess.block<6, 6>(i, j).transpose();
      }
    }
  }

  // Multithread for "acc_t_evaluate"
  void divide_thread(vector<SO3> &so3_ps, vector<Eigen::Vector3d> &t_ps,Eigen::MatrixXd &Hess, Eigen::VectorXd &JacT, double &residual)
  {
    Hess.setZero(); JacT.setZero(); residual = 0;

    vector<Eigen::MatrixXd> hessians(thd_num, Hess);
    vector<Eigen::VectorXd> jacobians(thd_num, JacT);
    vector<double> resis(thd_num, 0);

    uint gps_size = plvec_voxels.size();
    if(gps_size < (uint)thd_num)
    {
      acc_t_evaluate(so3_ps, t_ps, 0, gps_size, Hess, JacT, residual);
      Hess = hessians[0];
      JacT = jacobians[0];
      residual = resis[0];
      return;
    }
    
    vector<thread*> mthreads(thd_num);

    double part = 1.0*(gps_size)/thd_num;
    for(int i=0; i<thd_num; i++)
    {
      int np = part*i;
      int nn = part*(i+1);
      // 开辟子线程
      mthreads[i] = new thread(&LM_SLWD_VOXEL::acc_t_evaluate, this, ref(so3_ps), ref(t_ps), np, nn, ref(hessians[i]), ref(jacobians[i]), ref(resis[i]));
    }
    
    for(int i=0; i<thd_num; i++)
    {
      mthreads[i]->join();// 子线程汇合
      Hess += hessians[i];// 更新海森矩阵
      JacT += jacobians[i];// 更新雅克比矩阵
      residual += resis[i];// 更新残差
      delete mthreads[i];
    }

  }

  // Calculate residual
  void evaluate_only_residual(vector<SO3> &so3_ps, vector<Eigen::Vector3d> &t_ps, double &residual)
  {
    residual = 0;
    uint gps_size = plvec_voxels.size();
    Eigen::Vector3d vec_tran;

    for(uint a=0; a<gps_size; a++)
    {
      uint k = lam_types[a];
      SIG_VEC_CLASS &sig_vec = sig_vecs[a];// int &c = a 左值引用：将c作为a的别名，一个实体取了两个名字
      vector<Eigen::Vector3d> &plvec_voxel = *plvec_voxels[a];
      vector<int> &slwd_num = *slwd_nums[a];
      uint backnum = plvec_voxel.size();
      
      Eigen::Vector3d centor(Eigen::Vector3d::Zero());
      Eigen::Matrix3d covMat(Eigen::Matrix3d::Zero());

      for(uint i=0; i<backnum; i++)
      {
        vec_tran = so3_ps[slwd_num[i]].matrix()*plvec_voxel[i] + t_ps[slwd_num[i]];
        centor += vec_tran;
        covMat += vec_tran * vec_tran.transpose();
      }

      double N_points = backnum + sig_vec.sigma_size;
      centor += sig_vec.sigma_vi;
      covMat += sig_vec.sigma_vTv;

      covMat = covMat - centor*centor.transpose()/N_points;
      covMat = covMat / N_points;

      Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(covMat);
      Eigen::Vector3d eigen_value = saes.eigenvalues();
      
      if(k == 1)
      {
        residual += corn_less*eigen_value[k];
      }
      else
      {
        residual += eigen_value[k];
      }


    }
  }

  // LM process
  void damping_iter()
  {
    my_mutex.lock();
    map_refine_flag = 1;
    my_mutex.unlock();

    if(plvec_voxels.size()!=slwd_nums.size() || plvec_voxels.size()!=lam_types.size() || plvec_voxels.size()!=sig_vecs.size())
    {
      printf("size is not equal\n");
      exit(0);
    }

    double u = 0.01, v = 2;
    Eigen::MatrixXd D(jac_leng, jac_leng), Hess(jac_leng, jac_leng);
    Eigen::VectorXd JacT(jac_leng), dxi(jac_leng);

    Eigen::MatrixXd Hess2(jac_leng, jac_leng);
    Eigen::VectorXd JacT2(jac_leng);

    D.setIdentity();
    double residual1, residual2, q;
    bool is_calc_hess = true;
    // 初始化为0
    cv::Mat matA(jac_leng, jac_leng, CV_64F, cv::Scalar::all(0));
    cv::Mat matB(jac_leng, 1, CV_64F, cv::Scalar::all(0));
    cv::Mat matX(jac_leng, 1, CV_64F, cv::Scalar::all(0));

    for(int i=0; i<iter_max; i++)// 迭代
    {
      if(is_calc_hess)
      {
        // calculate Hessian, Jacobian, residual
        divide_thread(so3_poses, t_poses, Hess, JacT, residual1);
      }

      D = Hess.diagonal().asDiagonal();
      Hess2 = Hess + u*D;
      
      for(int j=0; j<jac_leng; j++)
      {
        matB.at<double>(j, 0) = -JacT(j, 0);
        for(int f=0; f<jac_leng; f++)
        {
          matA.at<double>(j, f) = Hess2(j, f);
        }
      }
      cv::solve(matA, matB, matX, cv::DECOMP_QR);
      for(int j=0; j<jac_leng; j++)
      {
        dxi(j, 0) = matX.at<double>(j, 0);
      }
  

      for(int j=0; j<slwd_size; j++)
      {
        // left multiplication 左乘
        so3_poses_temp[j] = SO3::exp(dxi.block<3, 1>(6*(j), 0)) * so3_poses[j];
        t_poses_temp[j] = t_poses[j] + dxi.block<3, 1>(6*(j)+3, 0);
      }

      // LM
      double q1 = 0.5*(dxi.transpose() * (u*D*dxi-JacT))[0];
      // double q1 = 0.5*dxi.dot(u*D*dxi-JacT);
      evaluate_only_residual(so3_poses_temp, t_poses_temp, residual2);// 仅计算残差

      q = (residual1-residual2);
      // printf("residual%d: %lf u: %lf v: %lf q: %lf %lf %lf\n", i, residual1, u, v, q/q1, q1, q);

      if(q > 0)
      {
        so3_poses = so3_poses_temp;
        t_poses = t_poses_temp;
        q = q / q1;
        v = 2;
        q = 1 - pow(2*q-1, 3);
        u *= (q<one_three ? one_three:q);
        is_calc_hess = true;// 继续计算海森矩阵
      }
      else
      {
        u = u * v;
        v = 2 * v;
        is_calc_hess = false;// 停止计算海森矩阵
      }
      
      if(fabs(residual1-residual2)<1e-9)// 收敛
      {
        break;
      }
    }
    
    my_mutex.lock();
    map_refine_flag = 2;
    my_mutex.unlock();
  }// LM end

  int read_refine_state()
  {
    int tem_flag;
    my_mutex.lock();
    tem_flag = map_refine_flag;
    my_mutex.unlock();
    return tem_flag;
  }

  void set_refine_state(int tem)
  {
    my_mutex.lock();
    map_refine_flag = tem;
    my_mutex.unlock();
  }

  void free_voxel()
  {
    uint a_size = plvec_voxels.size();
    for(uint i=0; i<a_size; i++)
    {
      delete (plvec_voxels[i]);
      delete (slwd_nums[i]);
    }

    plvec_voxels.clear();
    slwd_nums.clear();
    sig_vecs.clear();
    lam_types.clear();
  }

};// 优化

//  vector<Eigen::Matrix<>> 在Eigen管理内存和C++11中的方法是不一样的，所以需要单独强调元素的内存分配和管理
#define PLV(a) vector<Eigen::Matrix<double, a, 1>, Eigen::aligned_allocator<Eigen::Matrix<double, a, 1>>>
int win_size = 20;

class OCTO_TREE
{
public:
  static int voxel_windowsize;
  // 二维容器，存放的是滑动窗口中的所有帧点云(在该体素范围内)
  vector<PL_VEC*> plvec_orig;
  vector<PL_VEC*> plvec_tran;
  int octo_state; // 0 is end of tree, 1 is not
  int push_state;
  vector<PointCluster> sig_orig, sig_tran;

  PointCluster fix_point;
  Eigen::Vector3d center, direct, value_vector; // temporal
  int layer;
  int life;
  vector<int> each_num;


  PL_VEC sig_vec_points;
  SIG_VEC_CLASS sig_vec;
  int ftype;// 0:plane 1:edge
  int points_size, sw_points_size;
  double feat_eigen_ratio, feat_eigen_ratio_test;
  PointType ap_centor_direct;
  double voxel_center[3]; // x, y, z
  double quater_length;
  OCTO_TREE* leaves[8];// 指针数组，存放叶子节点指针
  bool is2opt;
  int capacity;
  pcl::PointCloud<PointType> root_centors;
  double decision, ref;
  vector<PLV(3)> vec_orig, vec_tran;
  OCTO_TREE()// 默认构造函数
  {
      octo_state = 0; push_state = 0;
      vec_orig.resize(win_size); vec_tran.resize(win_size);
      sig_orig.resize(win_size); sig_tran.resize(win_size);
      for(int i=0; i<8; i++) leaves[i] = nullptr;// 八叉树
      ref = 255.0*rand()/(RAND_MAX + 1.0f);
      layer = 0;

      is2opt = true;
      life = life_span;
      each_num.resize(win_size);
      for(int i=0; i<win_size; i++) each_num[i] = 0;
  }
  OCTO_TREE(int ft, int capa): ftype(ft), capacity(capa)// 构造函数
  {
    octo_state = 0;
    push_state = 0;
    ref = 255.0*rand()/(RAND_MAX + 1.0f);
    vec_orig.resize(win_size);
    vec_tran.resize(win_size);
    sig_orig.resize(win_size);
    sig_tran.resize(win_size);

    for(int i=0; i<8; i++)
    {
      leaves[i] = nullptr;
    }

    for(int i=0; i<capacity; i++)
    {
      plvec_orig.push_back(new PL_VEC());
      plvec_tran.push_back(new PL_VEC());
    }
    is2opt = true;
  }
    /*
    对于上述的线面特征，取整个体素内点云的质心和方差，，那么点面误差和点线误差就可以有如下的简化：
    点面误差：将特征平面上的参考点取成质心，法向量取成最大的特征值对应的特征向量时，最优化的位姿对应于最小化方差矩阵的最大特征值的问题
    点线误差：将线特征上的参考点取成质心，法向量取成最小的特征值对应的特征向量时，最优化的位姿对应于最小化方差矩阵的特征值的问题
    也就是因为这个特性，无论是线特征还是面特征，我们可以将优化问题简化为一个关于雷达位姿的“最小化体素中方差矩阵的特征值”问题
    其中是同一个体素中来自不同帧的点云坐标。区别于视觉SLAM中BA优化问题中每一轮的特征会随着位姿的变化而变化，在激光SLAM的一轮BA优化中，
    一旦位姿被确定，地图被构建，线面特征提取的结果也都是一样的，损失函数的只和待优化的位姿有关，特征点都已经被解析得计算出来，不再显式得参与计算。

    A = ∑(pi - p_)(pi - p_)┸ / N = ∑(pi - p_)(pi┸ - p_┸) / N = ∑ (pi*pi┸ - p_*pi┸ - pi*p_┸ + p_*p_┸) / N
    */
  // Used by "recut"
  void calc_eigen()// 对一个体素进行操作
  {
    Eigen::Matrix3d covMat(Eigen::Matrix3d::Zero());// 滑动窗口中(一个体素)的协方差矩阵
    Eigen::Vector3d center(0, 0, 0);// 中心

    uint asize;
    for(int i=0; i<OCTO_TREE::voxel_windowsize; i++)// 遍历滑动窗口中的当前一个体素
    {
      asize = plvec_tran[i]->size();// 当前体素中的某一帧的点云个数
      for(uint j=0; j<asize; j++)// 当前体素由很多帧组成，遍历其中一帧
      {
          // 计算点pi到滑窗中心的距离矩阵
          // TR(A) = || Pi - P_||²/N
        covMat += (*plvec_tran[i])[j] * (*plvec_tran[i])[j].transpose();// 所有帧的协方差，即一个体素的协方差
        center += (*plvec_tran[i])[j];
      }
    }

    covMat += sig_vec.sigma_vTv;
    center += sig_vec.sigma_vi;
    center /= points_size;
    covMat = covMat/points_size - center*center.transpose();
    
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(covMat);    // 模板类，专门计算特征值和特征向量
    // （三个特征值的大小接近为球或点，一个特征值远大于另外两个特征值为线状，两个特征值远大于一个特征值为平面）
    feat_eigen_ratio = saes.eigenvalues()[2] / saes.eigenvalues()[ftype];// 特征值，由小到大排序 // 0:plane 1:edge
    Eigen::Vector3d direct_vec = saes.eigenvectors().col(2*ftype);// 特征向量
    // λ3(A); if n* = u3, q* = p_
    // 更新 q，世界坐标系下
    ap_centor_direct.x = center.x();
    ap_centor_direct.y = center.y();
    ap_centor_direct.z = center.z();
    // n is the normal vector of the plane or the direction of the edge
    ap_centor_direct.normal_x = direct_vec.x();
    ap_centor_direct.normal_y = direct_vec.y();
    ap_centor_direct.normal_z = direct_vec.z();
  }


  // 对当前的一个体素进行操作
  // Cut root voxel into small pieces
  // frame_head: Position of newest scan in sliding window，位于滑动窗口的最新帧ID
  void recut(int layer, uint frame_head, pcl::PointCloud<PointType> &pl_feat_map)
  {
    // 0:初始状态、拟合成功状态(不需要八叉细分)
    // 1:未拟合成功状态(需要八叉树细分)
    if(octo_state == 0)// 若当前体素没有拟合
    {
      points_size = 0;
      // 遍历滑动窗口
      for(int i=0; i<OCTO_TREE::voxel_windowsize; i++)
      {
        points_size += plvec_orig[i]->size();// 滑动窗口(当前体素)所有点的数量
      }
      
      points_size += sig_vec.sigma_size;
      if(points_size < MIN_PS)
      {
        feat_eigen_ratio = -1;// 点太少未拟合成功
        return;
      }

      // 对一个体素进行的操作：
      // 首先，将三维空间划分为范围单位为1米的体素，然后计算体素内协方差矩阵的特征值，以确定体素内的点是否落在同一边缘/平面上，
      // 如果是，则保留当前体素，否则将体素划分为8个更小的体素，重复上述操作
      // 滑动窗口所有点的特征值
      calc_eigen(); // calculate eigenvalue ratio
      
      if(isnan(feat_eigen_ratio))// 特征值为空，未拟合成功则返回
      {
        feat_eigen_ratio = -1;
        return;
      }

      /*
       由粗到细的体素地图构建
       我们构建了一个由Hash组织的自适应体素图 以及用于每个哈希条目的八叉树。更具体地说， 我们首先将空间（在全局世界帧中）
       切割成体素， 每个具有粗略地图分辨率的大小。然后，对于 第一次激光雷达扫描，定义了世界帧 包含的点被分布到体素中。
       已填充体素 其中的点被索引到哈希表中。然后，对于每个填充体素，如果所有包含的点都位于一个平面上（ 点协方差矩阵的最小特征值小于指定的阈值），
       我们存储平面点并计算 （5）中的平面参数（n，q）及其不确定度 ∑n,q如（8）所示；否则，当前体素将进入八个八分之一和重复平面检查和体素切入每个层直到达到最大层数。
       请注意，体素具有不同的大小，每个体素包含一个根据所包含的激光雷达原始点拟合的平面特征。
       */
      // 说得更直白一点，LiDAR BA就是使得对应的边缘点落在同一边缘上，对应的平面点落在同一平面上。
      // 因此，作者对多帧之间的相应特征点（边缘/平面点）施加了以下约束：落在同一边缘或平面上。
      // 那么直观的结果是，在同一平面的点组成的平面越薄，优化效果越好。
      // 这个特征值越小，要优化的边越薄，线越细
      if(feat_eigen_ratio >= feat_eigen_limit[ftype])// 特征值大于阈值 plane:9、line:4
      {
        pl_feat_map.push_back(ap_centor_direct);// 储存滑窗的特征值和特征向量
        return;// 拟合成功，直接返回
      }

      // 到了第四层还没拟合成功就放弃
      if(layer == 4)// 1/2^3 = 0.125m
      {
        return;// 最大第四层，不考虑四层之后的
      }

      // 到了这还没返回，说明位于前三层，平面或直线拟合未成功，建立体素栅格未成功
      octo_state = 1;
      // All points in slidingwindow should be put into subvoxel
      frame_head = 0;
    }

    int leafnum;
    uint a_size;
    // 遍历滑动窗口
    for(int i=frame_head; i<OCTO_TREE::voxel_windowsize; i++)
    {
      a_size = plvec_tran[i]->size();// 滑窗中的当前帧点云数量
      for(uint j=0; j<a_size; j++)// 遍历当前帧点云
      {
        int xyz[3] = {0, 0, 0};
        for(uint k=0; k<3; k++)// 遍历xyz坐标
        {
          if((*plvec_tran[i])[j][k] > voxel_center[k])
          {
            xyz[k] = 1;
          }
        }
        // 当前点位于8个栅格中的哪一个栅格
        leafnum = 4*xyz[0] + 2*xyz[1] + xyz[2];// 4 * x + 2 * y + z
        if(leaves[leafnum] == nullptr)// 如果指向该栅格的指针为空
        {
          // 栅新建立体素格
          leaves[leafnum] = new OCTO_TREE(ftype, capacity);// octo_state = 0
          leaves[leafnum]->voxel_center[0] = voxel_center[0] + (2*xyz[0]-1)*quater_length;
          leaves[leafnum]->voxel_center[1] = voxel_center[1] + (2*xyz[1]-1)*quater_length;
          leaves[leafnum]->voxel_center[2] = voxel_center[2] + (2*xyz[2]-1)*quater_length;
          leaves[leafnum]->quater_length = quater_length / 2;// 划分为更小的体素
        }
        // 填充体素栅格
        // 点云填充栅格
        leaves[leafnum]->plvec_orig[i]->push_back((*plvec_orig[i])[j]);
        leaves[leafnum]->plvec_tran[i]->push_back((*plvec_tran[i])[j]);
      }
    }
    
    if(layer != 0)// 非根节点
    {
      for(int i=frame_head; i<OCTO_TREE::voxel_windowsize; i++)// 遍历滑动窗口
      {
        if(plvec_orig[i]->size() != 0)
        {// 释放内存，不然太占空间
          vector<Eigen::Vector3d>().swap(*plvec_orig[i]);
          vector<Eigen::Vector3d>().swap(*plvec_tran[i]);
        }
      }
    }
    // 注意：随着点云帧的不断加入，体素中的点会不断的增加
    layer++;
    for(uint i=0; i<8; i++)
    {
      if(leaves[i] != nullptr)// 如果该栅格中有点云才继续细分，否则停止递归
      {
        leaves[i]->recut(layer, frame_head, pl_feat_map);// 递归调用
      }
    }
  }

  // marginalize 5 scans in slidingwindow (assume margi_size is 5)
  void marginalize(int layer, int margi_size, vector<Eigen::Quaterniond> &q_poses, vector<Eigen::Vector3d> &t_poses, int window_base, pcl::PointCloud<PointType> &pl_feat_map)
  {
    if(octo_state!=1 || layer==0)
    {
      if(octo_state != 1)
      {
        for(int i=0; i<OCTO_TREE::voxel_windowsize; i++)
        {
          // Update points by new poses
          plvec_trans_func(*plvec_orig[i], *plvec_tran[i], q_poses[i+window_base].matrix(), t_poses[i+window_base]);
        }
      }

      // Push front 5 scans into P_fix
      uint a_size;
      if(feat_eigen_ratio > feat_eigen_limit[ftype])
      {
        for(int i=0; i<margi_size; i++)
        {
          sig_vec_points.insert(sig_vec_points.end(), plvec_tran[i]->begin(), plvec_tran[i]->end());
        }
        down_sampling_voxel(sig_vec_points, quater_length);
        
        a_size = sig_vec_points.size();
        sig_vec.tozero();
        sig_vec.sigma_size = a_size;
        for(uint i=0; i<a_size; i++)
        {
          sig_vec.sigma_vTv += sig_vec_points[i] * sig_vec_points[i].transpose();
          sig_vec.sigma_vi  += sig_vec_points[i];
        }
      }

      // Clear front 5 scans
      for(int i=0; i<margi_size; i++)
      {
        PL_VEC().swap(*plvec_orig[i]);
        PL_VEC().swap(*plvec_tran[i]);
        // plvec_orig[i].clear(); plvec_orig[i].shrink_to_fit();
      }

      if(layer == 0)
      {
        a_size = 0;
        for(int i=margi_size; i<OCTO_TREE::voxel_windowsize; i++)
        {
          a_size += plvec_orig[i]->size();
        }
        if(a_size == 0)
        {
          // Voxel has no points in slidingwindow
          is2opt = false;
        }
      }
      
      for(int i=margi_size; i<OCTO_TREE::voxel_windowsize; i++)
      {
        plvec_orig[i]->swap(*plvec_orig[i-margi_size]);
        plvec_tran[i]->swap(*plvec_tran[i-margi_size]);
      }
      
      if(octo_state != 1)
      {
        points_size = 0;
        for(int i=0; i<OCTO_TREE::voxel_windowsize-margi_size; i++)
        {
          points_size += plvec_orig[i]->size();
        }
        points_size += sig_vec.sigma_size;
        if(points_size < MIN_PS)
        {
          feat_eigen_ratio = -1;
          return;
        }

        calc_eigen();

        if(isnan(feat_eigen_ratio))
        {
          feat_eigen_ratio = -1;
          return;
        }
        if(feat_eigen_ratio >= feat_eigen_limit[ftype])
        {
          pl_feat_map.push_back(ap_centor_direct);
        }
      }
    }
    
    if(octo_state == 1)
    {
      layer++;
      for(int i=0; i<8; i++)
      {
        if(leaves[i] != nullptr)
        {
          leaves[i]->marginalize(layer, margi_size, q_poses, t_poses, window_base, pl_feat_map);
        }
      }
    }


  }
    bool judge_eigen(int win_count)
    {
        PointCluster covMat = fix_point;
        for(int i=0; i<win_count; i++)
            covMat += sig_tran[i];

        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(covMat.cov());
        value_vector = saes.eigenvalues();
        center = covMat.v / covMat.N;
        direct = saes.eigenvectors().col(0);

        decision = saes.eigenvalues()[0] / saes.eigenvalues()[1];
        float eigen_value_array[4] = {1.0/16, 1.0/16, 1.0/16, 1.0/16};
        return (decision < eigen_value_array[layer]);
    }
    void cut_func(int ci)
    {
        PLV(3) &pvec_orig = vec_orig[ci];
        PLV(3) &pvec_tran = vec_tran[ci];

        uint a_size = pvec_tran.size();
        for(uint j=0; j<a_size; j++)
        {
            int xyz[3] = {0, 0, 0};
            for(uint k=0; k<3; k++)
                if(pvec_tran[j][k] > voxel_center[k])
                    xyz[k] = 1;
            int leafnum = 4*xyz[0] + 2*xyz[1] + xyz[2];
            if(leaves[leafnum] == nullptr)
            {
                leaves[leafnum] = new OCTO_TREE();
                leaves[leafnum]->voxel_center[0] = voxel_center[0] + (2*xyz[0]-1)*quater_length;
                leaves[leafnum]->voxel_center[1] = voxel_center[1] + (2*xyz[1]-1)*quater_length;
                leaves[leafnum]->voxel_center[2] = voxel_center[2] + (2*xyz[2]-1)*quater_length;
                leaves[leafnum]->quater_length = quater_length / 2;
                leaves[leafnum]->layer = layer + 1;
            }

            leaves[leafnum]->vec_orig[ci].push_back(pvec_orig[j]);
            leaves[leafnum]->vec_tran[ci].push_back(pvec_tran[j]);

            if(leaves[leafnum]->octo_state != 1)
            {
                leaves[leafnum]->sig_orig[ci].push(pvec_orig[j]);
                leaves[leafnum]->sig_tran[ci].push(pvec_tran[j]);
            }
        }

        PLV(3)().swap(pvec_orig); PLV(3)().swap(pvec_tran);
    }

    void recut2(int win_count)
    {
        int layer_size[] = {30, 30, 30, 30};
        int layer_limit = 2;

        if(octo_state != 1)
        {
            int point_size = fix_point.N;
            for(int i=0; i<win_count; i++)
                point_size += sig_orig[i].N;

            push_state = 0;
            int min_ps = 15;
            if(point_size <= min_ps)
                return;

            if(judge_eigen(win_count))
            {
                if(octo_state==0 && point_size>layer_size[layer])
                    octo_state = 2;

                point_size -= fix_point.N;
                if(point_size > min_ps)
                    push_state = 1;
                return;
            }
            else if(layer == layer_limit)
            {
                octo_state = 2; return;
            }

            octo_state = 1;
            vector<PointCluster>().swap(sig_orig);
            vector<PointCluster>().swap(sig_tran);
            for(int i=0; i<win_count; i++)
                cut_func(i);
        }
        else
            cut_func(win_count-1);

        for(int i=0; i<8; i++)
            if(leaves[i] != nullptr)
                leaves[i]->recut2(win_count);
    }

    void tras_display(pcl::PointCloud<PointType> &pl_feat, int win_count)
    {
        if(octo_state != 1)
        {
            if(push_state != 1)
                return;

            PointType ap;
            ap.intensity = ref;

            int tsize = 0;
            for(int i=0; i<win_count; i++)
                tsize += vec_tran[i].size();
            if(tsize < 100) return;

            for(int i=0; i<win_count; i++)
                for(Eigen::Vector3d pvec : vec_tran[i])
                {
                    ap.x = pvec.x(); ap.y = pvec.y(); ap.z = pvec.z();
                    pl_feat.push_back(ap);
                }

        }
        else
        {
            for(int i=0; i<8; i++)
                if(leaves[i] != nullptr)
                    leaves[i]->tras_display(pl_feat, win_count);
        }
    }


    // Used by "traversal_opt"
  void traversal_opt_calc_eigen()
  {
    Eigen::Matrix3d covMat(Eigen::Matrix3d::Zero());
    Eigen::Vector3d center(0, 0, 0);
   
    uint asize;
    for(int i=0; i<OCTO_TREE::voxel_windowsize; i++)
    {
      asize = plvec_tran[i]->size();
      for(uint j=0; j<asize; j++)
      {
        covMat += (*plvec_tran[i])[j] * (*plvec_tran[i])[j].transpose();
        center += (*plvec_tran[i])[j];
      }
    }

    covMat -= center*center.transpose()/sw_points_size; 
    covMat /= sw_points_size;

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(covMat);
    feat_eigen_ratio_test = (saes.eigenvalues()[2] / saes.eigenvalues()[ftype]);
  }
    /*
     给定一个在具有姿态先验的世界坐标系中预测的激光雷达点Pw，
     我们首先通过它的哈希键找到它所在的根体素（具有粗糙的地图分辨率）。然后，对所有包含的子体素进行轮询，以此与点匹配。
     具体来说，让一个子体素包含一个具有法线ni和中心qi的平面，计算点到平面的距离
    */
  // Push voxel into "opt_lsv" (LM optimizer) 优化问题的递归创建
  void traversal_opt(LM_SLWD_VOXEL &opt_lsv)
  {
    if(octo_state != 1)// 如果当前体素为 非拟合体素
    {
      sw_points_size = 0;
      // 遍历滑动窗口
      for(int i=0; i<OCTO_TREE::voxel_windowsize; i++)
      {
        sw_points_size += plvec_orig[i]->size();// 滑动窗口当前体素中的总点云
      }

      if(sw_points_size < MIN_PS)// 点太少则返回
      {
        return;
      }

      traversal_opt_calc_eigen();// 计算滑窗中体素的特征值

      if(isnan(feat_eigen_ratio_test))// 特征值为空
      {
        return;
      }
      // 如果精拟合成功（设置了更加严苛的阈值）
      if(feat_eigen_ratio_test > opt_feat_eigen_limit[ftype])// 特征值大于阈值plane:16、line:9
      {
        // 输入滑窗中的拟合体素(lidar坐标系下)
        opt_lsv.push_voxel(plvec_orig, sig_vec, ftype);
      }
    }
    else
    {
      for(int i=0; i<8; i++)
      {
        if(leaves[i] != nullptr)// 栅格不为空
        {
          leaves[i]->traversal_opt(opt_lsv);// 遍历八叉树节点
        }
      }
    }
  }

};

int OCTO_TREE::voxel_windowsize = 0;

// Like "LM_SLWD_VOXEL"
// Scam2map optimizer
class VOXEL_DISTANCE
{
public:
  SO3 so3_pose, so3_temp;
  Eigen::Vector3d t_pose, t_temp;
  PL_VEC surf_centor, surf_direct;
  PL_VEC corn_centor, corn_direct;
  PL_VEC surf_gather, corn_gather;
  vector<double> surf_coeffs, corn_coeffs;

  void push_surf(Eigen::Vector3d &orip, Eigen::Vector3d &centor, Eigen::Vector3d &direct, double coeff)
  {
    direct.normalize();
    surf_direct.push_back(direct); surf_centor.push_back(centor);
    surf_gather.push_back(orip); surf_coeffs.push_back(coeff);
  }

  void push_line(Eigen::Vector3d &orip, Eigen::Vector3d &centor, Eigen::Vector3d &direct, double coeff)
  {
    direct.normalize();
    corn_direct.push_back(direct); corn_centor.push_back(centor);
    corn_gather.push_back(orip); corn_coeffs.push_back(coeff);
  }

  void evaluate_para(SO3 &so3_p, Eigen::Vector3d &t_p, Eigen::Matrix<double, 6, 6> &Hess, Eigen::Matrix<double, 6, 1> &g, double &residual)
  {
    Hess.setZero(); g.setZero(); residual = 0;
    uint a_size = surf_gather.size();
    for(uint i=0; i<a_size; i++)
    {
      Eigen::Matrix3d _jac = surf_direct[i] * surf_direct[i].transpose();
      Eigen::Vector3d vec_tran = so3_p.matrix() * surf_gather[i];
      Eigen::Matrix3d point_xi = -SO3::hat(vec_tran);
      vec_tran += t_p;

      Eigen::Vector3d v_ac = vec_tran - surf_centor[i];
      Eigen::Vector3d d_vec = _jac * v_ac;
      Eigen::Matrix<double, 3, 6> jacob;
      jacob.block<3, 3>(0, 0) = _jac * point_xi;
      jacob.block<3, 3>(0, 3) = _jac;

      residual += surf_coeffs[i] * d_vec.dot(d_vec);
      Hess += surf_coeffs[i] * jacob.transpose() * jacob;
      g += surf_coeffs[i] * jacob.transpose() * d_vec;
    }

    a_size = corn_gather.size();
    for(uint i=0; i<a_size; i++)
    {
      Eigen::Matrix3d _jac = Eigen::Matrix3d::Identity() - corn_direct[i] * corn_direct[i].transpose();
      Eigen::Vector3d vec_tran = so3_p.matrix() * corn_gather[i];
      Eigen::Matrix3d point_xi = -SO3::hat(vec_tran);
      vec_tran += t_p;

      Eigen::Vector3d v_ac = vec_tran - corn_centor[i];
      Eigen::Vector3d d_vec = _jac * v_ac;
      Eigen::Matrix<double, 3, 6> jacob;
      jacob.block<3, 3>(0, 0) = _jac * point_xi;
      jacob.block<3, 3>(0, 3) = _jac;

      residual += corn_coeffs[i] * d_vec.dot(d_vec);
      Hess += corn_coeffs[i] * jacob.transpose() * jacob;
      g += corn_coeffs[i] * jacob.transpose() * d_vec;
    }
  }

  void evaluate_only_residual(SO3 &so3_p, Eigen::Vector3d &t_p, double &residual)
  {
    residual = 0;
    uint a_size = surf_gather.size();
    for(uint i=0; i<a_size; i++)
    {
      Eigen::Matrix3d _jac = surf_direct[i] * surf_direct[i].transpose();
      Eigen::Vector3d vec_tran = so3_p.matrix() * surf_gather[i];
      vec_tran += t_p;

      Eigen::Vector3d v_ac = vec_tran - surf_centor[i];
      Eigen::Vector3d d_vec = _jac * v_ac;

      residual += surf_coeffs[i] * d_vec.dot(d_vec);
    }

    a_size = corn_gather.size();
    for(uint i=0; i<a_size; i++)
    {
      Eigen::Matrix3d _jac = Eigen::Matrix3d::Identity() - corn_direct[i] * corn_direct[i].transpose();
      Eigen::Vector3d vec_tran = so3_p.matrix() * corn_gather[i];
      vec_tran += t_p;

      Eigen::Vector3d v_ac = vec_tran - corn_centor[i];
      Eigen::Vector3d d_vec = _jac * v_ac;

      residual += corn_coeffs[i] * d_vec.dot(d_vec);
    }

  }

  void damping_iter()
  {
    double u = 0.01, v = 2;
    Eigen::Matrix<double, 6, 6> D; D.setIdentity();
    Eigen::Matrix<double, 6, 6> Hess, Hess2;
    Eigen::Matrix<double, 6, 1> g;
    Eigen::Matrix<double, 6, 1> dxi;
    double residual1, residual2;

    cv::Mat matA(6, 6, CV_64F, cv::Scalar::all(0));
    cv::Mat matB(6, 1, CV_64F, cv::Scalar::all(0));
    cv::Mat matX(6, 1, CV_64F, cv::Scalar::all(0));

    for(int i=0; i<20; i++)
    {
      evaluate_para(so3_pose, t_pose, Hess, g, residual1);
      D = Hess.diagonal().asDiagonal();
      
      // dxi = (Hess + u*D).bdcSvd(Eigen::ComputeFullU | Eigen::ComputeFullV).solve(-g);

      Hess2 = Hess + u*D;
      for(int j=0; j<6; j++)
      {
        matB.at<double>(j, 0) = -g(j, 0);
        for(int f=0; f<6; f++)
        {
          matA.at<double>(j, f) = Hess2(j, f);
        }
      }
      cv::solve(matA, matB, matX, cv::DECOMP_QR);
      for(int j=0; j<6; j++)
      {
        dxi(j, 0) = matX.at<double>(j, 0);
      }

      so3_temp = SO3::exp(dxi.block<3, 1>(0, 0)) * so3_pose;
      t_temp = t_pose + dxi.block<3, 1>(3, 0);
      evaluate_only_residual(so3_temp, t_temp, residual2);
      double q1 = dxi.dot(u*D*dxi-g);
      double q = residual1 - residual2;
      // printf("residual: %lf u: %lf v: %lf q: %lf %lf %lf\n", residual1, u, v, q/q1, q1, q);
      if(q > 0)
      {
        so3_pose = so3_temp;
        t_pose = t_temp;
        q = q / q1;
        v = 2;
        q = 1 - pow(2*q-1, 3);
        u *= (q<one_three ? one_three:q);
      }
      else
      {
        u = u * v;
        v = 2 * v;
      }

      if(fabs(residual1-residual2)<1e-9)
      {
        break;
      }

    }

  }

};


#endif

