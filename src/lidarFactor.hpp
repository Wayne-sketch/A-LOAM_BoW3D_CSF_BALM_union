// Author:   Tong Qin               qintonguav@gmail.com
// 	         Shaozu Cao 		    saozu.cao@connect.ust.hk

#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <eigen3/Eigen/Dense>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl_conversions/pcl_conversions.h>

struct LidarEdgeFactor
{
	// 构造函数
	LidarEdgeFactor(Eigen::Vector3d curr_point_, Eigen::Vector3d last_point_a_,
					Eigen::Vector3d last_point_b_, double s_)
		: curr_point(curr_point_), last_point_a(last_point_a_), last_point_b(last_point_b_), s(s_) {}
	// 仿函数
	template <typename T>
	// 参数块1：旋转矩阵，参数块2：平移矩阵，残差块
	bool operator()(const T *q, const T *t, T *residual) const
	{
		// 将double数组转成eigen的数据结构，注意这里必须都写成模板
		Eigen::Matrix<T, 3, 1> cp{T(curr_point.x()), T(curr_point.y()), T(curr_point.z())};
		Eigen::Matrix<T, 3, 1> lpa{T(last_point_a.x()), T(last_point_a.y()), T(last_point_a.z())};
		Eigen::Matrix<T, 3, 1> lpb{T(last_point_b.x()), T(last_point_b.y()), T(last_point_b.z())};

		//Eigen::Quaternion<T> q_last_curr{q[3], T(s) * q[0], T(s) * q[1], T(s) * q[2]};
		// 待优化变量：R_curr2last
		Eigen::Quaternion<T> q_last_curr{q[3], q[0], q[1], q[2]};
		Eigen::Quaternion<T> q_identity{T(1), T(0), T(0), T(0)};

		// 计算的是上一帧到当前帧的位姿变换，因此根据匀速运动模型(一帧中(100s)的运动都是匀速运动)，计算该点对应的位姿
		// R_curr2last  = R_end2stat ，end - start = 100ms
		// R_end2start * Δt/T  = R_curr2last
		q_last_curr = q_identity.slerp(T(s), q_last_curr);
		// t = t * Δt/T
		Eigen::Matrix<T, 3, 1> t_last_curr{T(s) * t[0], T(s) * t[1], T(s) * t[2]};

		Eigen::Matrix<T, 3, 1> lp;
		// 把当前点，根据当前计算的帧间位姿变换到上一帧的雷达坐标系，在同一个坐标系下操作
		// R_curr2last * P_curr  + t = P_last
		lp = q_last_curr * cp + t_last_curr;
		/*
		点到线的距离 = 平行四边形的面积/底边
		   . lpa
		  /↘		
		 /     .lp
		. lpb
              |(lp - lpa)×(lp - lpb)|
		dε = ------------------------- = 平行四边形的面积 / 底边
		           |(lpa - lpb)|
		loss = ∑dε_i + ∑dξ_i = D(lp_i) = D( G(p_curr, T_curr2last) )
		*/

		// 平行四边形的面积 = (lp - lpa)×(lp - lpb)
		Eigen::Matrix<T, 3, 1> nu = (lp - lpa).cross(lp - lpb);
		// 低边向量
		Eigen::Matrix<T, 3, 1> de = lpa - lpb;
		// 参数的模是该点到底边的垂线长度
		// 点到线的长度向量，该向量的摸长就是点到直线的距离
		residual[0] = nu.x() / de.norm();
		residual[1] = nu.y() / de.norm();
		residual[2] = nu.z() / de.norm();

		return true;
	}
	// 当前帧的角点、上一帧的最近角点、上一帧的次近角点、Δt/T
	static ceres::CostFunction *Create(const Eigen::Vector3d curr_point_, const Eigen::Vector3d last_point_a_,
									   const Eigen::Vector3d last_point_b_, const double s_)
	{
		// <自定义的类型, 残差维度, 四元数维度, 平移量维度>
		return (new ceres::AutoDiffCostFunction<
				LidarEdgeFactor, 3, 4, 3>(// 残差维度、旋转维度、平移维度
			// 调用构造函数
			new LidarEdgeFactor(curr_point_, last_point_a_, last_point_b_, s_)));
	}

	Eigen::Vector3d curr_point, last_point_a, last_point_b;
	double s;
};

struct LidarPlaneFactor
{
	LidarPlaneFactor(Eigen::Vector3d curr_point_, Eigen::Vector3d last_point_j_,
					 Eigen::Vector3d last_point_l_, Eigen::Vector3d last_point_m_, double s_)
		: curr_point(curr_point_), last_point_j(last_point_j_), last_point_l(last_point_l_),
		  last_point_m(last_point_m_), s(s_)
	{
		// a×b = |a||b|sinΘ，平面法向量垂直于平面
		// 平面单位法向量 = |(j - l)×(j - m)|， ljm ⊥ (j - l)，ljm ⊥ (j - m)
		ljm_norm = (last_point_j - last_point_l).cross(last_point_j - last_point_m);
		ljm_norm.normalize();// 归一化，模为1
	}

	template <typename T>
	bool operator()(const T *q, const T *t, T *residual) const
	{
		/*
		       m
			   .        
              / \     .lp   <<<T<<<  .cp
		-----.---.-----
			 j    l

		| ljm
	m	|→→→→→↗ .lp
		|  Θ↗ 
	j(l).↗
		|
		a·b = |a||b|cosΘ，|ljm| = 1
		(lp - lpj)·ljm = |(lp - lpj)|·1·cosΘ = 点到面的距离
		*/
		Eigen::Matrix<T, 3, 1> cp{T(curr_point.x()), T(curr_point.y()), T(curr_point.z())};
		// j
		Eigen::Matrix<T, 3, 1> lpj{T(last_point_j.x()), T(last_point_j.y()), T(last_point_j.z())};
		// ljm
		Eigen::Matrix<T, 3, 1> ljm{T(ljm_norm.x()), T(ljm_norm.y()), T(ljm_norm.z())};
		Eigen::Quaternion<T> q_last_curr{q[3], q[0], q[1], q[2]};
		Eigen::Quaternion<T> q_identity{T(1), T(0), T(0), T(0)};
		q_last_curr = q_identity.slerp(T(s), q_last_curr);
		Eigen::Matrix<T, 3, 1> t_last_curr{T(s) * t[0], T(s) * t[1], T(s) * t[2]};
		Eigen::Matrix<T, 3, 1> lp;
		// R_curr2last * P_curr  + t = P_last
		lp = q_last_curr * cp + t_last_curr;
		residual[0] = (lp - lpj).dot(ljm);
		return true;
	}
	
	// curr、j、l、m 
	static ceres::CostFunction *Create(const Eigen::Vector3d curr_point_, const Eigen::Vector3d last_point_j_,
									   const Eigen::Vector3d last_point_l_, const Eigen::Vector3d last_point_m_,
									   const double s_)
	{
		return (new ceres::AutoDiffCostFunction<
				LidarPlaneFactor, 1, 4, 3>(
			new LidarPlaneFactor(curr_point_, last_point_j_, last_point_l_, last_point_m_, s_)));
	}

	Eigen::Vector3d curr_point, last_point_j, last_point_l, last_point_m;
	Eigen::Vector3d ljm_norm;
	double s;
};


// 后端
struct LidarPlaneNormFactor
{
	// 构造函数初始化：当前帧的角点、地图点中的平面法向量、D模的倒数
	LidarPlaneNormFactor(Eigen::Vector3d curr_point_, Eigen::Vector3d plane_unit_norm_,
						 double negative_OA_dot_norm_) : curr_point(curr_point_), plane_unit_norm(plane_unit_norm_),
														 negative_OA_dot_norm(negative_OA_dot_norm_) {}

	template <typename T>
	bool operator()(const T *q, const T *t, T *residual) const
	{
		Eigen::Quaternion<T> q_w_curr{q[3], q[0], q[1], q[2]};
		Eigen::Matrix<T, 3, 1> t_w_curr{t[0], t[1], t[2]};
		// 当前帧的角点 P_curr
		Eigen::Matrix<T, 3, 1> cp{T(curr_point.x()), T(curr_point.y()), T(curr_point.z())};
		Eigen::Matrix<T, 3, 1> point_w;
		point_w = q_w_curr * cp + t_w_curr;
		// 平面法向量(A,B,C)，模长为1
		Eigen::Matrix<T, 3, 1> norm(T(plane_unit_norm.x()), T(plane_unit_norm.y()), T(plane_unit_norm.z()));
		// [(A,B,C) · P + D ]/ 1 = 点到面的距离
		residual[0] = norm.dot(point_w) + T(negative_OA_dot_norm);
		return true;
	}

	static ceres::CostFunction *Create(const Eigen::Vector3d curr_point_, const Eigen::Vector3d plane_unit_norm_,
									   const double negative_OA_dot_norm_)
	{
		return (new ceres::AutoDiffCostFunction<
				LidarPlaneNormFactor, 1, 4, 3>(
			new LidarPlaneNormFactor(curr_point_, plane_unit_norm_, negative_OA_dot_norm_)));
	}

	Eigen::Vector3d curr_point;
	Eigen::Vector3d plane_unit_norm;
	double negative_OA_dot_norm;
};


struct LidarDistanceFactor
{

	LidarDistanceFactor(Eigen::Vector3d curr_point_, Eigen::Vector3d closed_point_) 
						: curr_point(curr_point_), closed_point(closed_point_){}

	template <typename T>
	bool operator()(const T *q, const T *t, T *residual) const
	{
		Eigen::Quaternion<T> q_w_curr{q[3], q[0], q[1], q[2]};
		Eigen::Matrix<T, 3, 1> t_w_curr{t[0], t[1], t[2]};
		Eigen::Matrix<T, 3, 1> cp{T(curr_point.x()), T(curr_point.y()), T(curr_point.z())};
		Eigen::Matrix<T, 3, 1> point_w;
		point_w = q_w_curr * cp + t_w_curr;


		residual[0] = point_w.x() - T(closed_point.x());
		residual[1] = point_w.y() - T(closed_point.y());
		residual[2] = point_w.z() - T(closed_point.z());
		return true;
	}

	static ceres::CostFunction *Create(const Eigen::Vector3d curr_point_, const Eigen::Vector3d closed_point_)
	{
		return (new ceres::AutoDiffCostFunction<
				LidarDistanceFactor, 3, 4, 3>(
			new LidarDistanceFactor(curr_point_, closed_point_)));
	}

	Eigen::Vector3d curr_point;
	Eigen::Vector3d closed_point;
};