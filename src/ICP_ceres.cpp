#include "ICP_ceres/ICP_ceres.h"
using namespace std;
using namespace Eigen;

//ceres自动求导的结构体
ICPCeres::ICPCeres ( Vector3d uvw,Vector3d xyz ) : _uvw(uvw),_xyz(xyz) {}
// 残差的计算 重载括号运算符

template <typename T>
bool ICPCeres::operator() (
        const T* const camera,     // 模型参数，有6维 待优化的
        T* residual ) const     // 残差
{
    T p[3];
    T point[3];
    point[0]=T(_xyz.x());
    point[1]=T(_xyz.y());
    point[2]=T(_xyz.z());
    //算第二帧lidar坐标系下的点在第一帧lidar坐标系下的表示
    AngleAxisRotatePoint(camera, point, p);//计算RP
    p[0] += camera[3]; p[1] += camera[4]; p[2] += camera[5];//相机坐标2
    //计算残差
    residual[0] = T(_uvw.x())-p[0];
    residual[1] = T(_uvw.y())-p[1];
    residual[2] = T(_uvw.z())-p[2];
    return true;
}

ceres::CostFunction* ICPCeres::Create(const Vector3d uvw,const Vector3d xyz) {
    //残差3维，两帧lidar相对位姿6维
    return (new ceres::AutoDiffCostFunction<ICPCeres, 3, 6>(
            new ICPCeres(uvw,xyz)));
}


/**
 * @brief 通过两个link3d Frame帧和匹配的关键点索引，求出两帧相对位姿
 * 
 * @param[in] currentFrame 当前帧
 * @param[in] matchedFrame 上一帧
 * @param[in] vMatchedIndex 匹配关键点索引
 * @param[out] R 
 * @param[out] t 
 * @return int 1:成功 -1:失败
 */
int pose_estimation_3d3d(Frame* currentFrame, Frame* matchedFrame, vector<pair<int, int>> &vMatchedIndex,
Eigen::Matrix3d &R, Eigen::Vector3d &t, LinK3D_Extractor* pLinK3dExtractor)
{
    // 如果匹配的索引数量小于等于30，返回失败
    if(vMatchedIndex.size() <= 30)
    {
        return -1;
    }

    // 对当前帧和匹配帧的边缘关键点进行低曲率过滤
    ScanEdgePoints currentFiltered;
    ScanEdgePoints matchedFiltered;
    pLinK3dExtractor->filterLowCurv(currentFrame->mClusterEdgeKeypoints, currentFiltered);
    pLinK3dExtractor->filterLowCurv(matchedFrame->mClusterEdgeKeypoints, matchedFiltered);

    // 获取匹配的边缘关键点对
    vector<std::pair<PointXYZSCA, PointXYZSCA>> matchedEdgePt;
    pLinK3dExtractor->findEdgeKeypointMatch(currentFiltered, matchedFiltered, vMatchedIndex, matchedEdgePt);
    
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
    //至此获得了去质心的匹配边缘点坐标vRemoveCenterPt1、vRemoveCenterPt2

    //SVD方法
    Eigen::Matrix3d w = Eigen::Matrix3d::Zero();
    // 计算协方差矩阵
    for(int i = 0; i < corrSize; i++)
    {
        w += vRemoveCenterPt1[i] * vRemoveCenterPt2[i].transpose();
    }      
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(w, Eigen::ComputeFullU|Eigen::ComputeFullV);
    Eigen::Matrix3d U = svd.matrixU();
    Eigen::Matrix3d V = svd.matrixV();
    //计算旋转矩阵和平移向量
    //1对应当前帧点云 源点云 2对应上一帧点云 matchedFrame 目标点云
    //SVD解法求的是pi = R * pi' + t  W = sum( qi *qi'^t ) W = U * SIGMA * V^T R=U * V^T
    //这里求的其实是R^T 也就是pi' = R^T * (pi - t) = R^T * pi - R^T * t
    //捋清楚了，这里求的R t就是源点云到目标点云的变换
    R = V * U.transpose();
    t = center2 - R * center1;

    //todo ceres方法
    // ceres::Problem problem;
    // for (int i = 0; i < pts2.size(); ++i)
    // {
    //     ceres::CostFunction* cost_function =
    //             ICPCeres::Create(pts2[i],pts1[i]);
    //     problem.AddResidualBlock(cost_function,
    //                              NULL /* squared loss */,
    //                              camera);
    // }
    // ceres::Solver::Options options;
    // options.linear_solver_type = ceres::DENSE_SCHUR;
    // options.minimizer_progress_to_stdout = true;
    // ceres::Solver::Summary summary;
    // ceres::Solve(options, &problem, &summary);
    // std::cout << summary.FullReport() << "\n";
 
    // Mat R_vec = (Mat_<double>(3,1) << camera[0],camera[1],camera[2]);//数组转cv向量
    // Mat R_cvest;
    // Rodrigues(R_vec,R_cvest);//罗德里格斯公式，旋转向量转旋转矩阵
    // cout<<"R_cvest="<<R_cvest<<endl;
    // Eigen::Matrix<double,3,3> R_est;
    // cv2eigen(R_cvest,R_est);//cv矩阵转eigen矩阵
    // cout<<"R_est="<<R_est<<endl;
    // Eigen::Vector3d t_est(camera[3],camera[4],camera[5]);
    // cout<<"t_est="<<t_est<<endl;
    // Eigen::Isometry3d T(R_est);//构造变换矩阵与输出
    // T.pretranslate(t_est);
    // cout<<T.matrix()<<endl;

    return 1;
}

// //根据三维匹配点对估计两帧相对位姿
// void pose_estimation_3d3d (const vector<Point3f>& pts1, const vector<Point3f>& pts2, Mat& R, Mat& t)
// {
//     Point3f p1, p2;     // center of mass
//     int N = pts1.size();
//     for ( int i=0; i<N; i++ )
//     {
//         p1 += pts1[i];
//         p2 += pts2[i];
//     }
//     p1 = Point3f( Vec3f(p1) /  N);
//     p2 = Point3f( Vec3f(p2) / N);
//     //去中心化
//     vector<Point3f>     q1 ( N ), q2 ( N ); // remove the center
//     for ( int i=0; i<N; i++ )
//     {
//         q1[i] = pts1[i] - p1;
//         q2[i] = pts2[i] - p2;
//     }
 
//     // compute q1*q2^T
//     Eigen::Matrix3d W = Eigen::Matrix3d::Zero();
//     for ( int i=0; i<N; i++ )
//     {
//         W += Eigen::Vector3d ( q1[i].x, q1[i].y, q1[i].z ) * Eigen::Vector3d ( q2[i].x, q2[i].y, q2[i].z ).transpose();
//     }
//     cout<<"W="<<W<<endl;

//     // SVD on W
//     Eigen::JacobiSVD<Eigen::Matrix3d> svd ( W, Eigen::ComputeFullU|Eigen::ComputeFullV );
//     Eigen::Matrix3d U = svd.matrixU();
//     Eigen::Matrix3d V = svd.matrixV();
//     cout<<"U="<<U<<endl;
//     cout<<"V="<<V<<endl;
 
//     Eigen::Matrix3d R_ = U* ( V.transpose() );
//     Eigen::Vector3d t_ = Eigen::Vector3d ( p1.x, p1.y, p1.z ) - R_ * Eigen::Vector3d ( p2.x, p2.y, p2.z );
 
//     // convert to cv::Mat
//     R = ( Mat_<double> ( 3,3 ) <<
//                                R_ ( 0,0 ), R_ ( 0,1 ), R_ ( 0,2 ),
//             R_ ( 1,0 ), R_ ( 1,1 ), R_ ( 1,2 ),
//             R_ ( 2,0 ), R_ ( 2,1 ), R_ ( 2,2 )
//     );
//     t = ( Mat_<double> ( 3,1 ) << t_ ( 0,0 ), t_ ( 1,0 ), t_ ( 2,0 ) );
// }

// //todo: 使用方法，最后删除
// int main ( int argc, char** argv ) {
//     vector<Point3f> pts1, pts2;
//     //填入匹配的3D点
//     cout << "3d-3d pairs: " << pts1.size() << endl;
//     Mat R, t;
//     pose_estimation_3d3d(pts1, pts2, R, t);
//     cout << "ICP via SVD results: " << endl;
//     cout << "R = " << R << endl;
//     cout << "t = " << t << endl;
//     cout << "R_inv = " << R.t() << endl;
//     cout << "t_inv = " << -R.t() * t << endl;
 
//     cout << "calling bundle adjustment" << endl;
//     // verify p1 = R*p2 + t
//     for ( int i=0; i<5; i++ )
//     {
//         cout<<"p1 = "<<pts1[i]<<endl;
//         cout<<"p2 = "<<pts2[i]<<endl;
//         cout<<"(R*p2+t) = "<<
//             R * (Mat_<double>(3,1)<<pts2[i].x, pts2[i].y, pts2[i].z) + t
//             <<endl;
//         cout<<endl;
//     }

//     //利用ceres
//     ceres::Problem problem;
//     for (int i = 0; i < pts2.size(); ++i)
//     {
//         ceres::CostFunction* cost_function =
//                 ICPCeres::Create(pts2[i],pts1[i]);
//         problem.AddResidualBlock(cost_function,
//                                  NULL /* squared loss */,
//                                  camera);
//     }
//     ceres::Solver::Options options;
//     options.linear_solver_type = ceres::DENSE_SCHUR;
//     options.minimizer_progress_to_stdout = true;
//     ceres::Solver::Summary summary;
//     ceres::Solve(options, &problem, &summary);
//     std::cout << summary.FullReport() << "\n";
 
//     Mat R_vec = (Mat_<double>(3,1) << camera[0],camera[1],camera[2]);//数组转cv向量
//     Mat R_cvest;
//     Rodrigues(R_vec,R_cvest);//罗德里格斯公式，旋转向量转旋转矩阵
//     cout<<"R_cvest="<<R_cvest<<endl;
//     Eigen::Matrix<double,3,3> R_est;
//     cv2eigen(R_cvest,R_est);//cv矩阵转eigen矩阵
//     cout<<"R_est="<<R_est<<endl;
//     Eigen::Vector3d t_est(camera[3],camera[4],camera[5]);
//     cout<<"t_est="<<t_est<<endl;
//     Eigen::Isometry3d T(R_est);//构造变换矩阵与输出
//     T.pretranslate(t_est);
//     cout<<T.matrix()<<endl;
 
//     return 0;
// }

