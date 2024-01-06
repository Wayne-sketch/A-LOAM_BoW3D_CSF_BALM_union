cmake_minimum_required(VERSION 2.8.3)
project(aloam_velodyne)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++14")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  nav_msgs
  sensor_msgs
  roscpp
  rospy
  rosbag
  std_msgs
  image_transport
  cv_bridge
  tf
  #link3d
  pcl_ros
  roslib 
)

find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED)
find_package(OpenCV 4.2 REQUIRED)
find_package(Ceres REQUIRED)
# CSF算法添加
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    add_compile_definitions("CSF_USE_OPENMP")
endif()


include_directories(
  include
	${catkin_INCLUDE_DIRS} 
	${PCL_INCLUDE_DIRS}
  ${CERES_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  #link3d
  ${PROJECT_SOURCE_DIR}/include
  ${EIGEN3_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIR}
  ${CHOLMOD_INCLUDE_DIR})

add_subdirectory(include/CSF)

#其他功能包使用本功能包的时候定义，如当其他功能包调用这个功能包也需要pcl库的时候就不需要再次find_package(PCL REQUIRED)
catkin_package( 
  CATKIN_DEPENDS geometry_msgs nav_msgs roscpp rospy std_msgs pcl_ros roslib#项目依赖的其他catkin项目
  DEPENDS EIGEN3 PCL #该项目依赖的非catkin项目
  INCLUDE_DIRS include #导出包的include路径
)


add_executable(ascanRegistration src/scanRegistration.cpp)
target_link_libraries(ascanRegistration ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${CERES_LIBRARIES} ${OpenCV_LIBS} ${G2O_LIBS} ${Sophus_LIBRARIES})

# add_executable(FrontEnd src/FrontEnd.cpp)
# target_link_libraries(FrontEnd ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${CERES_LIBRARIES} ${OpenCV_LIBS} ${G2O_LIBS} ${Sophus_LIBRARIES})

add_executable(alaserOdometry src/laserOdometry.cpp src/Frame.cpp src/LinK3D_Extractor.cpp src/ICP_ceres.cpp)
target_link_libraries(alaserOdometry ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${CERES_LIBRARIES} ${OpenCV_LIBS} ${G2O_LIBS} ${Sophus_LIBRARIES})

# add_executable(BALM src/BALM.cpp)
# target_link_libraries(BALM ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${CERES_LIBRARIES} ${OpenCV_LIBS} ${G2O_LIBS} ${Sophus_LIBRARIES})

add_executable(alaserMapping src/laserMapping.cpp)
target_link_libraries(alaserMapping ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${CERES_LIBRARIES})
if(OpenMP_CXX_FOUND)
    target_link_libraries(alaserMapping PUBLIC CSF OpenMP::OpenMP_CXX)
endif()

# add_executable(ScanToMap src/ScanToMap.cpp)
# target_link_libraries(ScanToMap ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${CERES_LIBRARIES})
# if(OpenMP_CXX_FOUND)
#     target_link_libraries(ScanToMap PUBLIC CSF OpenMP::OpenMP_CXX)
# endif()

add_executable(kittiHelper src/kittiHelper.cpp)
target_link_libraries(kittiHelper ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBS})




