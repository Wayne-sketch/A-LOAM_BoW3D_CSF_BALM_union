cmake_minimum_required(VERSION 2.8.3)
# project(faloamBA)
project(aloam_velodyne)

set(CMAKE_BUILD_TYPE "Release")
# set(CMAKE_CXX_FLAGS "-std=c++11")
set(CMAKE_CXX_STANDARD 17)
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
  pcl_ros
)

find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Ceres REQUIRED)
find_package(Boost REQUIRED timer)
find_package(OpenMP REQUIRED)
find_package(GTSAM REQUIRED QUIET)

include_directories(
  include
	${catkin_INCLUDE_DIRS} 
	${PCL_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  ${CERES_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${GTSAM_INCLUDE_DIR}
)

catkin_package(
  CATKIN_DEPENDS geometry_msgs nav_msgs roscpp rospy std_msgs
  DEPENDS EIGEN3 PCL 
  INCLUDE_DIRS include
)

# add_executable(FrontEnd_Link3D src/FrontEnd_Link3D.cpp)

# target_link_libraries(FrontEnd_Link3D ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${CERES_LIBRARIES} ${OpenCV_LIBS})

add_executable(FrontEnd src/FrontEnd.cpp)

target_link_libraries(FrontEnd ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${CERES_LIBRARIES} ${OpenCV_LIBS})

add_executable(ScanToMap_CSF
		src/ScanToMap_CSF.cpp
		# include/utility/Scancontext.cpp
		# include/utility/VoxelGridLargeScale.cpp
		include/CSF/c2cdist.cpp
		include/CSF/Cloth.cpp
		include/CSF/Constraint.cpp
		include/CSF/Particle.cpp
		include/CSF/point_cloud.cpp
		include/CSF/Rasterization.cpp
		include/CSF/XYZReader.cpp
		include/CSF/CSF.cpp)
target_link_libraries(ScanToMap_CSF ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${CERES_LIBRARIES})

add_executable(BackEnd_BALM src/BackEnd_BALM.cpp include/utility/myso3.cpp)
target_link_libraries(BackEnd_BALM ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBS})

# add_executable(alaserPGO
# 		src/LoopOptimization.cpp
# 		include/utility/Scancontext.cpp
#     include/utility/VoxelGridLargeScale.cpp
# 		include/CSF/c2cdist.cpp
# 		include/CSF/Cloth.cpp
# 		include/CSF/Constraint.cpp
# 		include/CSF/Particle.cpp
# 		include/CSF/point_cloud.cpp
# 		include/CSF/Rasterization.cpp
# 		include/CSF/XYZReader.cpp
# 		include/CSF/CSF.cpp)

# target_compile_options(alaserPGO
#   PRIVATE ${OpenMP_CXX_FLAGS}
# )
# target_link_libraries(alaserPGO 
#   ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${CERES_LIBRARIES}
#   ${OpenMP_CXX_FLAGS}
#   gtsam
# )