echo "Building ROS nodes"

cd Examples/ROS/ROS_Demo_Feature_Tracking
mkdir build
cd build
cmake .. -DROS_BUILD_TYPE=Release
make -j12