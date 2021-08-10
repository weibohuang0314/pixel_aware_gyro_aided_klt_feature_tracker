#!/bin/bash

## Get real path ##
_script="$(readlink -f ${BASH_SOURCE[0]})"
echo "Script : $_script"

## Delete last component from $_script ##
_mydir="$(dirname $_script)"
echo "mydir : $_mydir"


#echo "Configuring and building glog ..."
#cd Thirdparty
##rm -f -r glog/
##cd glog-master
#xz -d glog-master.tar.xz
#tar -xvf glog-master.tar
#cd glog-master
#./autogen.sh
#./configure --prefix=$_mydir/Thirdparty/glog
#make
#make install


echo "Configuring and building pixel_aware_gyro_aided_klt_feature_tracker ..."
cd $_mydir
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j12


