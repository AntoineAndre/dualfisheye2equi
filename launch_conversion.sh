#!/bin/bash

clear
mkdir build
cd build
cmake ..
make -j8 -B

# Fill in the various paths involved in the conversion

calib_file=../../../data/Sequence3/calibration/calib_twin_fisheye.xml
image_dir=../../../data/Sequence3/images/Images/
mask=../../../data/Sequence3/calibration/maskFull.png
poses_fic=../../../data/Sequence3/poses.txt

echo "variables initialized"    

./dualfisheye2equi $calib_file $image_dir 0 20 1 $mask $poses_fic 1