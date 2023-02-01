project(dualfisheye2equi)

cmake_minimum_required(VERSION 2.6)
set(CMAKE_CXX_STANDARD 17)

find_package(PER REQUIRED per_core per_io)

if(PER_FOUND)
  include(${PER_USE_FILE})
endif(PER_FOUND)

find_package(OpenCV REQUIRED)

FIND_PACKAGE(VISP REQUIRED)

IF(VISP_FOUND)
  INCLUDE(${VISP_USE_FILE})
ENDIF(VISP_FOUND)

# Boost
FIND_PACKAGE(Boost REQUIRED)

include_directories(${Boost_INCLUDE_DIRS}) # /opt/local/include/ might be needed as well under MacOS

# link_directories(${Boost_LIBRARY_DIRS}) # /opt/local/lib/ might be needed as well under MacOS # similar /Users/guillaume/Developpement/librairies/visp-3.0.1/build/lib/Release/ might be needed as well under MacOS
set(dualfisheye2equi_cpp
  dualfisheye2equi.cpp
)

foreach(cpp ${dualfisheye2equi_cpp})
  per_add_target(${cpp})

  if(COMMAND pr_add_dependency)
    pr_add_dependency(${cpp} "dualfisheye2equi")
  endif()
endforeach()

target_link_libraries(dualfisheye2equi libboost_system.so libboost_filesystem.so libboost_regex.so)
target_link_libraries(dualfisheye2equi ${OpenCV_LIBS})