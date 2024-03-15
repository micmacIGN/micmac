cmake_minimum_required(VERSION 3.15)
project(mini2007 VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(MICMAC_PATH $ENV{HOME}/micmac)
set(MMVII_SOURCE_DIR $ENV{HOME}/micmac/MMVII)

set(mmv2_include_dir "${MMVII_SOURCE_DIR}/include")
set(mmv2_external_include_dir "${MMVII_SOURCE_DIR}/ExternalInclude")
set(EIGEN3_INCLUDE_PATH  "${mmv2_external_include_dir}/eigen-3.4.0")

add_executable(${CMAKE_PROJECT_NAME} main.cpp)
include_directories(${mmv2_include_dir} ${mmv2_external_include_dir} ${EIGEN3_INCLUDE_PATH})
target_link_libraries(${PROJECT_NAME} ${MICMAC_PATH}/MMVII/bin/libP2007.a)
target_link_libraries(${PROJECT_NAME} ${MICMAC_PATH}/lib/libelise.a)
target_link_libraries(${PROJECT_NAME} ${MICMAC_PATH}/lib/libANN.a)
target_link_libraries(${PROJECT_NAME} pthread X11 stdc++fs -fopenmp)
