cmake_minimum_required( VERSION 2.8.1 )

set (MICMAC_VERSION_MAJOR 1)
set (MICMAC_VERSION_MINOR 0)

include_directories ( ${PROJECT_SOURCE_DIR}/CodeExterne/ANN/include )
set(Install_Dir_lib ${PROJECT_SOURCE_DIR}/lib)

set(ANN_SRC_DIR ${PROJECT_SOURCE_DIR}/CodeExterne/ANN/src)

add_library( ANN 	${ANN_SRC_DIR}/ANN.cpp
					${ANN_SRC_DIR}/bd_pr_search.cpp
					${ANN_SRC_DIR}/bd_tree.cpp
					${ANN_SRC_DIR}/kd_dump.cpp
					${ANN_SRC_DIR}/kd_pr_search.cpp
					${ANN_SRC_DIR}/kd_split.cpp
					${ANN_SRC_DIR}/kd_util.cpp
					${ANN_SRC_DIR}/bd_fix_rad_search.cpp
					${ANN_SRC_DIR}/bd_search.cpp
					${ANN_SRC_DIR}/brute.cpp
					${ANN_SRC_DIR}/kd_fix_rad_search.cpp
					${ANN_SRC_DIR}/kd_search.cpp
					${ANN_SRC_DIR}/kd_tree.cpp
					${ANN_SRC_DIR}/perf.cpp )

INSTALL(TARGETS ANN ARCHIVE DESTINATION ${Install_Dir_lib} )
