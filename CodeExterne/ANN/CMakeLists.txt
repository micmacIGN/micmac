include_directories(${PROJECT_SOURCE_DIR}/CodeExterne/ANN/include)
set(Install_Dir_lib ${PROJECT_SOURCE_DIR}/lib)

set(ANN_SRC_DIR ${PROJECT_SOURCE_DIR}/CodeExterne/ANN/src)

set(ann_source_files ${ANN_SRC_DIR}/ANN.cpp
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
	${ANN_SRC_DIR}/perf.cpp)

add_library(ANN ${ann_source_files})

install(TARGETS ANN ARCHIVE DESTINATION ${Install_Dir_lib})

set(ann_source_files ${ann_source_files} PARENT_SCOPE)
