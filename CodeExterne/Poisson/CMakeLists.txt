cmake_minimum_required(VERSION 2.8.12)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
set(POISSON_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)

find_package(OpenMP QUIET)

if(MSVC)
	add_definitions(-D_CRT_SECURE_NO_WARNINGS -DMSVC)
else()
	set(CMAKE_CXX_FLAGS "-O3 -DRELEASE -DNDEBUG -funroll-loops -ffast-math -w")
endif()

add_library(poisson
	${POISSON_SRC_DIR}/PoissonRecon.cpp
	${POISSON_SRC_DIR}/CmdLineParser.cpp
	${POISSON_SRC_DIR}/Factor.cpp
	${POISSON_SRC_DIR}/Geometry.cpp
	${POISSON_SRC_DIR}/MarchingCubes.cpp
	${POISSON_SRC_DIR}/PlyFile.cpp)

add_executable(PoissonRecon ${POISSON_SRC_DIR}/PoissonRecon.cpp)
target_link_libraries(PoissonRecon poisson)

add_executable(SurfaceTrimmer ${POISSON_SRC_DIR}/SurfaceTrimmer.cpp)
target_link_libraries(SurfaceTrimmer poisson)

if(OPENMP_FOUND)
	message(STATUS "PoissonRecon/SurfaceTrimmer: OpenMP enabled")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

if(WIN32)
	target_link_libraries(PoissonRecon Psapi)
else()
	if(OPENMP_FOUND)
		target_link_libraries(PoissonRecon gomp)
		target_link_libraries(SurfaceTrimmer gomp)
	endif()
endif()

if(BIN_AUX_FULL_PATH)
	install(TARGETS PoissonRecon RUNTIME DESTINATION ${BIN_AUX_FULL_PATH})
	install(TARGETS SurfaceTrimmer RUNTIME DESTINATION ${BIN_AUX_FULL_PATH})
endif()
