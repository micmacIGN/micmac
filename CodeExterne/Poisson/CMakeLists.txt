cmake_minimum_required( VERSION 2.8.1 )

if (${BUILD_POISSON})
	include_directories ( ${PROJECT_SOURCE_DIR}/CodeExterne/Poisson/include )

	set(POISSON_SRC_DIR ${PROJECT_SOURCE_DIR}/CodeExterne/Poisson/src)
	set(BinAuxDir ${PROJECT_SOURCE_DIR}/binaire-aux)
	
	add_executable( PoissonRecon 	
					${POISSON_SRC_DIR}/CmdLineParser.cpp
					${POISSON_SRC_DIR}/Factor.cpp					
					${POISSON_SRC_DIR}/Geometry.cpp
					${POISSON_SRC_DIR}/MarchingCubes.cpp
					${POISSON_SRC_DIR}/PlyFile.cpp
					${POISSON_SRC_DIR}/PoissonRecon.cpp
					${POISSON_SRC_DIR}/PTime.cpp
					)				
						
	add_executable( SurfaceTrimmer 	
					${POISSON_SRC_DIR}/CmdLineParser.cpp
					${POISSON_SRC_DIR}/Factor.cpp					
					${POISSON_SRC_DIR}/Geometry.cpp
					${POISSON_SRC_DIR}/MarchingCubes.cpp
					${POISSON_SRC_DIR}/PlyFile.cpp
					${POISSON_SRC_DIR}/SurfaceTrimmer.cpp
					${POISSON_SRC_DIR}/PTime.cpp
					)				
					
	INSTALL(TARGETS PoissonRecon RUNTIME DESTINATION ${BinAuxDir} )
	IF (WIN32)
		target_link_libraries(PoissonRecon Psapi)
	ENDIF()
	
	INSTALL(TARGETS SurfaceTrimmer RUNTIME DESTINATION ${BinAuxDir} )
	

	
endif()
