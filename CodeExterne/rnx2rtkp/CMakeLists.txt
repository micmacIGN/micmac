cmake_minimum_required( VERSION 2.8.1 )

if (BUILD_RNX2RTKP)
	include_directories ( ${PROJECT_SOURCE_DIR}/CodeExterne/rnx2rtkp/include )

	set(RNX2RTKP_SRC_DIR ${PROJECT_SOURCE_DIR}/CodeExterne/rnx2rtkp/src)

	add_executable( Rnx2rtkp
					${RNX2RTKP_SRC_DIR}/rnx2rtkp.c
					${RNX2RTKP_SRC_DIR}/rtkcmn.c
					${RNX2RTKP_SRC_DIR}/rinex.c
					${RNX2RTKP_SRC_DIR}/rtkpos.c
					${RNX2RTKP_SRC_DIR}/postpos.c
					${RNX2RTKP_SRC_DIR}/solution.c
					${RNX2RTKP_SRC_DIR}/lambda.c
					${RNX2RTKP_SRC_DIR}/geoid.c
					${RNX2RTKP_SRC_DIR}/sbas.c
					${RNX2RTKP_SRC_DIR}/preceph.c
					${RNX2RTKP_SRC_DIR}/pntpos.c
					${RNX2RTKP_SRC_DIR}/ephemeris.c
					${RNX2RTKP_SRC_DIR}/options.c
					${RNX2RTKP_SRC_DIR}/ppp.c
					${RNX2RTKP_SRC_DIR}/ppp_ar.c
					${RNX2RTKP_SRC_DIR}/rtcm.c
					${RNX2RTKP_SRC_DIR}/rtcm2.c
					${RNX2RTKP_SRC_DIR}/rtcm3.c
					${RNX2RTKP_SRC_DIR}/rtcm3e.c
					${RNX2RTKP_SRC_DIR}/ionex.c
					${RNX2RTKP_SRC_DIR}/qzslex.c
					)
	
	target_compile_definitions(Rnx2rtkp PUBLIC -DTRACE -DENAGLO -DENAQZS -DENAGAL -DNFREQ=3)

	
	IF (${CMAKE_SYSTEM_NAME} MATCHES "Linux")
		TARGET_LINK_LIBRARIES(Rnx2rtkp m rt)
	ELSEIF (${CMAKE_SYSTEM_NAME} MATCHES "Windows")
		TARGET_LINK_LIBRARIES(Rnx2rtkp winmm.lib)
	ENDIF()

	INSTALL(TARGETS Rnx2rtkp RUNTIME DESTINATION ${BIN_AUX_FULL_PATH} )
	
endif()
