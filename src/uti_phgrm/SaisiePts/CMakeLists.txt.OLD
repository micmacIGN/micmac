IF (X11_FOUND)
	add_executable(SaisiePts

			cParamSaisiePts.h
			SaisiePts.cpp
			cImage.cpp
			cPointGlob.cpp
			cAppliSaisiePts.cpp
			cWCreatePt.cpp
			cWinIm.cpp
			cAppliInput.cpp
			cPointeIm.cpp
			cParamSaisiePts.cpp
	)

	target_link_libraries(SaisiePts elise ${X11_LIBRARIES})

	INSTALL(TARGETS SaisiePts RUNTIME DESTINATION ${Install_Dir})
ENDIF()
	
