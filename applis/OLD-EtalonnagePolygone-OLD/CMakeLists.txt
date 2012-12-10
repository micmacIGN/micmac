if (X11_FOUND)

file(GLOB  SRC . 
        cParamSaisiePts.h
        cEtalonnage.cpp
        cCamIncEtalonage.cpp
        cHypDetectCible.cpp
        UseParamCompl.cpp
        cCibleRechImage.cpp
        VerifcEtalonnage.cpp
        SauvParam.cpp
        cRechercheCDD.cpp
        PointeInit.cpp
        VisuFtm.cpp
        cBlockEtal.cpp
        cCpleCamEtal.cpp
        cCoordNormalizer.cpp
    )


add_library(libEtalPolyg  ${SRC})
INSTALL(TARGETS libEtalPolyg ARCHIVE DESTINATION ${Install_Dir})


add_executable(EPExePointeInitPolyg  ExePointeInitPolyg.cpp)
target_link_libraries(EPExePointeInitPolyg libEtalPolyg elise ${X11_LIBRARIES})
INSTALL(TARGETS EPExePointeInitPolyg RUNTIME DESTINATION ${Install_Dir})


add_executable(EPExeRechCibleInit  ExeRechCibleInit.cpp)
target_link_libraries(EPExeRechCibleInit libEtalPolyg elise ${X11_LIBRARIES})
INSTALL(TARGETS EPExeRechCibleInit RUNTIME DESTINATION ${Install_Dir})

add_executable(EPExeCalibInit  ExeCalibInit.cpp)
target_link_libraries(EPExeCalibInit libEtalPolyg elise ${X11_LIBRARIES})
INSTALL(TARGETS EPExeCalibInit RUNTIME DESTINATION ${Install_Dir})

add_executable(EPExeRechCibleDRad  ExeRechCibleDRad.cpp)
target_link_libraries(EPExeRechCibleDRad libEtalPolyg elise ${X11_LIBRARIES})
INSTALL(TARGETS EPExeRechCibleDRad RUNTIME DESTINATION ${Install_Dir})

add_executable(EPExeCalibFinale  ExeCalibFinale.cpp)
target_link_libraries(EPExeCalibFinale libEtalPolyg elise ${X11_LIBRARIES})
INSTALL(TARGETS EPExeCalibFinale RUNTIME DESTINATION ${Install_Dir})

add_executable(EPExeCompens  ExeCompens.cpp)
target_link_libraries(EPExeCompens libEtalPolyg elise ${X11_LIBRARIES})
INSTALL(TARGETS EPExeCompens RUNTIME DESTINATION ${Install_Dir})

add_executable(ScriptCalib  ScriptCalib.cpp)
target_link_libraries(ScriptCalib libEtalPolyg elise ${X11_LIBRARIES})
INSTALL(TARGETS ScriptCalib RUNTIME DESTINATION ${Install_Dir})

endif()


	
