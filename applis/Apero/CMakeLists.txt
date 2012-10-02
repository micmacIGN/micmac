SOURCE_GROUP(Param FILES 
	../../CodeGenere/File2String/Str_ParamApero.cpp)

add_executable(Apero
	../../CodeGenere/File2String/Str_ParamApero.cpp
	Apero.cpp
	PtsFlottant.cpp
	cAppliApero.cpp
	cPackMultipleLiaison.cpp
	Basculement.cpp
	Redressement.cpp
	cCalib.cpp
	cPackObsLiaison.cpp
	CarteProfExport.cpp
	Simule.cpp
	cChoixImagesMicMac.cpp
	cParamApero.cpp
	ExportNuage.cpp
	Tracage.cpp
	cCorrel.cpp
	cPonderateur.cpp
	#GenParamApero.cpp
	VisuMult.cpp
	cExport.cpp
	cPose.cpp
	GraphePose.cpp
	cAA_Compensation.cpp
	#cImageAppuis.cpp
	cSurfParam.cpp
	#ImProfAndL2App.cpp
	cAA_InitInconnues.cpp
	cLayersIm.cpp
	PowelOptimize.cpp
	cAA_LevenbergMarkard.cpp
	cPackAppuis.cpp
)

target_link_libraries(Apero elise ${X11_LIBRARIES})

INSTALL(TARGETS Apero RUNTIME DESTINATION ${Install_Dir})

	
