SOURCE_GROUP(Param FILES ../../CodeGenere/File2String/Str_ParamMICMAC.cpp)

set(MicMac_Src_Files
	../../CodeGenere/File2String/Str_ParamMICMAC.cpp
	cAppliMICMAC_CorrelPonctuelle.cpp
	Jp2ImageLoader.cpp
	MICMAC.cpp
	PartiesCachees.cpp
	cAppliMICMAC.cpp
	cAppliMICMAC_ChCorrel.cpp
	cAppliMICMAC_GPU.cpp
	cAppliMICMAC_MNE.cpp
	cAppliMICMAC_MEC.cpp
	cAppliMICMAC_Result1.cpp
	cAppliMICMAC_Result2.cpp
	cBasculeMnt.cpp
	cCaracOfDeZoom.cpp
	cEtapeMecComp.cpp
	cFilePx.cpp
	cGBV2_ProgDynOptimiseur.cpp
	cGeomImage.cpp
	cGeomXXX.cpp
	cLoadedImage.cpp
	cMicMacVisu.cpp
	cModeleAnalytiqueComp.cpp
	cNewProgDyn.cpp
	cOptimisationDiff.cpp
	cOrientationCon.cpp
	cOrientationGrille.cpp
	cOrientationRTO.cpp
	cParamMICMAC.cpp
	cPriseDeVue.cpp
	cProgDynOptimiseur.cpp
	cSimulation.cpp
	cStatNDistrib.cpp
	cStdTiffModuleImageLoader.cpp
	cSurfaceOptimiseur.cpp
	uti_MICMAC.cpp
	OrthoLocAnam.cpp
)

#enable_precompiled_headers(StdAfx.h MicMac_Src_Files)

add_executable(MICMAC ${MicMac_Src_Files})

target_link_libraries(MICMAC elise ${X11_LIBRARIES} ${KAKADU_LIBRARIES}  ${TICPP_LIBRARIES} ${Boost_LIBRARIES})

INSTALL(TARGETS MICMAC RUNTIME DESTINATION ${Install_Dir})
	
