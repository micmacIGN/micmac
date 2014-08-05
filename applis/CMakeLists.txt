ADD_SUBDIRECTORY(MICMAC)

add_executable(MpDcraw 
        MpDcraw/cArgMpDCRaw.cpp  
        MpDcraw/cNChanel.cpp  
        MpDcraw/cOneChanel.cpp  
        MpDcraw/MpDcraw.cpp
)
target_link_libraries(MpDcraw elise ${X11_LIBRARIES})

#Outils uti_phgrm
#add_executable(Malt uti_phgrm/Malt.cpp)
#target_link_libraries(Malt elise ${X11_LIBRARIES})

add_executable(Tawny uti_phgrm/Tawny.cpp)
target_link_libraries(Tawny elise ${X11_LIBRARIES})

add_executable(Tapioca uti_phgrm/Tapioca.cpp)
target_link_libraries(Tapioca elise ${X11_LIBRARIES})

add_executable(Pastis uti_phgrm/Pastis.cpp)
target_link_libraries(Pastis elise ${X11_LIBRARIES})

add_executable(Nuage2Ply uti_phgrm/Nuage2Ply.cpp)
target_link_libraries(Nuage2Ply elise ${X11_LIBRARIES})

add_executable(Pasta uti_phgrm/Pasta.cpp)
target_link_libraries(Pasta elise ${X11_LIBRARIES})

add_executable(Bascule uti_phgrm/Bascule.cpp)
target_link_libraries(Bascule elise ${X11_LIBRARIES})

add_executable(RepLocBascule uti_phgrm/RepLocBascule.cpp)
target_link_libraries(RepLocBascule elise ${X11_LIBRARIES})

add_executable(GCPBascule uti_phgrm/GCPBascule.cpp)
target_link_libraries(GCPBascule elise ${X11_LIBRARIES})

add_executable(SBGlobBascule uti_phgrm/SBGlobBascule.cpp)
target_link_libraries(SBGlobBascule elise ${X11_LIBRARIES})

add_executable(Tarama uti_phgrm/Tarama.cpp)
target_link_libraries(Tarama elise ${X11_LIBRARIES})

add_executable(MakeGrid uti_phgrm/MakeGrid.cpp)
target_link_libraries(MakeGrid elise ${X11_LIBRARIES})

add_executable(TestCam uti_phgrm/TestCam.cpp)
target_link_libraries(TestCam elise ${X11_LIBRARIES})

add_executable(Tapas uti_phgrm/Tapas.cpp)
target_link_libraries(Tapas elise ${X11_LIBRARIES})

#add_executable(AperiCloud uti_phgrm/AperiCloud.cpp)
#target_link_libraries(AperiCloud elise ${X11_LIBRARIES})

add_executable(CmpCalib uti_phgrm/CmpCalib.cpp)
target_link_libraries(CmpCalib elise ${X11_LIBRARIES})

#Outils uti_image
add_executable(GenXML2Cpp uti_image/GenXML2Cpp.cpp)
target_link_libraries(GenXML2Cpp elise ${X11_LIBRARIES})

add_executable(ScaleIm uti_image/ScaleIm.cpp)
target_link_libraries(ScaleIm elise ${X11_LIBRARIES})

add_executable(ElDcraw uti_image/ElDcraw.cpp)
target_link_libraries(ElDcraw elise ${X11_LIBRARIES})

add_executable(to8Bits uti_image/to8Bits.cpp)
target_link_libraries(to8Bits elise ${X11_LIBRARIES})

add_executable(GrShade uti_image/GrShade.cpp)
target_link_libraries(GrShade elise ${X11_LIBRARIES})

add_executable(PastDevlop uti_image/PastDevlop.cpp)
target_link_libraries(PastDevlop elise ${X11_LIBRARIES})

add_executable(Devlop uti_image/Devlop.cpp)
target_link_libraries(Devlop elise ${X11_LIBRARIES})

add_executable(tiff_info uti_image/tiff_info.cpp)
target_link_libraries(tiff_info elise ${X11_LIBRARIES})

add_executable(Dequant uti_image/Dequant.cpp)
target_link_libraries(Dequant elise ${X11_LIBRARIES})


IF (X11_FOUND)
	add_executable(SaisieBasc uti_image/SaisieBasc.cpp)
	target_link_libraries(SaisieBasc elise ${X11_LIBRARIES})
	INSTALL(TARGETS SaisieBasc
		RUNTIME DESTINATION ${Install_Dir}
	)
	add_executable(SaisieAppuisInit uti_image/SaisieAppuisInit.cpp)
	target_link_libraries(SaisieAppuisInit elise ${X11_LIBRARIES})
	INSTALL(TARGETS SaisieAppuisInit
		RUNTIME DESTINATION ${Install_Dir}
	)
	add_executable(SaisieAppuisPredic uti_image/SaisieAppuisPredic.cpp)
	target_link_libraries(SaisieAppuisPredic elise ${X11_LIBRARIES})
	INSTALL(TARGETS SaisieAppuisPredic
		RUNTIME DESTINATION ${Install_Dir}
	)


	add_executable(SaisieMasq uti_image/SaisieMasq.cpp)
	target_link_libraries(SaisieMasq elise ${X11_LIBRARIES})
	INSTALL(TARGETS SaisieMasq
		RUNTIME DESTINATION ${Install_Dir}
	)

	add_executable(SEL uti_phgrm/SEL.cpp)
	target_link_libraries(SEL elise ${X11_LIBRARIES})
        INSTALL(TARGETS SEL
                RUNTIME DESTINATION ${Install_Dir}
        )

	add_executable(MICMACSaisieLiaisons MICMAC/SaisieLiaisons.cpp
		../CodeGenere/File2String/Str_ParamMICMAC.cpp
		MICMAC/cNewProgDyn.cpp
        MICMAC/cAppliMICMAC_GPU.cpp
        MICMAC/cAppliMICMAC_CorrelPonctuelle.cpp
        MICMAC/Jp2ImageLoader.cpp
        MICMAC/PartiesCachees.cpp
        MICMAC/cAppliMICMAC.cpp
        MICMAC/cAppliMICMAC_ChCorrel.cpp
        MICMAC/cAppliMICMAC_MNE.cpp
        MICMAC/cAppliMICMAC_MEC.cpp
        MICMAC/cAppliMICMAC_Result1.cpp
        MICMAC/cAppliMICMAC_Result2.cpp
        MICMAC/cBasculeMnt.cpp
        MICMAC/cCaracOfDeZoom.cpp
        MICMAC/cEtapeMecComp.cpp
        MICMAC/cFilePx.cpp
        MICMAC/cGBV2_ProgDynOptimiseur.cpp
        MICMAC/cGeomImage.cpp
        MICMAC/cGeomXXX.cpp
        MICMAC/cLoadedImage.cpp
        MICMAC/cMicMacVisu.cpp
        MICMAC/cModeleAnalytiqueComp.cpp
        MICMAC/cOptimisationDiff.cpp
        MICMAC/cOrientationCon.cpp
        MICMAC/cOrientationGrille.cpp
        MICMAC/cOrientationRTO.cpp
        MICMAC/cParamMICMAC.cpp
        MICMAC/cPriseDeVue.cpp
        MICMAC/cProgDynOptimiseur.cpp
        MICMAC/cSimulation.cpp
        MICMAC/cStatNDistrib.cpp
        MICMAC/cStdTiffModuleImageLoader.cpp
        MICMAC/cSurfaceOptimiseur.cpp
        MICMAC/uti_MICMAC.cpp
		MICMAC/OrthoLocAnam.cpp
	)

	target_link_libraries(MICMACSaisieLiaisons elise ${X11_LIBRARIES} ${KAKADU_LIBRARIES} )
        
	INSTALL(TARGETS MICMACSaisieLiaisons RUNTIME DESTINATION ${Install_Dir})
	
ENDIF()

#Outils uti_files
add_executable(BatchFDC uti_files/BatchFDC.cpp)
target_link_libraries(BatchFDC elise ${X11_LIBRARIES})

add_executable(MapCmd uti_files/MapCmd.cpp)
target_link_libraries(MapCmd elise ${X11_LIBRARIES})

ADD_SUBDIRECTORY(Apero)
ADD_SUBDIRECTORY(ReducHom)
ADD_SUBDIRECTORY(SaisiePts)
ADD_SUBDIRECTORY(Ortho)
ADD_SUBDIRECTORY(EtalonnagePolygone)

IF ( MatisOrientation_FOUND)
	target_link_libraries(MICMAC elise ${MatisOrientation_LIBRARIES})
	IF (MICMAC_USE_PROJ4)
		target_link_libraries(MICMAC elise ${PROJ4_LIBRARIES})
	ENDIF()
ENDIF()

install(TARGETS MpDcraw Tawny  Dequant Tapioca Pastis Nuage2Ply Pasta Bascule MakeGrid TestCam Tapas GenXML2Cpp ScaleIm ElDcraw to8Bits GrShade PastDevlop Devlop tiff_info BatchFDC MapCmd RepLocBascule Tarama GCPBascule SBGlobBascule
	RUNTIME DESTINATION ${Install_Dir}
)
