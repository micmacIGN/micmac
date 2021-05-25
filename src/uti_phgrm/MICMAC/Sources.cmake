set(uti_phgrm_MICMAC_Src_Files
#    ${UTI_PHGRM_MICMAC_DIR}/BatchMICMAC.cpp
    ${UTI_PHGRM_MICMAC_DIR}/cCorrel2DPonctuel.cpp
    ${UTI_PHGRM_MICMAC_DIR}/SaisieLiaisons.cpp
    ${UTI_PHGRM_MICMAC_DIR}/cAppliMICMAC.cpp
    ${UTI_PHGRM_MICMAC_DIR}/cAppliMICMAC_ChCorrel.cpp
    ${UTI_PHGRM_MICMAC_DIR}/MMMaskByTP.cpp
    ${UTI_PHGRM_MICMAC_DIR}/cAppliMICMAC_CompareDirectRadiom.cpp
    ${UTI_PHGRM_MICMAC_DIR}/cAppliMICMAC_CorrelPonctuelle.cpp
    ${UTI_PHGRM_MICMAC_DIR}/cAppliMICMAC_GPU.cpp
    ${UTI_PHGRM_MICMAC_DIR}/cAppliMICMAC_MEC.cpp
    ${UTI_PHGRM_MICMAC_DIR}/cAppliMICMAC_MNE.cpp
    ${UTI_PHGRM_MICMAC_DIR}/cCorrelMulScale.cpp
    ${UTI_PHGRM_MICMAC_DIR}/cAppliMICMAC_Result1.cpp
    ${UTI_PHGRM_MICMAC_DIR}/cAppliMICMAC_Result2.cpp
    ${UTI_PHGRM_MICMAC_DIR}/cBasculeMnt.cpp
    ${UTI_PHGRM_MICMAC_DIR}/cCaracOfDeZoom.cpp
    ${UTI_PHGRM_MICMAC_DIR}/cEtapeMecComp.cpp
    ${UTI_PHGRM_MICMAC_DIR}/cFilePx.cpp
    ${UTI_PHGRM_MICMAC_DIR}/cGBV2_ProgDynOptimiseur.cpp
    ${UTI_PHGRM_MICMAC_DIR}/cGeomXXX.cpp
    ${UTI_PHGRM_MICMAC_DIR}/cLoadedImage.cpp
    ${UTI_PHGRM_MICMAC_DIR}/cMicMacVisu.cpp
    ${UTI_PHGRM_MICMAC_DIR}/cModeleAnalytiqueComp.cpp
    ${UTI_PHGRM_MICMAC_DIR}/cNewProgDyn.cpp
    ${UTI_PHGRM_MICMAC_DIR}/cOptimisationDiff.cpp
    ${UTI_PHGRM_MICMAC_DIR}/cCameraModuleOrientation.cpp
    ${UTI_PHGRM_MICMAC_DIR}/cOrientationGrille.cpp
    ${UTI_PHGRM_MICMAC_DIR}/cOrientationRTO.cpp
    ${UTI_PHGRM_MICMAC_DIR}/cParamMICMAC.cpp
    ${UTI_PHGRM_MICMAC_DIR}/cProgDynOptimiseur.cpp
    ${UTI_PHGRM_MICMAC_DIR}/cSimulation.cpp
    ${UTI_PHGRM_MICMAC_DIR}/cStatNDistrib.cpp
    ${UTI_PHGRM_MICMAC_DIR}/cStdTiffModuleImageLoader.cpp
    ${UTI_PHGRM_MICMAC_DIR}/cSurfaceOptimiseur.cpp
    ${UTI_PHGRM_MICMAC_DIR}/FusionCarteProf.cpp
    ${UTI_PHGRM_MICMAC_DIR}/cAppliMICMAC_Census.cpp
#    ${UTI_PHGRM_MICMAC_DIR}/GenParamMICMAC.cpp
#    ${UTI_PHGRM_MICMAC_DIR}/GrilleXml2Bin.cpp
    ${UTI_PHGRM_MICMAC_DIR}/Jp2ImageLoader.cpp
#    ${UTI_PHGRM_MICMAC_DIR}/Ori-cAppliMICMAC_GPU.cpp
    ${UTI_PHGRM_MICMAC_DIR}/OrthoLocAnam.cpp
    ${UTI_PHGRM_MICMAC_DIR}/PartiesCachees.cpp
    ${UTI_PHGRM_MICMAC_DIR}/SaisieLiaisons.cpp
#    ${UTI_PHGRM_MICMAC_DIR}/StdAfx.cpp
#    ${UTI_PHGRM_MICMAC_DIR}/SuperpositionImages.cpp
#    ${UTI_PHGRM_MICMAC_DIR}/TestMICMAC.cpp
#    ${UTI_PHGRM_MICMAC_DIR}/TestParamMICMAC.cpp
    ${UTI_PHGRM_MICMAC_DIR}/uti_MICMAC.cpp
#    ${UTI_PHGRM_MICMAC_DIR}/VisuSuperpGrid.cpp
	${UTI_PHGRM_MICMAC_DIR}/cPriseDeVue.cpp
	${UTI_PHGRM_MICMAC_DIR}/cGeomImage.cpp
	${UTI_PHGRM_MICMAC_DIR}/MM23DVariationnel.cpp
)

if (${WITH_IGN_ORI})
	set(uti_phgrm_MICMAC_Src_Files ${uti_phgrm_MICMAC_Src_Files}
		${UTI_PHGRM_MICMAC_DIR}/IgnSocleImageLoader.cpp
		${UTI_PHGRM_MICMAC_DIR}/cOrientationCon.cpp
	)
endif()
