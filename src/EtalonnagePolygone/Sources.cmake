if (X11_FOUND)

	set(UTI_ETAPOLYLIB_DIR ${UTI_ETAPOLY_DIR}/lib)

	set(SrcGrp_EtaPoly Eta_polygon)

	set(EtaPolyLib_Src_Files 
			#${UTI_ETAPOLYLIB_DIR}/cParamSaisiePts.h
			${UTI_ETAPOLYLIB_DIR}/cEtalonnage.cpp
			${UTI_ETAPOLYLIB_DIR}/cCamIncEtalonage.cpp
			${UTI_ETAPOLYLIB_DIR}/cHypDetectCible.cpp
			${UTI_ETAPOLYLIB_DIR}/UseParamCompl.cpp
			${UTI_ETAPOLYLIB_DIR}/cCibleRechImage.cpp
			${UTI_ETAPOLYLIB_DIR}/VerifcEtalonnage.cpp
			${UTI_ETAPOLYLIB_DIR}/SauvParam.cpp
			${UTI_ETAPOLYLIB_DIR}/cRechercheCDD.cpp
			${UTI_ETAPOLYLIB_DIR}/PointeInit.cpp
			${UTI_ETAPOLYLIB_DIR}/VisuFtm.cpp
			${UTI_ETAPOLYLIB_DIR}/cBlockEtal.cpp
			${UTI_ETAPOLYLIB_DIR}/cCpleCamEtal.cpp
			${UTI_ETAPOLYLIB_DIR}/cCoordNormalizer.cpp
		)

	set(EtaPoly_Src_Files 
		${UTI_ETAPOLY_DIR}/CPP_PointeInitPolyg.cpp
		${UTI_ETAPOLY_DIR}/CPP_RechCibleInit.cpp
		${UTI_ETAPOLY_DIR}/CPP_CalibInit.cpp
		${UTI_ETAPOLY_DIR}/CPP_RechCibleDRad.cpp
		${UTI_ETAPOLY_DIR}/CPP_CalibFinale.cpp
		${UTI_ETAPOLY_DIR}/CPP_Compens.cpp
		${UTI_ETAPOLY_DIR}/CPP_ScriptCalib.cpp
		${UTI_ETAPOLY_DIR}/CPP_ConvertPolygone.cpp
		${UTI_ETAPOLY_DIR}/CPP_CatImSaisie.cpp
	)
	
	source_group(${SrcGrp_EtaPoly} FILES ${EtaPoly_Src_Files})
	source_group(${SrcGrp_EtaPoly}\\Lib FILES ${EtaPolyLib_Src_Files})
	
	list(APPEND Elise_Src_Files ${EtaPolyLib_Src_Files})
	list(APPEND Elise_Src_Files ${EtaPoly_Src_Files})

endif()


	
