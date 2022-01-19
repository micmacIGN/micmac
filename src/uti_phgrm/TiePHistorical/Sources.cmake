set(uti_image_TiePHisto
		${UTI_PHGRM_TiePHisto_DIR}/cAppliTiepHistorical.cpp
		${UTI_PHGRM_TiePHisto_DIR}/cInterEp_SuperGlue.cpp
		${UTI_PHGRM_TiePHisto_DIR}/cInterEp_MergeTiePt.cpp
		${UTI_PHGRM_TiePHisto_DIR}/cInterEp_GetPatchPair.cpp
		${UTI_PHGRM_TiePHisto_DIR}/cInterEp_CreateGCPs.cpp
		${UTI_PHGRM_TiePHisto_DIR}/cInterEp_DSM_Equalization.cpp
		${UTI_PHGRM_TiePHisto_DIR}/cInterEp_GetOverlappedImages.cpp
		${UTI_PHGRM_TiePHisto_DIR}/cInterEp_GuidedSIFTMatch.cpp
		${UTI_PHGRM_TiePHisto_DIR}/cInterEp_CrossCorrelation.cpp
		${UTI_PHGRM_TiePHisto_DIR}/cInterEp_RANSAC.cpp
		${UTI_PHGRM_TiePHisto_DIR}/cInterEp_WallisFilter.cpp
		${UTI_PHGRM_TiePHisto_DIR}/cInterEp_TiePtEvaluation.cpp
		${UTI_PHGRM_TiePHisto_DIR}/cInterEp_MakeTrainingData.cpp
		${UTI_PHGRM_TiePHisto_DIR}/cInterEp_VisuTiePtIn3D.cpp
		${UTI_PHGRM_TiePHisto_DIR}/cInterEp_TiePtAddWeight.cpp
		${UTI_PHGRM_TiePHisto_DIR}/cInterEp_EnhancedSpG.cpp
		${UTI_PHGRM_TiePHisto_DIR}/cInterEp_SIFT2Step.cpp
		${UTI_PHGRM_TiePHisto_DIR}/cInterEp_GlobalR3D.cpp
)

list( APPEND uti_phgrm_Src_Files
		${uti_image_TiePHisto}
)


