set(uti_phgrm_MergeCloud_Src_Files
    ${UTI_PHGRM_MERGE_CLOUD}/cASAMG_StepSelection.cpp
    ${UTI_PHGRM_MERGE_CLOUD}/MergeCloud.cpp
    ${UTI_PHGRM_MERGE_CLOUD}/cASAMG.cpp
    ${UTI_PHGRM_MERGE_CLOUD}/cASAMG_RelVois.cpp
    ${UTI_PHGRM_MERGE_CLOUD}/cASAMG_ImageProcessing.cpp
    ${UTI_PHGRM_MERGE_CLOUD}/GesGraphe.cpp
    ${UTI_PHGRM_MERGE_CLOUD}/cMC_SmallClasses.cpp
)


list( APPEND uti_phgrm_Src_Files
	${uti_phgrm_MergeCloud_Src_Files}
)


