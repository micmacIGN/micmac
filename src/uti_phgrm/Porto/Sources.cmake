set(uti_phgrm_Porto_Src_Files
    ${UTI_PHGRM_PORTO_DIR}/cAppli_Ortho.cpp
    ${UTI_PHGRM_PORTO_DIR}/cLoadedIm.cpp
    ${UTI_PHGRM_PORTO_DIR}/cOneImOrhto.cpp
    ${UTI_PHGRM_PORTO_DIR}/Egalise.cpp
    ${UTI_PHGRM_PORTO_DIR}/Ortho_PC.cpp
)

#SOURCE_GROUP(uti_phgrm FILES ${uti_phgrm_Src_Files})

list( APPEND uti_phgrm_Src_Files
	${uti_phgrm_Porto_Src_Files}
)
