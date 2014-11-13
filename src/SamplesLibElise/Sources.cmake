set(SamplesLibElise_Src_Files
    ${SAMPLESLIBELISE_DIR}/CPP_Windows0Main.cpp
    ${SAMPLESLIBELISE_DIR}/CPP_LSQMain.cpp
    ${SAMPLESLIBELISE_DIR}/CPP_TestAbdou.cpp
    ${SAMPLESLIBELISE_DIR}/CPP_Test0LucGirod.cpp
    ${SAMPLESLIBELISE_DIR}/CPP_Tests_Vincent.cpp
    ${SAMPLESLIBELISE_DIR}/CPP_LucasModifNuage.cpp
    ${SAMPLESLIBELISE_DIR}/CPP_TestMatthieu.cpp
    ${SAMPLESLIBELISE_DIR}/CPP_RAW_test.cpp
)

# JE NE SAIS PAS SI CA SERT ???
SOURCE_GROUP(SamplesLibElise FILES ${SamplesLibElise_Src_Files})

set(Elise_Src_Files
	${Elise_Src_Files}
	${SamplesLibElise_Src_Files}
)
