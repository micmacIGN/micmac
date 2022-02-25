set(SamplesLibElise_Src_Files
    ${SAMPLESLIBELISE_DIR}/CPP_Windows0Main.cpp
    ${SAMPLESLIBELISE_DIR}/CPP_LSQMain.cpp
    ${SAMPLESLIBELISE_DIR}/CPP_TestAbdou.cpp
    ${SAMPLESLIBELISE_DIR}/CPP_Test0LucGirod.cpp
    ${SAMPLESLIBELISE_DIR}/CPP_Tests_Vincent.cpp
    ${SAMPLESLIBELISE_DIR}/CPP_LucasModifNuage.cpp
    ${SAMPLESLIBELISE_DIR}/CPP_TestMatthieu.cpp
    ${SAMPLESLIBELISE_DIR}/CPP_RAW_test.cpp
    ${SAMPLESLIBELISE_DIR}/CPP_TestER.cpp
    ${SAMPLESLIBELISE_DIR}/CPP_TestJB.cpp
    ${SAMPLESLIBELISE_DIR}/CPP_TestYZ.cpp
    ${SAMPLESLIBELISE_DIR}/CPP_TestPush.cpp
    #${SAMPLESLIBELISE_DIR}/CPP_CilliaImg.cpp
    #${SAMPLESLIBELISE_DIR}/CPP_CilliaXML.cpp
    #${SAMPLESLIBELISE_DIR}/CPP_CilliaAss.cpp
    #${SAMPLESLIBELISE_DIR}/CPP_CilliaImgt.cpp
    #${SAMPLESLIBELISE_DIR}/CPP_CilliaCol.cpp  
    #${SAMPLESLIBELISE_DIR}/CPP_CilliaMap.cpp 
    ${SAMPLESLIBELISE_DIR}/CPP_TestCamTOF.cpp 
    ${SAMPLESLIBELISE_DIR}/CPP_TestMH.cpp 
    ${SAMPLESLIBELISE_DIR}/CPP_TestLulin.cpp
)

# JE NE SAIS PAS SI CA SERT ???
source_group(SamplesLibElise FILES ${SamplesLibElise_Src_Files})

set(Elise_Src_Files
	${Elise_Src_Files}
	${SamplesLibElise_Src_Files}
)
