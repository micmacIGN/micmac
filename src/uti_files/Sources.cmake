set(uti_files_Src_Files
    ${UTI_FILES_DIR}/CPP_XifGps2Xml.cpp
    ${UTI_FILES_DIR}/CPP_MapCmd.cpp
    ${UTI_FILES_DIR}/CPP_BatchFDC.cpp
    ${UTI_FILES_DIR}/CPP_MyRename.cpp
    ${UTI_FILES_DIR}/CPP_cod.cpp
    ${UTI_FILES_DIR}/CPP_GGP_txt2Xml.cpp
    ${UTI_FILES_DIR}/CPP_Ori_txt2Xml.cpp
    ${UTI_FILES_DIR}/CPP_TestKeys.cpp
    ${UTI_FILES_DIR}/CPP_TestMTD.cpp
    ${UTI_FILES_DIR}/CPP_TestCmds.cpp
    ${UTI_FILES_DIR}/CPP_Apero2PMVS.cpp
    ${UTI_FILES_DIR}/CPP_Apero2Meshlab.cpp
    ${UTI_FILES_DIR}/CPP_GenHeadTifTile.cpp
    ${UTI_FILES_DIR}/CPP_Ori2Xml.cpp
    ${UTI_FILES_DIR}/CPP_Prep4masq.cpp
    ${UTI_FILES_DIR}/CPP_GenCode.cpp
    ${UTI_FILES_DIR}/CPP_Xml2Dmp.cpp
    ${UTI_FILES_DIR}/VersionedFileHeader.cpp
    ${UTI_FILES_DIR}/CPP_CheckChantier.cpp
    ${UTI_FILES_DIR}/CPP_Test_Apero2NVM.cpp
    ${UTI_FILES_DIR}/CPP_EditSetRel.cpp
    ${UTI_FILES_DIR}/CPP_GCP2D3D2Xml.cpp
)

source_group(uti_files FILES ${uti_files_Src_Files})

set(Elise_Src_Files
        ${Elise_Src_Files}
        ${uti_files_Src_Files}
)
