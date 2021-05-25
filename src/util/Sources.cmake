#QT mock visual mode
if(QT_ENABLED)
    set(Uti_Headers_ToMoc
        ../include/general/visual_mainwindow.h
        ../include/general/visual_buttons.h
    )

qt5_wrap_cpp(Uti_HEADERS_MOCED ${Uti_Headers_ToMoc})
endif()

set(Util_Src_Files
        ${UTIL_DIR}/affin2d.cpp
        ${UTIL_DIR}/all.cpp
        ${UTIL_DIR}/arg_main.cpp
        ${UTIL_DIR}/visual_arg_main.cpp
        ${UTIL_DIR}/cMMSpecArg.cpp
        ${UTIL_DIR}/bicub.cpp
        ${UTIL_DIR}/bits_flow.cpp
        ${UTIL_DIR}/box.cpp
        ${UTIL_DIR}/cElStatErreur.cpp
        ${UTIL_DIR}/cEquiv1D.cpp
        ${UTIL_DIR}/cGPAO.cpp
        ${UTIL_DIR}/cSysCoor.cpp
        ${UTIL_DIR}/current_fonc.cpp
        ${UTIL_DIR}/dates.cpp
        ${UTIL_DIR}/error.cpp
        ${UTIL_DIR}/externalToolHandler.cpp
        ${UTIL_DIR}/fifo.cpp
        ${UTIL_DIR}/files.cpp
        ${UTIL_DIR}/num.cpp
        ${UTIL_DIR}/num_tpl.cpp
        ${UTIL_DIR}/pt2di.cpp
        ${UTIL_DIR}/randomm.cpp
        ${UTIL_DIR}/regex.cpp
        ${UTIL_DIR}/sort.cpp
        ${UTIL_DIR}/string_dyn.cpp
        ${UTIL_DIR}/stringifie.cpp
        ${UTIL_DIR}/tabul.cpp
        ${UTIL_DIR}/prime_test.cpp 
        ${UTIL_DIR}/wildmatch.cpp
        ${UTIL_DIR}/xml.cpp
        ${UTIL_DIR}/xml2cpp.cpp
        ${UTIL_DIR}/parseline.cpp
        ${UTIL_DIR}/TD_Sol.cpp
        ${UTIL_DIR}/cElCommand.cpp
        #${UTIL_DIR}/win_regex.c
        ${UTIL_DIR}/visual_mainwindow.cpp
        ${UTIL_DIR}/visual_buttons.cpp
        ${UTIL_DIR}/errors.cpp
        ${UTIL_DIR}/MessageHandler.cpp
        ${UTIL_DIR}/GIT_defines.cpp
        ${UTIL_DIR}/PlyFile.cpp
)

source_group(Util FILES ${Util_Src_Files})

set(Elise_Src_Files
        ${Elise_Src_Files}
        ${Util_Src_Files}
)

