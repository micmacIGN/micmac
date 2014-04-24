if( ${qt_version} EQUAL 4 )

    set(Uti_Headers_ToMoc
        ../include/general/visual_mainwindow.h
        ../include/general/visual_buttons.h
    )

set(SAISIE_DIR "../saisieQT")

    set(Ui_ToWrap
        saisieQT/saisieQT_window.ui
        saisieQT/Settings.ui
    )

set(vmm_SRCS    ${SAISIE_DIR}/GLWidget.cpp
                ${SAISIE_DIR}/saisieQT_window.cpp
                ${SAISIE_DIR}/Cloud.cpp
                ${SAISIE_DIR}/Data.cpp
                ${SAISIE_DIR}/Engine.cpp
                ${SAISIE_DIR}/3DObject.cpp
                ${SAISIE_DIR}/GLWidgetSet.cpp
                ${SAISIE_DIR}/MatrixManager.cpp
                ${SAISIE_DIR}/HistoryManager.cpp
                ${SAISIE_DIR}/ContextMenu.cpp
                ${SAISIE_DIR}/Settings.cpp
                ${SAISIE_DIR}/QT_interface_Elise.cpp
                ${SAISIE_DIR}/Tree.cpp)

set( vmm_HEADERS_nomoc
   ${SAISIE_DIR}/HistoryManager.h
   ${SAISIE_DIR}/MatrixManager.h
   ${SAISIE_DIR}/Cloud.h
   ${SAISIE_DIR}/saisieQT_main.h
   ${SAISIE_DIR}/Data.h
   #SaisieGlsl.glsl
   ${SAISIE_DIR}/Engine.h
   ${SAISIE_DIR}/GLWidgetSet.h
   ${SAISIE_DIR}/3DObject.h )

set( vmm_HEADERS_tomoc
   ${SAISIE_DIR}/GLWidget.h
   ${SAISIE_DIR}/saisieQT_window.h
   ${SAISIE_DIR}/ContextMenu.h
   ${SAISIE_DIR}/Settings.h
   ${SAISIE_DIR}/QT_interface_Elise.h
   ${SAISIE_DIR}/Tree.h)

    QT4_WRAP_UI(Ui_VisualMode ${Ui_ToWrap})
    QT4_WRAP_CPP(Uti_HEADERS_MOCED ${Uti_Headers_ToMoc} ${vmm_HEADERS_tomoc})
    add_definitions(${QT_DEFINITIONS})
    INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR})
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
        ${UTIL_DIR}/wildmatch.cpp
        ${UTIL_DIR}/xml.cpp
        ${UTIL_DIR}/xml2cpp.cpp
        ${UTIL_DIR}/parseline.cpp
        ${UTIL_DIR}/TD_Sol.cpp
        ${UTIL_DIR}/cElCommand.cpp
        #${UTIL_DIR}/win_regex.c
        ${UTIL_DIR}/visual_mainwindow.cpp
        ${UTIL_DIR}/visual_buttons.cpp
        saisieQT/saisieQT_window.cpp
        ${Uti_HEADERS_MOCED}
)

SOURCE_GROUP(Util FILES ${Util_Src_Files})

set(Elise_Src_Files
        ${Elise_Src_Files}
        ${Util_Src_Files}
)

