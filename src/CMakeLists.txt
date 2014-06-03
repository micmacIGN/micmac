#SET (ELISE_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

set(SAN_DIR "SAN")
set(AMD_DIR "AMD")
set(HASSA_DIR "HassanArrangt")
set(XINTER_DIR "xinterf")
set(VECTO_DIR "vectorize")
set(UTIL_DIR "util")
set(DocEx "doc_ex")
set(TDPPMD_DIR "TpMMPD")
set(TIFF_DIR "tiff")
set(SIMULPH_DIR "SimulPhgr")
set(PSC_DIR "post_script")
set(PHOTOGR_DIR "photogram")
set(OPERATOR_DIR "operator")
set(ORIPHO_DIR "ori_phot")
set(OUTPUT_DIR "output")
set(XMLGEN_DIR "XML_GEN")
set(RECIPES_DIR "recipes")
set(RAS_DIR "ras_vect")
set(RADIOM_DIR "radiom")
set(PLOTTER_DIR "plotter")
set(TOP_DIR "top_level")
set(ALOG_SPE_DIR "algo_speciaux")
set(API_DIR "api")
set(BITM_DIR "bitm")
set(COMPRE_DIR "compression")
set(CORREL_DIR "correl")
set(FILE_IMAGE_DIR "file_image")
set(FLUX_PTS_DIR "flux_pts")
set(FONC_NUM_DIR "fonc_num")
set(GEOM2D_DIR "geom2d")
set(GEOM3D_DIR "geom3d")
set(HOUGH_DIR "hough")
set(MEMORY_DIR "memory")
set(MORPHO_DIR "morpho_cabl")
set(MULLGES_DIR "mullgesuhlig")
set(NEIGH_DIR "neighboor")
set(OP_BUF_DIR "op_buf")
set(OPTIM_DIR "optim")
set(SAMPLESLIBELISE_DIR "SamplesLibElise")
set(UTI_FILES_DIR "uti_files")
set(UTI_IMAGE_DIR "uti_image")
set(UTI_PHGRM_DIR "uti_phgrm")
set(UTI_ETAPOLY_DIR "EtalonnagePolygone")
set(POISSON_DIR "poisson")

INCLUDE (${AMD_DIR}/Sources.cmake)
INCLUDE (${HASSA_DIR}/Sources.cmake)
INCLUDE (${SAN_DIR}/Sources.cmake)
INCLUDE (${SIMULPH_DIR}/Sources.cmake)
INCLUDE (${XMLGEN_DIR}/Sources.cmake)
INCLUDE (${ALOG_SPE_DIR}/Sources.cmake)
INCLUDE (${API_DIR}/Sources.cmake)
INCLUDE (${BITM_DIR}/Sources.cmake)
INCLUDE (${COMPRE_DIR}/Sources.cmake)
INCLUDE (${CORREL_DIR}/Sources.cmake)
INCLUDE (${FILE_IMAGE_DIR}/Sources.cmake)
INCLUDE (${FLUX_PTS_DIR}/Sources.cmake)
INCLUDE (${FONC_NUM_DIR}/Sources.cmake)
INCLUDE (${GEOM2D_DIR}/Sources.cmake)
INCLUDE (${GEOM3D_DIR}/Sources.cmake)
INCLUDE (${HOUGH_DIR}/Sources.cmake)
INCLUDE (${MEMORY_DIR}/Sources.cmake)
INCLUDE (${MORPHO_DIR}/Sources.cmake)
#INCLUDE (${MULLGES_DIR}/Sources.cmake)
INCLUDE (${NEIGH_DIR}/Sources.cmake)
INCLUDE (${OP_BUF_DIR}/Sources.cmake)
INCLUDE (${OPERATOR_DIR}/Sources.cmake)
INCLUDE (${OPTIM_DIR}/Sources.cmake)
INCLUDE (${UTI_FILES_DIR}/Sources.cmake)
INCLUDE (${SAMPLESLIBELISE_DIR}/Sources.cmake)
INCLUDE (${UTI_IMAGE_DIR}/Sources.cmake)
INCLUDE (${UTI_PHGRM_DIR}/Sources.cmake)
INCLUDE (${ORIPHO_DIR}/Sources.cmake)
INCLUDE (${OUTPUT_DIR}/Sources.cmake)
INCLUDE (${PHOTOGR_DIR}/Sources.cmake)
INCLUDE (${PLOTTER_DIR}/Sources.cmake)
INCLUDE (${PSC_DIR}/Sources.cmake)
INCLUDE (${RADIOM_DIR}/Sources.cmake)
INCLUDE (${RAS_DIR}/Sources.cmake)
INCLUDE (${RECIPES_DIR}/Sources.cmake)
INCLUDE (${TIFF_DIR}/Sources.cmake)
INCLUDE (${TOP_DIR}/Sources.cmake)
INCLUDE (${UTIL_DIR}/Sources.cmake)
INCLUDE (${TDPPMD_DIR}/Sources.cmake)
INCLUDE (${DocEx}/Sources.cmake)
INCLUDE (${VECTO_DIR}/Sources.cmake)
INCLUDE (${XINTER_DIR}/Sources.cmake)
INCLUDE (${POISSON_DIR}/Sources.cmake)

if(${WITH_ETALONPOLY})
        INCLUDE (${UTI_ETAPOLY_DIR}/Sources.cmake)
        set( LIB_ETAL_POLYG libEtalPolyg )
endif()

# ORGANISATION des en-tetes ELISE et fichiers Xml
file(GLOB_RECURSE IncFiles ${PROJECT_SOURCE_DIR}/include/*.h  )
file(GLOB_RECURSE XmlFiles ${PROJECT_SOURCE_DIR}/include/*.xml  )
file(GLOB_RECURSE QssFiles ${PROJECT_SOURCE_DIR}/src/uti_qt/*.qss  )

list(APPEND IncFiles ${XmlFiles})

list(APPEND IncFiles ${QssFiles})

# Construction de l'arborescence des fichiers INCLUDE
foreach(source_file ${IncFiles})
        get_filename_component(VarName ${source_file}  ABSOLUTE)
        get_filename_component(NameFile ${source_file}  NAME)

        string(REGEX REPLACE "/${NameFile}" "" includeTree ${VarName})
        string(REGEX REPLACE "${PROJECT_SOURCE_DIR}/" "" includeTree ${includeTree})
        string(REGEX REPLACE "/" "\\\\\\\\" includeTree ${includeTree})
        source_group("${includeTree}" FILES ${source_file})
endforeach(source_file)

if(${CUDA_ENABLED})
         file(GLOB_RECURSE IncFilesGpGpu ${PROJECT_SOURCE_DIR}/include/GpGpu/*.h  )
         list(REMOVE_ITEM IncFiles ${IncFilesGpGpu})
endif()

# OPTION BUILD : En tetes precompiles visual studio

enable_precompiled_headers_msvc(StdAfx.h Elise_Src_Files)

# fichiers qui n'utilisent pas les en tetes precompilées
set(Elise_Src_Files
        ${Elise_Src_Files}
        ${UTIL_DIR}/win_regex.c
        ${TIFF_DIR}/el_dcraw.c
        ${OPTIM_DIR}/opt_mat_creuse.cpp
        ${OPTIM_DIR}/opt_sysl2.cpp
        ${MULLGES_DIR}/mubasic.cpp
        ${MULLGES_DIR}/muflaguer.cpp
        ${MULLGES_DIR}/mufmueller.cpp
        ${MULLGES_DIR}/muvmblock.cpp
        ${ORIPHO_DIR}/cOriMntCarto.cpp
)

set(libElise elise)

if ( NOT ${qt_version} EQUAL 0 )

    set(SAISIE_DIR ${PROJECT_SOURCE_DIR}/src/saisieQT)

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

    set( HEADERS_nomoc
       ${SAISIE_DIR}/HistoryManager.h
       ${SAISIE_DIR}/MatrixManager.h
       ${SAISIE_DIR}/Cloud.h
       ${SAISIE_DIR}/saisieQT_main.h
       ${SAISIE_DIR}/Data.h
       #SaisieGlsl.glsl
       ${SAISIE_DIR}/Engine.h
       ${SAISIE_DIR}/GLWidgetSet.h
       ${SAISIE_DIR}/3DObject.h )

    set( HEADERS_tomoc
       ${SAISIE_DIR}/GLWidget.h
       ${SAISIE_DIR}/saisieQT_window.h
       ${SAISIE_DIR}/ContextMenu.h
       ${SAISIE_DIR}/Settings.h
       ${SAISIE_DIR}/QT_interface_Elise.h
       ${SAISIE_DIR}/Tree.h
    )

    set (ui_toWrap
     ${SAISIE_DIR}/saisieQT_window.ui
     ${SAISIE_DIR}/Settings.ui
    )

   if ( ${qt_version} EQUAL 5 )

      set(CMAKE_INCLUDE_CURRENT_DIR ON)

      if ( NOT ${OPENGL_GLU_FOUND} )
              message( FATAL_ERROR "GLU library could not be found" )
              return( -1 )
      endif()

      set(CMAKE_AUTOMOC ON)

      set(Saisie_DIR "saisieQT")

      QT5_ADD_RESOURCES( RC_SRCS ${Saisie_DIR}/icones/icones.qrc )

      qt5_wrap_ui(saisie_ui ${ui_toWrap})

      add_definitions(-DTWEAK)
      if ( WIN32 )
         add_definitions(-DELISE_windows)
      ENDIF()

      find_package(Qt5Core REQUIRED)

      if( Qt5Core_FOUND )

        include_directories(${Qt5Widgets_INCLUDE_DIRS})
        include_directories(${Qt5Core_INCLUDE_DIRS})
        include_directories(${Qt5Concurrent_INCLUDE_DIRS})
        include_directories(${Qt5OpenGL_INCLUDE_DIRS})
        include_directories(${Qt5Xml_INCLUDE_DIRS})
        include_directories(${Qt5Gui_INCLUDE_DIRS})

         # Use the compile definitions defined in the Qt 5 Widgets module
         add_definitions(${Qt5Widgets_DEFINITIONS})

        # Add compiler flags for building executables (-fPIE)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${Qt5Widgets_EXECUTABLE_COMPILE_FLAGS}")

      endif()

   elseif ( ${qt_version} EQUAL 4 )
      QT4_WRAP_UI(saisie_ui ${ui_toWrap})
      QT4_WRAP_CPP(HEADERS_moced ${HEADERS_tomoc} )

      INCLUDE(${QT_USE_FILE})
      include_directories( ${PROJECT_BINARY_DIR} )
      INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR})
   endif()

endif()

# CONSTRUCTION : Lib ELISE

if(${CUDA_ENABLED})
    INCLUDE (GpGpu.cmake)

    cuda_add_library( ${libElise} ${Elise_Src_Files} ${IncFiles} ${vmm_SRCS} ${RC_SRCS} ${HEADERS_moced} ${saisie_ui} OPTIONS ${GENCODE_SM})

    target_link_libraries(${libElise} ${libStatGpGpuTools} ${libStatGpGpuInterfMicMac} ${libStatGpGpuOpt} ${CUDA_nvToolsExt_LIBRARY})

else()

    add_library( ${libElise}  ${Elise_Src_Files} ${IncFiles} ${vmm_SRCS} ${RC_SRCS} ${HEADERS_moced} ${saisie_ui})

endif()

target_link_libraries(${libElise} ${OPENGL_LIBRARIES} ${QT_LIBRARIES})

if( ${qt_version} EQUAL 5 )
    qt5_use_modules(${libElise} Widgets Core Gui Xml Concurrent OpenGL)
endif()

INSTALL(TARGETS ${libElise} ARCHIVE DESTINATION ${Install_Dir_lib} )


# OPTION BUILD : En tetes precompilees pour gcc
enable_precompiled_headers_GCC(StdAfx.h ${libElise} "${CMAKE_CXX_FLAGS}")


# DEFINITION
add_definitions(-DNO_JPEG -DNO_LCMS)

# INSTALLATION : Dossier de destination lors de l'install
#install(TARGETS elise ARCHIVE DESTINATION ${Install_Dir})

ADD_SUBDIRECTORY(CBinaires)

if (WITH_INTERFACE)
    ADD_SUBDIRECTORY(interface)
endif()

#if (WITH_SAISIEQT)
if ( NOT ${qt_version} EQUAL 0)
    ADD_SUBDIRECTORY(saisieQT)
endif()

if (WITH_MM3dSat)
    ADD_SUBDIRECTORY(mm3dSat)
endif()

