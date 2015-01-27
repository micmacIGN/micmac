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
set(POISSON_DIR ${PROJECT_SOURCE_DIR}/CodeExterne/Poisson/)

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

if(CMAKE_COMPILER_IS_GNUCC)

    execute_process(COMMAND ${CMAKE_C_COMPILER} -dumpversion OUTPUT_VARIABLE GCC_VERSION)

    if(GCC_VERSION VERSION_GREATER 4.8 OR GCC_VERSION VERSION_EQUAL 4.8)
      set_source_files_properties(${TIFF_DIR}/el_dcraw.c PROPERTIES COMPILE_FLAGS -Wno-aggressive-loop-optimizations)
    endif()

endif()

set(libElise elise)

set(SAISIE_DIR ${PROJECT_SOURCE_DIR}/src/saisieQT)

include(${SAISIE_DIR}/Elise_QT.cmake)

# CONSTRUCTION : Lib ELISE

if(${CUDA_ENABLED})
    INCLUDE (GpGpu.cmake)

    cuda_add_library( ${libElise} ${Elise_Src_Files} ${IncFiles} ${vmm_SRCS} ${RC_SRCS} ${HEADERS_moced} ${HEADERS_tomoc} ${HEADERS_nomoc} ${saisie_ui} OPTIONS ${GENCODE_SM})
	
        if(${CUDA_NVTOOLS})
		target_link_libraries(${libElise} ${libStatGpGpuTools} ${libStatGpGpuInterfMicMac} ${libStatGpGpuOpt} ${CUDA_nvToolsExt_LIBRARY} ${OPENCL_LIBRARY})
	else()
		target_link_libraries(${libElise} ${libStatGpGpuTools} ${libStatGpGpuInterfMicMac} ${libStatGpGpuOpt} ${OPENCL_LIBRARY})
	endif()

else()

    add_library( ${libElise}  ${Elise_Src_Files} ${IncFiles} ${vmm_SRCS} ${RC_SRCS} ${HEADERS_moced} ${HEADERS_tomoc} ${HEADERS_nomoc} ${saisie_ui})

endif()

target_link_libraries(${libElise} ${OPENGL_LIBRARIES} ${QT_LIBRARIES})

#if( ${qt_version} EQUAL 5 )
#    qt5_use_modules(${libElise} Widgets Core Gui Xml Concurrent OpenGL)
#endif()

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
