#set(ELISE_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(SAN_DIR "SAN")
set(AMD_DIR "AMD")
set(XERES_DIR "Xeres")
set(HASSA_DIR "HassanArrangt")
set(XINTER_DIR "xinterf")
set(VECTO_DIR "vectorize")
set(UTIL_DIR "util")
set(DocEx "doc_ex")
set(TDPPMD_DIR "TpMMPD")
set(TNR_DIR "TNR")
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
set(SATELIB_DIR "SateLib")

set(YANN_DIR "Yann")

include(${AMD_DIR}/Sources.cmake)
include(${XERES_DIR}/Sources.cmake)
include(${HASSA_DIR}/Sources.cmake)
include(${SAN_DIR}/Sources.cmake)
include(${SIMULPH_DIR}/Sources.cmake)
include(${XMLGEN_DIR}/Sources.cmake)
include(${ALOG_SPE_DIR}/Sources.cmake)
include(${API_DIR}/Sources.cmake)
include(${BITM_DIR}/Sources.cmake)
include(${COMPRE_DIR}/Sources.cmake)
include(${CORREL_DIR}/Sources.cmake)
include(${FILE_IMAGE_DIR}/Sources.cmake)
include(${FLUX_PTS_DIR}/Sources.cmake)
include(${FONC_NUM_DIR}/Sources.cmake)
include(${GEOM2D_DIR}/Sources.cmake)
include(${GEOM3D_DIR}/Sources.cmake)
include(${HOUGH_DIR}/Sources.cmake)
include(${MEMORY_DIR}/Sources.cmake)
include(${MORPHO_DIR}/Sources.cmake)
#include(${MULLGES_DIR}/Sources.cmake)
include(${NEIGH_DIR}/Sources.cmake)
include(${OP_BUF_DIR}/Sources.cmake)
include(${OPERATOR_DIR}/Sources.cmake)
include(${OPTIM_DIR}/Sources.cmake)
include(${UTI_FILES_DIR}/Sources.cmake)
include(${SAMPLESLIBELISE_DIR}/Sources.cmake)
include(${UTI_IMAGE_DIR}/Sources.cmake)
include(${YANN_DIR}/Sources.cmake)
include(${UTI_PHGRM_DIR}/Sources.cmake)
include(${SATELIB_DIR}/Sources.cmake)
include(${ORIPHO_DIR}/Sources.cmake)
include(${OUTPUT_DIR}/Sources.cmake)
include(${PHOTOGR_DIR}/Sources.cmake)
include(${PLOTTER_DIR}/Sources.cmake)
include(${PSC_DIR}/Sources.cmake)
include(${RADIOM_DIR}/Sources.cmake)
include(${RAS_DIR}/Sources.cmake)
include(${RECIPES_DIR}/Sources.cmake)
include(${TIFF_DIR}/Sources.cmake)
include(${TOP_DIR}/Sources.cmake)
include(${UTIL_DIR}/Sources.cmake)
include(${TDPPMD_DIR}/Sources.cmake)
include(${TNR_DIR}/Sources.cmake)
include(${DocEx}/Sources.cmake)
include(${VECTO_DIR}/Sources.cmake)
include(${XINTER_DIR}/Sources.cmake)

if(${WITH_ETALONPOLY})
	include(${UTI_ETAPOLY_DIR}/Sources.cmake)
	set(LIB_ETAL_POLYG libEtalPolyg)
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

# OPTION BUILD : En tetes precompiles pour Visual Studio
if (${WITH_HEADER_PRECOMP} AND MSVC)
	enable_precompiled_headers_msvc(StdAfx.h Elise_Src_Files)
endif()

# fichiers qui n'utilisent pas les en tetes precompilées
set(Elise_Src_Files
	${Elise_Src_Files}
	${TIFF_DIR}/el_dcraw.c
	${OPTIM_DIR}/opt_mat_creuse.cpp
	${OPTIM_DIR}/opt_sysl2.cpp
	${MULLGES_DIR}/mubasic.cpp
	${MULLGES_DIR}/muflaguer.cpp
	${MULLGES_DIR}/mufmueller.cpp
	${MULLGES_DIR}/muvmblock.cpp
	${ORIPHO_DIR}/cOriMntCarto.cpp
)

if(WINDOWS)
	set(Elise_Src_Files ${Elise_Src_Files} ${UTIL_DIR}/win_regex.c)
endif()

if(CMAKE_COMPILER_IS_GNUCC)
    execute_process(COMMAND ${CMAKE_C_COMPILER} -dumpversion OUTPUT_VARIABLE GCC_VERSION)
    if(GCC_VERSION VERSION_GREATER 4.8 OR GCC_VERSION VERSION_EQUAL 4.8)
    	set_source_files_properties(${TIFF_DIR}/el_dcraw.c PROPERTIES COMPILE_FLAGS -Wno-aggressive-loop-optimizations)
    endif()
endif()

if(QT_ENABLED)
	set(SAISIE_DIR ${PROJECT_SOURCE_DIR}/src/saisieQT)
	include(${SAISIE_DIR}/Elise_QT.cmake)
endif()

set(libElise elise)

# CONSTRUCTION : Lib ELISE
if(${CUDA_ENABLED})
    include(GpGpu.cmake)

    cuda_add_library( ${libElise} ${Elise_Src_Files} ${IncFiles} ${QT_ALLFILES} OPTIONS ${GENCODE_SM})

	if(${CUDA_NVTOOLS})
		target_link_libraries(${libElise} ${libStatGpGpuTools} ${libStatGpGpuInterfMicMac} ${libStatGpGpuOpt} ${CUDA_nvToolsExt_LIBRARY} ${OPENCL_LIBRARY})
	else()
		target_link_libraries(${libElise} ${libStatGpGpuTools} ${libStatGpGpuInterfMicMac} ${libStatGpGpuOpt} ${OPENCL_LIBRARY})
	endif()

else()
	add_library(${libElise} ${Elise_Src_Files} ${QT_ALLFILES})
endif()

if(QT_ENABLED)
	target_link_libraries(${libElise} Qt5::Widgets Qt5::Core Qt5::Gui Qt5::Xml Qt5::Concurrent Qt5::OpenGL)
	add_subdirectory(saisieQT)
endif()

install(TARGETS ${libElise} ARCHIVE DESTINATION ${BUILD_PATH_LIB})

if(${WITH_HEADER_PRECOMP} AND (CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_IS_CLANG))
	if(CMAKE_OSX_SYSROOT) # for macOS > 10.14, it contains the path to the SDK/headers. checking this fixes issue #113
		enable_precompiled_headers_GCC(StdAfx.h ${libElise} "-fPIC -std=gnu++11 -isysroot ${CMAKE_OSX_SYSROOT}")
	else()
		enable_precompiled_headers_GCC(StdAfx.h ${libElise} "-fPIC -std=gnu++11")
	endif()
	set(precompiled_header ${precompiled_header} PARENT_SCOPE)
endif()

# INSTALLATION : Dossier de destination lors de l'install
# install(TARGETS elise ARCHIVE DESTINATION ${Install_Dir})

add_subdirectory(CBinaires)

if(WITH_INTERFACE)
	add_subdirectory(interface)
endif()

if (QT_ENABLED AND ${WITH_GIMMI})
	set(GIMMI_DIR ${PROJECT_SOURCE_DIR}/src/GIMMI)
    add_subdirectory(GIMMI)
    set(GIMMI_ENABLED 1)
endif()

set(Elise_Src_Files ${Elise_Src_Files} PARENT_SCOPE)
set(saisieqt_files ${saisieqt_files} PARENT_SCOPE)


