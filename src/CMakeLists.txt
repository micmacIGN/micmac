#SET (ELISE_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

set(SAN_DIR "SAN")
set(AMD_DIR "AMD")
set(HASSA_DIR "HassanArrangt")
set(XINTER_DIR "xinterf")
set(VECTO_DIR "vectorize")
set(UTIL_DIR "util")
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

INCLUDE (${AMD_DIR}/Sources.cmake)

set(Elise_Src_Files
	${Elise_Src_Files}
	../CodeGenere/File2String/Str_DefautChantierDescripteur.cpp
	../CodeGenere/File2String/Str_ParamChantierPhotogram.cpp
	../CodeGenere/File2String/Str_SuperposImage.cpp
)

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
INCLUDE (${VECTO_DIR}/Sources.cmake)
INCLUDE (${XINTER_DIR}/Sources.cmake)

# OPTION BUILD : En tetes precompilées visual studio
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

# CONSTRUCTION : Lib ELISE
add_library( elise ${Elise_Src_Files})

# OPTION BUILD : En tetes precompilées pour gcc
enable_precompiled_headers_GCC(StdAfx.h elise)

# DEFINITION
add_definitions(-DNO_JPEG -DNO_LCMS)

# INSTALLATION : Dossier de destination lors de l'install
install(TARGETS elise ARCHIVE DESTINATION ${Install_Dir})
	
