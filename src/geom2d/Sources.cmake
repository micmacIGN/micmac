set(Geom2d_Src_Files
	${GEOM2D_DIR}/Shewchuk_api.cpp
	${GEOM2D_DIR}/Shewchuk_implem.cpp
	${GEOM2D_DIR}/basic.cpp
	${GEOM2D_DIR}/dist.cpp
	${GEOM2D_DIR}/env_conv.cpp
	${GEOM2D_DIR}/gpc.cpp
	${GEOM2D_DIR}/gpc_interf.cpp
	${GEOM2D_DIR}/inter_cerle_losange.cpp
	${GEOM2D_DIR}/intersection.cpp
	${GEOM2D_DIR}/proj.cpp
	${GEOM2D_DIR}/region_plan.cpp
	${GEOM2D_DIR}/seg_comp.cpp
	${GEOM2D_DIR}/triangle_comp.cpp
)

source_group(Geom2d FILES ${Geom2d_Src_Files})

set(Elise_Src_Files
	${Elise_Src_Files}
	${Geom2d_Src_Files}
)
