set(Flux_pts_Src_Files
	${FLUX_PTS_DIR}/curser_pack_pts.cpp
	${FLUX_PTS_DIR}/flx_gen_tpl.cpp
	${FLUX_PTS_DIR}/flx_general.cpp
	${FLUX_PTS_DIR}/flx_oper.cpp
	${FLUX_PTS_DIR}/front_to_surf.cpp
	${FLUX_PTS_DIR}/lineique_2d.cpp
	${FLUX_PTS_DIR}/rle_pack.cpp
	${FLUX_PTS_DIR}/select.cpp
	${FLUX_PTS_DIR}/surf_geom_2d.cpp
)

source_group(Flux_pts FILES ${Flux_pts_Src_Files})

set(Elise_Src_Files
	${Elise_Src_Files}
	${Flux_pts_Src_Files}
)
