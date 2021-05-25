set(Optim_Src_Files
	${OPTIM_DIR}/cox_roy.cpp
	${OPTIM_DIR}/func_mean_square.cpp
	${OPTIM_DIR}/nappes.cpp
	${OPTIM_DIR}/opt_cube_flux.cpp
	${OPTIM_DIR}/opt_dmr_f1v.cpp
	${OPTIM_DIR}/opt_elgrowingsetind.cpp
	${OPTIM_DIR}/opt_forml1.cpp
	#${OPTIM_DIR}/opt_mat_creuse.cpp
	${OPTIM_DIR}/opt_nvar.cpp
	${OPTIM_DIR}/opt_somme_formelle.cpp
	${OPTIM_DIR}/opt_sysl2.cpp
	${OPTIM_DIR}/opt_syssuresolu.cpp
	${OPTIM_DIR}/optim_etiq_binaire.cpp
)

source_group(Optim FILES ${Optim_Src_Files})

set(Elise_Src_Files
	${Elise_Src_Files}
	${Optim_Src_Files}
)
