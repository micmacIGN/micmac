set(Opera_Src_Files
	operator/assoc_mixte.cpp
	operator/binarie_int.cpp
	operator/binarie_mixte.cpp
	operator/colour.cpp
	operator/compar_op.cpp
	operator/complex.cpp
	operator/func_opbinmix_tpl.cpp
	operator/func_opun_tpl.cpp
	operator/inst_users_op.cpp
	operator/math_op.cpp
	operator/op_bin_ndim.cpp
	operator/op_def.cpp
	operator/oper_naire.cpp
	operator/users_op.cpp
)

source_group(Operator FILES ${Opera_Src_Files})

set(Elise_Src_Files
	${Elise_Src_Files}
	${Opera_Src_Files}
)
