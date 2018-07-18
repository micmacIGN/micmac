set(Vecto_Src_Files
	${VECTO_DIR}/aprrox_poly.cpp
	${VECTO_DIR}/cont_vect.cpp
	${VECTO_DIR}/inermats.cpp
	${VECTO_DIR}/sk_vect.cpp
)

source_group(Vectorize FILES ${Vecto_Src_Files})

set(Elise_Src_Files
	${Elise_Src_Files}
	${Vecto_Src_Files}
)
