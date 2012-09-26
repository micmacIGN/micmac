set(XML_GEN_Src_Files
	${XMLGEN_DIR}/ParamChantierPhotogram.cpp
	${XMLGEN_DIR}/SuperposImage.cpp
)

SOURCE_GROUP(${XMLGEN_DIR} FILES ${XML_GEN_Src_Files})

set(Elise_Src_Files
	${Elise_Src_Files}
	${XML_GEN_Src_Files}
)
