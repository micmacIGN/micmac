set(code_genere_file2string_dir ${code_genere_dir}/File2String)
#set(code_genere_photogram_dir ${code_genere_dir}/photogram)

list( APPEND code_genere_file2string_Src_Files
    ${code_genere_file2string_dir}/Str_SuperposImage.cpp
    ${code_genere_file2string_dir}/Str_ParamMICMAC.cpp
    ${code_genere_file2string_dir}/Str_ParamChantierPhotogram.cpp
    ${code_genere_file2string_dir}/Str_ParamApero.cpp
    ${code_genere_file2string_dir}/Str_ParamDigeo.cpp
    ${code_genere_file2string_dir}/Str_DefautChantierDescripteur.cpp
    ${code_genere_file2string_dir}/cParamXMLNew0.cpp
)

source_group(CodeGenere FILES ${code_genere_file2string_Src_Files})

list( APPEND Elise_Src_Files
	${code_genere_file2string_Src_Files}
)
