set(Compression_Src_Files
	${COMPRE_DIR}/huffman.cpp
	${COMPRE_DIR}/lzw.cpp
	${COMPRE_DIR}/pack_bits.cpp
	${COMPRE_DIR}/range_code.cpp
)

source_group(Compression FILES ${Compression_Src_Files})

set(Elise_Src_Files
	${Elise_Src_Files}
	${Compression_Src_Files}
)
