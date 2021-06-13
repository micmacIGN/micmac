set(Plotter_Src_Files
	${PLOTTER_DIR}/draw_plot1d.cpp
	${PLOTTER_DIR}/gen_plot1.cpp
)

source_group(Plotter FILES ${Plotter_Src_Files})

set(Elise_Src_Files
	${Elise_Src_Files}
	${Plotter_Src_Files}
)
