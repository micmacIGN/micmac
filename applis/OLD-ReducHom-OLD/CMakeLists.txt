file(GLOB
     ALL_SRC
     "*.cpp"
)

add_executable(ReducHom
	CalcHcImage.cpp
	cAppliReduc.cpp
        cImagH.cpp
        cPtHom.cpp
        IndexAppli.cpp
        ReducHom.cpp
        ReducHom.h
#        SimulMerge.cpp
)

target_link_libraries(ReducHom elise ${X11_LIBRARIES})

INSTALL(TARGETS ReducHom RUNTIME DESTINATION ${Install_Dir})

	
