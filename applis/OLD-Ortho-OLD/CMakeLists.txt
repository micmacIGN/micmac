add_executable(Porto

        Ortho.h
        Porto.cpp
        cOneImOrhto.cpp
        cLoadedIm.cpp
        cAppli_Ortho.cpp
        Egalise.cpp
        Ortho_PC.cpp
)

target_link_libraries(Porto elise ${X11_LIBRARIES})

INSTALL(TARGETS Porto RUNTIME DESTINATION ${Install_Dir})

	
