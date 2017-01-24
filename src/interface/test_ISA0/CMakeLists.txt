add_executable(test_ISA0 TestIsabelle.cpp
						 include_isa/Affichage.cpp
						 include_isa/Coord.cpp
#						 include_isa/exemples.cpp
						 include_isa/Filtres.cpp
						 include_isa/Image.cpp
						 include_isa/PaireImg.cpp
						 include_isa/Point.cpp
						 include_isa/addon_ParamChantierPhotogram.cpp
						 include_isa/Str_addon_ParamChantierPhotogram.cpp)
target_link_libraries(test_ISA0 ${OPENGL_LIBRARIES}
                                ${QT_LIBRARIES}
                                elise
                                ${X11_LIBRARIES}
                                ${WINSOCK2_LIBRAIRIE})
INSTALL(TARGETS test_ISA0 RUNTIME DESTINATION ${Install_Dir})
