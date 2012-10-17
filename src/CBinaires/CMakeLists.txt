add_executable(mm3d mm3d.cpp)
target_link_libraries(mm3d elise ${X11_LIBRARIES})
INSTALL(TARGETS mm3d RUNTIME DESTINATION ${Install_Dir})

	
add_executable(AperiCloud  AperiCloud.c)
target_link_libraries(AperiCloud   m)
INSTALL(TARGETS AperiCloud RUNTIME DESTINATION ${Install_Dir})

add_executable(Malt  Malt.c)
target_link_libraries(Malt   m)
INSTALL(TARGETS Malt RUNTIME DESTINATION ${Install_Dir})


