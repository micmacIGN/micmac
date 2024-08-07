# prevents: dereferencing type-punned pointer will break strict-aliasing rules
file(GLOB srcs CONFIGURE_DEPENDS "*.cpp")
file(GLOB headers "*.h")

set_source_files_properties(${srcs} PROPERTIES COMPILE_FLAGS "-O1")

add_library(TrianglePPobjs OBJECT ${srcs})
