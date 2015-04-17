set(bench_mm3d_source_directory ${PROJECT_SOURCE_DIR}/BenchElise/bench_mm3d)
add_executable( bench_mm3d
   ${bench_mm3d_source_directory}/VersionedFileHeader.cpp
   ${bench_mm3d_source_directory}/ChunkStream.cpp
   ${bench_mm3d_source_directory}/TracePack.cpp
   ${bench_mm3d_source_directory}/bench_mm3d.cpp
)
target_link_libraries(bench_mm3d elise ${X11_LIBRARIES} ${WINSOCK2_LIBRAIRIE})
INSTALL(TARGETS bench_mm3d RUNTIME DESTINATION ${Install_Dir})
