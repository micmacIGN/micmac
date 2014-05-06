 file(GLOB_RECURSE IncuhCudaFiles ${PROJECT_SOURCE_DIR}/include/GpGpu/*.cuh  )
 file(GLOB_RECURSE IncCudaFiles ${PROJECT_SOURCE_DIR}/include/GpGpu/*.h  )
 list(APPEND IncCudaFiles ${IncuhCudaFiles})


execute_process( COMMAND "${CUDA_NVCC_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/src/uti_phgrm/GpGpu/tools/FoundCapa.cu" "--run"
                 WORKING_DIRECTORY "${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CMakeTmp/"
                 RESULT_VARIABLE _resultNVCC OUTPUT_VARIABLE _outNVCC
                 ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)


if(NOT _resultNVCC EQUAL 0)
      message(STATUS "Error Cuda")
else()
      set(_cudaArch "${_outNVCC}")
      message("Cuda capabilities = ${_cudaArch}")
#      string(REPLACE "2.1" "2.1(2.0)" _cudaArch "${_cudaArch}")
      #message("${_cudaArch}")
endif()

string(FIND "${_cudaArch}" "2.1" arch_21)
string(FIND "${_cudaArch}" "2.0" arch_20)
string(FIND "${_cudaArch}" "3.0" arch_30)
string(FIND "${_cudaArch}" "3.5" arch_35)

if(${CUDA_LINEINFO})
    set(flag_Lineinfo  -lineinfo)
endif()

if(${CUDA_FASTMATH})
    set(flag_fastMath  -use_fast_math)
endif()

if((NOT ${arch_20} LESS 0) OR (NOT ${arch_21} LESS 0))

    set(cuda_arch_version 20 )

elseif((NOT ${arch_30} LESS 0))

    set(cuda_arch_version 30 )

elseif((NOT ${arch_30} LESS 0))

    set(cuda_arch_version 35 )

else()

    message("Cuda capabilities are not sufficient")

endif()

 set(GENCODE_SM -gencode=arch=compute_${cuda_arch_version},code=sm_${cuda_arch_version} ${flag_Lineinfo} ${flag_fastMath})

 set(libStatGpGpuTools GpGpuTools)
 set(libStatGpGpuInterfMicMac GpGpuInterfMicMac)
 set(libStatGpGpuOpt GpGpuOpt)

 find_cuda_helper_libs(nvToolsExt)

 cuda_add_library(${libStatGpGpuTools}  ${GpGpuTools_Src_Files} ${IncCudaFiles} STATIC OPTIONS ${GENCODE_SM})

 target_link_libraries(${libStatGpGpuTools} ${CUDA_nvToolsExt_LIBRARY})

 cuda_add_library(${libStatGpGpuInterfMicMac}  ${uti_phgrm_GpGpu_Src_Files} STATIC OPTIONS ${GENCODE_SM})

 cuda_add_library(${libStatGpGpuOpt}  ${uti_phgrm_Opt_GpGpu_Src_Files} ${IncCudaFiles} STATIC OPTIONS ${GENCODE_SM})

 if (Boost_FOUND)
          target_link_libraries(${libStatGpGpuInterfMicMac} ${libStatGpGpuTools} ${Boost_LIBRARIES} ${Boost_THREADAPI})
          if (NOT WIN32)
                target_link_libraries(${libStatGpGpuInterfMicMac}  rt pthread )
          endif()
 endif()

 set(GpGpu_UnitTesting GpGpuUnitTesting)
 cuda_add_executable(${GpGpu_UnitTesting} ${uti_Test_Opt_GpGpu_Src_Files})

 target_link_libraries(${GpGpu_UnitTesting}  ${libStatGpGpuInterfMicMac} ${libStatGpGpuOpt})

 if (NOT WIN32)
        target_link_libraries(${GpGpu_UnitTesting}  rt pthread )
 endif()
 INSTALL(TARGETS ${GpGpu_UnitTesting} RUNTIME DESTINATION ${Install_Dir})

link_directories(${PROJECT_SOURCE_DIR}/lib/)

if(${CUDA_ENABLED})
         file(GLOB_RECURSE IncFilesGpGpu ${PROJECT_SOURCE_DIR}/include/GpGpu/*.h  )
         list(REMOVE_ITEM IncFiles ${IncFilesGpGpu})
endif()

 cuda_add_library( ${libElise} ${Elise_Src_Files} ${IncFiles} OPTIONS ${GENCODE_SM})

 target_link_libraries(${libElise} ${libStatGpGpuTools} ${libStatGpGpuInterfMicMac} ${libStatGpGpuOpt} ${CUDA_nvToolsExt_LIBRARY})

 INSTALL(TARGETS ${libStatGpGpuTools} ${libStatGpGpuInterfMicMac} ${libStatGpGpuOpt}
            LIBRARY DESTINATION ${PROJECT_SOURCE_DIR}/lib
            ARCHIVE DESTINATION ${PROJECT_SOURCE_DIR}/lib)
