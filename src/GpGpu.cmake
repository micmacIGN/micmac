 file(GLOB_RECURSE IncuhCudaFiles ${PROJECT_SOURCE_DIR}/include/GpGpu/*.cuh  )
 file(GLOB_RECURSE IncCudaFiles ${PROJECT_SOURCE_DIR}/include/GpGpu/*.h  )
 list(APPEND IncCudaFiles ${IncuhCudaFiles})

 #set(GENCODE_SM20 -gencode=arch=compute_20,code=sm_20 -gencode=arch=compute_20,code=compute_20 -use_fast_math)
 #set(GENCODE_SM20 -gencode=arch=compute_20,code=sm_20 -gencode=arch=compute_20,code=compute_20)
set(GENCODE_SM20 -gencode=arch=compute_20,code=sm_20 -lineinfo)
  #set(GENCODE_SM20 -gencode=arch=compute_20,code=sm_20)

 set(libStatGpGpuTools GpGpuTools)
 set(libStatGpGpuInterfMicMac GpGpuInterfMicMac)
 set(libStatGpGpuOpt GpGpuOpt)

 find_cuda_helper_libs(nvToolsExt)

 cuda_add_library(${libStatGpGpuTools}  ${GpGpuTools_Src_Files} ${IncCudaFiles} STATIC OPTIONS ${GENCODE_SM20})

 target_link_libraries(${libStatGpGpuTools} ${CUDA_nvToolsExt_LIBRARY})

 cuda_add_library(${libStatGpGpuInterfMicMac}  ${uti_phgrm_GpGpu_Src_Files} STATIC OPTIONS ${GENCODE_SM20})

 cuda_add_library(${libStatGpGpuOpt}  ${uti_phgrm_Opt_GpGpu_Src_Files} ${IncCudaFiles} STATIC OPTIONS ${GENCODE_SM20})

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

 cuda_add_library( ${libElise} ${Elise_Src_Files} ${IncFiles} OPTIONS ${GENCODE_SM20})

 target_link_libraries(${libElise} ${libStatGpGpuTools} ${libStatGpGpuInterfMicMac} ${libStatGpGpuOpt} ${CUDA_nvToolsExt_LIBRARY})

 INSTALL(TARGETS ${libStatGpGpuTools} ${libStatGpGpuInterfMicMac} ${libStatGpGpuOpt}
            LIBRARY DESTINATION ${PROJECT_SOURCE_DIR}/lib
            ARCHIVE DESTINATION ${PROJECT_SOURCE_DIR}/lib)
