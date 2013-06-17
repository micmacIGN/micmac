set(GpGpuTools_Src_Files
   ${UTI_PHGRM_GPGPU_DIR}/GpGpuTools.cpp
   #${UTI_PHGRM_GPGPU_DIR}/data2Optimize.cpp
)

set(uti_phgrm_GpGpu_Src_Files
    ${UTI_PHGRM_GPGPU_DIR}/InterfaceMicMacGpGpu.cpp
    ${UTI_PHGRM_GPGPU_DIR}/cudaAppliMicMac.cu
    ${UTI_PHGRM_GPGPU_DIR}/cudaFilters.cu
    #${UTI_PHGRM_GPGPU_DIR}/cudaTextureTools.cu
)

set(uti_phgrm_Opt_GpGpu_Src_Files
	 ${UTI_PHGRM_GPGPU_DIR}/GpGpuOptimisation.cu
	 ${UTI_PHGRM_GPGPU_DIR}/GpGpuOptimisation.cpp	 
)

set(uti_Test_Opt_GpGpu_Src_Files
	 ${UTI_PHGRM_GPGPU_DIR}/TestGpGpuOptimisation.cpp
	 ${UTI_PHGRM_GPGPU_DIR}/GpGpuMultiThreadingCpu.cpp
)
