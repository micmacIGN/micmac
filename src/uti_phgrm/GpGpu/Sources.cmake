set(GpGpuTools_Src_Files
         ${UTI_PHGRM_GPGPU_DIR}/GpGpuObject.cpp
         ${UTI_PHGRM_GPGPU_DIR}/GpGpuData.cpp
	 ${UTI_PHGRM_GPGPU_DIR}/GpGpuTools.cpp
	 ${UTI_PHGRM_GPGPU_DIR}/GpGpuMultiThreadingCpu.cpp
)

set(uti_phgrm_GpGpu_Src_Files
	 ${UTI_PHGRM_GPGPU_DIR}/GpGpuInterfaceCorrel.cpp
	 ${UTI_PHGRM_GPGPU_DIR}/SData2Correl.cpp
	 ${UTI_PHGRM_GPGPU_DIR}/GpGpuCudaCorrelation.cu
	 ${UTI_PHGRM_GPGPU_DIR}/cudaFilters.cu
	 #${UTI_PHGRM_GPGPU_DIR}/GpGpuTextureTools.cu
)

set(uti_phgrm_Opt_GpGpu_Src_Files
         ${UTI_PHGRM_GPGPU_DIR}/GpGpuCudaOptimisation.cu
	 ${UTI_PHGRM_GPGPU_DIR}/GpGpuInterfaceOptimisation.cpp
)

set(uti_Test_Opt_GpGpu_Src_Files	 
	 ${UTI_PHGRM_GPGPU_DIR}/TestGpGpuInterfaceOptimisation.cpp
)
