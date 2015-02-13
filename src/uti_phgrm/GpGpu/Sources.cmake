set(GpGpuTools_Src_Files
         ${UTI_PHGRM_GPGPU_DIR}/GpGpu_Object.cpp
         ${UTI_PHGRM_GPGPU_DIR}/GpGpu_Data.cpp
         ${UTI_PHGRM_GPGPU_DIR}/GpGpu_Tools.cpp
         ${UTI_PHGRM_GPGPU_DIR}/GpGpu_MultiThreadingCpu.cpp
)

set(uti_phgrm_GpGpu_Src_Files
         ${UTI_PHGRM_GPGPU_DIR}/GpGpu_InterfaceCorrel.cpp
         ${UTI_PHGRM_GPGPU_DIR}/GpGpu_Interface_CorMultiScale.cpp
	 ${UTI_PHGRM_GPGPU_DIR}/SData2Correl.cpp
         ${UTI_PHGRM_GPGPU_DIR}/GpGpu_Cuda_Correlation.cu
         ${UTI_PHGRM_GPGPU_DIR}/GpGpu_CorMultiScale.cu
	 ${UTI_PHGRM_GPGPU_DIR}/cudaFilters.cu
	 #${UTI_PHGRM_GPGPU_DIR}/GpGpuTextureTools.cu
)

set(uti_phgrm_Opt_GpGpu_Src_Files        
        ${UTI_PHGRM_GPGPU_DIR}/GpGpu_Cuda_Optimisation.cu
        ${UTI_PHGRM_GPGPU_DIR}/GpGpu_InterfaceOptimisation.cpp
)

set(uti_Test_Opt_GpGpu_Src_Files
        ${UTI_PHGRM_GPGPU_DIR}/GpGpu_UnitTestingKernel.cu
        ${UTI_PHGRM_GPGPU_DIR}/GpGpu_UnitTesting.cpp
)
