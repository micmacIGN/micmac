
## Include gpugpu header .h and .cuh
file(GLOB_RECURSE IncuhCudaFiles ${PROJECT_SOURCE_DIR}/include/GpGpu/*.cuh  )
file(GLOB_RECURSE IncCudaFiles ${PROJECT_SOURCE_DIR}/include/GpGpu/*.h  )
list(APPEND IncCudaFiles ${IncuhCudaFiles})

##
if(MSVC12)
	get_filename_component(VS_DIR [HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\VisualStudio\\12.0\\Setup\\VS;ProductDir] REALPATH CACHE)
elseif (MSVC11)
	get_filename_component(VS_DIR [HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\VisualStudio\\11.0\\Setup\\VS;ProductDir] REALPATH CACHE)
elseif (MSVC10)
    get_filename_component(VS_DIR [HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\VisualStudio\\10.0\\Setup\\VS;ProductDir] REALPATH CACHE)
elseif (MSVC90)
    get_filename_component(VS_DIR [HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\VisualStudio\\9.0\\Setup\\VS;ProductDir] REALPATH CACHE)
elseif (MSVC80)
    get_filename_component(VS_DIR [HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\VisualStudio\\8.0\\Setup\\VS;ProductDir] REALPATH CACHE)
endif()

# FoundCapa bin for found capabilities of video cards

# Define variable global for build FoundCapa on windows
if(MSVC12 OR MSVC11 OR MSVC10 OR MSVC90 OR MSVC80)
    set(ENV{PATH} "${VS_DIR}\\VC\\bin;C:\\Windows\\System32;${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}\\CMakeTmp\\")
endif() 

# verif if FoundCapa.exe exists --
if(EXISTS "${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CMakeTmp/FoundCapa.exe")
    # run FoundCapa.exe windows
    execute_process(COMMAND "${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CMakeTmp/FoundCapa.exe"
                        WORKING_DIRECTORY "${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CMakeTmp/"
                        RESULT_VARIABLE _resultNVCC OUTPUT_VARIABLE _outNVCC
                        ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)
else()
# build and run FoundCapa mac and LINUX
    execute_process(COMMAND "${CUDA_NVCC_EXECUTABLE}" "-o=FoundCapa" "${PROJECT_SOURCE_DIR}/src/uti_phgrm/GpGpu/tools/FoundCapa.cu"  "--run" 
                        WORKING_DIRECTORY "${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CMakeTmp/"
                        RESULT_VARIABLE _resultNVCC OUTPUT_VARIABLE _outNVCC
                        ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)
endif()

#message(${_outNVCC}) 

if(NOT _resultNVCC EQUAL 0)
	if(MSVC12 OR MSVC11 OR MSVC10 OR MSVC90 OR MSVC80)    
		set(ENV{PATH} "${VS_DIR}\\VC\\bin;C:\\Windows\\System32;${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}\\CMakeTmp\\")
		
		message("Visual studio express edition")
		
		execute_process(COMMAND "vcvars32.bat"  WORKING_DIRECTORY "${VS_DIR}\\VC\\bin")

		#message("Start process 32 bit") 

		if(EXISTS "${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CMakeTmp/FoundCapa32.exe")
			#message("Found 32")
			execute_process( COMMAND "${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CMakeTmp/FoundCapa32.exe"
							 WORKING_DIRECTORY "${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CMakeTmp/"
							 RESULT_VARIABLE _resultNVCC OUTPUT_VARIABLE _outNVCC
							 ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)

		else()
			execute_process( COMMAND "${CUDA_NVCC_EXECUTABLE}" "-m32" "-o=FoundCapa32" "${PROJECT_SOURCE_DIR}/src/uti_phgrm/GpGpu/tools/FoundCapa.cu"  "--run"		
					 WORKING_DIRECTORY "${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CMakeTmp/"
					 RESULT_VARIABLE _resultNVCC OUTPUT_VARIABLE _outNVCC
					 ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)

		endif()
		#message(${_outNVCC}) 
	endif()

	if(NOT _resultNVCC EQUAL 0)		
		if("${CMAKE_GENERATOR}" MATCHES "Win64")
		    message("Error: 64 bits generation with cuda and visual express edition")
			message("2 options ")
			message("create with 32 bits generator")
			message("or")
			message("Copy the folder ${VS_DIR}\\VC\\bin\\x86_amd64 and rename x86_amd64 to amd64")
			message("Inside the new amd64 folder rename vcvarsx86_amd64.bat file to vcvars64.bat")
			message(FATAL_ERROR "64 bits generation with cuda and express edition")
		endif() 
		message( FATAL_ERROR "Error Cuda --- build in 32 bits" )
	endif()  
endif()

set(_cudaArch "${_outNVCC}")

string(FIND "${_cudaArch}" "2.1" arch_21)
string(FIND "${_cudaArch}" "2.0" arch_20)
string(FIND "${_cudaArch}" "3.0" arch_30)
string(FIND "${_cudaArch}" "3.2" arch_32)
string(FIND "${_cudaArch}" "3.5" arch_35)
string(FIND "${_cudaArch}" "5.0" arch_50)

if(${CUDA_LINEINFO})
    set(flag_Lineinfo  -lineinfo)
endif()

if(${CUDA_FASTMATH})
    set(flag_fastMath  -use_fast_math)
endif()

if((NOT ${arch_20} LESS 0) OR (NOT ${arch_21} LESS 0))

    set(cuda_arch_version 20 )
	if(NOT ${arch_20} LESS 0)
		set(cuda_arch_version_string 2.0 )
	else()
		set(cuda_arch_version_string 2.1 )
	endif()
    set(cuda_generation Fermi)

elseif((NOT ${arch_30} LESS 0) OR (NOT ${arch_32} LESS 0))

    set(cuda_arch_version 30 )
	set(cuda_arch_version_string 3.0 )
    set(cuda_generation Kepler)

elseif((NOT ${arch_35} LESS 0))

    set(cuda_arch_version 35 )
	set(cuda_arch_version_string 3.5 )
    set(cuda_generation Kepler)

elseif((NOT ${arch_50} LESS 0))

    set(cuda_arch_version 50 )
        set(cuda_arch_version_string 5.0 )
    set(cuda_generation Maxwell)

else()

    message("Cuda capabilities are not sufficient")

endif()

message("Cuda API Version ${CUDA_VERSION}")
message("Cuda card compute capabilities SM ${cuda_arch_version_string} (${cuda_generation} generation)")

set(GENCODE_SM -gencode=arch=compute_${cuda_arch_version},code=sm_${cuda_arch_version} ${flag_Lineinfo} ${flag_fastMath})
 
set(libStatGpGpuTools GpGpuTools)
set(libStatGpGpuInterfMicMac GpGpuInterfMicMac)
set(libStatGpGpuOpt GpGpuOpt)

if(${CUDA_NVTOOLS})
    find_cuda_helper_libs(nvToolsExt)
endif()

cuda_add_library(${libStatGpGpuTools}  ${GpGpuTools_Src_Files} ${IncCudaFiles} STATIC OPTIONS ${GENCODE_SM})


if(${CUDA_NVTOOLS})
    target_link_libraries(${libStatGpGpuTools} ${CUDA_nvToolsExt_LIBRARY})
endif()

cuda_add_library(${libStatGpGpuInterfMicMac}  ${uti_phgrm_GpGpu_Src_Files} STATIC OPTIONS ${GENCODE_SM})

cuda_add_library(${libStatGpGpuOpt}  ${uti_phgrm_Opt_GpGpu_Src_Files} ${IncCudaFiles} STATIC OPTIONS ${GENCODE_SM})

if(Boost_FOUND)
    target_link_libraries(${libStatGpGpuInterfMicMac} ${libStatGpGpuTools} ${Boost_LIBRARIES} ${Boost_THREADAPI})
endif()

if(NOT WIN32)
    target_link_libraries(${libStatGpGpuInterfMicMac}  rt pthread )
endif()
set(GpGpu_UnitTesting GpGpuUnitTesting)

cuda_add_executable(${GpGpu_UnitTesting} ${uti_Test_Opt_GpGpu_Src_Files})

if(${CUDA_NVTOOLS})
    target_link_libraries(${GpGpu_UnitTesting}  ${libStatGpGpuInterfMicMac} ${libStatGpGpuOpt} ${libStatGpGpuTools} ${CUDA_nvToolsExt_LIBRARY}   )
else()
    target_link_libraries(${GpGpu_UnitTesting}  ${libStatGpGpuInterfMicMac} ${libStatGpGpuOpt} ${libStatGpGpuTools})
endif()

if (NOT WIN32)
        target_link_libraries(${GpGpu_UnitTesting}  rt pthread )
endif()

install(TARGETS ${GpGpu_UnitTesting} RUNTIME DESTINATION ${Install_Dir})

link_directories(${PROJECT_SOURCE_DIR}/lib/) 

if(${CUDA_ENABLED})
         file(GLOB_RECURSE IncFilesGpGpu ${PROJECT_SOURCE_DIR}/include/GpGpu/*.h  )
         list(REMOVE_ITEM IncFiles ${IncFilesGpGpu})
endif()

#cuda_add_library( ${libElise} ${Elise_Src_Files} ${IncFiles} OPTIONS ${GENCODE_SM})

#target_link_libraries(${libElise} ${libStatGpGpuTools} ${libStatGpGpuInterfMicMac} ${libStatGpGpuOpt} ${CUDA_nvToolsExt_LIBRARY})

install(TARGETS ${libStatGpGpuTools} ${libStatGpGpuInterfMicMac} ${libStatGpGpuOpt}
            LIBRARY DESTINATION ${PROJECT_SOURCE_DIR}/lib
            ARCHIVE DESTINATION ${PROJECT_SOURCE_DIR}/lib)


#///////// OPENCL
#//////////////////////////////////////////
if(${WITH_OPENCL})

    message("OPENCL Doesn't work for the moment")
    find_path(OPENCL_INCLUDE_DIR
        NAMES
                CL/cl.h OpenCL/cl.h
        PATHS
                $ENV{AMDAPPSDKROOT}/include
				$ENV{CUDA_PATH}/include
                $ENV{INTELOCLSDKROOT}/include
                $ENV{NVSDKCOMPUTE_ROOT}/OpenCL/common/inc
                ${CUDA_TOOLKIT_INCLUDE}
                 # Legacy Stream SDK
                $ENV{ATISTREAMSDKROOT}/include)

    if(CMAKE_SIZEOF_VOID_P EQUAL 4)
        set(OPENCL_LIB_SEARCH_PATH
                ${OPENCL_LIB_SEARCH_PATH}
                $ENV{AMDAPPSDKROOT}/lib/x86
                $ENV{INTELOCLSDKROOT}/lib/x86
                $ENV{NVSDKCOMPUTE_ROOT}/OpenCL/common/lib/Win32
				$ENV{CUDA_PATH}//lib/Win32
                # Legacy Stream SDK
                $ENV{ATISTREAMSDKROOT}/lib/x86)
    elseif(CMAKE_SIZEOF_VOID_P EQUAL 8)
        set(OPENCL_LIB_SEARCH_PATH
                ${CUDA_TOOLKIT_INCLUDE}/lib64
                #${OPENCL_LIB_SEARCH_PATH}
                $ENV{AMDAPPSDKROOT}/lib/x86_64
                $ENV{INTELOCLSDKROOT}/lib/x64
                $ENV{NVSDKCOMPUTE_ROOT}/OpenCL/common/lib/x64
				$ENV{CUDA_PATH}//lib/x64
                # Legacy stream SDK
                $ENV{ATISTREAMSDKROOT}/lib/x86_64)
    endif(CMAKE_SIZEOF_VOID_P EQUAL 4)

    find_library(
        OPENCL_LIBRARY
        NAMES OpenCL
        PATHS ${OPENCL_LIB_SEARCH_PATH})

    include(FindPackageHandleStandardArgs)
    find_package_handle_standard_args(
      OpenCL
      DEFAULT_MSG
      OPENCL_LIBRARY OPENCL_INCLUDE_DIR)

    if(OPENCL_FOUND)
      set(OPENCL_LIBRARIES ${OPENCL_LIBRARY})
    else(OPENCL_FOUND)
      set(OPENCL_LIBRARIES)
    endif(OPENCL_FOUND)

    mark_as_advanced(
      OPENCL_INCLUDE_DIR
      OPENCL_LIBRARY
      )

if(NOT CUDA_ENABLED)
    set(filesopencl
            "${PROJECT_SOURCE_DIR}/src/uti_phgrm/GpGpu/GpGpu_OpenCL_Kernel.cu"
            "${PROJECT_SOURCE_DIR}/src/uti_phgrm/GpGpu/GpGpu_OpenCL.cpp"
        )

    add_executable(TestOpenCL ${filesopencl})

    target_link_libraries(TestOpenCL ${libStatGpGpuTools} ${OPENCL_LIBRARY})

    install(TARGETS TestOpenCL RUNTIME DESTINATION ${Install_Dir})
    message("OPENCL TEST")
else()
    set(filesCUDA
            "${PROJECT_SOURCE_DIR}/src/uti_phgrm/GpGpu/GpGpu_OpenCL.cpp"
            "${PROJECT_SOURCE_DIR}/src/uti_phgrm/GpGpu/GpGpu_CUDA_Define.cu"
            "${PROJECT_SOURCE_DIR}/src/uti_phgrm/GpGpu/GpGpu_OpenCL_Kernel.cu"
        )

    cuda_add_executable(TestCUDA ${filesCUDA})

     if(${CUDA_NVTOOLS})
            target_link_libraries(TestCUDA ${libStatGpGpuTools} ${CUDA_nvToolsExt_LIBRARY} ${OPENCL_LIBRARY})
     else()
            target_link_libraries(TestCUDA ${libStatGpGpuTools} ${OPENCL_LIBRARY})
     endif()

    install(TARGETS TestCUDA RUNTIME DESTINATION ${Install_Dir})
endif()

else()
    set(filesCUDA
            "${PROJECT_SOURCE_DIR}/src/uti_phgrm/GpGpu/GpGpu_OpenCL.cpp"
            "${PROJECT_SOURCE_DIR}/src/uti_phgrm/GpGpu/GpGpu_CUDA_Define.cu"
            "${PROJECT_SOURCE_DIR}/src/uti_phgrm/GpGpu/GpGpu_OpenCL_Kernel.cu"
        )

    cuda_add_executable(TestCUDA ${filesCUDA})

    if(${CUDA_NVTOOLS})
        target_link_libraries(TestCUDA ${libStatGpGpuTools} ${CUDA_nvToolsExt_LIBRARY} )
    else()
        target_link_libraries(TestCUDA ${libStatGpGpuTools} ${OPENCL_LIBRARY})
    endif()

    target_link_libraries(TestCUDA ${libStatGpGpuTools} )

    install(TARGETS TestCUDA RUNTIME DESTINATION ${Install_Dir})
endif()

#////////////////////////////