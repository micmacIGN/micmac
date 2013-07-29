#include "GpGpu/GpGpu_Data.h"

DecoratorImageCuda::DecoratorImageCuda(CData<cudaArray> *dataCudaArray):
    _dataCudaArray(dataCudaArray)
{

}

bool  DecoratorImageCuda::bindTexture( textureReference& texRef )
{
    cudaChannelFormatDesc desc;
//    bool bCha	= CData::ErrorOutput(cudaGetChannelDesc(&desc, GetCudaArray()),"Bind Texture / cudaGetChannelDesc");
//    bool bBind	= CData::ErrorOutput(cudaBindTextureToArray(&texRef,GetCudaArray(),&desc),"Bind Texture / Bind");
    bool bCha	= !cudaGetChannelDesc(&desc, GetCudaArray());
    bool bBind	= !cudaBindTextureToArray(&texRef,GetCudaArray(),&desc);

    return bCha && bBind;
}

cudaArray* DecoratorImageCuda::GetCudaArray()
{
    return _dataCudaArray->pData();
}

bool DecoratorImageCuda::Memset( int val )
{
    std::cout << "PAS DE MEMSET POUR CUDA ARRAY" << "\n";
    return true;
}

bool DecoratorImageCuda::abDealloc()
{
    return (cudaFreeArray( GetCudaArray()) == cudaSuccess) ? true : false;
}
