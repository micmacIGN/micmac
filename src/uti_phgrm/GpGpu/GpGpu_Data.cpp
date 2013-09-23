#include "GpGpu/GpGpu_Data.h"

DecoratorImageCuda::DecoratorImageCuda(CData<cudaArray> *dataCudaArray):
    _dataCudaArray(dataCudaArray)
{
}

bool  DecoratorImageCuda::bindTexture( textureReference& texRef )
{
    _textureReference = &texRef;

    cudaChannelFormatDesc desc;

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

bool DecoratorImageCuda::UnbindDealloc()
{

    if(_textureReference) cudaUnbindTexture(_textureReference);

    _textureReference = NULL;

    return _dataCudaArray->Dealloc();;
}



