#include"GpGpu/GpGpu_Interface_Census.h"


dataCorrelMS::dataCorrelMS():
    _texMaskErod(texture_Masq_Erod())
{
    for (int t = 0; t < NBEPIIMAGE; ++t)
    {
        _texImage[t] = pTexture_ImageEpi(t);
        GpGpuTools::SetParamterTexture(*_texImage[t]);
    }
}

void dataCorrelMS::transfertImage(uint2 sizeImage, float ***dataImage, int id)
{

    _HostImage[id].ReallocIfDim(sizeImage,3);

//    std::string numeEpi(GpGpuTools::conca("EPI_",id));

    for (int tScale = 0; tScale < 3; tScale++)
    {

        float ** buuf = dataImage[tScale];
        float *dest = _HostImage[0].pData() + size(sizeImage) * tScale;
        memcpy( dest , buuf[0],  size(sizeImage) * sizeof(float));
//        DUMP_INT(tScale)
//        std::string numec(GpGpuTools::conca("_IMAGES_",tScale));
//        std::string nameFile = numeEpi + numec + std::string(".pgm");
//        printf("%s\n",nameFile.c_str());
//        GpGpuTools::Array1DtoImageFile(dest,nameFile.c_str(),sizeImage,1.f/65536.f);

    }

   //getchar();
}

void dataCorrelMS::syncDeviceData()
{
    for (int t = 0; t < NBEPIIMAGE; ++t)
        _dt_Image[t].syncDevice(_HostImage[t],*_texImage[t]);

    _dt_MaskErod.syncDevice(_HostMaskErod,_texMaskErod);

    _DeviceInterval_Z.ReallocIf(_HostInterval_Z.GetDimension());
    _DeviceInterval_Z.CopyHostToDevice(_HostInterval_Z.pData());

}
