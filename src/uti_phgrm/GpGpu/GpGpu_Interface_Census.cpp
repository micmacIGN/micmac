#include"GpGpu/GpGpu_Interface_Census.h"


dataCorrelMS::dataCorrelMS():
    _texMaskErod(texture_Masq_Erod())
{
    for (int t = 0; t < NBEPIIMAGE; ++t)
    {
        _texImage[t]    = pTexture_ImageEpi(t);
        GpGpuTools::SetParamterTexture(*_texImage[t]);
    }

    _texMaskErod.addressMode[0]	= cudaAddressModeBorder;
    _texMaskErod.addressMode[1]	= cudaAddressModeBorder;
    _texMaskErod.filterMode     = cudaFilterModePoint; //cudaFilterModePoint cudaFilterModeLinear
    _texMaskErod.normalized     = false;
}

void dataCorrelMS::transfertImage(uint2 sizeImage, float ***dataImage, int id)
{
    _HostImage[id].ReallocIfDim(sizeImage,3);
    for (int tScale = 0; tScale < 3; tScale++)
    {
        float ** source   = dataImage[tScale];
        float *  dest     = _HostImage[id].pLData(tScale);
        memcpy( dest , source[0],  size(sizeImage) * sizeof(float));
    }
}

void dataCorrelMS::transfertMask(uint2 dimMask, pixel **mImMasqErod_0, pixel **mImMasqErod_1)
{
    uint2 dimMaskByte = make_uint2((dimMask.x+7)/8,dimMask.y);
    _HostMaskErod.ReallocIfDim(dimMaskByte,2);
    memcpy( _HostMaskErod.pData()   , mImMasqErod_0[0],  size(dimMaskByte) * sizeof(pixel));
    memcpy( _HostMaskErod.pLData(1) , mImMasqErod_1[0],  size(dimMaskByte) * sizeof(pixel));

//    for (uint y = 0; y < dimMask.y; ++y)
//    {
//        //pixel* yP = mImMasqErod_0[y];

//        for (uint x = 0; x < dimMask.x; ++x)
//        {
//            _HostMaskErod[make_uint3(x,y,0)] = mImMasqErod_0[y][x];
////            _HostMaskErod[make_uint3(x,y,0)] = ((yP[x/8] >> (7-x %8) ) & 1) ? 255 : 0;
//        }
//    }

//    _HostMaskErod.saveImage("Mask_",0);

}

void dataCorrelMS::transfertNappe(int mX0Ter, int mX1Ter, int mY0Ter, int mY1Ter, short **mTabZMin, short **mTabZMax)
{

    uint2 dimNappe = make_uint2(mX1Ter-mX0Ter,mY1Ter-mY0Ter);

    _HostInterval_Z.ReallocIfDim(dimNappe,1);

    for (int anX = mX0Ter ; anX <  mX1Ter ; anX++)
    {
        int X = anX - mX0Ter;
        for (int anY = mY0Ter ; anY < mY1Ter ; anY++)
            _HostInterval_Z[make_uint2(X,anY - mY0Ter)] = make_short2(mTabZMin[anY][anX],mTabZMax[anY][anX]);
    }
}

void dataCorrelMS::syncDeviceData()
{
    for (int t = 0; t < NBEPIIMAGE; ++t)
        _dt_Image[t].syncDevice(_HostImage[t],*_texImage[t]);

    _dt_MaskErod.syncDevice(_HostMaskErod,_texMaskErod);

    _DeviceInterval_Z.ReallocIf(_HostInterval_Z.GetDimension());
    _DeviceInterval_Z.CopyHostToDevice(_HostInterval_Z.pData());
}

void constantParameterCensus::transfertConstantCensus(const std::vector<std::vector<Pt2di> > &aVV, const std::vector<double> &aVPds, int2 offset0, int2 offset1)
{
    for (int s = 0; s < (int)aVV.size(); ++s)
    {
        short2 *lw = w[s];

        const std::vector<Pt2di> &vv = aVV[s];
        sizeW[s] = vv.size();
        poids[s] = aVPds[s];
        _offset0 = offset0;
        _offset1 = offset1;

        for (int p = 0; p < (int)vv.size(); ++p)
        {
            Pt2di pt = vv[p];
            lw[p] = make_short2(pt.x,pt.y);
        }
    }
}

void constantParameterCensus::transfertTerrain(Rect zoneTerrain)
{
    _zoneTerrain    = zoneTerrain;
    _dimTerrain     = _zoneTerrain.dimension();
}

void GpGpuInterfaceCensus::jobMask()
{
    paramCencus2Device(_cDataCMS);   
    _dataCMS.syncDeviceData();
    LaunchKernelCorrelationCensus(_dataCMS,_cDataCMS);
}
