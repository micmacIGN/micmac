#include"GpGpu/GpGpu_Interface_CorMultiScale.h"


dataCorrelMS::dataCorrelMS()
{
    for (int t = 0; t < NBEPIIMAGE; ++t)
    {
        _texImage[t]    = pTexture_ImageEpi(t);
        _texMaskErod[t] = ptexture_Masq_Erod(t);
        GpGpuTools::SetParamterTexture(*_texImage[t]);

        _texMaskErod[t]->addressMode[0]	= cudaAddressModeBorder;
        _texMaskErod[t]->addressMode[1]	= cudaAddressModeBorder;
        _texMaskErod[t]->filterMode     = cudaFilterModePoint; //cudaFilterModePoint cudaFilterModeLinear
        _texMaskErod[t]->normalized     = false;
    }

	_uInterval_Z.SetName("_uInterval_Z");
	_uCostf.SetName("_uCostf");
	_uCostu.SetName("_uCostu");
	_uCostp.SetName("_uCostu");
}

dataCorrelMS::~dataCorrelMS()
{

    dealloc();
}

//#define unitTestCorMS

void dataCorrelMS::unitT__CopyCoordInColor(uint2 sizeImage, float *dest)
{
    for (int y = 0; y < (int)sizeImage.y; ++y)
        for (int x = 0; x < (int)sizeImage.x; ++x)
            dest[to1D(x,y,sizeImage)] = (float)10000*x + y;
}

void dataCorrelMS::transfertImage(uint2 sizeImage, float ***dataImage, int id)
{
    _HostImage[id].ReallocIfDim(sizeImage,3);
    for (int tScale = 0; tScale < 3; tScale++)
    {
        float *  dest     = _HostImage[id].pLData(tScale);
#ifndef  unitTestCorMS
        float ** source   = dataImage[tScale];
        memcpy( dest , source[0],  size(sizeImage) * sizeof(float));
#else
        unitT__CopyCoordInColor(sizeImage,dest);
#endif
    }
}

template<>
float* dataCorrelMS::pDeviceCost()
{
	return _uCostf.deviceData.pData();
}

template<>
ushort* dataCorrelMS::pDeviceCost()
{
	return _uCostu.deviceData.pData();
}

void dataCorrelMS::transfertMask(uint2 dimMask0,uint2 dimMask1, pixel **mImMasqErod_0, pixel **mImMasqErod_1)
{
    uint2 dimMaskByte0 = make_uint2((dimMask0.x+7)/8,dimMask0.y);
    _HostMaskErod[0].ReallocIfDim(dimMaskByte0,1);

    uint2 dimMaskByte1 = make_uint2((dimMask1.x+7)/8,dimMask1.y);
    _HostMaskErod[1].ReallocIfDim(dimMaskByte1,1);

    memcpy( _HostMaskErod[0].pData() , mImMasqErod_0[0],  size(dimMaskByte0) * sizeof(pixel));
    memcpy( _HostMaskErod[1].pData() , mImMasqErod_1[0],  size(dimMaskByte1) * sizeof(pixel));

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

void dataCorrelMS::transfertNappe(int mX0Ter, int mX1Ter, int mY0Ter, int mY1Ter, short **mTabZMin, short **mTabZMax, bool dynGpu)
{

    uint2 dimNappe = make_uint2(mX1Ter-mX0Ter,mY1Ter-mY0Ter);

    _uInterval_Z.ReallocIfDim(dimNappe,1);

	_uPit.ReallocIfDim(dimNappe,1);

    _maxDeltaZ = 0;

	uint size = 0;

	// TODO Attention deja fait dans optimisation GPU!!!
    for (int anX = mX0Ter ; anX <  mX1Ter ; anX++)
    {
        int X = anX - mX0Ter;
        for (int anY = mY0Ter ; anY < mY1Ter ; anY++)
        {

			uint2	pt	= make_uint2(X,anY - mY0Ter);
			short2	ZZ  = make_short2(mTabZMin[anY][anX],mTabZMax[anY][anX]);

			_uInterval_Z.hostData[pt] = ZZ;

			uint deltaZ = abs(ZZ.x-ZZ.y);

			_maxDeltaZ  = max(_maxDeltaZ,deltaZ);

			_uPit.hostData[pt]    = size;
			size				 += deltaZ;
        }
    }

	_maxDeltaZ  = min(_maxDeltaZ,512); // TODO Attention

	if(dynGpu)
	{
		_uCostu.ReallocIfDim(size);
		_uCostp.ReallocIfDim(size);
//		_uCostu.ReallocIfDim(dimNappe,_maxDeltaZ);
//		_uCostp.ReallocIfDim(dimNappe,_maxDeltaZ);
	}
	else
		_uCostf.ReallocIfDim(dimNappe,_maxDeltaZ);

}

void dataCorrelMS::syncDeviceData()
{
    for (int t = 0; t < NBEPIIMAGE; ++t)
    {
        _dt_Image[t].syncDevice(_HostImage[t],*_texImage[t]);
        _dt_MaskErod[t].syncDevice(_HostMaskErod[t],*_texMaskErod[t]);
    }

    _uInterval_Z.syncDevice();
	_uPit.syncDevice();

    //_DeviceInterval_Z.ReallocIf(_HostInterval_Z.GetDimension());
    //_DeviceInterval_Z.CopyHostToDevice(_HostInterval_Z.pData());
}

void dataCorrelMS::dealloc()
{
    for (int t = 0; t < NBEPIIMAGE; ++t)
    {
        _HostImage[t].Dealloc();
        _HostMaskErod[t].Dealloc();
        _dt_MaskErod[t].UnbindDealloc();
        _dt_Image[t].UnbindDealloc();
    }

//    _HostInterval_Z.Dealloc();
    _uInterval_Z.Dealloc();
	_uPit.Dealloc();
	_uCostf.Dealloc();
	_uCostu.Dealloc();
	_uCostp.Dealloc();
}

InterfOptimizGpGpu*dataCorrelMS::getInterOpt(){return mInterOpt;}

void dataCorrelMS::setInterOpt(InterfOptimizGpGpu* InterOpt)

{
	mInterOpt = InterOpt;
}



void const_Param_Cor_MS::init(const std::vector<std::vector<Pt2di> > &VV,
							  const std::vector<double> &VPds,
		int2    offset0,
		int2    offset1,
		uint2	sIg0,
		uint2   sIg1,
		ushort  NbByPix,
		float   StepPix,
		float   nEpsilon,
		float   AhDefCost,
		float   SeuilHC,
		float   SeuilBC,
		bool    ModeMax,
		bool    mdoMixte,
		bool	dynRegulGpu,
		ushort nbscale)
{

    aNbScale    = nbscale;
    mNbByPix    = NbByPix;
    aStepPix    = StepPix;
    anEpsilon   = nEpsilon;
    mAhDefCost  = AhDefCost;
    aSeuilHC    = SeuilHC;
    aSeuilBC    = SeuilBC;
    aModeMax    = ModeMax;
    DoMixte     = mdoMixte;	
	mSIg0		= sIg0;
	mSIg1		= sIg1;	
	mDyRegGpu	= dynRegulGpu;	

    for (int s = 0; s < (int)VV.size(); ++s)
    {
        short2 *lw = aVV[s];

        const std::vector<Pt2di> &vv = VV[s];
        size_aVV[s] = vv.size();
        aVPds[s] = (float)VPds[s];
        anOff0 = offset0;
        anOff1 = offset1;

        for (int p = 0; p < (int)vv.size(); ++p)
        {
            Pt2di pt = vv[p];
            lw[p] = make_short2(pt.x,pt.y);
        }
    }
}

void const_Param_Cor_MS::setTerrain(Rect zoneTerrain)
{
    _zoneTerrain    = zoneTerrain;
    _dimTerrain     = _zoneTerrain.dimension();
    mDim3Cache      = make_uint3(_dimTerrain.x,_dimTerrain.y,aNbScale);
}

void const_Param_Cor_MS::dealloc()
{
    // TODO A Faire avec la liberation de symbole GPU
}

GpGpu_Interface_Cor_MS::GpGpu_Interface_Cor_MS():
    CSimpleJobCpuGpu(true)
{
    freezeCompute();
}

GpGpu_Interface_Cor_MS::~GpGpu_Interface_Cor_MS()
{
    dealloc();
}

void GpGpu_Interface_Cor_MS::Job_Correlation_MultiScale()
{
    paramCorMultiScale2Device(_cDataCMS);
    _dataCMS.syncDeviceData();

	LaunchKernel__Correlation_MultiScale(_dataCMS,_cDataCMS);

	//_dataCMS._uCostu.deviceData.CopyDevicetoHost(_dataCMS.getInterOpt()->_poInitCost._CostInit1D);
}

void GpGpu_Interface_Cor_MS::transfertImageAndMask(uint2 sI0, uint2 sI1, float ***dataImg0, float ***dataImg1, pixel **mask0, pixel **mask1)
{
    _dataCMS.transfertImage(sI0,dataImg0,0);
    _dataCMS.transfertImage(sI1,dataImg1,1);
    _dataCMS.transfertMask(sI0,sI1,mask0,mask1);
}

void GpGpu_Interface_Cor_MS::init(Rect                                    terrain,
		const std::vector<std::vector<Pt2di> > &aVV,
		const std::vector<double>              &aVPds,
		int2                                    offset0,
		int2                                    offset1,
		uint2                                   sIg0,
		uint2                                   sIg1,
		short                                 **mTabZMin,
		short                                 **mTabZMax,
		ushort                                  NbByPix,
		float                                   StepPix,
		float                                   nEpsilon,
		float                                   AhDefCost,
		float                                   aSeuilHC,
		float                                   aSeuilBC,
		bool                                    aModeMax,
		bool                                    DoMixte,
		bool									dynRegulGpu,
		InterfOptimizGpGpu*						interOpt,
		ushort                                  nbscale)
{   
	_dataCMS.setInterOpt(interOpt);

	_cDataCMS.init(aVV,aVPds,offset0,offset1,sIg0,sIg1,NbByPix,StepPix,nEpsilon,AhDefCost, aSeuilHC,aSeuilBC,aModeMax,DoMixte,dynRegulGpu);

	_dataCMS.transfertNappe(terrain.pt0.x, terrain.pt1.x, terrain.pt0.y, terrain.pt1.y, mTabZMin, mTabZMax,dynRegulGpu);
    _cDataCMS.setTerrain(terrain);
    _cDataCMS.maxDeltaZ = _dataCMS._maxDeltaZ;

    //_cDataCMS.transfertTerrain(Rect(mX0Ter,mY0Ter,mY1Ter,mX1Ter));
}

template<class T>
T GpGpu_Interface_Cor_MS::getCost(uint3 pt)
{   
	return 0.1f;
}

template<class T>
T* GpGpu_Interface_Cor_MS::getCost(uint2 pt)
{
	return NULL;
}

template<>
float GpGpu_Interface_Cor_MS::getCost(uint3 pt)
{
	int2 pt2 = make_int2(pt.x,pt.y);
	float *pcost = _dataCMS._uCostf.hostData.pData();
	//return pcost[to1D(pt,_dataCMS._uCostf.hostData.GetDimension3D())];
	return pcost[to1D(pt2,_cDataCMS._dimTerrain)*_cDataCMS.maxDeltaZ + pt.z];
}

template<>
ushort GpGpu_Interface_Cor_MS::getCost(uint3 pt)
{
	int2 pt2 = make_int2(pt.x,pt.y);
//	ushort *pcost = _dataCMS._uCostu.hostData.pData();
//	return pcost[to1D(pt,_dataCMS._uCostu.hostData.GetDimension3D())];
	ushort *pcost = _dataCMS._uCostu.hostData.pData();
	return pcost[to1D(pt2,_cDataCMS._dimTerrain)*_cDataCMS.maxDeltaZ + pt.z];
}

template<>
pixel GpGpu_Interface_Cor_MS::getCost(uint3 pt)
{
	int2 pt2 = make_int2(pt.x,pt.y);
	pixel *pcost = _dataCMS._uCostp.hostData.pData();
	//return pcost[to1D(pt,_dataCMS._uCostp.hostData.GetDimension3D())];
	return pcost[to1D(pt2,_cDataCMS._dimTerrain)*_cDataCMS.maxDeltaZ + pt.z];
}

template<>
ushort* GpGpu_Interface_Cor_MS::getCost(uint2 pt)
{
//	ushort *pcost = _dataCMS._uCostu.hostData.pData();
//	return pcost + to1D(pt,_cDataCMS._dimTerrain)*_cDataCMS.maxDeltaZ ;

	ushort *pcost = _dataCMS._uCostu.hostData.pData();
	return pcost + _dataCMS._uPit.hostData[pt];
}

template<>
pixel* GpGpu_Interface_Cor_MS::getCost(uint2 pt)
{

//	pixel *pcost = _dataCMS._uCostp.hostData.pData();
//	return pcost + to1D(pt,_cDataCMS._dimTerrain)*_cDataCMS.maxDeltaZ ;

	pixel *pcost = _dataCMS._uCostp.hostData.pData();
	return pcost + _dataCMS._uPit.hostData[pt];
}

void GpGpu_Interface_Cor_MS::dealloc()
{
    _dataCMS.dealloc();
    _cDataCMS.dealloc();
}

