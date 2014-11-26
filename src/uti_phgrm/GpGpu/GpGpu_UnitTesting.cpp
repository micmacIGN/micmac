#include "GpGpu/GpGpu.h"
/*
int Main_UnitTest_Realloc()
{
//    printf("Main_UnitTest_Realloc - START\n");

//    ImageLayeredCuda<float2>    _iVolume;
//    CuHostData3D<float2>        _hVolume;

//    CuDeviceData3D<float2>      _dVolume;

//    uint2   dim___01 = make_uint2(50,50);
//    uint    layer_01 = 5;



//    // =======================================================================
//    printf("Main_UnitTest_Realloc - STEP 1\n");

//    _hVolume.ReallocIf(dim___01,layer_01);
//    _iVolume.ReallocIfDim(dim___01,layer_01);
//    _dVolume.ReallocIf(dim___01,layer_01);

//    _hVolume.Fill(make_float2(2.f,3.f));

//    _iVolume.copyHostToDevice(_hVolume.pData());
//    _dVolume.CopyHostToDevice(_hVolume.pData());
//    _dVolume.CopyDevicetoHost(_hVolume);

//    // =======================================================================
//    printf("Main_UnitTest_Realloc - STEP 2\n");

//    _hVolume.Dealloc();
//    _iVolume.Dealloc();

//    _hVolume.ReallocIf(dim___01,layer_01);
//    _iVolume.ReallocIfDim(dim___01,layer_01);
//    _dVolume.ReallocIf(dim___01,layer_01);

//    _hVolume.Fill(make_float2(2.f,3.f));

//    _iVolume.copyHostToDevice(_hVolume.pData());
//    _dVolume.CopyHostToDevice(_hVolume.pData());
//    _dVolume.CopyDevicetoHost(_hVolume);

//    // =======================================================================
//    printf("Main_UnitTest_Realloc - STEP 3\n");

//    _hVolume.ReallocIf(dim___01,layer_01+1);
//    _iVolume.ReallocIfDim(dim___01,layer_01+1);
//    _dVolume.ReallocIf(dim___01,layer_01+1);

//    printf("Main_UnitTest_Realloc - STEP 3 : FILL\n");
//    _hVolume.Fill(make_float2(2.f,3.f));

//    printf("Main_UnitTest_Realloc - STEP 3 : COPY\n");
//    _iVolume.copyHostToDevice(_hVolume.pData());
//    _dVolume.CopyHostToDevice(_hVolume.pData());
//    _dVolume.CopyDevicetoHost(_hVolume);

//    // =======================================================================
//    printf("*************************************************************************\n");
//    printf("Main_UnitTest_Realloc - ****************************************   STEP 4\n");

//    uint2   dim___02 = make_uint2(80,100);

//    _hVolume.ReallocIf(dim___02,layer_01+1);
//    _iVolume.ReallocIfDim(dim___02,layer_01+1);
//    _dVolume.ReallocIf(dim___02,layer_01+1);

//    printf("Main_UnitTest_Realloc - STEP 4 : FILL\n");
//    _hVolume.Fill(make_float2(2.f,3.f));

//    printf("Main_UnitTest_Realloc - STEP 4 : COPY\n");
//    _iVolume.copyHostToDevice(_hVolume.pData());
//    _dVolume.CopyHostToDevice(_hVolume.pData());
//    _dVolume.CopyDevicetoHost(_hVolume);

//    // =======================================================================
//    printf("Main_UnitTest_Realloc - STEP 5\n");

//    _hVolume.ReallocIf(dim___02*2,layer_01+4);
//    _iVolume.ReallocIfDim(dim___02*2,layer_01+4);
//    _dVolume.ReallocIf(dim___02*2,layer_01+4);

//    printf("Main_UnitTest_Realloc - STEP 5 : FILL\n");
//    _hVolume.Fill(make_float2(2.f,3.f));

//    printf("Main_UnitTest_Realloc - STEP 5 : COPY\n");
//    _iVolume.copyHostToDevice(_hVolume.pData());
//    _dVolume.CopyHostToDevice(_hVolume.pData());
//    _dVolume.CopyDevicetoHost(_hVolume);

//    printf("Main_UnitTest_Realloc - END\n");

    return 0;
}

int Main_Test_Optimisation()
{
    // Creation du contexte GPGPU
    cudaDeviceProp deviceProp;
    // Obtention de l'identifiant de la carte la plus puissante
    int devID = gpuGetMaxGflopsDeviceId();
    // Initialisation du contexte
    checkCudaErrors(cudaSetDevice(devID));
    // Obtention des proprietes de la carte
    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, devID));
    // Affichage des proprietes de la carte
    printf("GPU Device %d: \"%s\" with compute capability %d.%d\n", devID, deviceProp.name, deviceProp.major, deviceProp.minor);


    cout << "Launch Data optimisation GpGpu ***" << endl;
    GpGpuTools::OutputInfoGpuMemory();

    srand ((uint)time(NULL));

    // Declaration des variables du cote du HOST
    HOST_Data2Opti h2O;
    // Declaration des variables du cote du DEVICE
    DEVC_Data2Opti d2O;

    bool random = false;

    uint nbLines        = random ? rand() % 1024 : 1;
    uint lenghtMaxLines = random ? rand() % 1024 : 2;
    uint depthMax       = NAPPEMAX;

    uint sizeMaxLine = (uint)(1.5f*sqrt((float)lenghtMaxLines * lenghtMaxLines + nbLines * nbLines));

    h2O.ReallocParam(sizeMaxLine);
    d2O.ReallocParam(sizeMaxLine);

    uint pit_Strm_DZ    = WARPSIZE;
    uint pit_Strm_ICost = NAPPEMAX;

    CuHostData3D<ushort> tabZ(nbLines,lenghtMaxLines,2);
    CuHostData3D<ushort> lenghtLines(nbLines);

    if(random)
    {
        tabZ.FillRandom(0,depthMax/2);
        lenghtLines.FillRandom(lenghtMaxLines-1,lenghtMaxLines);
    }
    else
    {
        tabZ.Fill(depthMax/2);
        lenghtLines.Fill(lenghtMaxLines);
    }

    for (uint p= 0 ; p < nbLines; p++)
    {

        h2O.SetParamLine(p,pit_Strm_ICost,pit_Strm_DZ,lenghtLines[p]);

        uint sizeStreamLine = 0;
        for (uint aK= 0 ; aK < lenghtLines[p]; aK++)
            sizeStreamLine += count(make_short2(-tabZ[make_uint3(p,aK,0)],tabZ[make_uint3(p,aK,1)]));

        pit_Strm_DZ     += iDivUp(lenghtLines[p],   WARPSIZE) * WARPSIZE;
        pit_Strm_ICost  += iDivUp(sizeStreamLine,   WARPSIZE) * WARPSIZE;
    }

    h2O.ReallocInputIf(pit_Strm_ICost + NAPPEMAX,pit_Strm_DZ + WARPSIZE);
    h2O.s_InitCostVol().Fill(0);

    // index

    for (uint idLine= 0 ; idLine < nbLines; idLine++)
    {
        uint    pitStrm = 0;

        for (uint aK= 0 ; aK < lenghtLines[idLine]; aK++)

        {
            short2 lDZ      = make_short2(-tabZ[make_uint3(idLine,aK,0)],tabZ[make_uint3(idLine,aK,1)]);
            ushort lDepth   = count(lDZ);

            //h2O._s_Index(h2O.param(0)[idLine].y + aK) = lDZ;

//            uint idStrm = h2O.param(0)[idLine].x + pitStrm - lDZ.x;    h2O.Dealloc();
            d2O.Dealloc();
            tabZ.Dealloc();
            lenghtLines.Dealloc();

            GpGpuTools::OutputInfoGpuMemory();
            checkCudaErrors(cudaDeviceReset());
            //GpGpuTools::OutputInfoGpuMemory();
            printf("Reset Device GpGpu.\n");


//            for ( int aPx = lDZ.x ; aPx < lDZ.y; aPx++)
//                //h2O._s_InitCostVol[idStrm + aPx]  = 10000 * (idLine + 1) + (aK+1) * 1000 + aPx - lDZ.x + 1;
//                h2O.s_InitCostVol(idStrm + aPx)  = 1;

            pitStrm += lDepth;
        }
    }

    h2O.SetNbLine(nbLines);
    d2O.SetNbLine(h2O.nbLines());

    h2O.ReallocOutputIf(h2O.s_InitCostVol().GetSize(),h2O.s_Index().GetSize());

    h2O.s_ForceCostVol(0).Fill(0);

    d2O.ReallocIf(h2O);

    //      Transfert des données vers le device                            ---------------		-
    d2O.CopyHostToDevice(h2O);
    d2O.s_ForceCostVol(0).CopyHostToDevice(h2O.s_ForceCostVol(0).pData());

    h2O.s_InitCostVol().OutputValues();

    Gpu_OptimisationOneDirection(d2O);

    d2O.CopyDevicetoHost(h2O);

    h2O.s_ForceCostVol(0).OutputValues();


    //
    uint errorCount = 0;


    for (uint idLine= 0 ; idLine < nbLines; idLine++)
    {
        uint    pitStrm = 0;

        for (uint aK= 0 ; aK < lenghtLines[idLine]; aK++)
        {
            short2 dZ = h2O._s_Index[h2O._param[0][idLine].y + aK ];

            uint idStrm = h2O._param[0][idLine].x + pitStrm - dZ.x;

            for ( int aPx = dZ.x ; aPx < dZ.y; aPx++)
                if( h2O._s_InitCostVol[idStrm + aPx]  != h2O._s_ForceCostVol[0][idStrm + aPx])
                {
                    //printf(" %d ",h2O._s_InitCostVol[idStrm + aPx]);
                    errorCount++;
                }

            pitStrm += count(dZ);
        }
    }


    printf("\nError Count   = %d/%d\n",errorCount,h2O._s_InitCostVol.GetSize()- 2*NAPPEMAX);
    printf("Error percent = %f\n",(((float)errorCount*100)/(h2O._s_InitCostVol.GetSize()- 2*NAPPEMAX)));


    h2O.Dealloc();
    d2O.Dealloc();
    tabZ.Dealloc();
    lenghtLines.Dealloc();

    GpGpuTools::OutputInfoGpuMemory();
    checkCudaErrors(cudaDeviceReset());
    //GpGpuTools::OutputInfoGpuMemory();
    printf("Reset Device GpGpu.\n");

    return 0;
}

int Main_UnitTest_ProjectionImage()
{
    pCorGpu param;

    param.invPC.SetParamInva(make_ushort2(5,5),make_ushort2(2,2),make_uint2(2,2),1.f,SAMPLETERR,1,4,1);
    param.SetDimension(Rect(0,0,48,48),8);

    CuDeviceData3D<float>       DeviImagesProj;

    LaunchKernelprojectionImage(param,DeviImagesProj);

//    param.SetDimension(Rect(0,0,48,40),8);

//    LaunchKernelprojectionImage(param,DeviImagesProj);

//    param.SetDimension(Rect(0,0,48,80),8);

//    LaunchKernelprojectionImage(param,DeviImagesProj);

//    param.SetDimension(Rect(0,0,48,80),4);

//    LaunchKernelprojectionImage(param,DeviImagesProj);


    return 0;
}

int Main_UnitTest_Memoire()
{
    //GpGpuTools::OutputInfoGpuMemory();

    return 0;
}*/
int main()
{

    return 0;//Main_UnitTest_Memoire();

}
