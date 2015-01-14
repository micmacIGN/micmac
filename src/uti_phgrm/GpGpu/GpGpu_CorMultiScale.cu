
#include "GpGpu/GpGpu_CommonHeader.h"
//#include "GpGpu/GpGpu_TextureTools.cuh"
#include "GpGpu/GpGpu_Interface_CorMultiScale.h"

// Algorithme Correlation multi echelle sur ligne epipolaire

// Données :
//  - 2 images avec différents niveaux de floutage

//
// * Pré-calcul et paramètres                                   |       GPU
// -------------------------------------------------------------|----------------------------------
// - Tableau de parcours des vignettes                          >>      constant 3d data short2
// - poids des echelles                                         >>      constant 2d data ???
// - Tableau du ZMin et ZMax de chaque coordonnées terrain      >>      global 2D data short2
// - les offsets Terrain <--> Image Epi                         >>      constant 2 x int2
// - le masque erodé de l'image 1                               >>      1 texture pixel
// - 2 images x N echelles                                      >>      2 textures layered float


//
// Phase mNbByPix // pas utilise en GPU


/*  CPU
 *
 *  pour chaque
 *      - calcul des images interpolées pour l'image 1
 *      - mise en vecteur des images interpolées
 *      - Precalcul somme et somme quad
 *      - Parcour du terrain
 *      - Calcul des images interpolé
 *      - Parcours des Z
 *          - Calcul de la projection image 1
 *          - Calcul de la correlation Quick_MS_CorrelBasic_Center
 *              - pour chaque echelle
 *                  - Calcul de correlation
 *
 *      - set cost dans la matrice de regularisation
 */


//#define unitTestCorMS_gpgpu

///
static __constant__ const_Param_Cor_MS     cstPCMS;

extern "C" void paramCorMultiScale2Device( const_Param_Cor_MS &param )
{
    checkCudaErrors(cudaMemcpyToSymbol(cstPCMS, &param, sizeof(const_Param_Cor_MS)));
}

texture< float,	cudaTextureType2DLayered >      texture_ImageEpi_00;
texture< float,	cudaTextureType2DLayered >      texture_ImageEpi_01;
texture< pixel,	cudaTextureType2D >             Texture_Masq_Erod_00;
texture< pixel,	cudaTextureType2D >             Texture_Masq_Erod_01;

extern "C" textureReference& texture_ImageEpi(int nEpi){return nEpi == 0 ? texture_ImageEpi_00 : texture_ImageEpi_01;}

extern "C" textureReference* pTexture_ImageEpi(int nEpi){return nEpi == 0 ? &texture_ImageEpi_00 : &texture_ImageEpi_01;}

extern "C" textureReference* ptexture_Masq_Erod(int nEpi){return nEpi == 0 ? &Texture_Masq_Erod_00 : &Texture_Masq_Erod_01;}

__device__
inline    bool IN_THREAD(uint x = 0,uint y = 0,uint z = 0,uint bx = 0,uint by = 0,uint bz = 0)
{
    return blockIdx.x == bx && blockIdx.y == by && blockIdx.z == bz && threadIdx.z == z && threadIdx.x == x && threadIdx.y == y;
}

__device__
inline    bool IN_BLOCK(uint bx = 0,uint by = 0,uint bz = 0)
{
    return blockIdx.x == bx && blockIdx.y == by && blockIdx.z == bz;
}


__device__
inline    void PRINT_THREAD()
{
    printf("[%d,%d,%d|%d,%d,%d]  ",blockIdx.x ,blockIdx.y ,blockIdx.z,threadIdx.x,threadIdx.y ,threadIdx.z);
}

__device__
inline    bool GET_Val_BIT(const U_INT1 * aData,int anX)
{
    return (aData[anX/8] >> (7-anX %8) ) & 1;
}

__device__
inline    texture< pixel,cudaTextureType2D>  getMask(ushort iDi)
{
    return iDi == 0 ? Texture_Masq_Erod_00 : Texture_Masq_Erod_01;
}

inline __device__ int dElise_div(int a,int b)
{
    int res = a / b;
    return res - ((res * b) > a);
}

__device__
inline    bool IsOkErod(int3 pt)
{
    // TODO peut etre simplifier % et division

    const int ptxBy8 = pt.x >> 3;           // pt.x >> 3 Division par 8
    const int modulo = pt.x - (ptxBy8 << 3)  ;// (ptxBy8<<3) multiplication par 8

    pixel mask8b = tex2D(getMask(pt.z),(float)(ptxBy8) + 0.5f,(float)pt.y + 0.5f);

    return (mask8b >> (7-modulo ) ) & 1;
}

__device__
inline    bool IsOkErod(int2 pt,ushort idi)
{
    return IsOkErod(make_int3(pt.x,pt.y,idi));
}

__device__
inline    texture< float,	cudaTextureType2DLayered >  getTexture(ushort iDi)
{
    return iDi == 0 ? texture_ImageEpi_00 : texture_ImageEpi_01;
}

__device__
inline    float getValImage(float2 pt,ushort iDi,ushort nScale)
{
    return tex2DLayered(getTexture(iDi),pt.x + 0.5f,pt.y + 0.5f ,nScale);
}

template<class T>
__device__ float getValImage(T pt,ushort iDi,ushort nScale)
{
    return tex2DLayered(getTexture(iDi),(float)pt.x + 0.5f,(float)pt.y + 0.5f ,nScale);
}

__global__
void projectionMasqImage(float * dataPixel,uint3 dTer)
{

    if(blockIdx.x > cstPCMS._dimTerrain.x || blockIdx.y > cstPCMS._dimTerrain.y)
        return;

    const int3 pt = make_int3(blockIdx.x,blockIdx.y,blockIdx.z);

    float valImage = tex2DLayered(pt.z == 0 ? texture_ImageEpi_00 : texture_ImageEpi_01 ,pt.x + 0.5f,pt.y + 0.5f ,0);

    dataPixel[to1D(pt,dTer)] = IsOkErod(pt) ? valImage/(32768.f) : 0;
}

extern "C" void LaunchKernelCorrelationMultiScalePreview(dataCorrelMS &data,const_Param_Cor_MS &param)
{
    dim3	threads( 1, 1, 1);
    dim3	blocks(param._dimTerrain.x , param._dimTerrain.y, 2);

    CuHostData3D<float>     hData;
    CuDeviceData3D<float>   dData;

    uint3 dTer  = make_uint3(param._dimTerrain.x , param._dimTerrain.y,2);
    uint2 dTer2 = make_uint2(dTer);

    hData.Malloc(dTer2,2);
    dData.Malloc(dTer2,2);
    hData.Fill(0.f);
    dData.Memset(0);

    projectionMasqImage<<<blocks, threads>>>(dData.pData(),dTer);

    dData.CopyDevicetoHost(hData);

    GpGpuTools::Array1DtoImageFile(hData.pData()    ,"ET_HOP_0.pmg",hData.GetDimension());
    GpGpuTools::Array1DtoImageFile(hData.pLData(1)  ,"ET_HOP_1.pmg",hData.GetDimension());
}

//  pre-calcul pour la correlation multi echelle et mise en cache dans mSom et mSomSqr
__global__
void KernelPrepareCorrel(ushort idImage,float aStepPix, ushort mNbByPix, float* mSom, float* mSomSqr)
{

    // point image
    const uint2     pt          =   make_uint2(blockIdx.x*blockDim.x + threadIdx.x,blockIdx.y*blockDim.y + threadIdx.y);

    if(oSE(pt,cstPCMS._dimTerrain))
        return;

    // indice de l'etape sub pixelaire, le maximum étant cPCencus.mNbByPix
    const ushort    aPhase   =   (ushort)blockIdx.z;

    // la dimension du cache, la cache stocke des precaluls pour la corrélation
    const uint3     dimCache =   make_uint3(cstPCMS._dimTerrain.x,cstPCMS._dimTerrain.y,mNbByPix*cstPCMS.aNbScale);

    // le décalage sub pixelaire
    const float     cStepPix =   ((float)aPhase)*aStepPix;

    // point de l'image pour cette etape sub pixelaire
    const float2    ptImage  =   make_float2((float)pt.x + cStepPix,(float)pt.y);

    float aGlobSom      = 0;
    float aGlobSomSqr   = 0;
    float aGlobPds      = 0;

    // pour toutes les echelles
    for (int aKS=0 ; aKS< cstPCMS.aNbScale ; aKS++)
    {
        float   aSom    = 0;
        float   aSomSqr = 0;
        short2 *aVP     = cstPCMS.aVV[aKS];
        ushort  aNbP    = cstPCMS.size_aVV[aKS];
        float   aPdsK   = cstPCMS.aVPds[aKS];

        // pour les éléments de la vignettes
        for (int aKP=0 ; aKP<aNbP ; aKP++)
        {
            const short2 aP = aVP[aKP];
            float aV = getValImage(ptImage+aP,idImage,aKS);
            aSom += aV;
            aSomSqr += aV*aV;
        }

        aGlobSom    += aSom     * aPdsK;
        aGlobSomSqr += aSomSqr  * aPdsK;
        aGlobPds    += aPdsK    * aNbP;

        // indice dans le cache
        const uint3     p3d        =   make_uint3(pt.x,pt.y,aPhase*mNbByPix + aKS);

        // Ecriture dans le cache des

#ifndef unitTestCorMS_gpgpu
        mSom    [to1D(p3d,dimCache)] = aGlobSom    / aGlobPds;
        mSomSqr [to1D(p3d,dimCache)] = aGlobSomSqr / aGlobPds;
#else
        mSom    [to1D(p3d,dimCache)] = 10000*ptImage.x + ptImage.y;
        mSomSqr [to1D(p3d,dimCache)] = 10000*ptImage.x + ptImage.y;
#endif

    }
}


// calcul rapide de la correlation multi-echelles centre sur une vignette
__device__
inline    float Quick_MS_CorrelBasic_Center(

        const int2 & aPG0,
        const int2 & aPG1,
        float*  aSom1,
        float*  aSom11,
        float*  aSom2,
        float*  aSom22,
        int     aPx2,
        bool    ModeMax,
        ushort  aPhase)
{

    float aMaxCor = -1;
    float aCovGlob = 0;
    float aPdsGlob = 0;

    // pt float dans l'image 1
    const float2      aFG1      =   f2X(cstPCMS.aStepPix*(float)aPhase /*+ (float)dElise_div(aPx2,cstP_CorMS.mNbByPix)*/)+  make_float2(aPG1.x,aPG1.y);

    int aNbScale = cstPCMS.aNbScale;
    for (int aKS=0 ; aKS< aNbScale ; aKS++)
    {
        bool   aLast   = (aKS==(aNbScale-1));
        short2*aVP     = cstPCMS.aVV[aKS];
        float  aPds    = cstPCMS.aVPds[aKS];
        float  aCov    = 0;
        ushort aNbP    = cstPCMS.size_aVV[aKS];

        aPdsGlob += aPds * aNbP;
        for (int aKP=0 ; aKP<aNbP ; aKP++)
        {
            const short2 aP = aVP[aKP];

            const float valima_0 = getValImage(aPG0 + aP,0,aKS);
            const float valima_1 = getValImage(aFG1 + aP,1,aKS);


#ifdef unitTestCorMS_gpgpu
            if(IN_THREAD(1,2,1,6,9,4) && !aKS && !aP.x && !aP.y)
            {

                // PRINT_THREAD();
                DUMP(aPG0)
                        DUMP(aPG1)
                        DUMP(aFG1)
                        DUMP((int)valima_0)
                        DUMP(valima_1)
                        DUMP(aPhase)
            }
            if(IN_THREAD(1,2,2,6,9,4) && !aKS && !aP.x && !aP.y)
            {

                // PRINT_THREAD();
                        DUMP(aPG0)
                        DUMP(aPG1)
                        DUMP(aFG1)
                        DUMP((int)valima_0)
                        DUMP(valima_1)
                        DUMP(aPhase)
            }
#endif
            aCov += valima_0*valima_1;

        }

        aCovGlob += aCov * aPds;

        if (ModeMax || aLast)
        {
            const uint  pit0   =   to1D(make_uint3(aPG0.x,aPG0.y,aKS),cstPCMS._dimTerrain);
            const uint  pit1   =   to1D(make_uint3(aPG1.x,aPG1.y,aKS + aNbScale*aPhase),cstPCMS._dimTerrain);

            const float aM1    =   aSom1 [pit0];
            const float aM2    =   aSom2 [pit1];

#ifdef unitTestCorMS_gpgpu
            {
                if(IN_THREAD(1,2,1,6,9,4) && !aKS )
                {
                    DUMP((int)aSom1  [pit0])
                            DUMP((int)aSom11 [pit0])
                            DUMP((int)aSom2  [pit1])
                            DUMP((int)aSom22 [pit1])
                }

                if(IN_THREAD(1,2,2,6,9,4) && !aKS )
                {
                    DUMP((int)aSom1  [pit0])
                            DUMP((int)aSom11 [pit0])
                            DUMP((int)aSom2  [pit1])
                            DUMP((int)aSom22 [pit1])
                }
            }
#endif

            const float aM11   =   aSom11[pit0] - aM1*aM1;
            const float aM22   =   aSom22[pit1] - aM2*aM2;
            const float aM12   =   aCovGlob / aPdsGlob   - aM1 * aM2;

            if (ModeMax)
            {
                float aCor = (aM12 * abs(aM12)) /max(cstPCMS.anEpsilon,aM11*aM22);
                aMaxCor = max(aMaxCor,aCor);
            }
            else
                return aM12 / sqrt(max(cstPCMS.anEpsilon,aM11*aM22));
        }

    }

    return (aMaxCor > 0) ? sqrt(aMaxCor) : - sqrt(-aMaxCor) ;
}

__global__
void Kernel__DoCorrel_MultiScale_Global(float* aSom1,float*  aSom11,float* aSom2,float*  aSom22,short2 *nappe, float *cost)
{

    // ??? TODO à cabler
//    bool    DoMixte     = false;
//    bool    aModeMax    = true;
//    float   aSeuilHC    = 1.0;
//    float   aSeuilBC    = 1.0;

    // point image
    const   int2  an  =   make_int2(blockIdx.x*blockDim.x + threadIdx.x,blockIdx.y*blockDim.y + threadIdx.y);

    // sortir si le point est en dehors du terrain
    if(oSE(an,cstPCMS._dimTerrain))
        return;

    //      pt int dans l'image 0
    const   int2     aPIm0       =   an + cstPCMS.anOff0;

    if (IsOkErod(aPIm0,0))
    {

        // Z relatif au thread
        const ushort thZ        =  blockIdx.z*blockDim.z + threadIdx.z;

        float&          _cost   =  cost[to1D(an,thZ,cstPCMS._dimTerrain)];

        const short2    _nappe  =  nappe[to1D(an,cstPCMS._dimTerrain)];

        short           aZ0     =  _nappe.x;

        const int       DeltaZ  =  abs(_nappe.y-aZ0);

        if(thZ>=DeltaZ)
            return; // TODO on pourrait eventuellement affacter la valeur du cout par defaut.... mais bof

        // z Absolu
        const short aZ = aZ0 + (short)thZ;

        // calcul de la phase
        // Attention probleme avec valeur negative et le modulo
        const ushort aPhase = (ushort)((abs((int)aZ))%cstPCMS.mNbByPix);

        /// peut etre precalcul  -- voir simplifier
        ///
//        while ((abs((int)aZ0))%cstP_CorMS.mNbByPix != aPhase)
//            aZ0++;


//        const int anOffset      = dElise_div((int)aZ0,cstP_CorMS.mNbByPix);

        const int gpu_anOffset  = dElise_div((int)aZ,cstPCMS.mNbByPix);


        //        if( aEq(an,10) && aPhase == 0 && thZ < cstP_CorMS.mNbByPix)
        //            DUMP(gpu_anOffset)

        //int anOffset = dElise_div((int)aZ0,cstP_CorMS.mNbByPix);
        //
        const   int2    aIm1SsPx   =   an + cstPCMS.anOff1;
        //      pt int dans l'image 1
        const   int2    aPIm1      =   aIm1SsPx + i2X(gpu_anOffset);

//        DUMP(gpu_anOffset)

        float           aCost      =   cstPCMS.mAhDefCost;

        if (IsOkErod(aPIm1,1))
        {
            float aGlobCostBasic    = 0;
            float aGlobCostCorrel   = 0;

            aCost = Quick_MS_CorrelBasic_Center(aPIm0,aPIm1,aSom1,aSom11,aSom2,aSom22,gpu_anOffset,cstPCMS.aModeMax,aPhase);

            aGlobCostCorrel = aCost;

            if (cstPCMS.DoMixte)
            {
                if(aGlobCostCorrel>cstPCMS.aSeuilHC)

                    aCost = aGlobCostCorrel;

                else if (aGlobCostCorrel>cstPCMS.aSeuilBC)
                {
                    float aPCor =  (aGlobCostCorrel - cstPCMS.aSeuilBC) / (cstPCMS.aSeuilHC-cstPCMS.aSeuilBC);
                    aCost       =  aPCor * aGlobCostCorrel + (1-aPCor) * cstPCMS.aSeuilBC *  aGlobCostBasic;
                }
                else
                    aCost =  cstPCMS.aSeuilBC *  aGlobCostBasic;
            }

            aCost = 1.f-aCost;
        }

        _cost = aCost;
    }
}
#include <stdio.h>
extern "C" void LaunchKernel__Correlation_MultiScale(dataCorrelMS &data,const_Param_Cor_MS &parCMS)
{
    // Cache device
    //CuUnifiedData3D<float>  aSom1;

    CuDeviceData3D<float>  aSom_0;
    CuDeviceData3D<float>  aSomSqr_0;

    CuDeviceData3D<float>  aSom_1;
    CuDeviceData3D<float>  aSomSqr_1;

    aSom_0   .Malloc (parCMS._dimTerrain,parCMS.aNbScale);                  //  pas de sous echantillonnage
    aSomSqr_0.Malloc (parCMS._dimTerrain,parCMS.aNbScale);

    aSom_1   .Malloc (parCMS._dimTerrain,parCMS.aNbScale*parCMS.mNbByPix);  // avec sous echantillonnage
    aSomSqr_1.Malloc (parCMS._dimTerrain,parCMS.aNbScale*parCMS.mNbByPix);

    dim3	threads( 32, 32, 1);

    uint    divDTerX = iDivUp32(parCMS._dimTerrain.x);
    uint    divDTerY = iDivUp32(parCMS._dimTerrain.y);

    dim3	blocks_00(divDTerX,divDTerY, 1);
    dim3	blocks_01(divDTerX,divDTerY, parCMS.mNbByPix);

    /// Les données sont structurées par calques
    /// les echelles (du même subpixel) sont regroupées par calques consécutifs
    KernelPrepareCorrel<<<blocks_00,threads>>>(0,1,1,aSom_0.pData(),aSomSqr_0.pData());
    KernelPrepareCorrel<<<blocks_01,threads>>>(1,parCMS.aStepPix,parCMS.mNbByPix,aSom_1.pData(),aSomSqr_1.pData());

    ushort  modThreadZ = 8;

    dim3	threads_CorMS( 32, 32, modThreadZ);

    uint    bC =  iDivUp(data._maxDeltaZ,modThreadZ);

    dim3    blocks__CorMS(divDTerX,divDTerY,bC);

    /// calcul des couts de correlation multi-echelles

    Kernel__DoCorrel_MultiScale_Global<<<threads_CorMS,blocks__CorMS>>>(
                                                                          aSom_0   .pData(),
                                                                          aSomSqr_0.pData(),
                                                                          aSom_1   .pData(),
                                                                          aSomSqr_1.pData(),
                                                                          data._uInterval_Z   .pData(),
                                                                          data._uCost         .pData());

    //    aSom1.syncHost();
    //    aSom1.hostData.OutputValues();

    data._uCost.syncHost();
//    data._uCost.hostData.OutputValues(4*8);

    aSom_0   .Dealloc();
    aSomSqr_0.Dealloc();
    aSom_1   .Dealloc();
    aSomSqr_1.Dealloc();

}
