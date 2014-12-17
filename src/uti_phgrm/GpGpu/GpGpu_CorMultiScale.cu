
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


///
static __constant__ const_Param_Cor_MS     cPCencus;

extern "C" void paramCorMultiScale2Device( const_Param_Cor_MS &param )
{
  checkCudaErrors(cudaMemcpyToSymbol(cPCencus, &param, sizeof(const_Param_Cor_MS)));
}

texture< float,	cudaTextureType2DLayered >      texture_ImageEpi_00;
texture< float,	cudaTextureType2DLayered >      texture_ImageEpi_01;
texture< pixel,	cudaTextureType2D >             Texture_Masq_Erod_00;
texture< pixel,	cudaTextureType2D >             Texture_Masq_Erod_01;

extern "C" textureReference& texture_ImageEpi(int nEpi){return nEpi == 0 ? texture_ImageEpi_00 : texture_ImageEpi_01;}

extern "C" textureReference* pTexture_ImageEpi(int nEpi){return nEpi == 0 ? &texture_ImageEpi_00 : &texture_ImageEpi_01;}

extern "C" textureReference* ptexture_Masq_Erod(int nEpi){return nEpi == 0 ? &Texture_Masq_Erod_00 : &Texture_Masq_Erod_01;}

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


__device__
inline    bool IsOkErod(uint3 pt)
{
    // TODO peut etre simplifier % et division

    const int ptxBy8 = pt.x >> 3;           // pt.x >> 3 Division par 8
    const int modulo = pt.x - (ptxBy8 << 3)  ;// (ptxBy8<<3) multiplication par 8

    pixel mask8b = tex2D(getMask(pt.z),(float)(ptxBy8) + 0.5f,(float)pt.y + 0.5f);

    return (mask8b >> (7-modulo ) ) & 1;
}

__device__
inline    bool IsOkErod(uint2 pt,ushort idi)
{
    return IsOkErod(make_uint3(pt.x,pt.y,idi));
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

/* Algorithme de precalcul de corrélation
__device__
inline    void correl(float2 pt,ushort iDi, float* mdata1, float* mdata2)
{
    float aGlobSom1 = 0;
    float aGlobSom2 = 0;
    float aGlobPds  = 0;

    for (int aKS=0 ; aKS< cPCencus.aNbScale ; aKS++)
    {
        float   aSom1   = 0;
        float   aSom2   = 0;
        short2 *aVP     = cPCencus.aVV[aKS];
        ushort  aNbP    = cPCencus.size_aVV[aKS];
        float   aPdsK   = cPCencus.aVPds[aKS];

        for (int aKP=0 ; aKP<aNbP ; aKP++)
        {
            const short2 aP = aVP[aKP];
            //const uint ptV  = make_uint2(pt.x+aP.x,)
            float aV = getValImage(pt+aP,iDi,aKS);
            aSom1 += aV;
            aSom2 += aV*aV;
        }

        aGlobSom1 += aSom1 * aPdsK;
        aGlobSom2 += aSom2 * aPdsK;
        aGlobPds += aPdsK * aNbP;

//        mData1[aKS][aYGlob][aXGlob] = aGlobSom1 / aGlobPds;
//        mData2[aKS][aYGlob][aXGlob] = aGlobSom2 / aGlobPds;

    }
}
*/

__device__
inline    float CorrelBasic_Center(

    const float2 & aPG0,
    const float2 & aPG1,

//    float ***  aSom1,
//    float ***  aSom11,
//    float ***  aSom2,
//    float ***  aSom22,
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

    const float2 aPG1_x2 = make_float2(aPG1.x + aPx2,aPG1.y);

    int aNbScale = cPCencus.aNbScale;
    for (int aKS=0 ; aKS< aNbScale ; aKS++)
    {
         bool   aLast   = (aKS==(aNbScale-1));
         short2*aVP     = cPCencus.aVV[aKS];
         float  aPds    = cPCencus.aVPds[aKS];
         float  aCov    = 0;
         ushort aNbP    = cPCencus.size_aVV[aKS];

//         float ** anIm1= aVBOI1[aKS]->data();
//         float ** anIm2= aVBOI2[aKS]->data();

         aPdsGlob += aPds * aNbP;
         for (int aKP=0 ; aKP<aNbP ; aKP++)
         {
             const short2 aP = aVP[aKP];

             const float valima_0 = getValImage(aPG0 + aP,0,aKS);
             const float valima_1 = getValImage(aPG1_x2 + aP,1,aKS);

             aCov += valima_0*valima_1;
             //aCov += anIm1[aP.y][aP.x]*anIm2[aP.y][aP.x+aPx2];
         }

         aCovGlob += aCov * aPds;

         if (ModeMax || aLast)
         {
//             float aM1  = aSom1 [aKS][aPG0.y][aPG0.x];
//             float aM2  = aSom2 [aKS][aPG1.y][aPG1.x];
//             float aM11 = aSom11[aKS][aPG0.y][aPG0.x] - aM1*aM1;
//             float aM22 = aSom22[aKS][aPG1.y][aPG1.x] - aM2*aM2;

             const uint3 pt0    =   make_uint3(aPG0.x,aPG0.y,aKS);
             const uint3 pt1    =   make_uint3(aPG1_x2.x,aPG1_x2.y,aKS + aNbScale*aPhase);
             const uint3 dim    =   make_uint3(cPCencus._dimTerrain.x,cPCencus._dimTerrain.x,1);

             const float aM1    =   aSom1 [to1D(pt0,dim)];
             const float aM2    =   aSom2 [to1D(pt1,dim)];

             const float aM11   =   aSom11[to1D(pt0,dim)] - aM1*aM1;
             const float aM22   =   aSom22[to1D(pt1,dim)] - aM2*aM2;

             const float aM12   =   aCovGlob / aPdsGlob   - aM1 * aM2;

             if (ModeMax)
             {
                float aCor = (aM12 * abs(aM12)) /max(cPCencus.anEpsilon,aM11*aM22);
                aMaxCor = max(aMaxCor,aCor);
             }
             else
                return aM12 / sqrt(max(cPCencus.anEpsilon,aM11*aM22));
        }

    }
    return (aMaxCor > 0) ? sqrt(aMaxCor) : - sqrt(-aMaxCor) ;
}

__global__
void projectionMasqImage(float * dataPixel,uint3 dTer)
{

    if(blockIdx.x > cPCencus._dimTerrain.x || blockIdx.y > cPCencus._dimTerrain.y)
        return;

    const uint3 pt = make_uint3(blockIdx.x,blockIdx.y,blockIdx.z);

    float valImage = tex2DLayered(pt.z == 0 ? texture_ImageEpi_00 : texture_ImageEpi_01 ,pt.x + 0.5f,pt.y + 0.5f ,0);

    dataPixel[to1D(pt,dTer)] = IsOkErod(pt) ? valImage/(32768.f) : 0;
}

__global__
void KernelDoCorrelMultiScale(float* aSom1,float*  aSom11,float* aSom2,float*  aSom22,short2 *nappe, float *cost)
{

    // ??? TODO à cabler
    bool    DoMixte     = false;
    bool    aModeMax    = false;
    float   aSeuilHC    = 1.0;
    float   aSeuilBC    = 1.0;
    // ???

    int2    pt  =   make_int2(blockIdx.x*blockDim.x + threadIdx.x,blockIdx.y*blockDim.y + threadIdx.y);
    uint    tZ  =   blockIdx.z*blockDim.z + threadIdx.z;

    if(oSE(pt,cPCencus._dimTerrain))
        return;

    const int2  aPIm0   =   pt.x+cPCencus.anOff0; // TODO Attention au unsigned
    const bool  OkIm0   =   IsOkErod(make_uint2(aPIm0),0);
    const short2 iZ     =   nappe[to1D(pt,cPCencus._dimTerrain)];
    int aZ0             =   iZ.x;
    const int aZ1       =   iZ.y;
    const int DeltaZ    =   abs(aZ1-aZ0);

    if(tZ>DeltaZ)
        return; // TODO on pourrait eventuellement affacter la valeur du cout par defaut.... mais bof

    int aZI = aZ0 + tZ;

    const int2 aIm1SsPx =  pt + cPCencus.anOff1;

    // float aGlobCostGraphe = 0;
    float aGlobCostBasic  = 0;
    float aGlobCostCorrel = 0;

    float aCost = cPCencus.mAhDefCost;

    if (OkIm0)
    {
        //
        // anOffset calcul de anOffset
        ///
        int aPhase = tZ%cPCencus.mNbByPix;

        while ((aZ0%cPCencus.mNbByPix) != aPhase) aZ0++;

        int anOffset    = aZ0 / cPCencus.mNbByPix;
        anOffset        = anOffset - ((anOffset * cPCencus.mNbByPix) > aZ0);
        int sOff        = abs(aZI-aZ0)/cPCencus.mNbByPix; // --> doit tomber juste
        anOffset       += sOff;

        const uint2 aPIm1 = make_uint2(aIm1SsPx.x+anOffset,aIm1SsPx.y);

        if (IsOkErod(aPIm1,1))
        {

            // TODO à cabler avec correl(uint2 pt,ushort iDi)
            //float*  aSom1;  // ---> peut precalculer dans un kernel precedent!
            //float*  aSom11; // ---> peut precalculer dans un kernel precedent!

            // TODO à cabler avec correl(uint2 pt,ushort iDi)
            //float*  aSom2; // ---> peut-etre precalculer dans un kernel precedent! A VERIFIER!!!
            //float*  aSom22;// ---> peut-etre precalculer dans un kernel precedent! A VERIFIER!!!

            const float2 faPIm0 = make_float2((float)aPIm0.x,(float)aPIm0.y); // TODO ajouter le pas sub pixelaire            
            const float2 faPIm1 = make_float2((float)aPIm1.x,(float)aPIm1.y); // TODO ajouter le pas sub pixelaire
            const int    aPx2   = aPhase*cPCencus.aStepPix;

            aCost = CorrelBasic_Center(faPIm0,faPIm1,aSom1,aSom11,aSom2,aSom22,aPx2,aModeMax,aPhase);

            aGlobCostCorrel = aCost;

            if (DoMixte)
            {
               if(aGlobCostCorrel>aSeuilHC)

                    aCost = aGlobCostCorrel;

               else if (aGlobCostCorrel>aSeuilBC)
               {
                    float aPCor =  (aGlobCostCorrel - aSeuilBC) / (aSeuilHC-aSeuilBC);
                    aCost       =  aPCor * aGlobCostCorrel + (1-aPCor) * aSeuilBC *  aGlobCostBasic;
               }
               else
                    aCost =  aSeuilBC *  aGlobCostBasic;
            }

//            aCost = 1.f-aCost;

            const uint3 ptCost  = make_uint3(pt.x,pt.y,tZ);
            const uint3 dimCost = make_uint3(cPCencus._dimTerrain.x,cPCencus._dimTerrain.y,1);

            cost[to1D(ptCost,dimCost)] = 1.f-aCost;

        }
        else return;
    }
    else
        return;

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

__global__
void KernelPrepareCorrel(ushort idImage,float aStepPix, ushort mNbByPix, float* mData1, float* mData2)
{

    // point image
    const uint2     pt          =   make_uint2(blockIdx.x*blockDim.x + threadIdx.x,blockIdx.y*blockDim.y + threadIdx.y);

    if(oSE(pt,cPCencus._dimTerrain))
        return;

    // indice de l'etape sub pixelaire, le maximum étant cPCencus.mNbByPix
    const ushort    etapeSub    =   (ushort)blockIdx.z;

    // la dimension du cache, la cache stocke des precaluls pour la corrélation
    const uint3     dimCache    =   make_uint3(cPCencus._dimTerrain.x,cPCencus._dimTerrain.y,mNbByPix*cPCencus.aNbScale);

    // le décalage sub pixelaire
    const float     cStepPix    =   ((float)etapeSub)*aStepPix;

    // point de l'image pour cette etape sub pixelaire
    const float2    ptImage     =   make_float2((float)pt.x + cStepPix,(float)pt.y);

    float aGlobSom1 = 0;
    float aGlobSom2 = 0;
    float aGlobPds  = 0;

    // pour toutes les echelles
    for (int aKS=0 ; aKS< cPCencus.aNbScale ; aKS++)
    {
        float   aSom1   = 0;
        float   aSom2   = 0;
        short2 *aVP     = cPCencus.aVV[aKS];
        ushort  aNbP    = cPCencus.size_aVV[aKS];
        float   aPdsK   = cPCencus.aVPds[aKS];

        // pour les éléments de la vignettes
        for (int aKP=0 ; aKP<aNbP ; aKP++)
        {
            const short2 aP = aVP[aKP];
            float aV = getValImage(ptImage+aP,idImage,aKS);
            aSom1 += aV;
            aSom2 += aV*aV;
        }

        aGlobSom1   += aSom1 * aPdsK;
        aGlobSom2   += aSom2 * aPdsK;
        aGlobPds    += aPdsK * aNbP;

        // indice dans le cache
        const uint3     p3d        =   make_uint3(pt.x,pt.y,etapeSub*mNbByPix + aKS);

        // Ecriture dans le cache des
        mData1[to1D(p3d,dimCache)] = aGlobSom1 / aGlobPds;
        mData2[to1D(p3d,dimCache)] = aGlobSom2 / aGlobPds;

    }
}


__global__
void Kernel__DoCorrel_MultiScale_Global(float* aSom1,float*  aSom11,float* aSom2,float*  aSom22,short2 *nappe, float *cost)
{

}


extern "C" void LaunchKernel__Correlation_MultiScale(dataCorrelMS &data,const_Param_Cor_MS &param)
{
    // Cache device
    CuDeviceData3D<float>  aSom1;
    //CuUnifiedData3D<float>  aSom1;
    CuDeviceData3D<float>  aSom11;
    CuDeviceData3D<float>  aSom2;
    CuDeviceData3D<float>  aSom22;

    aSom1. Malloc (param._dimTerrain,param.aNbScale); //  pas de sous echantillonnage
    aSom11.Malloc (param._dimTerrain,param.aNbScale);

    aSom2. Malloc (param._dimTerrain,param.aNbScale*param.mNbByPix); // avec sous echantillonnage
    aSom22.Malloc (param._dimTerrain,param.aNbScale*param.mNbByPix);

    dim3	threads( 32, 32, 1);
    dim3	blocks_00(iDivUp32(param._dimTerrain.x),iDivUp32(param._dimTerrain.y), 1);
    dim3	blocks_01(iDivUp32(param._dimTerrain.x),iDivUp32(param._dimTerrain.y), param.mNbByPix);

    KernelPrepareCorrel<<<blocks_00,threads>>>(0,1,1,aSom1.pData(),aSom11.pData());
    KernelPrepareCorrel<<<blocks_01,threads>>>(1,param.aStepPix,param.mNbByPix,aSom2.pData(),aSom22.pData());

    Kernel__DoCorrel_MultiScale_Global<<<blocks_00,threads>>>(
                                                        aSom1   .pData(),
                                                        aSom11  .pData(),
                                                        aSom2   .pData(),
                                                        aSom22  .pData(),
                                                        data._uInterval_Z   .pData(),
                                                        data._uCost         .pData());
//    aSom1.syncHost();
//    aSom1.hostData.OutputValues();

    aSom1. Dealloc();
    aSom11.Dealloc();
    aSom2. Dealloc();
    aSom22.Dealloc();

}
