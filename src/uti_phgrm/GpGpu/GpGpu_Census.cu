
#include "GpGpu/GpGpu_CommonHeader.h"
//#include "GpGpu/GpGpu_TextureTools.cuh"
#include "GpGpu/GpGpu_Interface_Census.h"

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
static __constant__ constantParameterCensus     cPCencus;

extern "C" void paramCencus2Device( constantParameterCensus &param )
{
  checkCudaErrors(cudaMemcpyToSymbol(cPCencus, &param, sizeof(constantParameterCensus)));
}

texture< float,	cudaTextureType2DLayered >      texture_ImageEpi_00;
texture< float,	cudaTextureType2DLayered >      texture_ImageEpi_01;
texture< pixel,	cudaTextureType2DLayered >      Texture_Masq_Erod;

extern "C" textureReference& texture_ImageEpi(int nEpi){return nEpi == 0 ? texture_ImageEpi_00 : texture_ImageEpi_01;}

extern "C" textureReference* pTexture_ImageEpi(int nEpi){return nEpi == 0 ? &texture_ImageEpi_00 : &texture_ImageEpi_01;}

extern "C" textureReference& texture_Masq_Erod(){return Texture_Masq_Erod;}

__device__
inline    bool GET_Val_BIT(const U_INT1 * aData,int anX)
{
    return (aData[anX/8] >> (7-anX %8) ) & 1;
}

__device__
inline    bool IsOkErod(uint3 pt)
{
    // TODO peut etre simplifier % et division
    pixel mask8b = tex2DLayered(Texture_Masq_Erod,pt.x/8 + 0.5f,pt.y + 0.5f ,pt.z);

    return (mask8b >> (7-pt.x %8) ) & 1;
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
inline    float getValImage(uint2 pt,ushort iDi,ushort nScale)
{
    return tex2DLayered(getTexture(iDi),pt.x + 0.5f,pt.y + 0.5f ,nScale);
}

__device__
inline    void correl(uint2 pt,ushort iDi)
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

__device__
inline    float CorrelBasic_Center(
    const uint2 & aPG0,
    const uint2 & aPG1,
//    float ***  aSom1,
//    float ***  aSom11,
//    float ***  aSom2,
//    float ***  aSom22,
    float*  aSom1,
    float*  aSom11,
    float*  aSom2,
    float*  aSom22,
    //int     aPx2, // ---> TODO Surement A virer : decalage sub pixel apriori il sera égale à 0!!
    bool    ModeMax)
{
    float aMaxCor = -1;
    float aCovGlob = 0;
    float aPdsGlob = 0;

    //const uint2 aPG1_x2 = make_uint2(aPG1.x + aPx2,aPG1.y);

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
             const float valima_1 = getValImage(aPG1 + aP,1,aKS);

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
             float aM1  = aSom1 [aKS];
             float aM2  = aSom2 [aKS];
             float aM11 = aSom11[aKS] - aM1*aM1;
             float aM22 = aSom22[aKS] - aM2*aM2;
             float aM12 = aCovGlob / aPdsGlob - aM1 * aM2;

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

    if(blockIdx.x > cPCencus._dimTerrain.y || blockIdx.y > cPCencus._dimTerrain.x)
        return;

    const uint3 pt = make_uint3(blockIdx.x,blockIdx.y,blockIdx.z);

    float valImage = tex2DLayered(pt.z == 0 ? texture_ImageEpi_00 : texture_ImageEpi_01 ,pt.x + 0.5f,pt.y + 0.5f ,0);

    dataPixel[to1D(pt,dTer)] = IsOkErod(pt) ? valImage/(32768.f) : 0;
}

__global__
void KernelDoCensusCorrel()
{

    // ??? TODO à cabler
    bool    DoMixte     = false;
    bool    aModeMax    = false;
    float   aSeuilHC    = 1.0;
    float   aSeuilBC    = 1.0;
    // ???

    uint anX = blockIdx.x;
    uint anY = blockIdx.y;
    uint Z   = threadIdx.x;

    if(anX > cPCencus._dimTerrain.y || anY > cPCencus._dimTerrain.x)
        return;

//    int** mTabZMin;
//    int** mTabZMax;

    const uint2 aPIm0   = make_uint2(anX+cPCencus.anOff0.x,anY+cPCencus.anOff0.x); // TODO Attention au unsigned
    const bool  OkIm0   = IsOkErod(aPIm0,0);

//    int aZ0 =  mTabZMin[anY][anX];
//    int aZ1 =  mTabZMax[anY][anX];

    int aXIm1SsPx = anX+cPCencus.anOff1.x;
    int aYIm1SsPx = anY+cPCencus.anOff1.y;

    // float aGlobCostGraphe = 0;
    float aGlobCostBasic  = 0;
    float aGlobCostCorrel = 0;

    float aCost = cPCencus.mAhDefCost;

    if (OkIm0)
    {
        int anOffset = Z;
        const uint2 aPIm1 = make_uint2(aXIm1SsPx+anOffset,aYIm1SsPx);

        if (IsOkErod(aPIm1,1))
        {

            // TODO à cabler avec correl(uint2 pt,ushort iDi)
            float*  aSom1;  // ---> peut precalculer dans un kernel precedent!
            float*  aSom11; // ---> peut precalculer dans un kernel precedent!

            // TODO à cabler avec correl(uint2 pt,ushort iDi)
            float*  aSom2; // ---> peut-etre precalculer dans un kernel precedent! A VERIFIER!!!
            float*  aSom22;// ---> peut-etre precalculer dans un kernel precedent! A VERIFIER!!!

            aCost = CorrelBasic_Center(aPIm0,aPIm1,aSom1,aSom11,aSom2,aSom22,aModeMax);

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

            aCost = 1.f-aCost;
        }
        else return;
    }
    else
        return;

}

extern "C" void LaunchKernelCorrelationCensus(dataCorrelMS &data,constantParameterCensus &param)
{
    dim3	threads( 1, 1, 1);
    dim3	blocks(param._dimTerrain.y , param._dimTerrain.x, 2);

    CuHostData3D<float>     hData;
    CuDeviceData3D<float>   dData;

    uint3 dTer  = make_uint3(param._dimTerrain.y , param._dimTerrain.x,2);
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
