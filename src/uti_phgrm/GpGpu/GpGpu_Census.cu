
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
static __constant__ constantParameterCensus     cParamCencus;

extern "C" void paramCencus2Device( constantParameterCensus &param )
{
  checkCudaErrors(cudaMemcpyToSymbol(cParamCencus, &param, sizeof(constantParameterCensus)));
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

__global__ void projectionMasq(float * dataPixel,uint3 dTer)
{
    const uint3 ptTer = make_uint3(blockIdx.x,blockIdx.y,blockIdx.z);
    const uint2 ptMTer = make_uint2(blockIdx.x/8,blockIdx.y);

    pixel val = tex2DLayered(Texture_Masq_Erod,ptMTer.x + 0.5f,ptMTer.y + 0.5f ,blockIdx.z);

    bool OkErod = (val >> (7-ptTer.x %8) ) & 1;

    dataPixel[to1D(ptTer,dTer)] = OkErod ? 1.f : 0;
}

extern "C" void LaunchKernelCorrelationCensus(dataCorrelMS &data,constantParameterCensus &param)
{
    dim3	threads( 1, 1, 1);
    dim3	blocks(param._dimTerrain.y , param._dimTerrain.x, 2);

    CuHostData3D<float>     hData;
    CuDeviceData3D<float>   dData;

    uint3 dTer = make_uint3(param._dimTerrain.y , param._dimTerrain.x,2);
    uint2 _2dTer = make_uint2(param._dimTerrain.y , param._dimTerrain.x);

    hData.Malloc(_2dTer,2);
    dData.Malloc(_2dTer,2);
    hData.Fill(0.f);
    dData.Memset(0);

    projectionMasq<<<blocks, threads>>>(dData.pData(),dTer);

    dData.CopyDevicetoHost(hData);

    GpGpuTools::Array1DtoImageFile(hData.pData(),"ET_HOP_0.pmg",hData.GetDimension());
    GpGpuTools::Array1DtoImageFile(hData.pData()+size(hData.GetDimension()),"ET_HOP_1.pmg",hData.GetDimension());
}
