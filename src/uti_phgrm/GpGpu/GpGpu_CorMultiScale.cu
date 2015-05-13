#include "GpGpu/GpGpu_CommonHeader.h"
//#include "GpGpu/GpGpu_TextureTools.cuh"
#include "GpGpu/GpGpu_Interface_CorMultiScale.h"
#include <stdio.h>


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


#define THREAD_CMS_PREPARE	32
#define THREAD_CMS			8

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
    printf("[%d,%d,%d|%d,%d,%d]  ",threadIdx.x,threadIdx.y ,threadIdx.z,blockIdx.x ,blockIdx.y ,blockIdx.z);
}

__device__
inline    bool GET_Val_BIT(const U_INT1 * aData,int anX)
{
    return (aData[anX/8] >> (7-anX %8) ) & 1;
}


template<ushort idTexture>
__device__
inline    texture< pixel,cudaTextureType2D>  getMask()
{
	return Texture_Masq_Erod_00;
}

template<>
__device__
inline    texture< pixel,cudaTextureType2D>  getMask<0>()
{
	return Texture_Masq_Erod_00;
}

template<>
__device__
inline    texture< pixel,cudaTextureType2D>  getMask<1>()
{
	return Texture_Masq_Erod_01;
}

inline __device__ int dElise_div(int a,int b)
{
    int res = a / b;
    return res - ((res * b) > a);
}

template<ushort id> inline
__device__ uint2 getSizeImage()
{
	return make_uint2(0);
}

template<> inline
__device__ uint2 getSizeImage<0>()
{
	return cstPCMS.mSIg0;
}

template<> inline
__device__ uint2 getSizeImage<1>()
{
	return cstPCMS.mSIg1;
}

template<ushort idTexture>
__device__
inline    bool IsOkErod(int2 pt)
{
    // TODO peut etre simplifier % et division

	const uint2 size = getSizeImage<idTexture>();

	const int ptxBy8 = sgpu::__div<8>(pt.x );				// pt.x >> 3 Division par 8
	const int modulo = pt.x - (sgpu::__mult<8>(ptxBy8 ))  ;// (ptxBy8<<3) multiplication par 8

	pixel mask8b = tex2D(getMask<idTexture>(),(float)(ptxBy8) + 0.5f,(float)pt.y + 0.5f);

	return ((mask8b >> (7-modulo ) ) & 1) && aI(pt,size);
}

template<ushort idTexture>
__device__
inline    texture< float,	cudaTextureType2DLayered >  getTexture()
{
	return idTexture == 0 ? texture_ImageEpi_00 : texture_ImageEpi_01;
}

template<>
__device__
inline    texture< float,	cudaTextureType2DLayered >  getTexture<0>()
{
	return texture_ImageEpi_00;
}

template<>
__device__
inline    texture< float,	cudaTextureType2DLayered >  getTexture<1>()
{
	return texture_ImageEpi_01;
}

template<ushort idTex>
__device__
inline    float getValImage(float2 pt,ushort nScale)
{
	return tex2DLayered(getTexture<idTex>(),pt.x + 0.5f,pt.y + 0.5f ,nScale);
}

template<ushort idTex,class T>
__device__ float getValImage(T pt,ushort nScale)
{
	return tex2DLayered(getTexture<idTex>(),(float)pt.x + 0.5f,(float)pt.y + 0.5f ,nScale);
}

template<class T> inline
__device__ void TO_COST(float cost,T& destCOST,pixel* pix)
{
	//destCOST = (T)cost;
}

template<> inline
__device__ void TO_COST(float cost,float& destCOST,pixel* pix )
{
	destCOST = cost;
}

template<> inline
__device__ void TO_COST(float cost,ushort& destCOST,pixel* pix)
{
	if(cost >= 0.f)
	{

		destCOST = (ushort)(rintf((float)cost*(float)1e4));
		pix[0]	 = (pixel)max(0.0,min(255.0,rintf(128.0*(2.0-cost)-0.5)));
	}
	else
	{
		destCOST = 10123;
		pix[0]	 = 123;
	}
}

__global__
void projectionMasqImage(float * dataPixel,uint3 dTer)
{

    if(blockIdx.x > cstPCMS._dimTerrain.x || blockIdx.y > cstPCMS._dimTerrain.y)
        return;

	const int2 pt	= make_int2(blockIdx.x,blockIdx.y);

//	float valImage	= tex2DLayered(getTexture<blockIdx.z>(),pt.x + 0.5f,pt.y + 0.5f ,0);

//	dataPixel[to1D(pt,dTer)] = IsOkErod<blockIdx.z>(pt) ? valImage/(32768.f) : 0;
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
template<ushort idTex>
__global__
void KernelPrepareCorrel(float aStepPix, ushort mNbByPix, float* mSom, float* mSomSqr)
{

    // point image
	const uint2 ptTer	= make_uint2((sgpu::__mult<THREAD_CMS_PREPARE>(blockIdx.x)) + threadIdx.x,(sgpu::__mult<THREAD_CMS_PREPARE>(blockIdx.y))+ threadIdx.y);

	const uint2 szImage = getSizeImage<idTex>();

	if(oSE(ptTer,szImage))
        return;

    // indice de l'etape sub pixelaire, le maximum étant cPCencus.mNbByPix
    const ushort    aPhase   =   (ushort)blockIdx.z;

    // la dimension du cache, la cache stocke des precaluls pour la corrélation
	const uint3     dimCache =   make_uint3(szImage.x,szImage.y,mNbByPix*cstPCMS.aNbScale);

    // le décalage sub pixelaire
    const float     cStepPix =   ((float)aPhase)*aStepPix;

    // point de l'image pour cette etape sub pixelaire
	const float2    ptImage  =   make_float2((float)ptTer.x + cStepPix,(float)ptTer.y);

    float aGlobSom      = 0;
    float aGlobSomSqr   = 0;
    float aGlobPds      = 0;

    // pour toutes les echelles
    for (int aKS=0 ; aKS< cstPCMS.aNbScale ; aKS++)
    {
        float   aSom    = 0;
        float   aSomSqr = 0;
		const short2 *aVP     = cstPCMS.aVV[aKS];
		const ushort  aNbP    = cstPCMS.size_aVV[aKS];
		const float   aPdsK   = cstPCMS.aVPds[aKS];

        // pour les éléments de la vignettes
        for (int aKP=0 ; aKP<aNbP ; aKP++)
        {
			const float	 aV = getValImage<idTex>(ptImage+aVP[aKP],aKS);
			aSom	+= aV;
            aSomSqr += aV*aV;
        }

        aGlobSom    += aSom     * aPdsK;
        aGlobSomSqr += aSomSqr  * aPdsK;
		aGlobPds    += aPdsK    * aNbP; // TODO peut etre précalculer

        // indice dans le cache
		const uint3     p3d        =   make_uint3(ptTer.x,ptTer.y,aPhase*mNbByPix + aKS);

        // Ecriture dans le cache des
		const uint pSom = to1D(p3d,dimCache);

		mSom    [pSom] = fdividef(aGlobSom,aGlobPds);
		mSomSqr [pSom] = fdividef(aGlobSomSqr,aGlobPds);

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
        bool    ModeMax,
        ushort  aPhase)
{

    float aMaxCor = -1;
    float aCovGlob = 0;
    float aPdsGlob = 0;

    // pt float dans l'image 1
	const float2      aFG1      =   f2X(cstPCMS.aStepPix*(float)aPhase)+  make_float2(aPG1.x,aPG1.y);

    int aNbScale = cstPCMS.aNbScale;
    for (int aKS=0 ; aKS< aNbScale ; aKS++)
    {
		//bool   aLast   = (aKS==(aNbScale-1));

		const short2*aVP	= cstPCMS.aVV[aKS];
		const float  aPds	= cstPCMS.aVPds[aKS];
		const ushort aNbP	= cstPCMS.size_aVV[aKS];
		float		 aCov   = 0;

		aPdsGlob += aPds * aNbP; // TODO peut etre pre calcul
        for (int aKP=0 ; aKP<aNbP ; aKP++)
        {
            const short2 aP = aVP[aKP];
			const float valima_0 = getValImage<0>(aPG0 + aP,aKS);
			const float valima_1 = getValImage<1>(aFG1 + aP,aKS);
            aCov += valima_0*valima_1;

        }

        aCovGlob += aCov * aPds;

		//        if (ModeMax || aLast)
		//        {
		const uint  pit0   =   to1D(make_uint3(aPG0.x,aPG0.y,aKS),cstPCMS.mSIg0);
		const uint  pit1   =   to1D(make_uint3(aPG1.x,aPG1.y,aKS + cstPCMS.mNbByPix*aPhase),cstPCMS.mSIg1);

		const float aM1    =   aSom1 [pit0];
		const float aM2    =   aSom2 [pit1];

		const float aM11   =   aSom11[pit0] - aM1*aM1;
		const float aM22   =   aSom22[pit1] - aM2*aM2;
		const float aM12   =   fdividef(aCovGlob,aPdsGlob)- aM1 * aM2;

		const float aCor = (aM12 * abs(aM12)) /max(cstPCMS.anEpsilon,aM11*aM22);
		aMaxCor = max(aMaxCor,aCor);

		{
			//		if (ModeMax)
			//		{
			//			float aCor = (aM12 * abs(aM12)) /max(cstPCMS.anEpsilon,aM11*aM22);
			//			aMaxCor = max(aMaxCor,aCor);
			//		}
			//		else
			//			return aM12 / sqrt(max(cstPCMS.anEpsilon,aM11*aM22));
			// }
		}
    }
    return (aMaxCor > 0) ? sqrt(aMaxCor) : - sqrt(-aMaxCor) ;
}


template<class T>
__global__
void Kernel__DoCorrel_MultiScale_Global(float* aSom1,float*  aSom11,float* aSom2,float*  aSom22,short2 *nappe, T *cost,pixel* pix = NULL,uint* pit = NULL)
{

    // point image
	const   int2  an  =   make_int2(sgpu::__mult<THREAD_CMS>(blockIdx.x)+ threadIdx.x,sgpu::__mult<THREAD_CMS>(blockIdx.y) + threadIdx.y);

    // sortir si le point est en dehors du terrain
    if(oSE(an,cstPCMS._dimTerrain))
        return;

    //      pt int dans l'image 0
    const   int2     aPIm0       =   an + cstPCMS.anOff0;

	if (IsOkErod<0>(aPIm0) && aPIm0.x < cstPCMS.mSIg0.x && aPIm0.y < cstPCMS.mSIg0.y)
    {

        // Z relatif au thread
		const ushort thZ        =	sgpu::__mult<THREAD_CMS>(blockIdx.z) + threadIdx.z;
		const uint	 pitTer		=	to1D(an,cstPCMS._dimTerrain);
		const short2 _nappe		=	nappe[pitTer];
		const uint	pitCost		=	pit[pitTer] + thZ;
		T&				_cost   =	cost[pitCost];

		pixel *locPix		    =	pix + pitCost;

		const short     aZ0     =  _nappe.x;

        const int       DeltaZ  =  abs(_nappe.y-aZ0);

		if(thZ>=DeltaZ || DeltaZ > 512) // TODO Attention 512 nappeMAX
            return; // TODO on pourrait eventuellement affacter la valeur du cout par defaut.... mais bof

        // z Absolu
        const short aZ = aZ0 + (short)thZ;

        // calcul de la phase
        // Attention probleme avec valeur negative et le modulo
        const ushort aPhase = (ushort)((abs((int)aZ))%cstPCMS.mNbByPix);

		const int anOffset  = dElise_div((int)aZ,cstPCMS.mNbByPix); // TODO peut etre simplifier avec l'opération précédente

        const   int2    aIm1SsPx   =   an + cstPCMS.anOff1;
        //      pt int dans l'image 1
		const   int2    aPIm1      =   aIm1SsPx + i2X(anOffset);

        float           aCost      =   cstPCMS.mAhDefCost;

		if (IsOkErod<1>(aPIm1) && aPIm1.x < cstPCMS.mSIg1.x && aPIm1.y < cstPCMS.mSIg1.y) // TODO ---> attention integrer les dimensions correctes de l'image
        {

#ifdef modeMixte
			aCost = Quick_MS_CorrelBasic_Center(aPIm0,aPIm1,aSom1,aSom11,aSom2,aSom22,/*gpu_anOffset,*/cstPCMS.aModeMax,aPhase);
			float aGlobCostCorrel   = 0;
			aGlobCostCorrel = aCost;

			if (cstPCMS.DoMixte)
			{

				float aGlobCostBasic    = 0;
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
#else
			aCost = 1.f-Quick_MS_CorrelBasic_Center(aPIm0,aPIm1,aSom1,aSom11,aSom2,aSom22,cstPCMS.aModeMax,aPhase);
#endif
        }

		TO_COST(aCost,_cost,locPix);
    }
}

extern "C" void LaunchKernel__Correlation_MultiScale(dataCorrelMS &data,const_Param_Cor_MS &parCMS)
{

    // Cache device
	CuDeviceData3D<float>  aSom_0;
    CuDeviceData3D<float>  aSomSqr_0;

    CuDeviceData3D<float>  aSom_1;
    CuDeviceData3D<float>  aSomSqr_1;

	aSom_0.CGObject::SetName("aSom_0");
	aSomSqr_0.CGObject::SetName("aSomSqr_0");

	aSom_1.CGObject::SetName("aSom_1");
	aSomSqr_1.CGObject::SetName("aSomSqr_1");

	aSom_0   .Malloc (parCMS.mSIg0,parCMS.aNbScale);                  //  pas de sous echantillonnage
	aSomSqr_0.Malloc (parCMS.mSIg0,parCMS.aNbScale);

	aSom_1   .Malloc (parCMS.mSIg1,parCMS.aNbScale*parCMS.mNbByPix);  // avec sous echantillonnage
	aSomSqr_1.Malloc (parCMS.mSIg1,parCMS.aNbScale*parCMS.mNbByPix);

	const dim3	threads( THREAD_CMS_PREPARE, THREAD_CMS_PREPARE, 1);

	const uint  divDTerX0 = sgpu::__iDivUp<THREAD_CMS_PREPARE>(parCMS.mSIg0.x);
	const uint  divDTerY0 = sgpu::__iDivUp<THREAD_CMS_PREPARE>(parCMS.mSIg0.y);

	const uint  divDTerX1 = sgpu::__iDivUp<THREAD_CMS_PREPARE>(parCMS.mSIg1.x);
	const uint  divDTerY1 = sgpu::__iDivUp<THREAD_CMS_PREPARE>(parCMS.mSIg1.y);

	const dim3	blocks_00(divDTerX0,divDTerY0, 1);
	const dim3	blocks_01(divDTerX1,divDTerY1, parCMS.mNbByPix);

    /// Les données sont structurées par calques
    /// les echelles (du même subpixel) sont regroupées par calques consécutifs
	KernelPrepareCorrel<0><<<blocks_00,threads>>>(1,1,aSom_0.pData(),aSomSqr_0.pData());

    getLastCudaError("KernelPrepareCorrel 0");

	KernelPrepareCorrel<1><<<blocks_01,threads>>>(parCMS.aStepPix,parCMS.mNbByPix,aSom_1.pData(),aSomSqr_1.pData());

    getLastCudaError("KernelPrepareCorrel 1");
//    aSom_0.syncHost();
//    aSom_0.hostData.OutputValues();
//    getchar();

	const ushort  modThreadZ	= THREAD_CMS;
	const ushort  modXTHread	= THREAD_CMS;
	const ushort  modYTHread	= THREAD_CMS;
	const dim3	threads_CorMS( modXTHread,modYTHread , modThreadZ);
	const uint  bC			= sgpu::__iDivUp<THREAD_CMS>(data._maxDeltaZ);
	const uint	divDTerX	= sgpu::__iDivUp<THREAD_CMS>(parCMS._dimTerrain.x);
	const uint	divDTerY	= sgpu::__iDivUp<THREAD_CMS>(parCMS._dimTerrain.y);
	const dim3  blocks__CorMS(divDTerX,divDTerY,bC);

//	data._uCost.hostData.Fill(parCMS.mAhDefCost);
//	data._uCost.syncDevice();

	if(parCMS.mDyRegGpu)
	{
		Kernel__DoCorrel_MultiScale_Global<<<blocks__CorMS,threads_CorMS>>>(
																			  aSom_0   .pData(),
																			  aSomSqr_0.pData(),
																			  aSom_1   .pData(),
																			  aSomSqr_1.pData(),
																			  data._uInterval_Z   .pData(),
																			  data._uCostu        .pData(),
																			  data._uCostp        .pData(),
																			  data._uPit		  .pData()
																			  );

		getLastCudaError("Kernel__DoCorrel_MultiScale_Global float");

		data._uCostu.syncHost();
		data._uCostp.syncHost();
	}
	else
	{
		Kernel__DoCorrel_MultiScale_Global<<<blocks__CorMS,threads_CorMS>>>(
																			  aSom_0   .pData(),
																			  aSomSqr_0.pData(),
																			  aSom_1   .pData(),
																			  aSomSqr_1.pData(),
																			  data._uInterval_Z   .pData(),
																			  data._uCostf         .pData());

		getLastCudaError("Kernel__DoCorrel_MultiScale_Global float");

		data._uCostf.syncHost();
	}

    aSom_0   .Dealloc();
    aSomSqr_0.Dealloc();
    aSom_1   .Dealloc();
    aSomSqr_1.Dealloc();

}
