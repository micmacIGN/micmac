#ifndef _SDATA2CORREL_H
#define _SDATA2CORREL_H
/** @addtogroup GpGpuDoc */
/*@{*/

#include "GpGpu/GpGpu_ParamCorrelation.cuh"

/// \cond
extern "C" textureReference&    getMaskGlobal();
extern "C" textureReference&	getMask();
extern "C" textureReference&	getImage();
extern "C" textureReference&	getProjection(int TexSel);
extern "C" textureReference&    getTexL_MaskImages();

#define SYNC    false
#define ASYNC   true
/// \endcond


///
/// \brief The cellules struct
/// Structure de cellules 3D
struct cellules
{
    ///
    /// \brief Zone
    /// La zone 2d
    ///
    Rect Zone;
    ///
    /// \brief Dz
    /// delta Z de la zone
    ///
	ushort Dz;

    cellules():
        Zone(MAXIRECT),
        Dz(INTERZ)
    {}
};

///
/// \brief The SData2Correl struct
///
struct SData2Correl
{

public:
    SData2Correl();

    ~SData2Correl();

	///
	/// \brief SetImages Initialise les images sur GPU
	/// \param dataImage
	/// \param dimImage
	/// \param nbLayer
	///
    void    SetImages( float* dataImage, uint2 dimImage, int nbLayer );

	///
	/// \brief SetGlobalMask Initialise les masques sur GPU
	/// \param dataMask
	/// \param dimMask
	///
    void    SetGlobalMask( pixel* dataMask, uint2 dimMask );

	///
	/// \brief MemsetHostVolumeProj Initialise la memoire des projections par une valeur iDef
	/// \param iDef
	///
    void    MemsetHostVolumeProj(int iDef);

	///
	/// \brief HostVolumeCost
	/// \param id
	/// \return le pointeur host du volume de corrélation
	///
    float*  HostVolumeCost(uint id);

	///
	/// \brief HostVolumeProj
	/// \return le pointeur host du volume de projection
	///
    float2* HostVolumeProj();

	///
	/// \brief HostRect
	/// \return le pointeur des rectangles images
	///
	uint2*	HostRect();

	///
	/// \brief DeviVolumeNOK
	/// \param s
	/// \return  le pointeur device des volumes des images correctes
	///
    uint*   DeviVolumeNOK(uint s);

	///
	/// \brief DeviVolumeCache
	/// \param s
	/// \return le pointeur device du cache des vecteurs centrés
	///
    float*  DeviVolumeCache(uint s);

	///
	/// \brief DeviVolumeCost
	/// \param s
	/// \return  Le pointeur device du volume de couts
	///
    float*  DeviVolumeCost(uint s);

	///
	/// \brief DeviRect
	/// \return Le pointeur device des rectangles images
	///
	uint2*	DeviRect();

	///
	/// \brief copyHostToDevice Copie les données vers le device
	/// \param param
	/// \param s
	///
    void    copyHostToDevice(pCorGpu param, uint s = 0);

	///
	/// \brief CopyDevicetoHost Copie les données vers le host
	/// \param idBuf
	/// \param s
	///
    void    CopyDevicetoHost(uint idBuf, uint s = 0);

	///
	/// \brief UnBindTextureProj relacher les textures sur le device
	/// \param s
	///
    void    UnBindTextureProj(uint s = 0);

	///
	/// \brief DeallocHostData Desalloue la mémoire host
	///
    void    DeallocHostData();

	///
	/// \brief DeallocDeviceData Désalloue la mémoire device
	///
    void    DeallocDeviceData();

	///
	/// \brief ReallocHostData réalloue la mémoire host
	/// \param zInter
	/// \param param
	///
    void    ReallocHostData(uint zInter, pCorGpu param);

	///
	/// \brief ReallocHostData réalloue la mémoire host
	/// \param zInter
	/// \param param
	/// \param idBuff
	///
    void    ReallocHostData(uint zInter, pCorGpu param, uint idBuff);

	///
	/// \brief ReallocDeviceData réalloue la mémoire device
	/// \param param
	///
    void    ReallocDeviceData(pCorGpu &param);   

	///
	/// \brief HostClassEqui
	/// \return Le pointeur des classes d'équivalence
	///
    ushort2 *HostClassEqui();

	///
	/// \brief ReallocConstData Réallocation des données constantes
	/// \param nbImages
	///
	void    ReallocConstData(uint nbImages);

	///
	/// \brief SyncConstData Synchronise les données constantes sur le device
	///
	void    SyncConstData();

	///
	/// \brief SetZoneImage Définir les dimensions des images
	/// \param idImage
	/// \param sizeImage
	/// \param r
	///
	void	SetZoneImage(const ushort& idImage, const uint2& sizeImage, const ushort2& r);

	///
	/// \brief DeviClassEqui
	/// \return Le pointeur device des classes d'équivalence
	///
    ushort2 *DeviClassEqui();

	///
	/// \brief SetMaskImages Transfert les masques sur le device
	/// \param dataMaskImages
	/// \param dimMaskImage
	/// \param nbLayer
	///
    void    SetMaskImages(pixel *dataMaskImages, uint2 dimMaskImage, int nbLayer);

private:

    void    ReallocDeviceData(int nStream, pCorGpu param);

    void    MallocInfo();

    textureReference& GetTeXProjection( int TexSel );

    CuHostData3D<float>         _hVolumeCost[2];
    CuHostData3D<float2>        _hVolumeProj;

    // TODO il semblerait qu'un uint2 suffirai....
    ///
    /// \brief _hRect   HOST     gestion des bords d'images
    ///
//    CuHostData3D<Rect>          _hRect;
    ///
    /// \brief _dRect   Device   gestion des bords d'images
    ///
//    CuDeviceData3D<Rect>        _dRect;

	CuUnifiedData3D<uint2>		_uRect;

    ///
    /// \brief _hClassEqui HOST     gestion des classes d'images
    ///
	//CuHostData3D<ushort2>       _hClassEqui;
    ///
    /// \brief _dClassEqui DEVICE    gestion des classes d'images
    ///
	//CuDeviceData3D<ushort2>     _dClassEqui;
	CuUnifiedData3D<ushort2>     _uClassEqui;


    CuDeviceData3D<float>       _d_volumeCost[NSTREAM];	// volume des couts
    CuDeviceData3D<float>       _d_volumeCach[NSTREAM];	// volume des calculs intermédiaires
    CuDeviceData3D<uint>        _d_volumeNIOk[NSTREAM];	// nombre d'image correct pour une vignette

    ImageGpGpu<pixel,cudaContext>           _dt_GlobalMask;
    ImageLayeredGpGpu<float,cudaContext>    _dt_LayeredImages;
    ImageLayeredGpGpu<pixel,cudaContext>    _dt_LayeredMaskImages;
    ImageLayeredGpGpu<float2,cudaContext>   _dt_LayeredProjection[NSTREAM];

    textureReference&           _texMaskGlobal;
    textureReference&           _TexMaskImages;
    textureReference&           _texImages;
    textureReference&           _texProjections_00;
    textureReference&           _texProjections_01;

    void DeviceMemset(pCorGpu &param, uint s = 0);
};

/*@}*/
#endif
