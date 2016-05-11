#ifndef ENGINE_H
#define ENGINE_H

//~ #define USE_MIPMAP_HANDLER

#ifdef USE_MIPMAP_HANDLER
	#include "MipmapHandler.h"
#endif

#include "cgldata.h"



class ViewportParameters
{
public:
    //! Default constructor
    ViewportParameters();

    //! Copy constructor
    ViewportParameters(const ViewportParameters& params);

    //! Destructor
    ~ViewportParameters();

    //!
    ViewportParameters & operator = (const ViewportParameters &);

    void    reset();

    void    ptSizeUp(bool up);

    float   changeZoom(float DChange)
    {
        if      (DChange > 0) m_zoom *= pow(2.f,  DChange *.05f);
        else if (DChange < 0) m_zoom /= pow(2.f, -DChange *.05f);

        return  m_zoom;
    }

    //! Current zoom
    float m_zoom;

    //! Point size
    int m_pointSize;

    //! Rotation and translation speed
    float m_speed;
};

class deviceIOTieFile
{
public:
	virtual void  load(QString aNameFile,QPolygonF& poly) = 0;
};

class deviceIOCamera
{
public:
	virtual cCamHandler*  loadCamera(QString aNameFile) = 0;
};

class deviceIOImage
{
public:
	virtual QImage*	loadImage(QString aNameFile,bool OPENGL = true) = 0;

	virtual QImage*	loadMask(QString aNameFile) = 0;

	virtual void	doMaskImage(QImage &mask,QString &aNameFile) = 0;
};

class cLoader
{

public:

    cLoader();

	cCamHandler* loadCamera(QString aNameFile);

    GlCloud*    loadCloud(string i_ply_file);

#ifdef USE_MIPMAP_HANDLER
	bool        reloadImage( MipmapHandler::Mipmap &aImage );
	bool        loadImage( MipmapHandler::Mipmap &aImage );
	MipmapHandler::Mipmap * loadImage( const std::string &aFilename, float scaleFactor = 1.f);
#else
	void loadImage(QString aNameFile, QMaskedImage *maskedImg, float scaleFactor = 1.f);
	void loadMask(QString aNameFile, QMaskedImage *maskedImg, float scaleFactor = 1.f);
#endif

    void        setFilenames(QStringList const &strl);
    void        setFilenameOut(QString str);

    QStringList& getFilenamesIn()        { return _FilenamesIn;  }
    QStringList& getFilenamesOut()       { return _FilenamesOut; }
    QStringList& getSelectionFilenames() { return _SelectionOut; }

    void        setPostFix(QString str);

	void memory();

	deviceIOCamera* devIOCamera() const;
	void setDevIOCamera(deviceIOCamera* devIOCamera);

	deviceIOImage* devIOImageAlter() const;
	void setDevIOImageAlter(deviceIOImage* devIOImageAlter);

#ifdef USE_MIPMAP_HANDLER
	void setMaxLoadMipmap( size_t aMaxLoadedMipmap ){ _mipmapHandler.setMaxLoaded(aMaxLoadedMipmap); }
	
	void setForceGrayMipmap( bool aForceGray ){ _forceGrayMipmap = aForceGray; }
	
	std::string getMaskFilename( const string &aImageFilename ) const;
#endif

private:
	QStringList _FilenamesIn;
	QStringList _FilenamesOut; //binary masks
	QStringList _SelectionOut; //selection infos
    QString     _postFix;

#ifdef USE_MIPMAP_HANDLER
	MipmapHandler _mipmapHandler;
	bool _forceGrayMipmap;
#endif

	deviceIOCamera* _devIOCamera;

	deviceIOImage*  _devIOImageAlter;
};

class cGLData;


enum idGPU_Vendor
{
    NVIDIA,
    AMD,
    ATI,
    INTEL,
    NOMODEL
};

class UpdateSignaler
{
public:
	virtual void operator()() = 0;
};

class cEngine
{
public:

    cEngine();
    ~cEngine();

    //! Set appli params
	#ifdef USE_MIPMAP_HANDLER
		void    setParams(cParameters *params);
	#else
		void    setParams(cParameters *params){ _params = params; }
	#endif

    //! Set input filenames
    void    setFilenames(QStringList const &strl){ _Loader->setFilenames(strl); }

    QStringList& getFilenamesIn(){return _Loader->getFilenamesIn();}

    QStringList& getFilenamesOut(){return _Loader->getFilenamesOut();}

    //! Set output filename
    void    setFilenameOut(QString filename){_Loader->setFilenameOut(filename);}

    QStringList& getSelectionFilenamesOut() { return _Loader->getSelectionFilenames(); }

    //! Set postfix
    void    setPostFix(){_Loader->setPostFix(_params->getPostFix());}

    //! Load point cloud .ply files
    void    loadClouds(QStringList);

    //! Load cameras .xml files
    void    loadCameras(QStringList);

    //! Load images  files
    void    loadImages(QStringList);

    //! Load image (and mask) file
	void    loadImage(QString imgName, float scaleFactor = 1.f);

    //void    reloadImage(int appMode, int aK);
    void    reloadMask(int appMode, int aK);

    //! Load object

    void    addObject(cObject*);

    void    unloadAll();

    void    unload(int aK);

    //! Compute mask binary images: projection of visible points into loaded cameras
//    void    do3DMasks();

    //! Creates binary image from selection and saves
    void    doMaskImage(ushort idCur, bool isFirstAction);

    void    saveBox2D(ushort idCur);

    void    saveMask(ushort idCur, bool isFirstAction);

    cData*  getData()  {return _Data;}

    //!looks for data and creates GLobjects
    void    allocAndSetGLData(int appMode, cParameters aParams);

    void    reallocAndSetGLData(int appMode, cParameters aParams, int aK);

    //!sends GLObjects to GLWidget
    cGLData* getGLData(int WidgetIndex);
    const cGLData * getGLData(int WidgetIndex) const;

    int     nbGLData() const {return _vGLData.size();}

	float	computeScaleFactor(QStringList &filenames);
    bool    extGLIsSupported(const char *strExt);
    void    setGLMaxTextureSize(int size) { _glMaxTextSize = size; }

	cLoader* Loader() const;
	void setLoader(cLoader* Loader);

#ifdef USE_MIPMAP_HANDLER
	int nbImages()
	{
		ELISE_DEBUG_ERROR(_Data == NULL, "Engine::nbImages", "_Data == NULL");
		return _Data->getNbImages();
	}

	int minLoadedGLDataId() const;

	void getGLDataIdSet( int aI0, int aI1, bool aIsLoaded, size_t aNbRequestedWidgets, std::vector<int> &oIds ) const;
#endif

	void setUpdateSignaler(UpdateSignaler *aSignaler);
	void signalUpdate();

private:

	cLoader*            _Loader;
	cData*              _Data;
	UpdateSignaler    * _updateSignaler;

    QVector <cGLData*>  _vGLData;

    cParameters*        _params;

    int                 _glMaxTextSize;
};

#endif // ENGINE_H
