#ifndef CGLDATA_H
#define CGLDATA_H

//~ #define DUMP_GL_DATA

class cData;
class GlCloud;
class MatrixManager;

#include "Data.h"
#include "MatrixManager.h"

class cGLData : public cObjectGL
{
public:
	#ifdef USE_MIPMAP_HANDLER
		cGLData( int aId, cData *data, cParameters aParams, int appMode = MASK2D, MaskedImage aSrcImage = MaskedImage(NULL, NULL) );
	#else
		cGLData(cData *data, QMaskedImage *qMaskedImage, cParameters aParams, int appMode = MASK2D);

		cGLData(cData *data, cParameters aParams, int appMode = MASK2D);
	#endif

    ~cGLData();

    void        draw();

    //~ bool        is3D() const                            { return ((!_vClouds.empty()) || (!_vCams.empty())); 
    bool        is3D() const                            { return glImageMasked().glImage() == NULL;   }

    bool        isImgEmpty()                            { return _glMaskedImage._m_image == NULL; }

	#ifndef USE_MIPMAP_HANDLER
		QImage * getMask()                               { return _glMaskedImage.getMaskedImage()->_m_rescaled_mask; }

		// this method may crash, for unknown reason, probably because of the multiple cObject inheritance
		QString imageName() { return _glMaskedImage.cObjectGL::name(); }
	#endif

    void        setPolygon(int aK, cPolygon *aPoly)     { _vPolygons[aK] = aPoly; }

    void        setCurrentPolygonIndex(int id)          { _currentPolygon = id;   }
    int         getCurrentPolygonIndex()                { return _currentPolygon; }

    void        normalizeCurrentPolygon(bool nrm);

	void        clearCurrentPolygon();

    bool        isNewMask()                             { return !isImgEmpty() ? _glMaskedImage._m_newMask : true; }

    //info coming from cData
    float       getBBoxMaxSize(){return _diam;}

    void        setBBoxMaxSize(float aS){_diam = aS;}

	void        setBBoxCenter(QVector3D aPt){_bbox_center = aPt;}

	void        setCloudsCenter(QVector3D aPt){_clouds_center = aPt;}

	void        setGlobalCenter(QVector3D aCenter);

    void        switchCenterByType(int val);

    bool        position2DClouds(MatrixManager &mm,QPointF pos);

	void        editImageMask(int mode, cPolygon* polyg, bool m_bFirstAction);

	void        editCloudMask(int mode, cPolygon*polyg, bool m_bFirstAction, MatrixManager &mm);

    void        replaceCloud(GlCloud* cloud, int id = 0);

    enum Option {
      OpNO          = 0x00,
      OpShow_Ball   = 0x01,
      OpShow_Axis   = 0x02,
      OpShow_BBox   = 0x04,
      OpShow_Mess   = 0x08,
	  OpShow_Cams   = 0x10,
      OpShow_Grid   = 0x20,
      //OpShow_Cent   = 0x40
      // ...
    };

    Q_DECLARE_FLAGS(options, Option)

    options     _options;

    void        GprintBits(size_t const size, void const * const ptr);

    void        setOption(QFlags<Option> option,bool show);

    bool        stateOption(QFlags<Option> option){ return _options & option; }

    bool        mode() { return _modePt; }

    void        setData(cData *data, bool setCam = true, int centerType=eCentroid);

    bool        incFirstCloud() const;

    void        setIncFirstCloud(bool incFirstCloud);

	cMaskedImageGL &glImageMasked();
	const cMaskedImageGL &glImageMasked() const;
    QVector <cMaskedImageGL*> glTiles();

    cPolygon*   polygon(int id = 0);

    cPolygon*   currentPolygon();

    QVector<cPolygon*> polygons() { return _vPolygons; }

    GlCloud*    getCloud(int iC);

    int         cloudCount();

    int         camerasCount();

    int         polygonCount();

    void        clearClouds(){ _vClouds.clear(); }

    cCamGL*       camera(int iC){ return _vCams[iC]; }

    void        setPolygons(cData *data);

	void		addPolygon(cPolygon *polygon);

    void        setOptionPolygons(cParameters aParams);

    void        drawCenter(bool white);

    void        createTiles();

//    void        setDrawTiles(bool val) { _bDrawTiles = val; }
//    bool        getDrawTiles() { return _bDrawTiles; }

	cBall*		pBall() const;

	void		saveLockRule();

	void		applyLockRule();

	#ifdef USE_MIPMAP_HANDLER
		int id() const { return mId; }

		MipmapHandler::Mipmap & getMask()
		{
			ELISE_DEBUG_ERROR( !_glMaskedImage.hasSrcMask(), "cMaskedImageGL::getMask()", "!_glMaskedImage.hasSrcMask()");
			return _glMaskedImage.srcMask();
		}

		bool isLoaded() const { return mNbLoaded != 0; }

		void setLoaded(bool aLoad)
		{
			if (aLoad)
				mNbLoaded++;
			else
			{
				ELISE_DEBUG_ERROR(mNbLoaded == 0, "cGLData::setLoaded", "aLoad == false && mNbLoaded == 0");
				mNbLoaded--;
			}

			if (mNbLoaded == 0) _glMaskedImage.deleteTextures();
		}

		void dump( std::string aPrefix, std::ostream &aStream ) const;
	#endif
private:
	#ifdef USE_MIPMAP_HANDLER
		int mId;
		unsigned int mNbLoaded;
	#endif

	cMaskedImageGL      _glMaskedImage;
	QVector <cMaskedImageGL*> _glMaskedTiles;

    cBall*              _pBall;

    cAxis*              _pAxis;

    cBBox*              _pBbox;

    cGrid*              _pGrid;

	QVector3D               _bbox_center;

	QVector3D               _clouds_center;

    bool                _modePt;

    int                 _appMode;

    QVector<GlCloud*>   _vClouds;

    QVector<cCamGL*>      _vCams;

    //! Point list for polygonal selection
    QVector<cPolygon*>  _vPolygons;

    int         _currentPolygon;

    void        initOptions(int appMode = MASK2D);

    float       _diam;

    bool        _incFirstCloud;
//    bool        _bDrawTiles;

	QPointF     _locksRule[2];
};

#ifdef __DEBUG
	string eToString( QImage::Format e );
#endif

#ifdef DUMP_GL_DATA
	extern list<cGLData *> __all_cGLData;

	bool __exist_cGLData( cGLData *aData );

	void __add_cGLData( cGLData *aData );

	void __remove_cGLData( cGLData *aData );
#endif

Q_DECLARE_OPERATORS_FOR_FLAGS(cGLData::options)
//====================================================================================

#endif // CGLDATA_H
