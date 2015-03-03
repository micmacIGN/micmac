#ifndef CGLDATA_H
#define CGLDATA_H

class cData;
class GlCloud;
class MatrixManager;

#include "Data.h"
#include "MatrixManager.h"

class cGLData : public cObjectGL
{
public:

    cGLData(cData *data, QMaskedImage *qMaskedImage, cParameters aParams, int appMode = MASK2D);

    cGLData(cData *data, cParameters aParams, int appMode = MASK2D);

    ~cGLData();

    void        draw();

    bool        is3D()                                  { return _vClouds.size() || _vCams.size();   }

    bool        isImgEmpty()                            { return _glMaskedImage._m_image == NULL; }

    QImage*     getMask()                               { return _glMaskedImage.getMaskedImage()->_m_rescaled_mask; }

    void        setPolygon(int aK, cPolygon *aPoly)     { _vPolygons[aK] = aPoly; }

    void        setCurrentPolygonIndex(int id)          { _currentPolygon = id;   }
    int         getCurrentPolygonIndex()                { return _currentPolygon; }

    void        normalizeCurrentPolygon(bool nrm);

    void        clearPolygon();

    bool        isNewMask()                             { return !isImgEmpty() ? _glMaskedImage._m_newMask : true; }

    QString     imageName() { return _glMaskedImage.cObjectGL::name(); }

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

    void        setOptionPolygons(cParameters aParams);

    void        drawCenter(bool white);

    void        createTiles();

//    void        setDrawTiles(bool val) { _bDrawTiles = val; }
//    bool        getDrawTiles() { return _bDrawTiles; }

	cBall*		pBall() const;


private:

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

};

Q_DECLARE_OPERATORS_FOR_FLAGS(cGLData::options)
//====================================================================================

#endif // CGLDATA_H
