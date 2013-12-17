#ifndef ENGINE_H
#define ENGINE_H

#include "qiodevice.h"
#include <QFileDialog>
#include <QMessageBox>
#include <QDir>
#include <QDomDocument>
#include <QTextStream>

#include "Cloud.h"
#include "Data.h"
#include "general/bitm.h"

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

    //! Current zoom
    float m_zoom;

    //! Point size
    int m_PointSize;

    //! Line width
    float m_LineWidth;

    //! Translation matrix
    float m_translationMatrix[3];

	float m_speed;
};

struct selectInfos
{
    //! polyline infos
    QVector <QPointF> poly;

    //! selection mode
    int         selection_mode;

    GLdouble    mvmatrix[16];
    GLdouble    projmatrix[16];
    GLint       glViewport[4];
};

//! Selection mode
enum SELECTION_MODE { SUB,
                      ADD,
                      INVERT,
                      ALL,
                      NONE
                    };

class cLoader : QObject   
{
    Q_OBJECT
public:

    cLoader();

    CamStenope* loadCamera(QString aNameFile);

    Cloud*      loadCloud(string i_ply_file , int *incre = NULL);

    void        loadImage(QString aNameFile, QMaskedImage &maskedImg);

    void        setDir(QDir aDir){_Dir = aDir;}
    QDir        getDir(){return _Dir;}

    void        setFilenamesIn(QStringList const &strl){_FilenamesIn = strl;}
    void        setFilenamesOut();
    void        setFilenameOut(QString str);
    void        setSelectionFilename();

    QStringList getFilenamesOut() {return _FilenamesOut;}
    QString     getSelectionFilename() {return _SelectionOut;}


    void        setPostFix(QString str);

private:
    QStringList _FilenamesIn;
    QStringList _FilenamesOut; //binary masks
    QString     _SelectionOut; //selection infos
    QString     _postFix;

    //! Working directory
    QDir        _Dir;
};

class cGLData : cObjectGL
{
public:

    cGLData();
    cGLData(QMaskedImage &qMaskedImage);
    cGLData(cData *data);

    ~cGLData();

    bool        is3D(){return Clouds.size() || Cams.size();}
    bool        isDataLoaded(){return (!isImgEmpty()) || is3D();}

    cMaskedImageGL glMaskedImage;

    QImage      *pQMask;

    //! Point list for polygonal selection
    cPolygon    m_polygon;

    bool        isImgEmpty(){return maskedImage._m_image == NULL;}

    bool        isMaskEmpty(){ return maskedImage._m_mask == NULL;}

    QImage*     getMask(){return pQMask;}

    void        setPolygon(cPolygon const &aPoly){m_polygon = aPoly;}

    //3D

    void        draw();

    QVector < cCam* > Cams;

    cBall       *pBall;
    cAxis       *pAxis;
    cBBox       *pBbox;

    QVector < Cloud* > Clouds;

    //info coming from cData
    float       getScale(){return _diam;}
    void        setScale(float aS){_diam = aS;}

    Pt3dr       getCenter(){return _center;}
    void        setCenter(Pt3dr aCenter){_center = aCenter;}

private:

    float       _diam;
    Pt3dr       _center;
};

class cEngine
{    
public:

    cEngine();
    ~cEngine();

    //! Set working directory
    void    setDir(QDir aDir){_Loader->setDir(aDir);}

    //! Set working directory
    void    setFilename(){_Loader->setSelectionFilename();}

    //! Set input filenames
    void    setFilenamesIn(QStringList const &strl){_Loader->setFilenamesIn(strl);}

    //! Set output filenames
    void    setFilenamesOut(){_Loader->setFilenamesOut();}

    //! Set output filename
    void    setFilenameOut(QString filename){_Loader->setFilenameOut(filename);}

    //! Set postfix
    void    setPostFix(QString filename){_Loader->setPostFix(filename);}

    //! Load point cloud .ply files
    void    loadClouds(QStringList, int *incre = NULL);

    //! Load cameras .xml files
    void    loadCameras(QStringList);

    //! Load images  files
    void    loadImages(QStringList);

    //! Load image (and mask) file
    void    loadImage(QString imgName);

    void    unloadAll();

    //! Compute mask binary images: projection of visible points into loaded cameras
    void    do3DMasks();

    //! Creates binary image from selection and saves
    void    doMaskImage(ushort idCur);

    void    saveMask(ushort idCur);

    void    saveSelectInfos(QVector <selectInfos> const &Infos);

    cData*  getData()  {return _Data;}

    //!looks for data and creates GLobjects
    void    AllocAndSetGLData();

    //!sends GLObjects to GLWidget
    cGLData* getGLData(int WidgetIndex);

    void    applyGammaToImage(int aK);

    void    setGamma(float aGamma) {_Gamma = aGamma;}

private:

    cLoader*            _Loader;
    cData*              _Data;

    QVector <cGLData*>  _vGLData;

    float               _Gamma;
};




#endif // ENGINE_H
