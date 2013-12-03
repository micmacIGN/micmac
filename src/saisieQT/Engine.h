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

    void    setGamma(float aGamma) {m_gamma = aGamma;}
    float   getGamma() {return m_gamma;}

    void    ptSizeUp(bool up);

    //! Current zoom
    float m_zoom;

    //! Point size
    int m_PointSize;

    //! Line width
    float m_LineWidth;

    //! Rotation angles
    float m_angleX;
    float m_angleY;
    float m_angleZ;

    //! Translation matrix
    float m_translationMatrix[3];

    float m_gamma;

	float m_speed;
};

struct selectInfos
{
    //! Ortho camera infos
    ViewportParameters params;

    //! polyline infos
    QVector <QPointF>  poly;

    //! selection mode
    int                selection_mode;
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

    void        loadImage(QString aNameFile, QImage* &aImg, QImage* &aImgMask);

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

class cGLData
{
    public:

    cGLData();
    ~cGLData();

    //2D
    cImageGL    *pImg;
    cImageGL    *pMask;

    //! Point list for polygonal selection
    cPolygon    m_polygon;

    //! Point list for polygonal insertion
    cPolygon    m_dihedron;

    //3D
    QVector < cCam* > Cams;

    cBall       *pBall;
    cAxis       *pAxis;
    cBBox       *pBbox;

    //QVector < Cloud *> Clouds;
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
    void    doMasks();

    //! Creates binary image from selection and saves
    void    doMaskImage();

    void    saveSelectInfos(QVector <selectInfos> const &Infos);

    cData*  getData()  {return _Data;}

    //!looks for data and creates GLobjects
    void    setGLData();

    bool    isGLDataSet(){return _bGLDataSet;}

    //!sends GLObjects to GLWidget
    cGLData* getGLData(int WidgetIndex);

private:

    cLoader*         _Loader;
    cData*           _Data;

    QVector <cGLData*> _GLData;

    bool             _bGLDataSet;
};




#endif // ENGINE_H
