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

    //! Current zoom
    float zoom;

    //! Point size
    float PointSize;

    //! Line width
    float LineWidth;

    //! Rotation angles
    float angleX;
    float angleY;
    float angleZ;

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

    void        SetFilenamesIn(QStringList const &strl){_FilenamesIn = strl;}
    void        SetFilenamesOut();
    void        SetFilenameOut(QString str);
    void        SetSelectionFilename();

    QStringList GetFilenamesOut() {return _FilenamesOut;}
    QString     GetSelectionFilename() {return _SelectionOut;}


    void        SetPostFix(QString str);

private:
    QStringList _FilenamesIn;
    QStringList _FilenamesOut; //binary masks
    QString     _SelectionOut; //selection infos
    QString     _postFix;

    //! Working directory
    QDir        _Dir;
};

class cEngine
{    
public:

    cEngine();
    ~cEngine();

    //! Set working directory
    void setDir(QDir aDir){_Loader->setDir(aDir);}

    //! Set working directory
    void setFilename(){_Loader->SetSelectionFilename();}

    //! Set input filenames
    void SetFilenamesIn(QStringList const &strl){_Loader->SetFilenamesIn(strl);}

    //! Set output filenames
    void setFilenamesOut(){_Loader->SetFilenamesOut();}

    //! Set output filename
    void setFilenameOut(QString filename){_Loader->SetFilenameOut(filename);}

    //! Set postfix
    void setPostFix(QString filename){_Loader->SetPostFix(filename);}

    //! Load point cloud .ply files
    void loadClouds(QStringList, int *incre = NULL);

    //! Load cameras .xml files
    void loadCameras(QStringList);

    //! Load images  files
    void loadImages(QStringList);

    //! Load image (and mask) file
    void loadImage(QString imgName);

    void unloadAll();

    //! Compute mask binary images: projection of visible points into loaded cameras
    void doMasks();

    //! Creates binary image from selection and saves
    void doMaskImage(QImage* pImg);

    void saveSelectInfos(QVector <selectInfos> const &Infos);

    cData*   getData()  {return _Data;}

private:

    cLoader *_Loader;
    cData   *_Data;
};




#endif // ENGINE_H
