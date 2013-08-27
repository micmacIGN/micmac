#ifndef ENGINE_H
#define ENGINE_H

#include "qiodevice.h"
#include <QFileDialog>
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

    void reset();

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
};

struct selectInfos
{

    selectInfos(){}
    ~selectInfos(){}
    //! Ortho camera infos
    ViewportParameters params;
    //! polyline infos
    QVector <QPoint>   poly;
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

    CamStenope * loadCamera(QString aNameFile);
   //vector <CamStenope *> loadCameras();

    Cloud*      loadCloud(string i_ply_file , int *incre = NULL);

    QImage*     loadImage(QString aNameFile);

    void        setDir(QDir aDir){m_Dir = aDir;}
    QDir        getDir(){return m_Dir;}

    void        SetFilenamesIn(QStringList const &strl){m_FilenamesIn = strl;}
    void        SetFilenamesOut();
    void        SetSelectionFilename();

    QStringList GetFilenamesOut() {return m_FilenamesOut;}
    QString     GetSelectionFilename() {return m_SelectionOut;}


private:
    QStringList m_FilenamesIn;
    QStringList m_FilenamesOut; //binary masks
    QString     m_SelectionOut; //selection infos

    //! Working directory
    QDir        m_Dir;
};

class cEngine
{    
public:

    cEngine();
    ~cEngine();

    //! Set working directory
    void setDir(QDir aDir){m_Loader->setDir(aDir);}

    //! Set working directory
    void setFilename(){m_Loader->SetSelectionFilename();}

    //! Set input filenames
    void SetFilenamesIn(QStringList const &strl){m_Loader->SetFilenamesIn(strl);}

    //! Set output filenames
    void setFilenamesOut(){m_Loader->SetFilenamesOut();}

    //! Load point cloud .ply files
    void loadClouds(QStringList, int *incre = NULL);

    //! Load cameras .xml files
    void loadCameras(QStringList);

    //! Load cameras .xml files
    void loadImages(QStringList);

    void unloadAll();

    //! Compute mask binary images: projection of visible points into loaded cameras
    void doMasks();

    //! Creates binary image from selection and saves
    void doMaskImage(QImage* pImg);

    void saveSelectInfos(QVector <selectInfos> const &Infos);

    cData*   getData()  {return m_Data;}

private:

    cLoader *m_Loader;
    cData   *m_Data;
};




#endif // ENGINE_H
