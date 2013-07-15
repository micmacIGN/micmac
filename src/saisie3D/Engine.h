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

class cSelectInfos
{
public:

    cSelectInfos();
    ~cSelectInfos();

    cSelectInfos(ViewportParameters aParams, QVector <QPoint> aPolyline, int selection_mode);

    ViewportParameters getParams(){return m_params;}
    QVector <QPoint>   getPoly(){return m_poly;}
    int                getSelectionMode(){return m_selection_mode;}

private:
    //Ortho camera infos
    ViewportParameters m_params;

    //polyline infos
    QVector <QPoint>   m_poly;
    int                m_selection_mode;
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
public:

    cLoader();

    CamStenope * loadCamera(string aNameFile);
    vector <CamStenope *> loadCameras();

    Cloud*      loadCloud(string i_ply_file , void (*incre)(int, void *) = NULL, void* obj = NULL);

    void        setDir(QDir aDir){m_Dir = aDir;}
    QDir        getDir(){return m_Dir;}

    void        SetFilenamesOut();
    QStringList GetFilenamesOut() {return m_FilenamesOut;}

private:
    QStringList m_FilenamesIn;
    QStringList m_FilenamesOut;

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

    //! Load point cloud .ply files
    void loadClouds(QStringList, void (*incre)(int, void *) = NULL, void *obj = NULL);

    //! Load cameras .xml files
    void loadCameras(QStringList);

    //! Load cameras orientation files
    void loadCameras();

    void unloadAll();

    //! Compute mask binary images: projection of visible points into loaded cameras
    void doMasks();

    void saveSelectInfos(QVector <cSelectInfos> const &Infos, QString FileName);

    cData*   getData()  {return m_Data;}

    void loadCloudsWin(QStringList filenames);
private:

    cLoader *m_Loader;
    cData   *m_Data;
};




#endif // ENGINE_H
