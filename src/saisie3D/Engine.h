#ifndef ENGINE_H
#define ENGINE_H

#include <QFileDialog>
#include <QDir>

#include "Cloud.h"
#include "Data.h"
#include "general/bitm.h"

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

    Cloud* loadCloud(string i_ply_file , void (*incre)(int, void *) = NULL, void* obj = NULL);
    vector <Cloud *> loadClouds();

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

    //! Load point cloud .ply files
    void loadPlys();

    //! Load cameras .xml files
    void loadCameras(QStringList);

    //! Load cameras orientation files
    void loadCameras();

    void unloadAll();

    //! Compute mask binary images: projection of visible points into loaded cameras
    void doMasks();

    void saveSelectInfos(string);

    cData*   getData()  {return m_Data;}

private:

    cLoader *m_Loader;
    cData   *m_Data;
};

class ViewportParameters
{
public:
    //! Default constructor
    ViewportParameters();

    //! Copy constructor
    ViewportParameters(const ViewportParameters& params);

    //! Destructor
    ~ViewportParameters();

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

    cSelectInfos(ViewportParameters aParams, std::vector <Pt2df> aPolyline, SELECTION_MODE);

private:
    //Ortho camera infos
    ViewportParameters  m_params;

    //polyline infos
    std::vector <Pt2df>    m_poly;
    SELECTION_MODE      m_selection_mode;
};


#endif // ENGINE_H
