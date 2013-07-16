#ifndef ENGINE2D_H
#define ENGINE2D_H

#include "qiodevice.h"
#include <QFileDialog>
#include <QDir>

/*class cSelectInfos
{
public:

    cSelectInfos();
    ~cSelectInfos();

    cSelectInfos(ViewportParameters aParams, QVector <QPoint> aPolyline, int selection_mode);

    ViewportParameters getParams(){return m_params;}
    QVector <QPoint>   getPoly(){return m_poly;}
    int                getSelectionMode(){return m_selection_mode;}

private:

    //polyline infos
    QVector <QPoint>   m_poly;
    int                m_selection_mode;
};*/

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

    /*CamStenope * loadCamera(string aNameFile);
    vector <CamStenope *> loadCameras();

    Cloud*      loadCloud(string i_ply_file , int *incre = NULL);*/

    void        setDir(QDir aDir){m_Dir = aDir;}
    QDir        getDir(){return m_Dir;}

    void        SetFilenamesOut();

    QStringList GetFilenamesOut() {return m_FilenamesOut;}

private:
    QStringList m_FilenamesIn;
    QStringList m_FilenamesOut; //binary masks

    //! Working directory
    QDir        m_Dir;
};

class cEngine2D
{
public:

    cEngine2D();
    ~cEngine2D();

    //! Set working directory
    void setDir(QDir aDir){m_Loader->setDir(aDir);}

    //! Set working directory
    void setFilename(){m_Loader->SetSelectionFilename();}

    //! Load point cloud .ply files
    void loadClouds(QStringList, int *incre = NULL);

    //! Load cameras .xml files
    void loadCameras(QStringList);

    //! Load cameras orientation files
    void loadCameras();

    void unloadAll();

    //! Compute mask binary images: projection of visible points into loaded cameras
    void doMasks();

    //void saveSelectInfos(QVector <cSelectInfos> const &Infos);

    //cData*   getData()  {return m_Data;}


private:

    cLoader *m_Loader;
    //cData   *m_Data;
};

#endif // ENGINE2D_H
