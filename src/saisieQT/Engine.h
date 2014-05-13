#ifndef ENGINE_H
#define ENGINE_H

#include "qiodevice.h"
#include <QFileDialog>
#include <QMessageBox>
#include <QDir>

#include "Data.h"
#include "general/bitm.h"

#include "HistoryManager.h"
#include "Settings.h"

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

class cLoader
{

public:

    cLoader();

    CamStenope* loadCamera(QString aNameFile);

    GlCloud*    loadCloud(string i_ply_file , int *incre = NULL);

    void        loadImage(QString aNameFile, QMaskedImage &maskedImg);

    //! Check if georeferencing data exists (for box2d mode)
    void        checkGeoref(QString aNameFile, QMaskedImage &maskedImg);

    void        setDir(QDir aDir){_Dir = aDir;}
    void        setDir(QStringList const &list);
    QDir        getDir(){return _Dir;}

    void        setFilenamesAndDir(QStringList const &strl);
    void        setFilenameOut(QString str);

    QStringList& getFilenamesIn()        { return _FilenamesIn; }
    QStringList  getFilenamesOut()       { return _FilenamesOut; }
    QStringList& getSelectionFilenames() { return _SelectionOut; }

    void        setPostFix(QString str);

private:
    QStringList _FilenamesIn;
    QStringList _FilenamesOut; //binary masks
    QStringList _SelectionOut; //selection infos
    QString     _postFix;

    //! Working directory
    QDir        _Dir;
};

class cGLData;

class cEngine
{
public:

    cEngine();
    ~cEngine();

    //! Set appli params
    void    setParams(cParameters *params){ _params = params; }

    //! Set input filenames
    void    setFilenamesAndDir(QStringList const &strl){ _Loader->setFilenamesAndDir(strl); }

    QStringList& getFilenamesIn(){return _Loader->getFilenamesIn();}

    //! Set output filename
    void    setFilenameOut(QString filename){_Loader->setFilenameOut(filename);}

    //! Set postfix
    void    setPostFix(){_Loader->setPostFix(_params->getPostFix());}

    //! Load point cloud .ply files
    void    loadClouds(QStringList, int *incre = NULL);

    //! Load cameras .xml files
    void    loadCameras(QStringList);

    //! Load images  files
    void    loadImages(QStringList);

    //! Load image (and mask) file
    void    loadImage(QString );
    void    loadImage(int aK);

    void    reloadImage(int appMode, int aK);

    //! Load object

    void    loadObject(cObject*);

    void    unloadAll();

    void    unload(int aK);

    //! Compute mask binary images: projection of visible points into loaded cameras
    void    do3DMasks();

    //! Creates binary image from selection and saves
    void    doMaskImage(ushort idCur, bool isFirstAction);

    void    saveBox2D(ushort idCur);

    void    saveMask(ushort idCur, bool isFirstAction);

    cData*  getData()  {return _Data;}

    //!looks for data and creates GLobjects
    void    allocAndSetGLData(int appMode, QString ptName);

    void    reallocAndSetGLData(int appMode, int aK);

    //!sends GLObjects to GLWidget
    cGLData* getGLData(int WidgetIndex);

    int     nbGLData(){return (int)_vGLData.size();}

private:

    cLoader*            _Loader;
    cData*              _Data;

    QVector <cGLData*>  _vGLData;

    cParameters*        _params;
};




#endif // ENGINE_H
