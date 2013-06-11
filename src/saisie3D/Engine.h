#ifndef ENGINE_H
#define ENGINE_H

#include <QFileDialog>
#include <QDir>

#include "Cloud.h"
#include "Data.h"
#include "StdAfx.h"

class cLoader : QObject
{
    public:

        cLoader();
        ~cLoader();

        CamStenope * loadCamera(string aNameFile);
        vector <CamStenope *> loadCameras();

        Cloud* loadCloud( string i_ply_file );

        void setDir(QDir aDir){m_Dir = aDir;}
        QDir getDir(){return m_Dir;}

        void SetFilenamesOut();
        QStringList GetFilenamesOut() {return m_FilenamesOut;}

    private:
        QStringList m_FilenamesIn;
        QStringList m_FilenamesOut;

        QDir        m_Dir;
};

class cEngine
{
    public:

        cEngine();
        ~cEngine();

        void addFiles(QStringList);
        void setDir(QDir aDir){m_Loader->setDir(aDir);}
        void loadCameras();

        void doMasks();

        cData*   getData()  {return m_Data;}

    private:

        cLoader *m_Loader;
        cData   *m_Data;
};

#endif // ENGINE_H
