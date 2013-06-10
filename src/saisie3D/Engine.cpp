#include "Engine.h"

cLoader::cLoader()
 : m_FilenamesIn(),
   m_FilenamesOut()
{}

cLoader::~cLoader(){}

void cLoader::SetFilenamesOut()
{
    m_FilenamesOut.clear();

    for (int aK=0;aK < m_FilenamesIn.size();++aK)
    {
        QFileInfo fi(m_FilenamesIn[aK]);

        m_FilenamesOut.push_back(fi.path() + QDir::separator() + fi.completeBaseName() + "_mask.tif");
    }
}

Cloud* cLoader::loadCloud( string i_ply_file )
{
    return Cloud::loadPly( i_ply_file );
}

vector <cElNuage3DMaille *> cLoader::loadCameras()
{
   vector <cElNuage3DMaille *> a_res;

   m_FilenamesIn = QFileDialog::getOpenFileNames(NULL, tr("Open Camera Files"),m_Dir.path(), tr("Files (*.xml)"));

   for (int aK=0;aK < m_FilenamesIn.size();++aK)
   {
       a_res.push_back(loadCamera(m_FilenamesIn[aK].toStdString()));
   }

   SetFilenamesOut();

   return a_res;
}

cElNuage3DMaille *  cLoader::loadCamera(string aFile)
{
   return cElNuage3DMaille::FromFileIm(aFile);
}

cEngine::cEngine():
    m_Data(new cData),
    m_Loader(new cLoader)
{}

cEngine::~cEngine()

{}

void cEngine::loadCameras()
{
    m_Data->addCameras(m_Loader->loadCameras());
}

void cEngine::doMasks()
{
    if ((m_Data->NbCameras()==0) || (m_Data->NbClouds()==0)) return;

    //for (int i=0; i < m_glWidget->m_ply)
    for (int aK=0;aK < m_Data->NbCameras();++aK)
    {
        Pt3dr pt(0,0,0);
        //Pt2dr ptIm = (Camera(aK))->Terrain2Index(pt);
    }
}

void cEngine::addFiles(QStringList filenames)
{
    for (int i=0;i<filenames.size();++i)
    {
        getData()->centerCloud(m_Loader->loadCloud(filenames[i].toStdString()));
    }
}
