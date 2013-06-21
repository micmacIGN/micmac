#include "Engine.h"

cLoader::cLoader()
 : m_FilenamesIn(),
   m_FilenamesOut()
{}

void cLoader::SetFilenamesOut()
{
    m_FilenamesOut.clear();

    for (int aK=0;aK < m_FilenamesIn.size();++aK)
    {
        QFileInfo fi(m_FilenamesIn[aK]);

        m_FilenamesOut.push_back(fi.path() + QDir::separator() + fi.completeBaseName() + "_masq.tif");
    }
}

Cloud* cLoader::loadCloud( string i_ply_file )
{
    return Cloud::loadPly( i_ply_file );
}

vector <Cloud *> cLoader::loadClouds()
{
   vector <Cloud *> a_res;

   QStringList FilenamesIn = QFileDialog::getOpenFileNames(NULL, tr("Open Ply Files"), m_Dir.path(), tr("Files (*.ply)"));

   for (int aK=0;aK < FilenamesIn.size();++aK)
   {
       a_res.push_back(loadCloud(FilenamesIn[aK].toStdString()));
   }

   if (FilenamesIn.size())
   {
       QFileInfo fi(FilenamesIn[0]);
       QDir Dir = fi.dir();
       Dir.cdUp();
       m_Dir = Dir;
   }

   return a_res;
}


vector <CamStenope *> cLoader::loadCameras()
{
   vector <CamStenope *> a_res;

   m_FilenamesIn = QFileDialog::getOpenFileNames(NULL, tr("Open Camera Files"), m_Dir.path(), tr("Files (*.xml)"));

   for (int aK=0;aK < m_FilenamesIn.size();++aK)
   {
       a_res.push_back(loadCamera(m_FilenamesIn[aK].toStdString()));
   }

   SetFilenamesOut();

   return a_res;
}

// File structure is assumed to be a typical Micmac workspace structure:
// .ply files are in /MEC folder and orientations files in /Ori- folder
// /MEC and /Ori- are in the main working directory (m_Dir)

CamStenope* cLoader::loadCamera(string aNameFile)
{
    string DirChantier = (m_Dir.absolutePath()+ QDir::separator()).toStdString();

    cInterfChantierNameManipulateur * anICNM = cInterfChantierNameManipulateur::BasicAlloc(DirChantier);

    return CamOrientGenFromFile(aNameFile.substr(DirChantier.size(),aNameFile.size()),anICNM);
}

//****************************************
//   cEngine

cEngine::cEngine():
    m_Data(new cData),
    m_Loader(new cLoader)
{}

cEngine::~cEngine()
{
   delete m_Data;
   delete m_Loader;
}

void cEngine::loadClouds(QStringList filenames)
{
    for (int i=0;i<filenames.size();++i)
    {
        getData()->centerCloud(m_Loader->loadCloud(filenames[i].toStdString()));
    }
}

void cEngine::loadPlys()
{
    m_Data->addClouds(m_Loader->loadClouds());
}

void cEngine::loadCameras()
{
    m_Data->addCameras(m_Loader->loadCameras());
}

void cEngine::doMasks()
{
    if (m_Data->NbClouds()==0) return;

    CamStenope* pCam;
    Cloud *pCloud, *pOCloud;
    Vertex vert, orig_vert;
    Pt2dr ptIm;

    for (int cK=0;cK < m_Data->NbCameras();++cK)
    {
        pCam = m_Data->getCamera(cK);

        Im2D_BIN mask = Im2D_BIN (pCam->Sz(), 0);

        for (int aK=0; aK < m_Data->NbClouds();++aK)
        {
            pCloud  = m_Data->getCloud(aK);
            pOCloud = m_Data->getOriginalCloud(aK);

            for (int bK=0; bK < pCloud->size();++bK)
            {
                vert = pCloud->getVertex(bK);                

                if (vert.isVisible())  //visible = selected in GLWidget
                {
                    orig_vert = pOCloud->getVertex(bK);

                    Pt3dr pt(orig_vert.getCoord());

                    if (pCam->PIsVisibleInImage(pt)) //visible = projected inside image
                    {
                        ptIm = pCam->Ter2Capteur(pt);
                        mask.set(floor(ptIm.x), floor(ptIm.y), 1);
                    }
                }
            }
        }

        string aOut = m_Loader->GetFilenamesOut()[cK].toStdString();
        #ifdef _DEBUG
            printf ("Saving %s\n", aOut);
        #endif

        Tiff_Im::CreateFromIm(mask, aOut);

        #ifdef _DEBUG
            printf ("Done\n");
        #endif  
    }
}
