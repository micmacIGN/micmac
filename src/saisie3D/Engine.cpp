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

        m_FilenamesOut.push_back(fi.path() + QDir::separator() + fi.completeBaseName() + "_masq.tif");
    }
}

Cloud* cLoader::loadCloud( string i_ply_file )
{
    return Cloud::loadPly( i_ply_file );
}

vector <CamStenope *> cLoader::loadCameras()
{
   vector <CamStenope *> a_res;

   m_FilenamesIn = QFileDialog::getOpenFileNames(NULL, tr("Open Camera Files"),m_Dir.path(), tr("Files (*.xml)"));

   for (int aK=0;aK < m_FilenamesIn.size();++aK)
   {
       a_res.push_back(loadCamera(m_FilenamesIn[aK].toStdString()));
   }

   SetFilenamesOut();

   return a_res;
}


CamStenope*  cLoader::loadCamera(string aNameFile)
{
    // cInterfChantierNameManipulateur * anICNM = cInterfChantierNameManipulateur::BasicAlloc(m_Dir.dirName().toStdString());
    cInterfChantierNameManipulateur * anICNM = cInterfChantierNameManipulateur::BasicAlloc("D:/data/Boudha_dataset/Boudha/");
    aNameFile = "Ori-RadialBasic/Orientation-IMG_5582.tif.xml";
    CamStenope * elCam =  CamOrientGenFromFile(aNameFile,anICNM);
    return elCam;
}

cEngine::cEngine():
    m_Data(new cData),
    m_Loader(new cLoader)
{}

cEngine::~cEngine()
{
   delete m_Data;
   delete m_Loader;
}

void cEngine::loadCameras()
{
    m_Data->addCameras(m_Loader->loadCameras());
}

void cEngine::doMasks()
{
    if ((m_Data->NbCameras()==0) || (m_Data->NbClouds()==0)) return;

    if (m_Data->NbCameras() != m_Loader->GetFilenamesOut().size()) return;

    Cloud *pCloud, *pOCloud;
    Vertex vert, orig_vert;
    Pt2dr ptIm;

    for (int cK=0;cK < m_Data->NbCameras();++cK)
    {
        CamStenope* pCam = m_Data->getCamera(cK);

        Im2D_BIN mask = Im2D_BIN (pCam->Sz(), 0);

        for (int aK=0; aK < m_Data->NbClouds();++aK)
        {
            pCloud = m_Data->getCloud(aK);
            pOCloud = m_Data->getOriginalCloud(aK);

            for (int bK=0; bK < pCloud->size();++bK)
            {
                vert = pCloud->getVertex(bK);                

                if (vert.isVisible())
                {
                    orig_vert = pOCloud->getVertex(bK);
                    Pt3dr pt(orig_vert.x(),orig_vert.y(),orig_vert.z());

                    if (pCam->PIsVisibleInImage(pt))
                    {

                        ptIm = pCam->Ter2Capteur(pt);
                        //cout << "ptIm: " << ptIm.x <<" " << ptIm.y << endl;
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

        delete pCam;
        delete pCloud;
        delete pOCloud;
    }
}

void cEngine::loadClouds(QStringList filenames)
{
    for (int i=0;i<filenames.size();++i)
    {
        getData()->centerCloud(m_Loader->loadCloud(filenames[i].toStdString()));
    }
}
