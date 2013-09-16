#include "Engine.h"
#include "general/bitm.h"

cLoader::cLoader()
 : m_FilenamesIn(),
   m_FilenamesOut(),
   m_postFix("_Masq")
{}

void cLoader::SetFilenamesOut()
{
    m_FilenamesOut.clear();

    for (int aK=0;aK < m_FilenamesIn.size();++aK)
    {
        QFileInfo fi(m_FilenamesIn[aK]);

        m_FilenamesOut.push_back(fi.path() + QDir::separator() + fi.completeBaseName() + m_postFix + ".tif");
    }
}

void cLoader::SetFilenameOut(QString str)
{
    m_FilenamesOut.clear();

    m_FilenamesOut.push_back(str);
}

void cLoader::SetPostFix(QString str)
{
    m_postFix = str;
}

void cLoader::SetSelectionFilename()
{
    m_SelectionOut = m_Dir.absolutePath() + QDir::separator() + "SelectionInfos.xml";
}

Cloud* cLoader::loadCloud( string i_ply_file, int* incre )
{
    return Cloud::loadPly( i_ply_file, incre );
}

int	ByP=-1;
std::string MkFT;

void DoMkT()
{
    if (ByP)
    {
        std::string aSMkSr = g_externalToolHandler.get( "make" ).callName()+" all -f " + MkFT + string(" -j")+ToString(ByP)/*+" -s"*/;
        System(aSMkSr,true);
    }
}

void cLoader::loadImage(QString aNameFile , QImage* &aImg, QImage* &aImgMask)
{
    QImage* img = new QImage( aNameFile );

    QFileInfo fi(aNameFile);

    QString mask_filename = fi.path() + QDir::separator() + fi.completeBaseName() + "_Masq.tif";

    SetFilenameOut(mask_filename);

    if (img->isNull())
    {
       //Tiff_Im aTifIn = Tiff_Im::BasicConvStd(aNameFile.toStdString().c_str());
    }
    else
        aImg = img;

    if (!QFile::exists(mask_filename))
    {
        if (!img->isNull())
        {
            QImage* pDest = new QImage( img->width(), img->height(), QImage::Format_ARGB32 );
            pDest->fill(QColor(255,255,255,255));

            aImgMask = pDest;
        }
    }
    else
    {
        Tiff_Im img( mask_filename.toStdString().c_str() );

        if( img.can_elise_use() )
        {
            int w = img.sz().x;
            int h = img.sz().y;

            QImage* pDest = new QImage( w, h, QImage::Format_ARGB32 );

            Im2D_Bits<1> aOut(w,h,1);
            ELISE_COPY(img.all_pts(),img.in(),aOut.out());

            for (int x=0;x< w;++x)
            {
                for (int y=0; y<h;++y)
                {
                    if (aOut.get(x,y) == 0 )
                    {
                        QColor c(0,0,0,0);
                        pDest->setPixel(x,y,c.rgba());
                    }
                    else
                    {
                        QColor c(255,255,255,255);
                        pDest->setPixel(x,y,c.rgba());
                    }
                }
            }
            aImgMask = pDest;
        }
        else
        {
            QMessageBox::critical(NULL, "cLoader::loadMask","Cannot load mask image");
        }
    }
}

/*vector <CamStenope *> cLoader::loadCameras()
{
   vector <CamStenope *> a_res;

   m_FilenamesIn = QFileDialog::getOpenFileNames(NULL, tr("Open Camera Files"), m_Dir.path(), tr("Files (*.xml)"));

   for (int aK=0;aK < m_FilenamesIn.size();++aK)
   {
       a_res.push_back(loadCamera(m_FilenamesIn[aK].toStdString()));
   }

   SetFilenamesOut();

   return a_res;
}*/

// File structure is assumed to be a typical Micmac workspace structure:
// .ply files are in /MEC folder and orientations files in /Ori- folder
// /MEC and /Ori- are in the main working directory (m_Dir)

CamStenope* cLoader::loadCamera(QString aNameFile)
{
    string DirChantier = (m_Dir.absolutePath()+ QDir::separator()).toStdString();
    string filename    = aNameFile.toStdString();

    QFileInfo fi(aNameFile);

    m_FilenamesOut.push_back(fi.path() + QDir::separator() + fi.completeBaseName() + "_Masq.tif");

    cInterfChantierNameManipulateur * anICNM = cInterfChantierNameManipulateur::BasicAlloc(DirChantier);

    return CamOrientGenFromFile(filename.substr(DirChantier.size(),filename.size()),anICNM);
}

//****************************************
//   cEngine

cEngine::cEngine():    
    m_Loader(new cLoader),
    m_Data(new cData)
{}

cEngine::~cEngine()
{
   delete m_Data;
   delete m_Loader;
}

void cEngine::loadClouds(QStringList filenames, int* incre)
{
    for (int i=0;i<filenames.size();++i)
    {
        getData()->getBB(m_Loader->loadCloud(filenames[i].toStdString(), incre));
    }
}

void cEngine::loadCameras(QStringList filenames)
{
    for (int i=0;i<filenames.size();++i)
    {
        m_Data->addCamera(m_Loader->loadCamera(filenames[i]));
    }
}

void cEngine::loadImages(QStringList filenames)
{
    for (int i=0;i<filenames.size();++i)
    {
        loadImage(filenames[i]);
    }

    m_Loader->SetFilenamesOut();
}

void  cEngine::loadImage(QString imgName)
{
    QImage* img, *mask;
    img = mask = NULL;

    m_Loader->loadImage(imgName, img, mask);

    if (!img->isNull()) m_Data->addImage(img);
    if (!mask->isNull()) m_Data->addMask(mask);
}

void cEngine::doMasks()
{

    CamStenope* pCam;
    Cloud *pCloud;
    Vertex vert;
    Pt2dr ptIm;

    for (int cK=0;cK < m_Data->NbCameras();++cK)
    {
        pCam = m_Data->getCamera(cK);

        Im2D_BIN mask = Im2D_BIN(pCam->Sz(), 0);

        for (int aK=0; aK < m_Data->NbClouds();++aK)
        {
            pCloud  = m_Data->getCloud(aK);

            for (int bK=0; bK < pCloud->size();++bK)
            {
                vert = pCloud->getVertex(bK);

                if (vert.isVisible())  //visible = selected in GLWidget
                {
                    Pt3dr pt(vert.getCoord());

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

void cEngine::doMaskImage(QImage* pImg)
{
    QColor c;
    uint w,h;
    w = pImg->width();
    h = pImg->height();

    Im2D_BIN mask = Im2D_BIN(w, h, 0);

    for (uint aK=0; aK < w;++aK)
    {
        for (uint bK=0; bK < h;++bK)
        {
            c = QColor::fromRgba(pImg->pixel(aK,bK));
            if (c.alpha() == 255) mask.set(aK, h-bK-1, 1);
        }
    }

    string aOut = m_Loader->GetFilenamesOut()[0].toStdString();
#ifdef _DEBUG
    printf ("Saving %s\n", aOut);
#endif

    Tiff_Im::CreateFromIm(mask, aOut);

    cout << "saved " << aOut <<endl;
#ifdef _DEBUG
    printf ("Done\n");
#endif

    std::string aNameXML = StdPrefix(aOut)+".xml";
    if (!ELISE_fp::exist_file(aNameXML))
    {
        cFileOriMnt anOri;

        anOri.NameFileMnt() = aOut;
        anOri.NombrePixels() = mask.sz();
        anOri.OriginePlani() = Pt2dr(0,0);
        anOri.ResolutionPlani() = Pt2dr(1.0,1.0);
        anOri.OrigineAlti() = 0.0;
        anOri.ResolutionAlti() = 1.0;
        anOri.Geometrie() = eGeomMNTFaisceauIm1PrCh_Px1D;

        MakeFileXML(anOri,aNameXML);
    }

    cout << "saved " << aNameXML << endl;

}

void cEngine::saveSelectInfos(const QVector<selectInfos> &Infos)
{
    QDomDocument doc;

    QFile outFile(m_Loader->GetSelectionFilename());
    if (!outFile.open(QIODevice::WriteOnly)) return;

    QDomElement SI = doc.createElement("SelectionInfos");

    QDomText t;
    for (int i = 0; i < Infos.size(); ++i)
    {
        QDomElement SII         = doc.createElement("Item");
        QDomElement Scale       = doc.createElement("Scale");
        QDomElement Rotation	= doc.createElement("Rotation");
        QDomElement Translation	= doc.createElement("Translation");
        QDomElement Mode        = doc.createElement("Mode");

        selectInfos SInfo = Infos[i];

        t = doc.createTextNode(QString::number(SInfo.params.zoom));
        Scale.appendChild(t);

        t = doc.createTextNode(QString::number(SInfo.params.angleX) + " " + QString::number(SInfo.params.angleY) + " " + QString::number(SInfo.params.angleZ));
        Rotation.appendChild(t);

        t = doc.createTextNode(QString::number(SInfo.params.m_translationMatrix[0]) + " " + QString::number(SInfo.params.m_translationMatrix[1]) + " " + QString::number(SInfo.params.m_translationMatrix[2]));
        Translation.appendChild(t);

        SII.appendChild(Scale);
        SII.appendChild(Rotation);
        SII.appendChild(Translation);

        QVector <QPoint> pts = SInfo.poly;

        for (int aK=0; aK <pts.size(); ++aK)
        {
            QDomElement Point    = doc.createElement("Pt");
            QString str = QString::number(pts[aK].x()) + " "  + QString::number(pts[aK].y());

            t = doc.createTextNode( str );
            Point.appendChild(t);
            SII.appendChild(Point);
        }

        t = doc.createTextNode(QString::number(SInfo.selection_mode));
        Mode.appendChild(t);

        SII.appendChild(Mode);

        SI.appendChild(SII);
    }

    doc.appendChild(SI);

    QTextStream content(&outFile);
    content << doc.toString();
    outFile.close();

    #ifdef _DEBUG
       printf ( "File saved in: %s\n", m_Loader->GetSelectionFilename().toStdString());
    #endif
}

void cEngine::unloadAll()
{
    m_Data->clearClouds();
    m_Data->clearCameras();
    m_Data->clearImages();
    m_Data->reset();
}

//********************************************************************************

ViewportParameters::ViewportParameters()
    : zoom(1.f)
    , PointSize(1.f)
    , LineWidth(1.f)
    , angleX(0.f)
    , angleY(0.f)
    , angleZ(0.f)
{
    m_translationMatrix[0] = m_translationMatrix[1] = m_translationMatrix[2] = 0.f;
}

ViewportParameters::ViewportParameters(const ViewportParameters& params)
    : zoom(params.zoom)
    , PointSize(params.PointSize)
    , LineWidth(params.LineWidth)
    , angleX(params.angleX)
    , angleY(params.angleY)
    , angleZ(params.angleZ)
{
    m_translationMatrix[0] = params.m_translationMatrix[0];
    m_translationMatrix[1] = params.m_translationMatrix[1];
    m_translationMatrix[2] = params.m_translationMatrix[2];
}

ViewportParameters::~ViewportParameters(){}

ViewportParameters& ViewportParameters::operator =(const ViewportParameters& par)
{
    if (this != &par)
    {
        this->zoom = par.zoom;
        this->PointSize = par.PointSize;

        this->angleX = par.angleX;
        this->angleY = par.angleY;
        this->angleZ = par.angleZ;

        this->m_translationMatrix[0] = par.m_translationMatrix[0];
        this->m_translationMatrix[1] = par.m_translationMatrix[1];
        this->m_translationMatrix[2] = par.m_translationMatrix[2];
    }

    return *this;
}

void ViewportParameters::reset()
{
    zoom = PointSize = LineWidth = 1.f;
    angleX = angleY = angleZ = 0.f;

    m_translationMatrix[0] = m_translationMatrix[1] = m_translationMatrix[2] = 0.f;
}


