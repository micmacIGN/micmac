#include "Engine.h"
#include "general/bitm.h"

cLoader::cLoader()
 : _FilenamesIn(),
   _FilenamesOut(),
   _postFix("_Masq")
{}

void cLoader::SetFilenamesOut()
{
    _FilenamesOut.clear();

    for (int aK=0;aK < _FilenamesIn.size();++aK)
    {
        QFileInfo fi(_FilenamesIn[aK]);

        _FilenamesOut.push_back(fi.path() + QDir::separator() + fi.completeBaseName() + _postFix + ".tif");
    }
}

void cLoader::SetFilenameOut(QString str)
{
    _FilenamesOut.clear();

    _FilenamesOut.push_back(str);
}

void cLoader::SetPostFix(QString str)
{
    _postFix = str;
}

void cLoader::SetSelectionFilename()
{
    _SelectionOut = _Dir.absolutePath() + QDir::separator() + "SelectionInfos.xml";
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
        Tiff_Im aTF= Tiff_Im::StdConvGen(aNameFile.toStdString(),3,false);

        Pt2di aSz = aTF.sz();

        Im2D_U_INT1  aImR(aSz.x,aSz.y);
        Im2D_U_INT1  aImG(aSz.x,aSz.y);
        Im2D_U_INT1  aImB(aSz.x,aSz.y);

        ELISE_COPY
        (
           aTF.all_pts(),
           aTF.in(),
           Virgule(aImR.out(),aImG.out(),aImB.out())
        );

        U_INT1 ** aDataR = aImR.data();
        U_INT1 ** aDataG = aImG.data();
        U_INT1 ** aDataB = aImB.data();

        aImg = new QImage(aSz.x, aSz.y, QImage::Format_ARGB32);

        for (int y=0; y<aSz.y; y++)
        {
            for (int x=0; x<aSz.x; x++)
            {
                QColor col(aDataR[y][x],aDataG[y][x],aDataB[y][x],255);

                aImg->setPixel(x,y,col.rgba());
            }
        }
    }
    else
        aImg = img;

    if (QFile::exists(mask_filename))
    {
        QImage* imgM = new QImage( mask_filename );

        if (img->isNull())
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
        else
            aImgMask = imgM;
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
    string DirChantier = (_Dir.absolutePath()+ QDir::separator()).toStdString();
    string filename    = aNameFile.toStdString();

    #ifdef _DEBUG
        cout << "DirChantier : " << DirChantier << endl;
        cout << "filename : "    << filename << endl;
    #endif

    QFileInfo fi(aNameFile);

    _FilenamesOut.push_back(fi.path() + QDir::separator() + fi.completeBaseName() + "_Masq.tif");

    cInterfChantierNameManipulateur * anICNM = cInterfChantierNameManipulateur::BasicAlloc(DirChantier);

    return CamOrientGenFromFile(filename.substr(DirChantier.size(),filename.size()),anICNM);
}

//****************************************
//   cEngine

cEngine::cEngine():    
    _Loader(new cLoader),
    _Data(new cData)
{}

cEngine::~cEngine()
{
   delete _Data;
   delete _Loader;
}

void cEngine::loadClouds(QStringList filenames, int* incre)
{
    for (int i=0;i<filenames.size();++i)
    {
        getData()->getBB(_Loader->loadCloud(filenames[i].toStdString(), incre));
    }
}

void cEngine::loadCameras(QStringList filenames)
{
    for (int i=0;i<filenames.size();++i)
    {
        _Data->addCamera(_Loader->loadCamera(filenames[i]));
    }
}

void cEngine::loadImages(QStringList filenames)
{
    for (int i=0;i<filenames.size();++i)
    {
        loadImage(filenames[i]);
    }

    _Loader->SetFilenamesOut();
}

void  cEngine::loadImage(QString imgName)
{
    QImage* img, *mask;
    img = mask = NULL;

    _Loader->loadImage(imgName, img, mask);

    if (img!=NULL) _Data->addImage(img);
    if (mask!=NULL) _Data->addMask(mask);
}

void cEngine::doMasks()
{

    CamStenope* pCam;
    Cloud *pCloud;
    Vertex vert;
    Pt2dr ptIm;

    for (int cK=0;cK < _Data->NbCameras();++cK)
    {
        pCam = _Data->getCamera(cK);

        Im2D_BIN mask = Im2D_BIN(pCam->Sz(), 0);

        for (int aK=0; aK < _Data->NbClouds();++aK)
        {
            pCloud  = _Data->getCloud(aK);

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

        string aOut = _Loader->GetFilenamesOut()[cK].toStdString();
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
            if (c.red() == 255) mask.set(aK, h-bK-1, 1);
        }
    }

    string aOut = _Loader->GetFilenamesOut()[0].toStdString();
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

    QFile outFile(_Loader->GetSelectionFilename());
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

        QVector <QPointF> pts = SInfo.poly;

        for (int aK=0; aK <pts.size(); ++aK)
        {
            QDomElement Point    = doc.createElement("Pt");
            QString str = QString::number(pts[aK].x()) + " "  + QString::number(pts[aK].y(), 'f',1);

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
    _Data->clearClouds();
    _Data->clearCameras();
    _Data->clearImages();
    _Data->reset();
}

//********************************************************************************

ViewportParameters::ViewportParameters()
    : zoom(1.f)
    , PointSize(1.f)
    , LineWidth(1.f)
    , angleX(0.f)
    , angleY(0.f)
    , angleZ(0.f)
    , _gamma(1.f)
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
		this->LineWidth = par.LineWidth;
		this->_gamma 	= par._gamma;
		
    }

    return *this;
}

void ViewportParameters::reset()
{
    zoom = PointSize = LineWidth = 1.f;
    angleX = angleY = angleZ = 0.f;

    m_translationMatrix[0] = m_translationMatrix[1] = m_translationMatrix[2] = 0.f;
}


