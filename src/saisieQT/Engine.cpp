#include "Engine.h"
#include "general/bitm.h"

cLoader::cLoader()
 : _FilenamesIn(),
   _FilenamesOut(),
   _postFix("_Masq")
{}

void cLoader::setFilenamesOut()
{
    _FilenamesOut.clear();

    for (int aK=0;aK < _FilenamesIn.size();++aK)
    {
        QFileInfo fi(_FilenamesIn[aK]);

        _FilenamesOut.push_back(fi.path() + QDir::separator() + fi.completeBaseName() + _postFix + ".tif");
    }
}

void cLoader::setFilenameOut(QString str)
{
    _FilenamesOut.clear();

    _FilenamesOut.push_back(str);
}

void cLoader::setPostFix(QString str)
{
    _postFix = str;
}

void cLoader::setSelectionFilename()
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

    setFilenameOut(mask_filename);

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

    for (uint aK=0; aK<_GLData.size();++aK)
        delete _GLData[aK];
    _GLData.clear();
}

void cEngine::loadClouds(QStringList filenames, int* incre)
{
    for (int i=0;i<filenames.size();++i)
    {
        _Data->addCloud(_Loader->loadCloud(filenames[i].toStdString(), incre));
    }

    _Data->getBB();
}

void cEngine::loadCameras(QStringList filenames)
{
    for (int i=0;i<filenames.size();++i)
    {
        _Data->addCamera(_Loader->loadCamera(filenames[i]));
    }

    _Data->getBB();
}

void cEngine::loadImages(QStringList filenames)
{
    for (int i=0;i<filenames.size();++i)
    {
        loadImage(filenames[i]);
    }

    _Loader->setFilenamesOut();
}

void  cEngine::loadImage(QString imgName)
{
    QImage* img, *mask;
    img = mask = NULL;

    _Loader->loadImage(imgName, img, mask);

    if (img !=NULL) _Data->addImage(img);
    if (mask!=NULL) _Data->addMask(mask);
#ifdef _DEBUG
    else cout << "mask null" << endl;
#endif
}

void cEngine::doMasks()
{
    CamStenope* pCam;
    Cloud *pCloud;
    Vertex vert;
    Pt2dr ptIm;

    for (int cK=0;cK < _Data->getNbCameras();++cK)
    {
        pCam = _Data->getCamera(cK);

        Im2D_BIN mask = Im2D_BIN(pCam->Sz(), 0);

        for (int aK=0; aK < _Data->getNbClouds();++aK)
        {
            pCloud  = _Data->getCloud(aK);

            for (int bK=0; bK < pCloud->size();++bK)
            {
                vert = pCloud->getVertex(bK);

                if (vert.isVisible())  //visible = selected in GLWidget
                {
                    Pt3dr pt(vert.getPosition());

                    if (pCam->PIsVisibleInImage(pt)) //visible = projected inside image
                    {
                        ptIm = pCam->Ter2Capteur(pt);
                        mask.set(floor(ptIm.x), floor(ptIm.y), 1);
                    }
                }
            }
        }

        string aOut = _Loader->getFilenamesOut()[cK].toStdString();
#ifdef _DEBUG
        printf ("Saving %s\n", aOut);
#endif

        Tiff_Im::CreateFromIm(mask, aOut);

#ifdef _DEBUG
        printf ("Done\n");
#endif
    }
}

void cEngine::doMaskImage()
{
    QImage* pImg = _Data->getCurMask();

	if (pImg->hasAlphaChannel())
	{
		QColor c;
		uint w = pImg->width();
		uint h = pImg->height();

		QImage qMask(w, h, QImage::Format_Mono);
		qMask.fill(0);

		for (uint aK=0; aK < w;++aK)
		{
			for (uint bK=0; bK < h;++bK)
			{
				c = QColor::fromRgba(pImg->pixel(aK,bK));
				if (c.red() == 255)
					qMask.setPixel(aK, h-bK-1, 1);
			}
		}

        QString aOut = _Loader->getFilenamesOut()[0];
		string sOut = aOut.toStdString();

		#ifdef _DEBUG
			printf ("Saving %s\n", sOut);
		#endif

		qMask.save(aOut);

		#ifdef _DEBUG
			printf ("Done\n");
		#endif

		cFileOriMnt anOri;

		anOri.NameFileMnt()		= sOut;
		anOri.NombrePixels()	= Pt2di(w,h);
		anOri.OriginePlani()	= Pt2dr(0,0);
		anOri.ResolutionPlani() = Pt2dr(1.0,1.0);
		anOri.OrigineAlti()		= 0.0;
		anOri.ResolutionAlti()	= 1.0;
		anOri.Geometrie()		= eGeomMNTFaisceauIm1PrCh_Px1D;

		MakeFileXML(anOri, StdPrefix(sOut) + ".xml");
		
		#ifdef _DEBUG
            printf("saved %s.xml\n", StdPrefix(sOut));
		#endif
	}
	else
    {
        QMessageBox::critical(NULL, "cEngine::doMaskImage","No alpha channel!!!");
    }
}

void cEngine::saveSelectInfos(const QVector<selectInfos> &Infos)
{
    QDomDocument doc;

    QFile outFile(_Loader->getSelectionFilename());
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

        t = doc.createTextNode(QString::number(SInfo.params.m_zoom));
        Scale.appendChild(t);

        t = doc.createTextNode(QString::number(SInfo.params.m_angleX) + " " + QString::number(SInfo.params.m_angleY) + " " + QString::number(SInfo.params.m_angleZ));
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
            QString str = QString::number(pts[aK].x(), 'f',1) + " "  + QString::number(pts[aK].y(), 'f',1);

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
        printf ( "File saved in: %s\n", _Loader->GetSelectionFilename().toStdString().c_str());
#endif
}

void cEngine::unloadAll()
{
    _Data->clearClouds();
    _Data->clearCameras();
    _Data->clearImages();
    _Data->clearMasks();
    _Data->reset();
}

void cEngine::setGLData()
{
    _GLData.clear();

    for (int aK = 0; aK < _Data->getNbImages();++aK)
    {
        cGLData *theData = new cGLData();

        if (_Data->getNbMasks()>aK)
        {
            if(_Data->getMask(aK) == NULL)
                glGenTextures(1, theData->pMask->getTexture() );   
            _Data->setEmptymask(false);

            theData->pMask->ImageToTexture(_Data->getMask(aK));
        }
        else if (_Data->getNbMasks() == 0)
        {
            QImage *mask;
            mask = new QImage(_Data->getImage(aK)->size(),QImage::Format_Mono);
            _Data->addMask(mask);
            _Data->fillMask(aK);
            _Data->setEmptymask(true);
        }

        _GLData.push_back(theData);
    }

    if (_Data->is3D())
    {

        cGLData *theData = new cGLData();

        for (int aK = 0; aK < _Data->getNbClouds();++aK)
        {
           /* Cloud *pCloud = new Cloud();
            pCloud = _Data->getCloud(aK);
            theData->Clouds.push_back(pCloud);*/

            _Data->getCloud(aK)->setBufferGl();
            //theData->Clouds[aK]->setBufferGl();
        }

        for (int aK = 0; aK < _Data->getNbCameras();++aK)
        {
            cCam *pCam = new cCam(_Data->getCamera(aK));

            theData->Cams.push_back(pCam);
        }

        float scale = _Data->m_diam / 1.5f;

        theData->pBall->setPosition(_Data->getCenter());
        theData->pBall->setScale(scale);
        theData->pBall->setVisible(true);

        theData->pAxis->setPosition(_Data->getCenter());
        theData->pAxis->setScale(scale);

        theData->pBbox->setPosition(_Data->getCenter());
        theData->pBbox->set(_Data->m_minX,_Data->m_minY,_Data->m_minZ,_Data->m_maxX,_Data->m_maxY,_Data->m_maxZ);

        for (int i=0; i<_Data->getNbCameras();i++)
        {
            cCam *pCam = new cCam(_Data->getCamera(i));

            pCam->setScale(scale);
            pCam->setVisible(true);

            theData->Cams.push_back(pCam);
        }

        _GLData.push_back(theData);
    }
}
cGLData* cEngine::getGLData(uint WidgetIndex)
{
    if ((_GLData.size() > 0) && (WidgetIndex < _GLData.size()))
        return _GLData[WidgetIndex];
    else
        return NULL;
}

cGLData::cGLData()
{
    //2D
    pImg  = new cImageGL();
    pMask = new cImageGL();

    //3D
    pBall = new cBall();
    pAxis = new cAxis();
    pBbox = new cBBox();
}

cGLData::~cGLData()
{
    delete pImg;
    delete pMask;

    for (int aK = 0; aK< Cams.size(); ++aK) delete Cams[aK];
    //qDeleteAll(Cams);
    Cams.clear();

    delete pBall;
    delete pAxis;
    delete pBbox;

   // qDeleteAll(Clouds);
   // Clouds.clear();
}

//********************************************************************************

ViewportParameters::ViewportParameters()
    : m_zoom(1.f)
    , m_PointSize(1)
    , m_LineWidth(1.f)
    , m_angleX(0.f)
    , m_angleY(0.f)
    , m_angleZ(0.f)
    , m_gamma(1.f)
    , m_speed(2.f)
{
    m_translationMatrix[0] = m_translationMatrix[1] = m_translationMatrix[2] = 0.f;
}

ViewportParameters::ViewportParameters(const ViewportParameters& params)
    : m_zoom(params.m_zoom)
    , m_PointSize(params.m_PointSize)
    , m_LineWidth(params.m_LineWidth)
    , m_angleX(params.m_angleX)
    , m_angleY(params.m_angleY)
    , m_angleZ(params.m_angleZ)
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
        m_zoom = par.m_zoom;
        m_PointSize = par.m_PointSize;

        m_angleX = par.m_angleX;
        m_angleY = par.m_angleY;
        m_angleZ = par.m_angleZ;

        m_translationMatrix[0] = par.m_translationMatrix[0];
        m_translationMatrix[1] = par.m_translationMatrix[1];
        m_translationMatrix[2] = par.m_translationMatrix[2];
        m_LineWidth = par.m_LineWidth;
        m_gamma 	= par.m_gamma;

    }

    return *this;
}

void ViewportParameters::reset()
{
    m_zoom = m_LineWidth = m_gamma = 1.f;
    m_angleX = m_angleY = m_angleZ = 0.f;
    m_PointSize = 1;

    m_translationMatrix[0] = m_translationMatrix[1] = m_translationMatrix[2] = 0.f;
}

void ViewportParameters::ptSizeUp(bool up)
{
    if (up)
        m_PointSize++;
    else
        m_PointSize--;

    if (m_PointSize == 0)
        m_PointSize = 1;

    glPointSize((GLfloat) m_PointSize);
}


