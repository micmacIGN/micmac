#include "Engine.h"

cLoader::cLoader()
 : _FilenamesIn(),
   _FilenamesOut(),
   _postFix("_Masq")
{}


void cLoader::setPostFix(QString str)
{
    _postFix = str;
}

GlCloud* cLoader::loadCloud( string i_ply_file, int* incre )
{
    return GlCloud::loadPly( i_ply_file, incre );
}

void cLoader::loadImage(QString aNameFile , QMaskedImage &maskedImg)
{    

    maskedImg._m_image = new QImage( aNameFile );


    QFileInfo fi(aNameFile);

    QString mask_filename = fi.path() + QDir::separator() + fi.completeBaseName() + "_Masq.tif";

    maskedImg.setName(fi.fileName());

    setFilenameOut(mask_filename);

    // TODO factoriser le chargement d'image
    if (maskedImg._m_image->isNull())
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

        delete maskedImg._m_image;
        maskedImg._m_image = new QImage(aSz.x, aSz.y, QImage::Format_RGB32);

        for (int y=0; y<aSz.y; y++)
        {
            for (int x=0; x<aSz.x; x++)
            {
                QColor col(aDataR[y][x],aDataG[y][x],aDataB[y][x]);

                maskedImg._m_image->setPixel(x,y,col.rgb());
            }
        }
    }

    *(maskedImg._m_image) = QGLWidget::convertToGLFormat( *(maskedImg._m_image) );

    if (QFile::exists(mask_filename))
    {
        maskedImg._m_newMask = false;

        maskedImg._m_mask = new QImage( mask_filename );

        if (maskedImg._m_mask->isNull())
        {
            Tiff_Im imgMask( mask_filename.toStdString().c_str() );

            if( imgMask.can_elise_use() )
            {
                int w = imgMask.sz().x;
                int h = imgMask.sz().y;

                delete maskedImg._m_mask;
                maskedImg._m_mask = new QImage( w, h, QImage::Format_Mono);
                maskedImg._m_mask->fill(0);

                Im2D_Bits<1> aOut(w,h,1);
                ELISE_COPY(imgMask.all_pts(),imgMask.in(),aOut.out());

                for (int x=0;x< w;++x)
                    for (int y=0; y<h;++y)
                        if (aOut.get(x,y) == 1 )
                            maskedImg._m_mask->setPixel(x,y,1);

                *(maskedImg._m_mask) = QGLWidget::convertToGLFormat(*(maskedImg._m_mask));
            }
            else
            {
                QMessageBox::critical(NULL, "cLoader::loadMask","Cannot load mask image");
            }
        }
        else
            *(maskedImg._m_mask) = QGLWidget::convertToGLFormat(*(maskedImg._m_mask));

    }
    else
    {
        maskedImg._m_mask = new QImage(maskedImg._m_image->size(),QImage::Format_Mono);
        *(maskedImg._m_mask) = QGLWidget::convertToGLFormat(*(maskedImg._m_mask));
        maskedImg._m_mask->fill(Qt::white);
    }

}

void cLoader::setFilenamesAndDir(const QStringList &strl)
{
    _FilenamesIn = strl;

    setDir(strl);

    _FilenamesOut.clear();

    for (int aK=0;aK < _FilenamesIn.size();++aK)
    {
        QFileInfo fi(_FilenamesIn[aK]);

        _FilenamesOut.push_back(fi.path() + QDir::separator() + fi.completeBaseName() + _postFix + ".tif");
    }

    _SelectionOut.clear();

    for (int aK=0;aK < _FilenamesIn.size();++aK)
    {
        QFileInfo fi(_FilenamesIn[aK]);

        _SelectionOut.push_back(fi.path() + QDir::separator() + fi.completeBaseName() + "_selectionInfos.xml");
    }
}

void cLoader::setFilenameOut(QString str)
{
    _FilenamesOut.clear();

    _FilenamesOut.push_back(str);
}

void cLoader::setDir(const QStringList &list)
{
    QFileInfo fi(list[0]);

    //set default working directory as first file subfolder
    QDir Dir = fi.dir();
    Dir.cdUp();

    _Dir = Dir;
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

    delete _Loader;
    unloadAll();
    delete _Data;

}



void cEngine::loadClouds(QStringList filenames, int* incre)
{
    for (int i=0;i<filenames.size();++i)
    {
        _Data->addCloud(_Loader->loadCloud(filenames[i].toStdString(), incre));
    }

    _Data->computeBBox();
}

void cEngine::loadCameras(QStringList filenames)
{
    for (int i=0;i<filenames.size();++i)
    {
        _Data->addCamera(_Loader->loadCamera(filenames[i]));
    }

    _Data->computeBBox();
}

void cEngine::loadImages(QStringList filenames)
{
    for (int i=0;i<filenames.size();++i)
    {
        loadImage(filenames[i]);
    }
}

void  cEngine::loadImage(int aK)
{
    loadImage(getFilenamesIn()[aK]);
}

void  cEngine::loadImage(QString imgName)
{
    QMaskedImage maskedImg(_params->getGamma());

    _Loader->loadImage(imgName, maskedImg);

    _Data->pushBackMaskedImage(maskedImg);
}

void cEngine::reloadImage(int aK)
{
    QString imgName = getFilenamesIn()[aK];

    QMaskedImage maskedImg(_params->getGamma());

    _Loader->loadImage(imgName, maskedImg);

    if (aK < _Data->getNbImages())
        _Data->getMaskedImage(aK) = maskedImg;

    reallocAndSetGLData(aK);
}

void cEngine::do3DMasks()
{
    CamStenope* pCam;
    GlCloud *pCloud;
    GlVertex vert;
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

void cEngine::doMaskImage(ushort idCur)
{
    QImage pMask = _vGLData[idCur]->getMask()->mirrored().convertToFormat(QImage::Format_Mono);

    if (!pMask.isNull())
    {
        QString aOut = _Loader->getFilenamesOut()[idCur];

        pMask.save(aOut);

		cFileOriMnt anOri;

        anOri.NameFileMnt()		= aOut.toStdString();
        anOri.NombrePixels()	= Pt2di(pMask.width(),pMask.height());
		anOri.OriginePlani()	= Pt2dr(0,0);
		anOri.ResolutionPlani() = Pt2dr(1.0,1.0);
		anOri.OrigineAlti()		= 0.0;
		anOri.ResolutionAlti()	= 1.0;
		anOri.Geometrie()		= eGeomMNTFaisceauIm1PrCh_Px1D;

        MakeFileXML(anOri, StdPrefix(aOut.toStdString()) + ".xml");
	}
	else
    {
        QMessageBox::critical(NULL, "cEngine::doMaskImage","No alpha channel!!!");
    }
}

void cEngine::saveMask(ushort idCur)
{
    if (getData()->getNbImages())
        doMaskImage(idCur);
    else
        do3DMasks();
}

void cEngine::unloadAll()
{
    _Data->clearAll();
    qDeleteAll(_vGLData);
    _vGLData.clear();
}

void cEngine::unload(int aK)
{    
    if(_vGLData[aK])
    {
        delete _vGLData[aK];
        _vGLData[aK] = NULL;
    }
    _Data->clear(aK);
}

void cEngine::allocAndSetGLData(bool modePt, QString ptName)
{
    _vGLData.clear();

    for (int aK = 0; aK < _Data->getNbImages();++aK)
        _vGLData.push_back(new cGLData(_Data->getMaskedImage(aK), modePt, ptName));

    if (_Data->is3D())
        _vGLData.push_back(new cGLData(_Data));
}

void cEngine::reallocAndSetGLData(int aK)
{
    bool modePt = _vGLData[aK]->mode();
    delete _vGLData[aK];

    if (_Data->is3D())
        _vGLData[aK] = new cGLData(_Data);
    else
        _vGLData[aK] = new cGLData(_Data->getMaskedImage(aK), modePt);
}

cGLData* cEngine::getGLData(int WidgetIndex)
{
    if ((_vGLData.size() > 0) && (WidgetIndex < _vGLData.size()))
        return _vGLData[WidgetIndex];
    else
        return NULL;
}

//********************************************************************************


ViewportParameters::ViewportParameters()
    : m_zoom(1.f)
    , m_PointSize(1)
    , m_LineWidth(1.f)
    , m_speed(2.f)
{}

ViewportParameters::ViewportParameters(const ViewportParameters& params)
    : m_zoom(params.m_zoom)
    , m_PointSize(params.m_PointSize)
    , m_LineWidth(params.m_LineWidth)
{}

ViewportParameters::~ViewportParameters(){}

ViewportParameters& ViewportParameters::operator =(const ViewportParameters& par)
{
    if (this != &par)
    {
        m_zoom = par.m_zoom;
        m_PointSize = par.m_PointSize;
        m_LineWidth = par.m_LineWidth;
    }

    return *this;
}

void ViewportParameters::reset()
{
    m_zoom = m_LineWidth = 1.f;
    m_PointSize = 1;
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


