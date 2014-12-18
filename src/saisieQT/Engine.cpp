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

void cLoader::loadImage(QString aNameFile, QMaskedImage &maskedImg)
{
    //QTime *chronometre = new QTime(0,0,0,0) ;
    //chronometre->start();

    QImageReader *reader = new QImageReader(aNameFile);
    maskedImg._m_image = new QImage(reader->size(), QImage::Format_RGB888);
    if (!reader->read(maskedImg._m_image));
        maskedImg._m_image = new QImage( aNameFile );

    //cout << "chrono = " << chronometre->elapsed() << endl;

    // TODO: message d'erreur (non bloquant)
    // foo: Can not read scanlines from a tiled image.
    // see QTBUG-12636 => QImage load error on tiff tiled with lzw compression https://bugreports.qt-project.org/browse/QTBUG-12636
    // bug Qt non resolu
    // work around by creating an untiled and uncompressed temporary file with a system call to "tiffcp.exe" from libtiff library tools.

    if (maskedImg._m_image->isNull())
    {
        Tiff_Im aTF= Tiff_Im::StdConvGen(aNameFile.toStdString(),3,false);

        Pt2di aSz = aTF.sz();

        delete maskedImg._m_image;
        maskedImg._m_image = new QImage(aSz.x, aSz.y, QImage::Format_RGB888);

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

        for (int y=0; y<aSz.y; y++)
        {
            for (int x=0; x<aSz.x; x++)
            {
                QColor col(aDataR[y][x],aDataG[y][x],aDataB[y][x]);

                maskedImg._m_image->setPixel(x,y,col.rgb());
            }
        }
    }

    checkGeoref(aNameFile, maskedImg);

    *(maskedImg._m_image) = QGLWidget::convertToGLFormat( *(maskedImg._m_image) );

    //MASK

    QFileInfo fi(aNameFile);

    QString mask_filename = fi.path() + QDir::separator() + fi.completeBaseName() + _postFix + ".tif";

    maskedImg.setName(fi.fileName());

    loadMask(mask_filename, maskedImg);
}

void cLoader::loadMask(QString aNameFile, cMaskedImage<QImage> &maskedImg)
{
    setFilenameOut(aNameFile);

    if (QFile::exists(aNameFile))
    {
        maskedImg._m_newMask = false;

        QImageReader *reader = new QImageReader(aNameFile);
        maskedImg._m_mask = new QImage(reader->size(), QImage::Format_Mono);
        if (!reader->read(maskedImg._m_mask))
            maskedImg._m_mask = new QImage( aNameFile );

        if (maskedImg._m_mask->isNull())
        {
            Tiff_Im imgMask( aNameFile.toStdString().c_str() );

            if( imgMask.can_elise_use() )
            {
                int w = imgMask.sz().x;
                int h = imgMask.sz().y;

                delete maskedImg._m_mask;
                maskedImg._m_mask = new QImage( w, h, QImage::Format_Mono);
                maskedImg._m_mask->fill(1);

                Im2D_Bits<1> aOut(w,h,1);
                ELISE_COPY(imgMask.all_pts(),imgMask.in(),aOut.out());

                for (int x=0;x< w;++x)
                    for (int y=0; y<h;++y)
                        if (aOut.get(x,y) == 1 )
                            maskedImg._m_mask->setPixel(x,y,0);

                *(maskedImg._m_mask) = QGLWidget::convertToGLFormat(*(maskedImg._m_mask));
            }
            else
            {
                QMessageBox::critical(NULL, "cLoader::loadMask",QObject::tr("Cannot load mask image"));
            }
        }
        else
        {
            maskedImg._m_mask->invertPixels(QImage::InvertRgb);
            *(maskedImg._m_mask) = QGLWidget::convertToGLFormat(*(maskedImg._m_mask));
        }
    }
    else
    {
        //cout << "No mask found: " << aNameFile.toStdString().c_str() << endl;
        maskedImg._m_mask = new QImage(maskedImg._m_image->size(),QImage::Format_Mono);
        *(maskedImg._m_mask) = QGLWidget::convertToGLFormat(*(maskedImg._m_mask));
        maskedImg._m_mask->fill(Qt::white);
    }
}

void cLoader::checkGeoref(QString aNameFile, QMaskedImage &maskedImg)
{
    if (!maskedImg._m_image->isNull())
    {
        QFileInfo fi(aNameFile);

        QString suffix = fi.suffix();
        QString xmlFile = fi.absolutePath() + QDir::separator() + fi.baseName() + ".xml";

        if ((suffix == "tif") && (QFile(xmlFile).exists()))
        {
            std::string aNameTif = aNameFile.toStdString();

            maskedImg._m_FileOriMnt = StdGetObjFromFile<cFileOriMnt>
                                   (
                                        StdPrefix(aNameTif)+".xml",
                                        StdGetFileXMLSpec("ParamChantierPhotogram.xml"),
                                       "FileOriMnt",
                                       "FileOriMnt"
                                   );
        }
    }
}

void cLoader::setFilenames(const QStringList &strl)
{
    _FilenamesIn = strl;

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

        _SelectionOut.push_back(fi.path() + QDir::separator() + fi.completeBaseName() + "_selectionInfo.xml");
    }
}

void cLoader::setFilenameOut(QString str)
{
    _FilenamesOut.clear();

    _FilenamesOut.push_back(str);

    _SelectionOut.clear();

    QFileInfo fi(str);

    _SelectionOut.push_back(fi.path() + QDir::separator() + fi.completeBaseName() + "_selectionInfo.xml");
}

CamStenope* cLoader::loadCamera(QString aNameFile)
{
    QFileInfo fi(aNameFile);
    string DirChantier = (fi.dir().absolutePath()+ QDir::separator()).toStdString();

    #ifdef _DEBUG
        cout << "DirChantier : " << DirChantier << endl;
        cout << "filename : "    << aNameFile.toStdString() << endl;
    #endif

    cInterfChantierNameManipulateur * anICNM = cInterfChantierNameManipulateur::BasicAlloc(DirChantier);

    return CamOrientGenFromFile(fi.fileName().toStdString(),anICNM, false);
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
        _Data->addCloud(_Loader->loadCloud(filenames[i].toStdString(), incre));

    _Data->computeBBox();
    _Data->computeCloudsCenter();
}

void cEngine::loadCameras(QStringList filenames, int *incre)
{
    for (int i=0;i<filenames.size();++i)
    {
         if (incre) *incre = 100.0f*(float)i/filenames.size();
         CamStenope* cam = _Loader->loadCamera(filenames[i]);
         if (cam) _Data->addCamera(cam);
    }

    _Data->computeBBox();
}

//#define GL_GPU_MEM_INFO_TOTAL_AVAILABLE_MEM_NVX 0x9048
//#define GL_GPU_MEM_INFO_CURRENT_AVAILABLE_MEM_NVX 0x9049
//#define GL_RENDERBUFFER_FREE_MEMORY_ATI   0x87FD

bool cEngine::extGLIsSupported(const char* strExt)
{
#if ELISE_QT_VERSION == 5
    QOpenGLContext * contextHGL = QGLContext::currentContext()->contextHandle();
    return contextHGL->hasExtension(strExt);
#else
    const GLubyte *str;
    str = glGetString (GL_EXTENSIONS);
    //qDebug() << strExt;
    return (strstr((const char *)str, strExt) != NULL);
#endif
}

void cEngine::loadImages(QStringList filenames, int* incre)
{
    for (int i=0;i<filenames.size();++i)
    {
        if (incre) *incre = 100.0f*(float)i/filenames.size();
        loadImage(filenames[i]);
    }

    computeScaleFactor();
}

void  cEngine::loadImage(QString imgName)
{
    QMaskedImage maskedImg(_params->getGamma());

    _Loader->loadImage(imgName, maskedImg);

    _Data->pushBackMaskedImage(maskedImg);
}

/*void cEngine::reloadImage(int appMode, int aK)
{
    QMaskedImage maskedImg(_params->getGamma());

    _Loader->loadImage(getFilenamesIn()[aK], maskedImg);

    if (aK < _Data->getNbImages())
        _Data->getMaskedImage(aK) = maskedImg;

    reallocAndSetGLData(appMode, *_params, aK);
}*/

void cEngine::reloadMask(int appMode, int aK)
{
    if (aK < _Data->getNbImages())
        _Loader->loadMask(getFilenamesOut()[aK], _Data->getMaskedImage(aK));

    reallocAndSetGLData(appMode, *_params, aK);
}

void cEngine::addObject(cObject * aObj)
{
    getData()->addObject(aObj);
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

void cEngine::doMaskImage(ushort idCur, bool isFirstAction)
{
    if (!isFirstAction)
        _vGLData[idCur]->getMask()->invertPixels(QImage::InvertRgb);

    QImage Mask = _vGLData[idCur]->getMask()->mirrored().convertToFormat(QImage::Format_Mono);

    if (!Mask.isNull())
    {
        QString aOut = _Loader->getFilenamesOut()[idCur];

        float scaleFactor = _vGLData[idCur]->glImage().getLoadedImageRescaleFactor();

        if (scaleFactor != 1.f)
        {
            int width  = (int) ((float) Mask.width() / scaleFactor);
            int height = (int) ((float) Mask.height() / scaleFactor);

            Mask = Mask.scaled(width, height,Qt::KeepAspectRatio);
        }

#ifdef _DEBUG
        cout << "Saving mask to: " << aOut.toStdString().c_str() << endl;
#endif

        if (!Mask.save(aOut))
        {
            QMessageBox::critical(NULL, "cEngine::doMaskImage",QObject::tr("Error saving mask"));
            return;
        }

        cFileOriMnt anOri;

        anOri.NameFileMnt()		= aOut.toStdString();
        anOri.NombrePixels()	= Pt2di(Mask.width(),Mask.height());
        anOri.OriginePlani()	= Pt2dr(0,0);
        anOri.ResolutionPlani() = Pt2dr(1.0,1.0);
        anOri.OrigineAlti()		= 0.0;
        anOri.ResolutionAlti()	= 1.0;
        anOri.Geometrie()		= eGeomMNTFaisceauIm1PrCh_Px1D;

        MakeFileXML(anOri, StdPrefix(aOut.toStdString()) + ".xml");

        if (!isFirstAction)
            _vGLData[idCur]->getMask()->invertPixels(QImage::InvertRgb);
    }
    else
    {
        QMessageBox::critical(NULL, "cEngine::doMaskImage",QObject::tr("Mask is Null"));
    }
}

void cEngine::saveBox2D(ushort idCur)
{
    cPolygon* poly = _vGLData[idCur]->polygon(1);

    for (int aK=0; aK < poly->size(); ++aK)
    {
        //TODO
        //if (_FileOriMnt != NULL)
    }
}

void cEngine::saveMask(ushort idCur, bool isFirstAction)
{
    if (getData()->getNbImages())
        doMaskImage(idCur, isFirstAction);
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

void cEngine::allocAndSetGLData(int appMode, cParameters aParams)
{
    _vGLData.clear();

    for (int aK = 0; aK < _Data->getNbImages();++aK)
        _vGLData.push_back(new cGLData(_Data, &(_Data->getMaskedImage(aK)), aParams, appMode));

    if (_Data->is3D())
        _vGLData.push_back(new cGLData(_Data, aParams,appMode));
}

void cEngine::reallocAndSetGLData(int appMode, cParameters aParams, int aK)
{
    delete _vGLData[aK];

    if (_Data->is3D())
        _vGLData[aK] = new cGLData(_Data, aParams,appMode);
    else
        _vGLData[aK] = new cGLData(_Data, &(_Data->getMaskedImage(aK)), aParams, appMode);
}

cGLData* cEngine::getGLData(int WidgetIndex)
{
    if ((_vGLData.size() > 0) && (WidgetIndex < _vGLData.size()))
    {
        return _vGLData[WidgetIndex];
    }
    else
        return NULL;
}

void cEngine::computeScaleFactor()
{
    bool verbose = false;
    float scaleFactor = 1.f;

#if ELISE_QT_VERSION == 5
#ifdef COMPUTE_AVAILABLEVRAM
    if (QGLContext::currentContext())
    {

        QString  sGLVendor((char*)glGetString(GL_VENDOR));

        // GPU Model

        int GPUModel = NOMODEL;

        if ( sGLVendor.contains("AMD"))
            GPUModel = AMD;
        else if ( sGLVendor.contains("ATI"))
            GPUModel = ATI;
        else if ( sGLVendor.contains("NVIDIA"))
            GPUModel = NVIDIA;
        else if ( sGLVendor.contains("INTEL"))
            GPUModel = INTEL;

        GLint cur_avail_mem_kb      = 0;

        switch (GPUModel)
        {
        case NVIDIA:
            if(extGLIsSupported("GL_NVX_gpu_memory_info"))
            {
                glGetIntegerv(0x9049/*GL_GPU_MEM_INFO_CURRENT_AVAILABLE_MEM_NVX*/,&cur_avail_mem_kb);
            }
            break;
        case ATI: //TODO A RE TESTER
        case AMD:
            if(extGLIsSupported("GL_ATI_meminfo"))
                glGetIntegerv(0x87FD/*GL_TEXTURE_FREE_MEMORY_ATI*/,&cur_avail_mem_kb);
            break;
        default:
            cur_avail_mem_kb = 0;
            break;
        }

        //cout << sGLVendor.toStdString().c_str() << " - current available memory " << cur_avail_mem_kb/1024 << " MB" << endl;

        int sizeMemoryTexture_kb = 0;

        //for (int i=0;i<filenames.size();++i)
        for (int aK=0; aK< getData()->getNbImages();++aK)
        {
            QSize imageSize = getData()->getMaskedImage(aK)._m_image->size();

            sizeMemoryTexture_kb += imageSize.width()*imageSize.width()*4/1024;
        }

        //sizeMemoryTexture_kb = widthMax*heightMax*4/1024;

        //cur_avail_mem_kb = 5 * 1024;

        /*float scaleFactorVRAM = 1.f;
        // TODO delete texture car il y a un fuite dans la VRAM!!!

        if(cur_avail_mem_kb !=0)
        {
            // TODO GERER le MASK... car pas forcememt afficher
            if (appMode == MASK2D) sizeMemoryTexture_kb *= 2; // Image + masque
            if(sizeMemoryTexture_kb > cur_avail_mem_kb)
            {
                scaleFactorVRAM = (float) cur_avail_mem_kb / sizeMemoryTexture_kb;
            }
        }*/
    }
    else
        cout << "No GLContext" << endl;

#endif //COMPUTE_AVAILABLEVRAM
#endif

    int widthMax              = 0;
    int heightMax             = 0;

    //for (int i=0;i<filenames.size();++i)
    int nbImgs = getData()->getNbImages();
    for (int aK=0; aK< nbImgs;++aK)
    {
        //QImageReader(filenames[i]).size();
        QSize imageSize = getData()->getMaskedImage(aK)._m_image->size();

        widthMax  = max(imageSize.width(),widthMax);
        heightMax = max(imageSize.height(),heightMax);
    }

    //int maxImagesDraw = min(_params->getNbFen().x()*_params->getNbFen().y(),filenames.size());
    int maxImagesByRow = min(_params->getNbFen().x(),nbImgs);
    int maxImagesByCol = min(_params->getNbFen().y(),nbImgs);

   // widthMax    *= maxImagesDraw;
   // heightMax   *= maxImagesDraw;

    widthMax    *= maxImagesByRow;
    heightMax   *= maxImagesByCol;

    if ( widthMax > _glMaxTextSize || heightMax > _glMaxTextSize )
    {
        QSize totalSize(widthMax, heightMax);

        totalSize.scale(QSize(_glMaxTextSize,_glMaxTextSize), Qt::KeepAspectRatio);

        scaleFactor = (float) totalSize.width() / widthMax;

        //if (appMode == MASK2D) scaleFactor /= 2.f; //Image + Masque

        //cout << "scale factor = " << scaleFactor << endl;
    }

    //scaleFactor = min(scaleFactor,scaleFactorVRAM); // TODO A GERER

    for (int aK=0; aK< nbImgs;++aK)
    {
        getData()->getMaskedImage(aK)._loadedImageRescaleFactor = scaleFactor;
    }

    if (scaleFactor != 1.f)
    {
        if (verbose)
        {
            QString msg = QObject::tr("Rescaling images with ") + QString::number(scaleFactor,'f', 2) + QObject::tr(" factor");
            QMessageBox* msgBox = new QMessageBox(QMessageBox::Warning, QObject::tr("GL_MAX_TEXTURE_SIZE exceeded"),  msg);
            msgBox->setWindowFlags(Qt::WindowStaysOnTopHint);

            msgBox->exec();
        }

        //Rescale image and mask
        for (int aK=0; aK< nbImgs;++aK)
        {
            QImage * image = getData()->getMaskedImage(aK)._m_image;
            QImage * mask  = getData()->getMaskedImage(aK)._m_mask;

            QSize newSize = image->size()*scaleFactor;

            //cout << "new size: " << newSize.width() << " " << newSize.height() << endl;

            getData()->getMaskedImage(aK)._m_rescaled_image = new QImage(newSize, QImage::Format_RGB888);
            *(getData()->getMaskedImage(aK)._m_rescaled_image) = image->scaled(newSize,Qt::IgnoreAspectRatio);

            getData()->getMaskedImage(aK)._m_rescaled_mask = new QImage(newSize, QImage::Format_Mono);
            *(getData()->getMaskedImage(aK)._m_rescaled_mask) = mask->scaled(newSize,Qt::IgnoreAspectRatio);
        }
    }
    else
    {
        for (int aK=0; aK< nbImgs;++aK)
        {
            getData()->getMaskedImage(aK)._m_rescaled_image = getData()->getMaskedImage(aK)._m_image;
            getData()->getMaskedImage(aK)._m_rescaled_mask = getData()->getMaskedImage(aK)._m_mask;
        }
    }
}


//********************************************************************************

ViewportParameters::ViewportParameters()
    : m_zoom(1.f)
    , m_pointSize(1)
    , m_speed(2.f)
{}

ViewportParameters::ViewportParameters(const ViewportParameters& params)
    : m_zoom(params.m_zoom)
    , m_pointSize(params.m_pointSize)
    , m_speed(params.m_speed)
{}

ViewportParameters::~ViewportParameters(){}

ViewportParameters& ViewportParameters::operator =(const ViewportParameters& par)
{
    if (this != &par)
    {
        m_zoom      = par.m_zoom;
        m_pointSize = par.m_pointSize;
        m_speed     = par.m_speed;
    }

    return *this;
}

void ViewportParameters::reset()
{
    m_zoom = 1.f;
    m_pointSize = 1;
    m_speed = 2.f;
}

void ViewportParameters::ptSizeUp(bool up)
{
    if (up)
        m_pointSize++;
    else
        m_pointSize--;

    if (m_pointSize == 0)
        m_pointSize = 1;

    glPointSize((GLfloat) m_pointSize);
}
