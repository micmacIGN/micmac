#include "../src/uti_image/Digeo/MultiChannel.h"
#include "Engine.h"

cLoader::cLoader()
 : _FilenamesIn(),
   _FilenamesOut(),
   _postFix("_Masq"),
   _devIOCamera(NULL),
   _devIOImageAlter(NULL)
{}


void cLoader::setPostFix(QString str)
{
    _postFix = str;
}
deviceIOCamera* cLoader::devIOCamera() const
{
    return _devIOCamera;
}

void cLoader::setDevIOCamera(deviceIOCamera* devIOCamera)
{
    _devIOCamera = devIOCamera;
}
deviceIOImage* cLoader::devIOImageAlter() const
{
    return _devIOImageAlter;
}

void cLoader::setDevIOImageAlter(deviceIOImage* devIOImageAlter)
{
    _devIOImageAlter = devIOImageAlter;
}

GlCloud* cLoader::loadCloud(string i_ply_file)
{
    return GlCloud::loadPly(i_ply_file);
}

#ifdef USE_MIPMAP_HANDLER
	bool cLoader::loadImage( MipmapHandler::Mipmap &aImage )
	{
		bool result = _mipmapHandler.read(aImage);
		if ( !result) cerr << ELISE_RED_ERROR << "cLoader::readImage: failed to read image [" << aImage.mFilename << ']' << endl;
		return result;
	}

	bool cLoader::reloadImage( MipmapHandler::Mipmap &aImage )
	{
		if (aImage.mData != NULL) _mipmapHandler.release(aImage);
		return loadImage(aImage);
	}

	string cLoader::getMaskFilename( const string &aImageFilename ) const
	{
		QFileInfo fi(QString(aImageFilename.c_str()));
		return (fi.path() + QDir::separator() + fi.completeBaseName() + _postFix + ".tif").toStdString();
	}

	MipmapHandler::Mipmap * cLoader::loadImage( const string &aFilename, float scaleFactor )
	{
		MipmapHandler::Mipmap *image = _mipmapHandler.ask(aFilename, 0, _forceGrayMipmap);
		if (image == NULL) return NULL;
		return image;
	}
#else
	void cLoader::loadImage(QString aNameFile, QMaskedImage *maskedImg, float scaleFactor)
	{
	//	QTime *chro = new QTime(0,0,0,0) ;
	//	chro->start();

	//	qDebug() << chro->elapsed() << " begin";

		QImageReader reader(aNameFile);

	//	qDebug() << chro->elapsed() << " start read";

	//	DUMP(reader.supportsOption(QImageIOHandler::ClipRect))
	//	DUMP(reader.supportsOption(QImageIOHandler::ScaledSize))

		QSize rescaledSize	= reader.size()*scaleFactor;
		QSize FullSize		= reader.size();

		//reader.setScaledSize(rescaledSize );

		maskedImg->_m_image			= new QImage(FullSize,QImage::Format_RGB888);

		maskedImg->_loadedImageRescaleFactor = scaleFactor;
		maskedImg->_fullSize = reader.size();

		QSize tempSize = FullSize;

		QImage tempImage(tempSize,QImage::Format_RGB888);

		if (!reader.read(&tempImage))
		{
			if(maskedImg->_m_image )
			    delete maskedImg->_m_image;

			maskedImg->_m_image = new QImage( aNameFile);
		}

		// TODO: message d'erreur (non bloquant)
		// foo: Can not read scanlines from a tiled image.
		// see QTBUG-12636 => QImage load error on tiff tiled with lzw compression https://bugreports.qt-project.org/browse/QTBUG-12636
		// bug Qt non resolu
		// work around by creating an untiled and uncompressed temporary file with a system call to "tiffcp.exe" from libtiff library tools.

		if (maskedImg->_m_image->isNull() && _devIOImageAlter)
		{
			delete maskedImg->_m_image;

			maskedImg->_m_image = _devIOImageAlter->loadImage(aNameFile);
			maskedImg->_fullSize = maskedImg->_m_image->size(); // TODO ATTENTION _fullSize surement inutile

		}
		else
		{
			if(maskedImg->_m_image )
			    delete maskedImg->_m_image;

			maskedImg->_m_image = new QImage(QGLWidget::convertToGLFormat( tempImage ));
			maskedImg->_fullSize = maskedImg->_m_image->size();
		}

		if(scaleFactor <1.f)
			maskedImg->_m_rescaled_image = new QImage(maskedImg->_m_image->scaled(rescaledSize,Qt::IgnoreAspectRatio));

		//MASK
		QFileInfo fi(aNameFile);

		QString mask_filename = fi.path() + QDir::separator() + fi.completeBaseName() + _postFix + ".tif";

		maskedImg->setName(fi.fileName());

		loadMask(mask_filename, maskedImg,scaleFactor);

	}

	void cLoader::loadMask(QString aNameFile, cMaskedImage<QImage> *maskedImg,float scaleFactor)
	{
		setFilenameOut(aNameFile);

		if (QFile::exists(aNameFile))
		{

			maskedImg->_m_newMask = false;

			QImageReader reader(aNameFile);

			reader.setScaledSize(reader.size()*scaleFactor);

			if(maskedImg->_m_rescaled_mask )
			    delete maskedImg->_m_rescaled_mask;

			maskedImg->_m_rescaled_mask = new QImage(reader.scaledSize(), QImage::Format_Mono);
			QImage tempMask(reader.scaledSize(), QImage::Format_Mono);

			if ((!reader.read(&tempMask) || tempMask.isNull()) && _devIOImageAlter)
			{

			    maskedImg->_m_rescaled_mask = _devIOImageAlter->loadMask(aNameFile);

			    if(maskedImg->_m_rescaled_mask == NULL)

			        QMessageBox::critical(NULL, "cLoader::loadMask",QObject::tr("Cannot load mask image"));

			}
			else

			    *(maskedImg->_m_rescaled_mask) = QGLWidget::convertToGLFormat(tempMask);

		}
		else
		{
			if(scaleFactor<1.f) // TODO Mask c'est quoi la diff�rence ....
			{
			    QImage tempMask(maskedImg->_m_rescaled_image->size(),QImage::Format_Mono);
			    maskedImg->_m_rescaled_mask = new QImage(tempMask.size(),QImage::Format_Mono);
			    tempMask.fill(Qt::white);
			    *(maskedImg->_m_rescaled_mask) = QGLWidget::convertToGLFormat(tempMask);
			}
			else
			{
			    #ifdef USE_MIPMAP_HANDLER
			        QImage tempMask(maskedImg->_fullSize, QImage::Format_Mono);
			    #else
			        QImage tempMask(maskedImg->_m_image->size(),QImage::Format_Mono);
			    #endif
			    maskedImg->_m_rescaled_mask = new QImage(tempMask.size(),QImage::Format_Mono);
			    tempMask.fill(Qt::white);
			    *(maskedImg->_m_rescaled_mask) = QGLWidget::convertToGLFormat(tempMask);
			}
		}
	}
#endif

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

cCamHandler* cLoader::loadCamera(QString aNameFile)
{
    if(_devIOCamera)
        return _devIOCamera->loadCamera(aNameFile);
    else
        return NULL;
}

//****************************************
//   cEngine

cEngine::cEngine():
    _Loader(new cLoader),
    _Data(new cData),
    _updateSignaler(NULL)
{}

cEngine::~cEngine()
{
    delete _Loader;
    unloadAll();
    delete _Data;
}

#ifdef USE_MIPMAP_HANDLER
	void cEngine::setParams(cParameters *params)
	{
		_params = params;
		if (_Loader != NULL)
		{
			_Loader->setForceGrayMipmap(params->getForceGray());
			_Loader->setMaxLoadMipmap(size_t(params->getNbFen().x()) * size_t(params->getNbFen().y()));

			cout << "setForceGray(" << params->getForceGray() << ")" << endl;
		}
	}
#endif

void cEngine::loadClouds(QStringList filenames)
{
	for (int i=0;i<filenames.size();++i)
	{
		_Data->addCloud(_Loader->loadCloud(filenames[i].toStdString()));
		signalUpdate();
	}
	_Data->computeCenterAndBBox();
}

void cEngine::loadCameras(QStringList filenames)
{
	for (int i=0;i<filenames.size();++i)
	{
		cCamHandler* cam = _Loader->loadCamera(filenames[i]);
		if (cam) _Data->addCamera(cam);
		signalUpdate();
	}

	_Data->computeCenterAndBBox();
}

//#define GL_GPU_MEM_INFO_TOTAL_AVAILABLE_MEM_NVX 0x9048
//#define GL_GPU_MEM_INFO_CURRENT_AVAILABLE_MEM_NVX 0x9049
//#define GL_RENDERBUFFER_FREE_MEMORY_ATI   0x87FD

bool cEngine::extGLIsSupported(const char* strExt)
{
    QOpenGLContext * contextHGL = QGLContext::currentContext()->contextHandle();
    return contextHGL->hasExtension(strExt);
}
cLoader* cEngine::Loader() const
{
    return _Loader;
}

void cEngine::setLoader(cLoader* Loader)
{
    _Loader = Loader;
}


void cEngine::loadImages(QStringList filenames)
{
    float scaleFactor = computeScaleFactor(filenames);

    for (int i=0; i<filenames.size(); ++i)
    {
        loadImage(filenames[i], scaleFactor);
        signalUpdate();
    }
}

#ifdef USE_MIPMAP_HANDLER
	void cEngine::loadImage( QString imgName, float scaleFactor )
	{
		const string imageFilename = imgName.toStdString();
		MipmapHandler::Mipmap *image = _Loader->loadImage(imgName.toStdString(), scaleFactor);

		if (image == NULL)
		{
			cerr << ELISE_RED_ERROR << "skipping image [" << imgName.toStdString() << "]: loading failed" << endl;
			return;
		}

		const string maskFilename = _Loader->getMaskFilename(imageFilename);
		MipmapHandler::Mipmap *mask = _Loader->loadImage(maskFilename, scaleFactor);
		_Data->addImage(MaskedImage(image, mask));
	}
#else
	void  cEngine::loadImage(QString imgName,float scaleFactor )
	{
		QMaskedImage *maskedImg =  new QMaskedImage(_params->getGamma());

		_Loader->loadImage(imgName, maskedImg,scaleFactor);

		_Data->pushBackMaskedImage(maskedImg);
	}
#endif

/*void cEngine::reloadImage(int appMode, int aK)
{
    QMaskedImage maskedImg(_params->getGamma());

    _Loader->loadImage(getFilenamesIn()[aK], maskedImg);

    if (aK < _Data->getNbImages())
        _Data->getMaskedImage(aK) = maskedImg;

    reallocAndSetGLData(appMode, *_params, aK);
}*/

#ifdef USE_MIPMAP_HANDLER
	void cEngine::reloadMask(int appMode, int aK)
	{
		if (_Data != NULL && aK >= 0 && aK < _Data->getNbImages())
		{
			MaskedImage &maskedImage = _Data->getMaskedImage(aK);
			if (maskedImage.second != NULL) _Loader->reloadImage(*maskedImage.second);
		}

		reallocAndSetGLData(appMode, *_params, aK);
	}
#else
	void cEngine::reloadMask(int appMode, int aK)
	{
		if (aK < _Data->getNbImages())
		    _Loader->loadMask(getFilenamesOut()[aK], _Data->getMaskedImage(aK), _Data->getMaskedImage(aK)->_loadedImageRescaleFactor);

		reallocAndSetGLData(appMode, *_params, aK);
	}
#endif

void cEngine::addObject(cObject * aObj)
{
    getData()->addObject(aObj);
}
/*
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
        printf ("Saving %s\n", aOut.c_str());
#endif

        Tiff_Im::CreateFromIm(mask, aOut);

#ifdef _DEBUG
        printf ("Done\n");
#endif
    }
}
*/

#ifdef USE_MIPMAP_HANDLER
	void cEngine::doMaskImage(ushort idCur, bool isFirstAction)
	{
		if ( !isFirstAction) return;
		ELISE_DEBUG_ERROR(_vGLData[idCur] == NULL, "cEngine::doMaskImage", "_vGLData[" << idCur << "] == NULL");
		cMaskedImageGL &maskedImageGL = _vGLData[idCur]->glImageMasked();
		if ( !maskedImageGL.hasSrcMask()) return;
		string errorMesage;
		if ( !maskedImageGL.srcMask().writeTiff(errorMesage)) cerr << ELISE_RED_ERROR << "failed to save mask to [" << maskedImageGL.srcMask().mFilename << "] (" << errorMesage << ')' << endl;
	}
#else
	void scanlineCopy_binary( const unsigned char *aSrc, const size_t aSrcLineSize, size_t aHeight, unsigned char **aDst, const size_t aDstLineSize )
	{
		while (aHeight--)
		{
			memcpy(*aDst++, aSrc, aDstLineSize);
			aSrc += aSrcLineSize;
		}
	}

	void cEngine::doMaskImage(ushort idCur, bool isFirstAction)
	{
	//    if (!isFirstAction)
	//        _vGLData[idCur]->getMask()->invertPixels(QImage::InvertRgb);

		QImage Mask = _vGLData[idCur]->getMask()->mirrored().convertToFormat(QImage::Format_Mono);
		Mask.invertPixels();

		if (!Mask.isNull())
		{
			QString aOut = _Loader->getFilenamesOut()[idCur];
			float scaleFactor = _vGLData[idCur]->glImageMasked().getLoadedImageRescaleFactor();

		    if (scaleFactor != 1.f)
		    {
		        int width  = (int) ((float) Mask.width() / scaleFactor);
		        int height = (int) ((float) Mask.height() / scaleFactor);

		        Mask = Mask.scaled(width, height,Qt::KeepAspectRatio);
		    }

	#ifdef _DEBUG
		    cout << "Saving mask to: " << aOut.toStdString().c_str() << endl;
	#endif

			Im2D_Bits<1> outImage(Mask.width(), Mask.height());
			const string outFilename = aOut.toStdString();
			const size_t dstLineSize = (size_t)ceil((double)(outImage.tx()) / 8.);
			scanlineCopy_binary(Mask.constBits(), Mask.bytesPerLine(), Mask.height(), outImage.data(), dstLineSize);
			ELISE_COPY(
				outImage.all_pts(),
				outImage.in(),
			Tiff_Im(
				outFilename.c_str(),
				outImage.sz(),
				GenIm::bits1_msbf,
				Tiff_Im::LZW_Compr,
				Tiff_Im::BlackIsZero,
				Tiff_Im::Empty_ARG).out()
			);

			if ( !Tiff_Im::IsTiff(outFilename.c_str()))
			{
				QMessageBox::critical(NULL, "cEngine::doMaskImage",QObject::tr("Error saving mask"));
				return;
			}

			// write the xml associated to the mask
			if(Loader()->devIOImageAlter()) Loader()->devIOImageAlter()->doMaskImage(Mask,aOut);

		/*
		    cFileOriMnt anOri;

		    anOri.NameFileMnt()		= aOut.toStdString();
		    anOri.NombrePixels()	= Pt2di(Mask.width(),Mask.height());
		    anOri.OriginePlani()	= Pt2dr(0,0);
		    anOri.ResolutionPlani() = Pt2dr(1.0,1.0);
		    anOri.OrigineAlti()		= 0.0;
		    anOri.ResolutionAlti()	= 1.0;
		    anOri.Geometrie()		= eGeomMNTFaisceauIm1PrCh_Px1D;

		    MakeFileXML(anOri, StdPrefix(aOut.toStdString()) + ".xml");
		    */

	//        if (!isFirstAction)
	//            _vGLData[idCur]->getMask()->invertPixels(QImage::InvertRgb);
		}
		else
		{
		    QMessageBox::critical(NULL, "cEngine::doMaskImage",QObject::tr("Mask is Null"));
		}
	}
#endif

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
    //qDeleteAll(_vGLData);

    // TODO ATTENTION le delete n'est pas fait

    for (int var = 0; var < _vGLData.size(); ++var)
    {
        if(_vGLData[var])
            delete _vGLData[var];

        _vGLData[var] = NULL;
    }

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

#ifdef USE_MIPMAP_HANDLER
	void cEngine::allocAndSetGLData(int appMode, cParameters aParams)
	{
		_vGLData.clear();

		for (int aK = 0; aK < _Data->getNbImages();++aK)
		{
			ELISE_DEBUG_ERROR(_Loader == NULL, "cEngine::allocAndSetGLData", "_Loader == NULL");
			_vGLData.push_back(new cGLData(_vGLData.size(), _Data, aParams, appMode, _Data->getMaskedImage(aK)));
		}

		if (_Data->is3D()) _vGLData.push_back(new cGLData(_vGLData.size(), _Data, aParams, appMode));
	}
#else
	void cEngine::allocAndSetGLData(int appMode, cParameters aParams)
	{
		_vGLData.clear();

		for (int aK = 0; aK < _Data->getNbImages();++aK)
		    _vGLData.push_back(new cGLData(_Data, _Data->getMaskedImage(aK), aParams, appMode));

		if (_Data->is3D())
		    _vGLData.push_back(new cGLData(_Data, aParams,appMode));
	}
#endif

#ifdef USE_MIPMAP_HANDLER
	void cEngine::reallocAndSetGLData(int appMode, cParameters aParams, int aK)
	{
		delete _vGLData[aK];

		if (_Data->is3D())
			_vGLData[aK] = new cGLData(aK, _Data, aParams,appMode);
		else
			_vGLData[aK] = new cGLData(aK, _Data, aParams, appMode, _Data->getMaskedImage(aK));
	}
#else
	void cEngine::reallocAndSetGLData(int appMode, cParameters aParams, int aK)
	{


		delete _vGLData[aK];

		if (_Data->is3D())
		    _vGLData[aK] = new cGLData(_Data, aParams,appMode);
		else
		    _vGLData[aK] = new cGLData(_Data, _Data->getMaskedImage(aK), aParams, appMode);
	}
#endif

cGLData* cEngine::getGLData(int WidgetIndex)
{
    if ((_vGLData.size() > 0) && (WidgetIndex < _vGLData.size()))
    {
        return _vGLData[WidgetIndex];
    }
    else
        return NULL;
}

const cGLData * cEngine::getGLData(int WidgetIndex) const
{
    if ((_vGLData.size() > 0) && (WidgetIndex < _vGLData.size()))
    {
        return _vGLData[WidgetIndex];
    }
    else
        return NULL;
}

float cEngine::computeScaleFactor(QStringList& filenames)
{
//    bool verbose = false;
    float scaleFactor = 1.f;

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

//        int sizeMemoryTexture_kb = 0;

        //for (int i=0;i<filenames.size();++i)
        for (int aK=0; aK< getData()->getNbImages();++aK)
        {
            QSize imageSize = getData()->getMaskedImage(aK)._m_image->size();

            //sizeMemoryTexture_kb += imageSize.width()*imageSize.width()*4/1024;
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

    int widthMax              = 0;
    int heightMax             = 0;

    //for (int i=0;i<filenames.size();++i)
    int nbImgs = filenames.size();


    for (int aK=0; aK< nbImgs;++aK)
    {
        QImageReader imageReader(filenames[aK]);
//        QSize imageSize = getData()->getMaskedImage(aK)._m_image->size();

        QSize imageSize = imageReader.size();

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

    if (scaleFactor != 1.f && nbImgs==1)
    {
        scaleFactor /= 2.f;
    }

    //scaleFactor = min(scaleFactor,scaleFactorVRAM); // TODO A GERER


    return scaleFactor;

//    for (int aK=0; aK< nbImgs;++aK)
//    {
//        getData()->getMaskedImage(aK)._loadedImageRescaleFactor = scaleFactor;
//    }

//	QTime *chro = new QTime(0,0,0,0) ;
//	chro->start();

//    if (scaleFactor != 1.f)
//    {
//		qDebug() << chro->elapsed() << " rescale image and mask";
//        if (verbose)
//        {
//            QString msg = QObject::tr("Rescaling images with ") + QString::number(scaleFactor,'f', 2) + QObject::tr(" factor");
//            QMessageBox* msgBox = new QMessageBox(QMessageBox::Warning, QObject::tr("GL_MAX_TEXTURE_SIZE exceeded"),  msg);
//            msgBox->setWindowFlags(Qt::WindowStaysOnTopHint);

//            msgBox->exec();
//        }

//        //Rescale image and mask
//        for (int aK=0; aK< nbImgs;++aK)
//        {
//            QImage * image = getData()->getMaskedImage(aK)._m_image;
//			QImage * mask  = getData()->getMaskedImage(aK)._m_mask;

//            QSize newSize = image->size()*scaleFactor;

//            //cout << "new size: " << newSize.width() << " " << newSize.height() << endl;

//			getData()->getMaskedImage(aK)._m_rescaled_image = new QImage(newSize, QImage::Format_Mono);
//			*(getData()->getMaskedImage(aK)._m_rescaled_image) = image->scaled(newSize,Qt::IgnoreAspectRatio);

//			//DUMP(getData()->getMaskedImage(aK)._m_rescaled_image->byteCount()/(1024*1024))

//			getData()->getMaskedImage(aK)._m_rescaled_mask = new QImage(newSize, QImage::Format_Mono);
//			*(getData()->getMaskedImage(aK)._m_rescaled_mask) = mask->scaled(newSize,Qt::IgnoreAspectRatio);

//			//DUMP(getData()->getMaskedImage(aK)._m_rescaled_mask->byteCount()/(1024*1024))
//		}

//		qDebug() << chro->elapsed() << " rescale end";
//    }
//    else
//    {
//        for (int aK=0; aK< nbImgs;++aK) // TODO 2015 ne charge t'on pas deux fois l'image
//        {
//            getData()->getMaskedImage(aK)._m_rescaled_image = getData()->getMaskedImage(aK)._m_image;
//            getData()->getMaskedImage(aK)._m_rescaled_mask = getData()->getMaskedImage(aK)._m_mask;
//        }
//    }
}

#ifdef USE_MIPMAP_HANDLER
	int cEngine::minLoadedGLDataId() const
	{
		for (int i = 0; i < _vGLData.size(); i++)
			if (_vGLData[i]->isLoaded()) return i;
		return -1;
	}

	void cEngine::getGLDataIdSet( int aI0, int aI1, bool aIsLoaded, size_t aNbRequestedWidgets, std::vector<int> &oIds ) const
	{
		int di;
		if (aI0 <= aI1)
		{
			aI0 = max<int>(0, aI0);
			aI1 = min<int>(nbGLData(), aI1) + 1;
			di = 1;
		}
		else
		{
			aI0 = min<int>(nbGLData(), aI0);
			aI1 = max<int>(0, aI1) - 1;
			di = -1;
		}

		oIds.resize(aNbRequestedWidgets, -1);
		int *itDst = oIds.data();
		int * const dst_end = oIds.data() + oIds.size();
		for (int i = aI0; i != aI1 && itDst < dst_end; i += di)
		{
			const cGLData &data = *getGLData(i);
			// we consider only GLData with a source image
			if (data.glImageMasked().hasSrcImage() && data.isLoaded() == aIsLoaded) *itDst++ = i;
		}
		oIds.resize(itDst - oIds.data());
	}
#endif

void cEngine::setUpdateSignaler(UpdateSignaler *aSignaler)
{
	_updateSignaler = aSignaler;
}

void cEngine::signalUpdate()
{
	if (_updateSignaler != NULL) (*_updateSignaler)();
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
