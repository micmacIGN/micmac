#include "cgldata.h"
#include <limits>

#ifdef __DEBUG
	string eToString( QImage::Format e )
	{
		switch (e)
		{
		case QImage::Format_Invalid: return "Invalid";
		case QImage::Format_Mono: return "Mono";
		case QImage::Format_MonoLSB: return "MonoLSB";
		case QImage::Format_Indexed8: return "Indexed8";
		case QImage::Format_RGB32: return "RGB32";
		case QImage::Format_ARGB32: return "ARGB32";
		case QImage::Format_ARGB32_Premultiplied: return "ARGB32_Premultiplied";
		case QImage::Format_RGB16: return "RGB16";
		case QImage::Format_ARGB8565_Premultiplied: return "ARGB8565_Premultiplied";
		case QImage::Format_RGB666: return "RGB666";
		case QImage::Format_ARGB6666_Premultiplied: return "ARGB6666_Premultiplied";
		case QImage::Format_RGB555: return "RGB555";
		case QImage::Format_ARGB8555_Premultiplied: return "ARGB8555_Premultiplied";
		case QImage::Format_RGB888: return "RGB888";
		case QImage::Format_RGB444: return "RGB444";
		case QImage::Format_ARGB4444_Premultiplied: return "ARGB4444_Premultiplied";
		case QImage::Format_RGBX8888: return "RGBX8888";
		case QImage::Format_RGBA8888: return "RGBA8888";
		case QImage::Format_RGBA8888_Premultiplied: return "RGBA8888_Premultiplied";
		//~ case QImage::Format_BGR30: return "BGR30";
		//~ case QImage::Format_A2BGR30_Premultiplied: return "A2BGR30_Premultiplied";
		//~ case QImage::Format_RGB30: return "RGB30";
		//~ case QImage::Format_A2RGB30_Premultiplied: return "A2RGB30_Premultiplied";
		//~ case QImage::Format_Alpha8: return "Alpha8";
		//~ case QImage::Format_Grayscale8: return "Grayscale8";
		case QImage::NImageFormats: return "NImageFormats";
		}
		return "unknown";
	}
#endif

#ifdef DUMP_GL_DATA
	list<cGLData *> __all_cGLData;

	size_t __dump( QImage &aQImage, const string &aName, const string &aPrefix )
	{
		size_t total = size_t(aQImage.bytesPerLine()) * size_t(aQImage.height());
		cout << aPrefix << "QImage " << aName << ' ' << aQImage.width() << 'x' << aQImage.height() << ' ' << eToString(aQImage.format()) << ": " << humanReadable(total) << endl;
		return total;
	}

	#ifdef USE_MIPMAP_HANDLER
		size_t __dump( MipmapHandler::Mipmap &aMipmap, const string &aPrefix )
		{
			size_t total = (aMipmap.mData == NULL ? 0 : aMipmap.mNbBytes);
			cout << aPrefix << "Mipmap " << aMipmap.mWidth << 'x' << aMipmap.mHeight << 'x' << aMipmap.mNbChannels << '(' << aMipmap.mNbBitsPerChannel << "): " << humanReadable(total) << endl;
			return total;
		}
	#else
		size_t __dump( QMaskedImage &aQMaskedImage, const string &aPrefix )
		{
			cout << aPrefix << ">>>QMaskedImage:" << endl;
			size_t total = 0;
			if (aQMaskedImage._m_image != NULL)          total += __dump(*aQMaskedImage._m_image, "_m_image", aPrefix + "\t");
			if (aQMaskedImage._m_mask != NULL)           total += __dump(*aQMaskedImage._m_mask, "_m_mask", aPrefix + "\t");
			if (aQMaskedImage._m_rescaled_image != NULL) total += __dump(*aQMaskedImage._m_rescaled_image, "_m_rescaled_image", aPrefix + "\t");
			if (aQMaskedImage._m_rescaled_mask != NULL)  total += __dump(*aQMaskedImage._m_rescaled_mask, "_m_rescaled_mask", aPrefix + "\t");
			cout << aPrefix << "<<<QMaskedImage: total = " << humanReadable(total) << endl;
			return total;
		}
	#endif

	size_t __dump( cImageGL &aImageGL, const string &aPrefix )
	{
		cout << aPrefix << "cImageGL: unknown" << endl;
		return 0;
	}

	size_t __dump( cMaskedImageGL &aMaskedImageGL, const string &aPrefix )
	{
		cout << aPrefix << ">>>cMaskedImageGL:" << endl;
		size_t total = 0;
		if (aMaskedImageGL._m_image != NULL)          total += __dump(*aMaskedImageGL._m_image, aPrefix + "\t");
		if (aMaskedImageGL._m_mask != NULL)           total += __dump(*aMaskedImageGL._m_mask, aPrefix + "\t");
		if (aMaskedImageGL._m_rescaled_image != NULL) total += __dump(*aMaskedImageGL._m_rescaled_image, aPrefix + "\t");
		if (aMaskedImageGL._m_rescaled_mask != NULL)  total += __dump(*aMaskedImageGL._m_rescaled_mask, aPrefix + "\t");
		#ifdef USE_MIPMAP_HANDLER
			if (aMaskedImageGL.hasSrcImage())               total += __dump(aMaskedImageGL.srcImage(), aPrefix + "\t");
			if (aMaskedImageGL.hasSrcMask())               total += __dump(aMaskedImageGL.srcMask(), aPrefix + "\t");
		#else
			if (aMaskedImageGL.hasQImage())               total += __dump(*aMaskedImageGL.getMaskedImage(), aPrefix + "\t");
		#endif
		cout << aPrefix << "<<<cMaskedImageGL: total = " << humanReadable(total) << endl;
		return total;
	}

	size_t __dump( const QVector <cMaskedImageGL*> &aData, const string &aPrefix )
	{
		cout << aPrefix << ">>>QVector<cMaskedImageGL*>:" << endl;
		size_t total = 0;
		foreach(cMaskedImageGL *ptr, aData)
			total += __dump(*ptr, aPrefix + "\t");
		cout << aPrefix << "<<<QVector<cMaskedImageGL*>: total = " << humanReadable(total) << endl;
		return total;
	}

	size_t __dump( cGLData &aData, const string &aPrefix )
	{
		cout << aPrefix << ">>>cGLData{" << &aData << "}:" << endl;
		size_t total = __dump(aData.glImageMasked(), aPrefix + "\t");
		total += __dump(aData.glTiles(), aPrefix + "\t");
		cout << aPrefix << "<<<cGLData: total = " << humanReadable(total) << endl;
		return total;
	}

	string formatedLine( string aText, bool aAppend = true, size_t aLineSize = 60, char aFillCharacter = '-' )
	{
		if ( !aText.empty() && aText.length() < aLineSize)
		{
			if (aAppend)
				aText.append(" ");
			else
				aText = string(" ") + aText;
		}
		if (aText.length() >= aLineSize) return aText;
		if (aAppend) return aText + string(aLineSize - aText.length(), aFillCharacter);
		return string(aLineSize - aText.length(), aFillCharacter) + aText;
	}

	void __dump_used_memory( const string &aName = string() )
	{
		cout << formatedLine(aName) << endl;
		size_t total = 0;
		list<cGLData *>::iterator it = __all_cGLData.begin();
		while (it != __all_cGLData.end())
			total += __dump(**it++, "\t");
		cout << formatedLine(string("total = ") + humanReadable(total), false) << endl; // false = append -> prepend
	}

	bool __exist_cGLData( cGLData *aData )
	{
		list<cGLData *>::iterator it = __all_cGLData.begin();
		while (it != __all_cGLData.end())
		{
			if (*it == aData) return true;
			it++;
		}
		return false;
	}

	void __add_cGLData( cGLData *aData )
	{
		ELISE_DEBUG_ERROR(__exist_cGLData(aData), "__add_cGLData", aData << " already exists");
		__all_cGLData.push_back(aData);
		__dump_used_memory("add_cGLData");
	}

	void __remove_cGLData( cGLData *aData )
	{
		list<cGLData *>::iterator it = __all_cGLData.begin();
		while (it != __all_cGLData.end())
		{
			if (*it == aData)
			{
				__all_cGLData.erase(it);
				return;
			}
			it++;
		}
		__dump_used_memory("remove_cGLData");
		ELISE_DEBUG_ERROR(true, "__remove_cGLData", aData << " does not exist");
	}
#endif

void cGLData::setOptionPolygons(cParameters aParams)
{
    for (int aK=0; aK < _vPolygons.size(); ++aK)
    {
        polygon(aK)->showLines(!_modePt);
        polygon(aK)->showNames(_modePt);

        polygon(aK)->setDefaultName(aParams.getDefPtName());
        polygon(aK)->setPointSize(aParams.getPointDiameter());
        polygon(aK)->setLineWidth(aParams.getLineThickness());
    }
}

#ifdef USE_MIPMAP_HANDLER
	cGLData::cGLData( int aId, cData *data, cParameters aParams, int appMode, MaskedImage aSrcImage ):
		mId(aId),
		mNbLoaded(0),
		_glMaskedImage(aSrcImage.first, aSrcImage.second),
		_bbox_center(QVector3D(0.,0.,0.)),
		_clouds_center(QVector3D(0.,0.,0.)),
		_appMode(appMode),
		_currentPolygon(-1)
	{
		#ifdef DUMP_GL_DATA
			__add_cGLData(this);
		#endif

		_modePt = false;

		#ifdef USE_MIPMAP_HANDLER
			if (aSrcImage.first != NULL)
		#else
			if (qMaskedImage != NULL)
		#endif
		{
			_pBall = NULL;
			_pAxis = NULL;
			_pBbox = NULL;
			_pGrid = NULL;

			if (appMode != MASK2D) _glMaskedImage._m_mask->setVisible(aParams.getShowMasks());
			else _glMaskedImage._m_mask->setVisible(true);

			initOptions(appMode);
			setPolygons(data);
			setOptionPolygons(aParams);

			#ifdef USE_MIPMAP_HANDLER
				setName(QString(aSrcImage.first->mFilename.c_str()));
			#endif

			return;
		}

		_pBall = new cBall;
		_pAxis = new cAxis;
		_pBbox = new cBBox;
		_pGrid = new cGrid;
		_diam = 1.f;
		_incFirstCloud = false;

		setData(data, true, aParams.getSceneCenterType());
		setPolygons(data);
		setOptionPolygons(aParams);
	}
#else
	cGLData::cGLData(cData *data, QMaskedImage *qMaskedImage, cParameters aParams, int appMode):
		_glMaskedImage(qMaskedImage),
		_pBall(NULL),
		_pAxis(NULL),
		_pBbox(NULL),
		_pGrid(NULL),
		_bbox_center(QVector3D(0.,0.,0.)),
		_clouds_center(QVector3D(0.,0.,0.)),
		_appMode(appMode)
	//    _bDrawTiles(false)
	{
		#ifdef DUMP_GL_DATA
			__add_cGLData(this);
		#endif

		if (appMode != MASK2D) _glMaskedImage._m_mask->setVisible(aParams.getShowMasks());
		else _glMaskedImage._m_mask->setVisible(true);

		initOptions(appMode);

		setPolygons(data);

		setOptionPolygons(aParams);
	}

	cGLData::cGLData(cData *data, cParameters aParams, int appMode):
		_pBall(new cBall),
		_pAxis(new cAxis),
		_pBbox(new cBBox),
		_pGrid(new cGrid),
		_bbox_center(QVector3D(0.,0.,0.)),
		_clouds_center(QVector3D(0.,0.,0.)),
		_appMode(appMode),
		_diam(1.f),
		_incFirstCloud(false)
	//    _bDrawTiles(false)
	{
		#ifdef DUMP_GL_DATA4
			__add_cGLData(this);4
		#endif

		initOptions(appMode);

		setData(data, true, aParams.getSceneCenterType());

		setPolygons(data);

		setOptionPolygons(aParams);
	}
#endif

void cGLData::setPolygons(cData *data)
{
    for (int aK = 0; aK < data->getNbPolygons(); ++aK)
    {
        if (_appMode == BOX2D)
        {
            cRectangle* polygon = new cRectangle();
            polygon->setHelper(new cPolygonHelper(polygon, 4));
            _vPolygons.push_back(polygon);
        }
        else
        {
            cPolygon* polygon = new cPolygon(*(data->getPolygon(aK)));
            polygon->setHelper(new cPolygonHelper(polygon, 3));
            _vPolygons.push_back(polygon);
        }
	}
}

void cGLData::addPolygon(cPolygon* polygon)
{
	_vPolygons.push_back(polygon);
}

void cGLData::setData(cData *data, bool setCam, int centerType	)
{
    for (int aK = 0; aK < data->getNbClouds(); ++aK)
    {
        GlCloud *pCloud = data->getCloud(aK);
        _vClouds.push_back(pCloud);
        pCloud->setBufferGl();
    }

    float sc = data->getBBoxMaxSize() / 1.5f;
	QVector3D scale(sc, sc, sc);

    _pBall->setScale(scale);
    _pAxis->setScale(scale);
    _pBbox->setScale(scale);
    _pBbox->set(data->getMin(), data->getMax());
    _pGrid->setScale(scale*2.f);

    if(setCam)
        for (int i=0; i< data->getNbCameras(); i++)
        {
			cCamGL *pCam = new cCamGL(data->getCamera(i), sc);

            _vCams.push_back(pCam);
        }

    setBBoxMaxSize(data->getBBoxMaxSize());
    setBBoxCenter(data->getBBoxCenter());
    setCloudsCenter(data->getCloudsCenter());

    switchCenterByType(centerType);

}

bool cGLData::incFirstCloud() const
{
    return _incFirstCloud;
}

void cGLData::setIncFirstCloud(bool incFirstCloud)
{
    _incFirstCloud = incFirstCloud;
}

cMaskedImageGL &cGLData::glImageMasked()
{
    return _glMaskedImage;
}

const cMaskedImageGL &cGLData::glImageMasked() const
{
    return _glMaskedImage;
}

QVector<cMaskedImageGL *> cGLData::glTiles()
{
    return _glMaskedTiles;
}

cPolygon *cGLData::polygon(int id)
{
    if (id < 0 || id >= _vPolygons.size()) return NULL;
    return _vPolygons[id];
}

cPolygon *cGLData::currentPolygon()
{
    return polygon(_currentPolygon);
}

GlCloud* cGLData::getCloud(int iC)
{
    return _vClouds[iC];
}

int cGLData::cloudCount()
{
    return _vClouds.size();
}

int cGLData::camerasCount()
{
    return _vCams.size();
}

int cGLData::polygonCount()
{
    return _vPolygons.size();
}

void cGLData::initOptions(int appMode)
{
    //TODO: retirer BASC si on saisit des vraies lignes...
    if ((appMode == POINT2D_INIT) || (appMode == POINT2D_PREDIC) || (appMode == BASC))
        _modePt = true;
    else
        _modePt = false;

    _currentPolygon = 0;
    _options = options(OpShow_Mess);
}

cGLData::~cGLData()
{
	#ifdef DUMP_GL_DATA
		__remove_cGLData(this);
	#endif

	_glMaskedImage.deleteTextures();
	_glMaskedImage.deallocImages();

	for (int aK=0; aK < _glMaskedTiles.size();++aK)
	{
		_glMaskedTiles[aK]->deleteTextures();
		_glMaskedTiles[aK]->deallocImages();

		if(_glMaskedTiles[aK])
			delete _glMaskedTiles[aK];

		_glMaskedTiles[aK] = NULL;

	}

	_glMaskedTiles.clear();

	for (int aK=0; aK < _vCams.size();++aK)
	{

		if(_vCams[aK])
			delete _vCams[aK];

		_vCams[aK] = NULL;

	}

	_vCams.clear();

	for (int aK=0; aK < _vPolygons.size();++aK)
	{

		if(_vPolygons[aK])
			delete _vPolygons[aK];

		_vPolygons[aK] = NULL;

	}

	_vPolygons.clear();

	if(_pBall != NULL) delete _pBall;
	if(_pAxis != NULL) delete _pAxis;
	if(_pBbox != NULL) delete _pBbox;
	if(_pGrid != NULL) delete _pGrid;

	//pas de delete des pointeurs dans Clouds c'est Data qui s'en charge
	_vClouds.clear();
}

void outMatrix4X4(GLdouble *mvMatrix)
{

    QString esp = "       ";
    qDebug() << mvMatrix[0] << esp << mvMatrix[1] << esp << mvMatrix[2] << esp <<  mvMatrix[3] << endl;
    qDebug() << mvMatrix[4] << esp << mvMatrix[5] << esp << mvMatrix[6] << esp <<  mvMatrix[7] << endl;
    qDebug() << mvMatrix[8] << esp << mvMatrix[9] << esp << mvMatrix[10] << esp <<  mvMatrix[11] << endl;
    qDebug() << mvMatrix[12] << esp << mvMatrix[13] << esp << mvMatrix[14] << esp <<  mvMatrix[15] << endl;
}

void cGLData::draw()
{
	if(!is3D())
	{
		if (glImageMasked().glImage()->isVisible())
			glImageMasked().draw();
		else
		{
			for (int aK=0; aK< glTiles().size(); ++aK)
				glTiles()[aK]->draw();

			 glImageMasked().glImage()->setVisible(false);
			 glImageMasked().draw();
		}
	}
	else
	{
		enableOptionLine();

		glMatrixMode(GL_MODELVIEW);
		glPushMatrix();
		glTranslated(getPosition().x(),getPosition().y(),getPosition().z());
		glRotatef(cObject::getRotation().x(),1.f,0.f,0.f);
		glRotatef(cObject::getRotation().y(),0.f,1.f,0.f);
		glRotatef(cObject::getRotation().z(),0.f,0.f,1.f);
		glTranslated(-getPosition().x(),-getPosition().y(),-getPosition().z());

		for (int i=0; i<_vClouds.size();i++)
		{
			GLfloat oldPointSize;
			glGetFloatv(GL_POINT_SIZE,&oldPointSize);

			if (_incFirstCloud && i == 0) glPointSize(oldPointSize * 3.f);

			_vClouds[i]->draw();

			glPointSize(oldPointSize);
		}

		//cameras
		for (int i=0; i< _vCams.size();i++) _vCams[i]->draw();

		glMatrixMode(GL_MODELVIEW);
		glPopMatrix();

		_pBall->draw();
		_pAxis->draw();
		_pBbox->draw();
		_pGrid->draw();

		disableOptionLine();
	}
}

void cGLData::drawCenter(bool white)
{
    //TODO: check if a point is drawn close to center

    float radius = 6.f;
    float mini   = 1.f;

    GLint       glViewport[4];
    glGetIntegerv(GL_VIEWPORT, glViewport);

    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    glScalef(2.f/(float)glViewport[2],2.f/(float)glViewport[3],1.f);
    if (white)
        glColor3f(1.f,1.f,1.f);
    else
        glColor3f(0.f,0.f,0.f);
    glDrawEllipse( 0.f, 0.f, radius, radius);
    glDrawEllipse( 0.f, 0.f, mini, mini);
    glPopMatrix();

}

void cGLData::createTiles()
{
    int maxTextureSize;
    glGetIntegerv(GL_MAX_TEXTURE_SIZE, &maxTextureSize);
	maxTextureSize /= 2;


	QSize fullSize	= _glMaskedImage.fullSize();


	unsigned int nbTilesX = 6*qCeil((float) fullSize.width() / maxTextureSize);
	unsigned int nbTilesY = 6*qCeil((float) fullSize.height()/ maxTextureSize);

//	nbTilesX /= 2;
//	nbTilesY /= 2;
//    cout << "tile size : " << fullRes_image_sizeX/ nbTilesX << " " <<  fullRes_image_sizeY/ nbTilesY << endl;

	QSize stile(fullSize.width()/nbTilesX,fullSize.height()/nbTilesY);


//    cout << "NB TILES (ROW - COL) = " << nbTilesX << "x" << nbTilesY << endl;

    for (unsigned int aK=0; aK< nbTilesX; aK++)
        for (unsigned int bK=0; bK< nbTilesY; bK++)
        {
			QRectF rect(QPointF(aK*stile.width(), bK*stile.height()),QPointF((aK+1)*stile.width(), (bK+1)*stile.height()));

            cMaskedImageGL* tile = new cMaskedImageGL(rect);

            _glMaskedTiles.push_back(tile);
        }

	#ifdef DUMP_GL_DATA
		__dump_used_memory("createTiles");
	#endif
}
cBall* cGLData::pBall() const
{
	return _pBall;
}

void cGLData::saveLockRule()
{
	_locksRule[0] = QPointF(-1,-1);
	_locksRule[1] = QPointF(-1,-1);

	if(polygon(1))
		for (int i = 0; i < polygon(1)->size(); ++i)
			if(polygon(1)->point(i).parent())
				_locksRule[i] = *((cPoint*)polygon(1)->point(i).parent());
}

void cGLData::applyLockRule()
{
	if(polygon(1))
		for (int i = 0; i < polygon(1)->size(); ++i)
		{
			if(_locksRule[i] != QPointF(-1,-1))
			{
				for (int p = 0; p < polygon(0)->size(); ++p)
				{
					if(_locksRule[i] ==  polygon(0)->point(p))
					{
						polygon(1)->point(i).setParent(&polygon(0)->point(p));
						break;
					}
				}
			}
		}
}

void cGLData::normalizeCurrentPolygon(bool nrm)
{
	cPolygon *p = currentPolygon();
	if (p == NULL) return;
	p->normalize(nrm);
}

void cGLData::clearCurrentPolygon()
{
	cPolygon *p = currentPolygon();
	if (p == NULL) return;
	p->clear();
}

void cGLData::setGlobalCenter(QVector3D aCenter)
{
    setPosition(aCenter);
    _pBall->setPosition(aCenter);
    _pAxis->setPosition(aCenter);
    _pBbox->setPosition(aCenter);
    _pGrid->setPosition(aCenter);
}

void cGLData::switchCenterByType(int val)
{
    switch(val)
    {
        case eCentroid:
        case eDefault:
            setGlobalCenter(_clouds_center);
            break;
        case eBBoxCenter:
            setGlobalCenter(_bbox_center);
            break;
        case eOriginCenter:
			setGlobalCenter(QVector3D(0.,0.,0.));
            break;
    }
}

bool cGLData::position2DClouds(MatrixManager &mm, QPointF pos)
{
    mm.setMatrices();

    int idx1 = -1;
    int idx2;

    pos.setY(mm.vpHeight() - pos.y());

    for (int aK=0; aK < _vClouds.size();++aK)
    {

		float dist = std::numeric_limits<float>::max();
        idx2 = -1; // TODO a verifier, pourquoi init a -1 , probleme si plus 2 nuages...
        QPointF proj;

        GlCloud *a_cloud = _vClouds[aK];

        for (int bK=0; bK < a_cloud->size();++bK)
        {
            mm.getProjection(proj, a_cloud->getVertex( bK ).getPosition());

			const float sqrD = (proj.x()-pos.x())*(proj.x()-pos.x()) + (proj.y()-pos.y())*(proj.y()-pos.y());

            if (sqrD < dist )
            {
                dist = sqrD;
                idx1 = aK;
                idx2 = bK;
            }
        }
    }

    if ((idx1>=0) && (idx2>=0))
    {
        //final center:
        GlCloud *a_cloud = _vClouds[idx1];
		QVector3D Pt = a_cloud->getVertex( idx2 ).getPosition();

        setGlobalCenter(Pt);
		mm.resetAllMatrix(Pt,false);

        return true;
    }

    return false;
}

#ifdef USE_MIPMAP_HANDLER
	void cGLData::editImageMask(int mode, cPolygon *polyg, bool m_bFirstAction)
	{
		QPainter    p;

		MipmapHandler::Mipmap &mask = getMask();
		QImage qimage((int)mask.mWidth, (int)mask.mHeight, QImage::Format_RGB888);
		QRect rect = qimage.rect();
		unsigned int padding = (unsigned int)(qimage.bytesPerLine() / (qimage.width() * 3));
		gray8_to_rgb888(mask.mData, mask.mWidth, mask.mHeight, qimage.bits(), padding);
		p.begin(&qimage);

		QRectF rectPoly;

		p.setCompositionMode(QPainter::CompositionMode_Source);
		p.setPen(Qt::NoPen);

		QPolygonF polyDraw(polyg->getVector());
		QPainterPath path;

		float scaleFactor = _glMaskedImage.getLoadedImageRescaleFactor();
		QTransform trans;

		if ( scaleFactor < 1.f )
		{
		    rectPoly = polyDraw.boundingRect();

		    trans = trans.scale(scaleFactor,scaleFactor);

		    polyDraw = trans.map(polyDraw);
		}

		if(mode == ADD_INSIDE || mode == SUB_INSIDE)
		{
		    path.addPolygon(polyDraw);
		}
		else if((mode == ADD_OUTSIDE || mode == SUB_OUTSIDE))
		{
		    path.addRect(rect);
		    QPainterPath inner;
		    inner.addPolygon(polyDraw);
		    path = path.subtracted(inner);
		}

		QColor colorSelect(Qt::black);
		QColor colorUnSelect(Qt::white);

		if(mode == ADD_INSIDE || mode == ADD_OUTSIDE)
		{
		    if (m_bFirstAction)
				p.fillRect(rect, colorSelect);

			p.setBrush(QBrush(colorUnSelect));
		    p.drawPath(path);
		}
		else if(mode == SUB_INSIDE || mode == SUB_OUTSIDE)
		{
			p.setBrush(QBrush(colorSelect));
		    p.drawPath(path);
		}
		else if(mode == ALL)

			p.fillRect(rect, colorUnSelect);

		else if(mode == NONE)

			p.fillRect(rect, colorSelect);

		p.end();

		if (mode == INVERT) qimage.invertPixels();
		rgb888_to_red8(qimage.bits(), mask.mWidth, mask.mHeight, padding, mask.mData);

		_glMaskedImage._m_mask->deleteTexture(); // TODO verifier l'utilit� de supprimer la texture...
		_glMaskedImage._m_mask->createTexture(getMask());
	}
#else
	void cGLData::editImageMask(int mode, cPolygon *polyg, bool m_bFirstAction)
	{
		QPainter    p;

		QRect  rect = getMask()->rect();
		QRectF rectPoly;

		p.begin(getMask());
		p.setCompositionMode(QPainter::CompositionMode_Source);
		p.setPen(Qt::NoPen);

		QPolygonF polyDraw(polyg->getVector());
		QPainterPath path;

		float scaleFactor = _glMaskedImage.getLoadedImageRescaleFactor();
		QTransform trans;

		if ( scaleFactor < 1.f )
		{
		    rectPoly = polyDraw.boundingRect();

		    trans = trans.scale(scaleFactor,scaleFactor);

		    polyDraw = trans.map(polyDraw);
		}

		if(mode == ADD_INSIDE || mode == SUB_INSIDE)
		{
		    path.addPolygon(polyDraw);
		}
		else if((mode == ADD_OUTSIDE || mode == SUB_OUTSIDE))
		{
		    path.addRect(rect);
		    QPainterPath inner;
		    inner.addPolygon(polyDraw);
		    path = path.subtracted(inner);
		}

	//	QColor colorSelect(Qt::white);
	//	QColor colorUnSelect(Qt::black);


		QColor colorSelect(Qt::black);
		QColor colorUnSelect(Qt::white);

		if(mode == ADD_INSIDE || mode == ADD_OUTSIDE)
		{
		    if (m_bFirstAction)
				p.fillRect(rect, colorSelect);

			p.setBrush(QBrush(colorUnSelect));
		    p.drawPath(path);
		}
		else if(mode == SUB_INSIDE || mode == SUB_OUTSIDE)
		{
			p.setBrush(QBrush(colorSelect));
		    p.drawPath(path);
		}
		else if(mode == ALL)

			p.fillRect(rect, colorUnSelect);

		else if(mode == NONE)

			p.fillRect(rect, colorSelect);

		p.end();

		if (mode == INVERT)
		    getMask()->invertPixels(QImage::InvertRgb);

		_glMaskedImage._m_mask->deleteTexture(); // TODO verifier l'utilit� de supprimer la texture...
		_glMaskedImage._m_mask->createTexture(getMask());

	//    if ( getDrawTiles() )
	//    {

	//        for (int aK=0; aK < glTiles().size(); ++aK)
	//        {
	//            cMaskedImageGL * tile = glTiles()[aK];
	//            cImageGL * glMaskTile = tile->glMask();

	//            QVector3D pos = glMaskTile->getPosition();
	//            QSize sz  = glMaskTile->getSize();
	//            QRectF rectImg(QPointF(pos.x,pos.y), QSizeF(sz));

	//            if (rectImg.intersects(rectPoly))
	//            {
	//                QRect rescaled_rect = trans.mapRect(rectImg.toAlignedRect());

	//                QImage mask_crop = getMask()->copy(rescaled_rect).scaled(sz, Qt::KeepAspectRatio);

	//                tile->getMaskedImage()->_m_mask = &mask_crop;

	//                glMaskTile->createTexture(tile->getMaskedImage()->_m_mask);
	//            }
	//        }
	//    }
	}
#endif

void cGLData::editCloudMask(int mode, cPolygon *polyg, bool m_bFirstAction, MatrixManager &mm)
{

    QPointF P2D;
    bool pointInside;

    for (int aK=0; aK < _vClouds.size(); ++aK)
    {
        GlCloud *a_cloud = _vClouds[aK];

        for (uint bK=0; bK < (uint) a_cloud->size();++bK)
        {
            GlVertex &P  = a_cloud->getVertex( bK );
			QVector3D  Pt = P.getPosition();

			if(getRotation().x() != 0)
            {
                Pt = Pt - getPosition() ;
				Pt = QVector3D(Pt.x(),Pt.z(),-Pt.y());
                Pt = Pt + getPosition() ;
            }

            switch (mode)
            {
            case ADD_INSIDE:
                mm.getProjection(P2D, Pt);
				pointInside = polyg->isPointInsidePoly(P2D);
                if (m_bFirstAction)
                    P.setVisible(pointInside);
                else
                    P.setVisible(pointInside||P.isVisible());
                break;
            case ADD_OUTSIDE:
                mm.getProjection(P2D, Pt);
				pointInside = polyg->isPointInsidePoly(P2D);
                if (m_bFirstAction)
                    P.setVisible(!pointInside);
                else
                    P.setVisible(!pointInside||P.isVisible());
                break;
            case SUB_INSIDE:
                if (P.isVisible())
                {
                    mm.getProjection(P2D, Pt);
					pointInside = polyg->isPointInsidePoly(P2D);
                    P.setVisible(!pointInside);
                }
                break;
            case SUB_OUTSIDE:
                if (P.isVisible())
                {
                    mm.getProjection(P2D, Pt);
					pointInside = polyg->isPointInsidePoly(P2D);
                    P.setVisible(pointInside);
                }
                break;
            case INVERT:
                P.setVisible(!P.isVisible());
                break;
            case ALL:
            {
                m_bFirstAction = true;
                P.setVisible(true);
            }
                break;
            case NONE:
                P.setVisible(false);
                break;
            }
        }

        a_cloud->setBufferGl(true);
    }
}

void cGLData::replaceCloud(GlCloud *cloud, int id)
{
    if(id<_vClouds.size())
        _vClouds[id] = cloud;
    else
        _vClouds.insert(_vClouds.begin(),cloud);

    cloud->setBufferGl();
}

void cGLData::GprintBits(const size_t size, const void * const ptr)
{
    unsigned char *b = (unsigned char*) ptr;
    unsigned char byte;
    int i, j;

    for (i=(int)(size-1);i>=0;i--)
    {
        for (j=7;j>=0;j--)
        {
            byte = b[i] & (1<<j);
            byte >>= j;
            printf("%u", byte);
        }
    }
    puts("");
}

void cGLData::setOption(QFlags<cGLData::Option> option, bool show)
{

    if(show)
        _options |=  option;
    else
        _options &= ~option;

    //GprintBits(sizeof(QFlags<Option>),&_options);

    if(isImgEmpty())
    {
        _pBall->setVisible(stateOption(OpShow_Ball));
        _pAxis->setVisible(stateOption(OpShow_Axis));
        _pBbox->setVisible(stateOption(OpShow_BBox));
        _pGrid->setVisible(stateOption(OpShow_Grid));

        for (int i=0; i < _vCams.size();i++)
            _vCams[i]->setVisible(stateOption(OpShow_Cams));
    }
}

#ifdef USE_MIPMAP_HANDLER
	void cGLData::dump( std::string aPrefix, std::ostream &aStream ) const
	{
		aStream << aPrefix << mId << " [" << name().toStdString() << ']';
		if (_glMaskedImage.hasSrcImage()) aStream << " [" << _glMaskedImage.srcImage().mFilename << ']';
		if (isLoaded()) aStream << " loaded";
		aStream << endl;
	}
#endif
