#include "GLWidget.h"

GLWidget::GLWidget(int idx,  const QGLWidget *shared) : QGLWidget(QGLFormat(QGL::SampleBuffers),NULL,shared)
  , m_interactionMode(TRANSFORM_CAMERA)
  , m_bFirstAction(true)
  , m_GLData(NULL)
  , m_bDisplayMode2D(false)
  , _vp_Params(ViewportParameters())
  , _frameCount(0)
  , _previousTime(0)
  , _currentTime(0)
  , _messageManager(this)
  , _widgetId(idx)
{
    _matrixManager.resetAllMatrix();

    _time.start();

    setAcceptDrops(true);           //drag & drop handling

    setMouseTracking(true);

    setOption(cGLData::OpShow_Mess);

	#if ELISE_QT_VERSION==5 
        _painter = new QPainter();
		QGLFormat tformGL(QGL::SampleBuffers);
		tformGL.setSamples(16);
		setFormat(tformGL);
	#endif

    _contextMenu.createContextMenuActions();
}

void GLWidget::resizeGL(int width, int height)
{
    if (width==0 || height==0) return;

    _matrixManager.setGLViewport(0,0,width, height);
    _messageManager.wh(width, height);

    zoomFit();
}

//-------------------------------------------------------------------------
// Computes the frames rate
//-------------------------------------------------------------------------

void GLWidget::computeFPS(MessageToDisplay &dynMess)
{
    float       fps;

    //  Increase frame count
    _frameCount++;

    _currentTime = _time.elapsed();

    //  Compute elapsed time
    int deltaTime = _currentTime - _previousTime;

    if(deltaTime > 1000)
    {
        //  compute the number of frames per second
        fps = _frameCount * 1000.f / deltaTime;

        //  Set time
        _previousTime = _currentTime;

        //  Reset frame count
        _frameCount = 0;

        if (fps > 1e-3)
            dynMess.message = "fps: " + QString::number(fps,'f',1);
    }
}

ContextMenu* GLWidget::contextMenu()
{
    return &_contextMenu;
}

void GLWidget::setGLData(cGLData * aData, bool showMessage, bool doZoom, bool setPainter, bool resetPoly)
{
    if (aData != NULL)
    {
        m_GLData = aData;

        if(setPainter)
            m_GLData->setPainter(_painter);

        m_bDisplayMode2D = !m_GLData->isImgEmpty();
        m_bFirstAction   =  m_GLData->isNewMask();        

        _contextMenu.setPolygon( &m_GLData->m_polygon);

        resetView(doZoom, showMessage, true, resetPoly);
    }
}

void GLWidget::addGlPoint(QPointF pt, QString name, int  state, QPointF pt1, QPointF pt2, bool hightlight)
{
    cPoint point(_painter,pt,name,true,state);

    point.setHighlight(hightlight);

    if (pt1 != QPointF(0.f,0.f))
        point.setEpipolar(pt1, pt2);

    getGLData()->m_polygon.add(point);
}

bool GLWidget::imageLoaded()
{
    return hasDataLoaded() &&  m_bDisplayMode2D;
}

void GLWidget::paintGL()
{
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    //gradient color background
    cImageGL::drawGradientBackground(vpWidth(), vpHeight(), _BGColor0, !hasDataLoaded() || imageLoaded() ? _BGColor0 : _BGColor1);

    glClear(GL_DEPTH_BUFFER_BIT);

    if (hasDataLoaded())
    {
        if (m_bDisplayMode2D)
        {
            //TODO: virer dependance taille viewport / image Quad [1,1] puis scale dans glImage drawQuad()
            m_GLData->setDimensionImage(vpWidth(),vpHeight());
            //END TODO

            _matrixManager.doProjection(m_lastClickZoom, _vp_Params.m_zoom);

            m_GLData->glMaskedImage.draw();
        }
        else
        {
            _matrixManager.zoom(_vp_Params.m_zoom,2.f*m_GLData->getBBoxMaxSize());
            _matrixManager.applyTransfo();

            m_GLData->draw();        
        }

        //QPointF r = _matrixManager.WindowToImage(QPointF(0,_matrixManager.vpHeight()-10),_vp_Params.m_zoom);

        //cImageGL::drawQuad(r.x(),r.y(),10,10);

        //cImageGL::drawQuad(10,5,10,10);

        glPopMatrix();

        if (_messageManager.drawMessages() && !m_bDisplayMode2D)
            computeFPS(_messageManager.LastMessage());
    }

    _messageManager.draw();
	
    if (_widgetId >= 0) overlay();
    else        drawCenter();
}

void GLWidget::keyPressEvent(QKeyEvent* event)
{
    if(event->modifiers().testFlag(Qt::ControlModifier))
    {
        if(event->key() == Qt::Key_1)    zoomFactor(50);
        else if(event->key() == Qt::Key_2)    zoomFactor(25);
    }
    else
    {
        if (hasDataLoaded())
        {
            switch(event->key())
            {
            case Qt::Key_Delete:
                m_GLData->m_polygon.removeSelectedPoint();
                emit removePoint(NS_SaisiePts::eEPI_Deleted, m_GLData->m_polygon.idx());
                break;
            case Qt::Key_Escape:
                m_GLData->clearPolygon();
                break;
            case Qt::Key_1:
                zoomFactor(100);
                break;
            case Qt::Key_2:
                zoomFactor(200);
                break;
            case Qt::Key_4:
                zoomFactor(400);
                break;
            case Qt::Key_9:
                zoomFit();
                break;
            case Qt::Key_G:
                m_GLData->glMaskedImage._m_image->incGamma(0.2f);
                emit gammaChanged(m_GLData->glMaskedImage._m_image->getGamma());
                break;
            case Qt::Key_H:
                m_GLData->glMaskedImage._m_image->incGamma(-0.2f);
                emit gammaChanged(m_GLData->glMaskedImage._m_image->getGamma());
                break;
            case Qt::Key_J:
                m_GLData->glMaskedImage._m_image->setGamma(1.f);
                emit gammaChanged(1.f);
                break;
            case Qt::Key_Plus:
                if (m_bDisplayMode2D)
                {
                    m_lastClickZoom = m_lastPosWindow;
                    setZoom(_vp_Params.m_zoom*1.5f);
                }
                else
                    _vp_Params.ptSizeUp(true);
                break;
            case Qt::Key_Minus:
                if (m_bDisplayMode2D)
                {
                    m_lastClickZoom = m_lastPosWindow;
                    setZoom(_vp_Params.m_zoom/1.5f);
                }
                else
                    _vp_Params.ptSizeUp(false);
                break;
            case Qt::Key_W:
                    setCursor(Qt::SizeAllCursor);
                    polygon().helper()->clear();
                    polygon().setSelected(true);
                break;
            default:
                event->ignore();
                break;
            }
        }
    }

    update();
}

void GLWidget::keyReleaseEvent(QKeyEvent* event)
{
    if(hasDataLoaded())
    {
        if (event->key() == Qt::Key_Shift )
        {
            polygon().helper()->clear();
            polygon().resetSelectedPoint();
        }
        polygon().setSelected(false);
        update();
    }
    setCursor(Qt::ArrowCursor);
}

void GLWidget::dragEnterEvent(QDragEnterEvent *event)
{
    const QMimeData* mimeData = event->mimeData();

    if (mimeData->hasFormat("text/uri-list"))
        event->acceptProposedAction();
}

void GLWidget::dropEvent(QDropEvent *event)
{
    const QMimeData* mimeData = event->mimeData();

    if (mimeData->hasFormat("text/uri-list")) // TODO peut etre deplacer factoriser la gestion de drop fichier!!!
    {
        QByteArray data = mimeData->data("text/uri-list");
        QStringList fileNames = QUrl::fromPercentEncoding(data).split(QRegExp("\\n+"),QString::SkipEmptyParts);

        for (int i=0;i<fileNames.size();++i)
        {
            fileNames[i] = fileNames[i].trimmed();

#if defined(_WIN32) || defined(WIN32)
            fileNames[i].remove("file:///");
#else
            fileNames[i].remove("file://");
#endif
        }

        if (!fileNames.empty())
            emit filesDropped(fileNames);

        setFocus();

        event->acceptProposedAction();
    }

    event->ignore();
}

void GLWidget::overlay()
{
    if (hasDataLoaded() && (m_bDisplayMode2D || (m_interactionMode == SELECTION)))
    {
		#if ELISE_QT_VERSION==5
			_painter->begin(this);
		#else
			QPainter painter(this);
			_painter = &painter;
			m_GLData->setPainter(_painter);
		#endif
				
        if (m_bDisplayMode2D)
        {
            float zoom = _vp_Params.m_zoom;
            _painter->scale(zoom,-zoom);
            _painter->translate(_matrixManager.translateImgToWin(zoom));
		}

        polygon().draw();

        QPen pen(QColor(hasFocus() ? "#ffa02f" : "#707070"));
        _painter->setPen(pen);

        _painter->resetTransform();
        _painter->drawRect(this->rect());

        _painter->end();
    }
}

void GLWidget::setInteractionMode(int mode, bool showmessage)
{
    m_interactionMode = mode;

    resetView(false,showmessage,false);
}

void GLWidget::setView(VIEW_ORIENTATION orientation)
{
    if (hasDataLoaded())    
       _matrixManager.setView(orientation,m_GLData->getBBoxCenter());
}

void GLWidget::onWheelEvent(float wheelDelta_deg)
{
    //convert degrees in zoom 'power'
    float zoomFactor = pow(1.1f,wheelDelta_deg *.05f);

    setZoom(_vp_Params.m_zoom*zoomFactor);
}

void GLWidget::centerViewportOnImagePosition(QPointF pt)
{
    float vpCenterX = vpWidth() *.5f;
    float vpCenterY = vpHeight()*.5f;

    m_lastClickZoom = QPoint((int) vpCenterX, (int) vpCenterY);

    _matrixManager.translate(-pt.x() / vpCenterX, -pt.y() / vpCenterY);

    update();
}

void GLWidget::setZoom(float value)
{
    if (imageLoaded())  zoomClip( value );

    _vp_Params.m_zoom = value;

    if(imageLoaded() && _messageManager.drawMessages())
        _messageManager.GetLastMessage()->message = QString::number(_vp_Params.m_zoom*100,'f',1) + "%";

    update();
}

void GLWidget::zoomFit()
{
    if (hasDataLoaded())
    {
        if(m_bDisplayMode2D)
        {
            QPointF imCenter(imWidth()*.5f, imHeight()*.5f);
            centerViewportOnImagePosition(imCenter);

            float rw = (float) (1.05f*imWidth())  / vpWidth();
            float rh = (float) (1.05f*imHeight()) / vpHeight();

            if(rw>rh)
                setZoom(1.f/rw); //orientation landscape
            else
                setZoom(1.f/rh); //orientation portrait
        }

        else
            setZoom(m_GLData->getBBoxMaxSize());
    }
}

void GLWidget::zoomFactor(int percent)
{
    if (m_bDisplayMode2D)
    {
        m_lastClickZoom = m_lastPosWindow;
        setZoom(0.01f * percent);
    }
    else if (hasDataLoaded())
        setZoom(m_GLData->getBBoxMaxSize() / (float) percent * 100.f);
}

void GLWidget::wheelEvent(QWheelEvent* event)
{
    if ((m_interactionMode == SELECTION)&&(!m_bDisplayMode2D))
    {
        event->ignore();
        return;
    }

    m_lastClickZoom = event->pos();

#if ELISE_QT_VERSION==5
    setZoom(_vp_Params.m_zoom*pow(1.1f,event->angleDelta().y() / 160.0f ));
#else
    setZoom(_vp_Params.m_zoom*pow(1.1f,event->delta() / 160.0f ));
#endif
}

void GLWidget::mousePressEvent(QMouseEvent *event)
{
    if(hasDataLoaded())
    {
        m_lastPosWindow = event->pos();

        m_lastPosImage =  m_bDisplayMode2D ? _matrixManager.WindowToImage(m_lastPosWindow, _vp_Params.m_zoom) : m_lastPosWindow;

        if (event->button() == Qt::LeftButton)
        {
            if (m_bDisplayMode2D || (m_interactionMode == SELECTION))
            {

                if(!polygon().isClosed())             // ADD POINT

                    polygon().addPoint(m_lastPosImage);

                else if (polygon().isLinear() && (event->modifiers() & Qt::ShiftModifier)) // INSERT POINT

                    polygon().insertPoint();

                else if (polygon().idx() != -1) // SELECT POINT

                    polygon().setPointSelected();

                else if (!polygon().isLinear() && isPtInsideIm(m_lastPosImage))
                {
                    polygon().add(m_lastPosImage);

                    addPoint(m_lastPosImage);
                }
            }
        }
        else if (event->button() == Qt::RightButton)
        {
            if (polygon().isLinear())
            {
                if (event->modifiers() & Qt::ControlModifier)

                    polygon().removeLastPoint();

                else

                    polygon().removeNearestOrClose(m_lastPosImage);
            }            
        }
        else if (event->button() == Qt::MiddleButton)

            m_lastClickZoom = m_lastPosWindow;
    }
}

void GLWidget::mouseReleaseEvent(QMouseEvent *event)
{
    if ( event->button() == Qt::LeftButton && hasDataLoaded())
    {
        int idMovePoint = polygon().finalMovePoint(); //ne pas factoriser

        polygon().findNearestPoint(m_lastPosImage);

        update();

        emit movePoint(idMovePoint);
    }
}

void GLWidget::setCursorShape(QPointF pos)
{
    QCursor c = cursor();

    if ( imageLoaded() && !polygon().isLinear() && isPtInsideIm(pos) )

        c.setShape(Qt::CrossCursor);

    else

        c.setShape(Qt::ArrowCursor);

    setCursor(c);
}

void GLWidget::drawCenter()
{
    QPointF center(((float)vpWidth())*.5f,((float)vpHeight())*.5f);

    QPainter p;
    p.begin(this);

    QPen pen(QColor(0,0,0));
    pen.setCosmetic(true);
    p.setPen(pen);

    p.drawEllipse(center,5,5);
    p.drawEllipse(center,1,1);
    p.end();
}

void GLWidget::mouseMoveEvent(QMouseEvent *event)
{
    if (hasDataLoaded())
    {

#if ELISE_QT_VERSION == 5
        QPointF pos = m_bDisplayMode2D ?  _matrixManager.WindowToImage(event->localPos(), _vp_Params.m_zoom) : event->localPos();
#else
        QPointF pos = m_bDisplayMode2D ?  _matrixManager.WindowToImage(event->posF(), _vp_Params.m_zoom) : event->posF();
#endif

        setCursorShape(pos);

        if (m_bDisplayMode2D)

            m_lastMoveImage = pos;

        if (m_bDisplayMode2D || (m_interactionMode == SELECTION))
        {

            if (polygon().isSelected())                    // MOVE POLYGON

                polygon().translate(pos - _matrixManager.WindowToImage(m_lastPosWindow, _vp_Params.m_zoom));

            else if ((m_bDisplayMode2D && isPtInsideIm(pos)) || (m_interactionMode == SELECTION)) // REFRESH HELPER POLYGON
            {
                int id = polygon().idx();

                bool insertMode = polygon().isLinear() ? (event->modifiers() & Qt::ShiftModifier) : event->type() == QMouseEvent::MouseButtonPress;

                polygon().refreshHelper(pos, insertMode, _vp_Params.m_zoom);

                if(id != polygon().idx())
                    selectPoint(polygon().idx());
            }
        }
        if (m_interactionMode == TRANSFORM_CAMERA)
        {
            QPointF dPWin = QPointF(event->pos() - m_lastPosWindow);

            if ( event->buttons())
            {
                Pt3dr r(0,0,0);

                if ( event->buttons() == Qt::LeftButton )               // ROTATION X et Y
                {
                    r.x = dPWin.y() / vpWidth();
                    r.y = dPWin.x() / vpHeight();
                }
                else if ( event->buttons() == Qt::MiddleButton )
                {
                    if (event->modifiers() & Qt::ShiftModifier)         // ZOOM VIEW

                        setZoom(_vp_Params.changeZoom(dPWin.y()));

                    else if( vpWidth() && vpHeight())                   // TRANSLATION VIEW
                    {
                        QPointF dp = m_bDisplayMode2D ? pos - m_lastPosImage : QPointF(dPWin .x(),-dPWin .y()) * m_GLData->getBBoxMaxSize();
                        _matrixManager.translate(dp.x()/vpWidth(),dp.y()/vpHeight(),0.0,_vp_Params.m_speed);
                    }
                }
                else if (event->buttons() == Qt::RightButton)           // ROTATION Z
                    r.z = (float)dPWin.x() / vpWidth();

                _matrixManager.rotate(r.x, r.y, r.z, 50.f *_vp_Params.m_speed);
            }

            emit newImagePosition( m_lastMoveImage );
        }

        m_lastPosWindow = event->pos();

        update();
    }
}

void GLWidget::mouseDoubleClickEvent(QMouseEvent *event)
{
    if (hasDataLoaded() && m_GLData->Clouds.size())
    {
#if ELISE_QT_VERSION == 5
        QPointF pos = event->localPos();
#else
        QPointF pos = event->posF();
#endif

        if (m_GLData->position2DClouds(_matrixManager,pos))
            update();
    }
}

void GLWidget::Select(int mode, bool saveInfos)
{
    if (hasDataLoaded())
    {
        cPolygon polyg = polygon();

        if(mode == ADD || mode == SUB)
        {
            if ((polyg.size() < 3) || (!polyg.isClosed()))
                return;

            if (!m_bDisplayMode2D)
                polyg.flipY((float)_matrixManager.vpHeight());
        }

        if (m_bDisplayMode2D)
            m_GLData->editImageMask(mode,polyg,m_bFirstAction);
        else
            m_GLData->editCloudMask(mode,polyg,m_bFirstAction,_matrixManager);

        if (mode == ADD || mode == SUB) m_bFirstAction = false;

        if (saveInfos)
        {
            selectInfos info(polygon().getVector(),mode);

            _matrixManager.exportMatrices(info);

            _historyManager.push_back(info);
        }

        m_GLData->clearPolygon();

        update();
    }
}

void GLWidget::applyInfos()
{
    if (hasDataLoaded())
    {
        int actionIdx = _historyManager.getActionIdx();

        QVector <selectInfos> vInfos = _historyManager.getSelectInfos();

        if (actionIdx < 0 || actionIdx > vInfos.size()) return;

        for (int aK = 0; aK < actionIdx ; aK++)
        {
            selectInfos &infos = vInfos[aK];

            m_GLData->setPolygon(cPolygon(infos.poly, true));

            if (!m_bDisplayMode2D)

                _matrixManager.importMatrices(infos);

            Select(infos.selection_mode, false);
        }
    }
}

void GLWidget::setOption(QFlags<cGLData::Option> option, bool show)
{
    if (hasDataLoaded()) m_GLData->setOption(option,show);

    if( option & cGLData::OpShow_Mess) _messageManager.constructMessagesList(show, m_interactionMode, m_bDisplayMode2D, hasDataLoaded());

    update();
}

void GLWidget::reset()
{
    _vp_Params.reset();
    _historyManager.reset();

    m_bFirstAction = true;

    m_GLData = NULL;

    resetView();
}

void GLWidget::resetView(bool zoomfit, bool showMessage, bool resetMatrix, bool resetPoly)
{
    if (resetMatrix)
        _matrixManager.resetAllMatrix( hasDataLoaded() ? m_GLData->getBBoxCenter() : Pt3dr(0.f,0.f,0.f) );

    if (hasDataLoaded() && resetPoly) m_GLData->clearPolygon();

    setOption(cGLData::OpShow_Mess,showMessage);

    if (!m_bDisplayMode2D)
    {
        setOption(cGLData::OpShow_Ball, m_interactionMode == TRANSFORM_CAMERA);

        setOption(cGLData::OpShow_Axis, false);

        if (m_interactionMode == SELECTION)
        {
            _matrixManager.setMatrices();

            setOption(cGLData::OpShow_BBox | cGLData::OpShow_Cams,false);
        }
    }

    if (zoomfit)
        zoomFit(); //update already done in zoomFit
    else
        update();
}

void GLWidget::contextMenuEvent(QContextMenuEvent * event)
{
    QMenu menu(this);

    if (hasDataLoaded())
    {
        if (polygon().findNearestPoint(m_lastPosImage, polygon().radius()/getZoom()))
        {
            menu.addAction(_contextMenu._validate   );
            menu.addAction(_contextMenu._dubious    );
            menu.addAction(_contextMenu._refuted    );
            menu.addAction(_contextMenu._noSaisie   );
            menu.addAction(_contextMenu._highLight  );
            menu.addSeparator();
            menu.addAction(_contextMenu._rename     );
        }
        else
        {
            //menu.setWindowTitle(tr("Switch"));

            //menu.setWindowFlags(Qt::Tool | Qt::WindowTitleHint | Qt::WindowStaysOnTopHint);

            menu.addAction( _contextMenu._AllW  );
            menu.addAction( _contextMenu._ThisW );
            menu.addAction( _contextMenu._ThisP );
        }

        _contextMenu.setPos(m_lastPosImage);

        menu.exec(event->globalPos());

        polygon().resetSelectedPoint();

        emit selectPoint(-1);
    }
}

void GLWidget::enterEvent(QEvent *event)
{
    setFocus(Qt::ActiveWindowFocusReason);
    setFocusPolicy(Qt::StrongFocus);

    emit overWidget(this);
}
