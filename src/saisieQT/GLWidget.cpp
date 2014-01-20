#include "GLWidget.h"

//Min and max zoom ratio (relative)
const float GL_MAX_ZOOM = 50.f;
const float GL_MIN_ZOOM = 0.01f;

GLWidget::GLWidget(int idx, GLWidgetSet *theSet, const QGLWidget *shared) : QGLWidget(NULL,shared)
  , m_interactionMode(TRANSFORM_CAMERA)
  , m_bFirstAction(true)
  , m_GLData(NULL)
  , m_bDisplayMode2D(false)
  , _params(ViewportParameters())
  , _frameCount(0)
  , _previousTime(0)
  , _currentTime(0)
  , _messageManager(this)
  , _widgetId(idx)
  , _parentSet(theSet)
{
    _matrixManager.resetAllMatrix();

    _time.start();

    setFocusPolicy(Qt::StrongFocus);

    setAcceptDrops(true);           //drag & drop handling

    setMouseTracking(true);

    setOption(cGLData::OpShow_Mess);

    _painter = new QPainter();

    QGLFormat tformGL(QGL::SampleBuffers);
    tformGL.setSamples(16);
    setFormat(tformGL);

    createContexMenuActions();
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

void GLWidget::setGLData(cGLData * aData, bool showMessage, bool doZoom)
{
    if (aData != NULL)
    {
        m_GLData = aData;
        m_GLData->setPainter(_painter);

        m_bDisplayMode2D = !m_GLData->isImgEmpty();
        m_bFirstAction   =  m_GLData->isNewMask();

        resetView(showMessage, doZoom);

//        if (!aData->m_polygon.bShowLines())
//        {
//            m_GLData->m_polygon.setClosed(true); //ne pas mettre dans le constructeur car mis a false dans resetView
//        }
    }
}

void GLWidget::paintGL()
{
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    //gradient color background
    cImageGL::drawGradientBackground(vpWidth(), vpHeight(), _BGColor0, _BGColor1);

    glClear(GL_DEPTH_BUFFER_BIT);

    if (hasDataLoaded())
    {
        if (m_bDisplayMode2D)
        {
            _matrixManager.doProjection(m_lastClickZoom, _params.m_zoom);

            m_GLData->glMaskedImage.draw();            
        }
        else
        {
            _matrixManager.zoom(_params.m_zoom,2.f*m_GLData->getBBoxMaxSize());
            _matrixManager.applyTransfo();

            m_GLData->draw();        
        }

        glPopMatrix();

        if (_messageManager.DrawMessages() && !m_bDisplayMode2D)
            computeFPS(_messageManager.LastMessage());
    }

    _messageManager.draw();

    Overlay();
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
                break;
            case Qt::Key_H:
                m_GLData->glMaskedImage._m_image->incGamma(-0.2f);
                break;
            case Qt::Key_J:
                m_GLData->glMaskedImage._m_image->setGamma(1.0f);
                break;
            case Qt::Key_Plus:
                if (m_bDisplayMode2D)
                {
                    m_lastClickZoom = m_lastPosWindow;
                    setZoom(_params.m_zoom*1.5f);
                }
                else
                    _params.ptSizeUp(true);
                break;
            case Qt::Key_Minus:
                if (m_bDisplayMode2D)
                {
                    m_lastClickZoom = m_lastPosWindow;
                    setZoom(_params.m_zoom/1.5f);
                }
                else
                    _params.ptSizeUp(false);
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

void GLWidget::Overlay()
{
    if (hasDataLoaded() && (m_bDisplayMode2D || (m_interactionMode == SELECTION)))
    {
        _painter->begin(this);

        if (m_bDisplayMode2D)
        {
            _painter->scale(_params.m_zoom,-_params.m_zoom);
            _painter->translate(_matrixManager.translateImgToWin(_params.m_zoom));
        }

        polygon().draw();

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

    setZoom(_params.m_zoom*zoomFactor);
}

void GLWidget::setZoom(float value)
{
    if (value < GL_MIN_ZOOM)
        value = GL_MIN_ZOOM;
    else if (value > GL_MAX_ZOOM)
        value = GL_MAX_ZOOM;

    _params.m_zoom = value;

    if(m_bDisplayMode2D && _messageManager.DrawMessages())
        _messageManager.GetLastMessage()->message = QString::number(_params.m_zoom*100,'f',1) + "%";

    update();
}

void GLWidget::zoomFit()
{
    if (hasDataLoaded())
    {
        if(m_bDisplayMode2D)
        {
            float rw = (float) imWidth()  / vpWidth();
            float rh = (float) imHeight() / vpHeight();

            if(rw>rh)
                setZoom(1.f/rw); //orientation landscape
            else
                setZoom(1.f/rh); //orientation portrait

            _matrixManager.scaleAndTranslate(-rw, -rh, _params.m_zoom);

            m_GLData->glMaskedImage.setDimensions(2.f*rh,2.f*rw);
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

#if QT_VERSION==5
    setZoom(_params.m_zoom*pow(1.1f,event->angleDelta().y() / 160.0f ));
#else
    setZoom(_params.m_zoom*pow(1.1f,event->delta() / 160.0f ));
#endif
}

void GLWidget::mousePressEvent(QMouseEvent *event)
{
    if(hasDataLoaded())
    {
        m_lastPosWindow = event->pos();

        m_lastPosImage =  m_bDisplayMode2D ? _matrixManager.WindowToImage(m_lastPosWindow, _params.m_zoom) : m_lastPosWindow;

        if (event->button() == Qt::LeftButton)
        {
            if (m_bDisplayMode2D || (m_interactionMode == SELECTION))
            {

                if(!polygon().isClosed())             // ADD POINT

                    polygon().addPoint(m_lastPosImage);

                else if ((event->modifiers() & Qt::ShiftModifier)) // INSERT POINT

                    polygon().insertPoint();

                else if (polygon().idx() != -1) // SELECT POINT

                    polygon().setPointSelected();

                else if (!polygon().bShowLines())

                    polygon().add(m_lastPosImage);
            }
        }
        else if (event->button() == Qt::RightButton)
        {
            if (polygon().bShowLines())
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
    if ( event->button() == Qt::LeftButton && hasDataLoaded() && polygon().bShowLines())
    {
        polygon().finalMovePoint(); //ne pas factoriser

        polygon().findNearestPoint(m_lastPosImage);

        update();
    }
}

void GLWidget::refreshPositionMessage(QPointF pos)
{
    if (_messageManager.DrawMessages() && (pos.x()>=0.f)&&(pos.y()>=0.f)&&(pos.x()<imWidth())&&(pos.y()<imHeight()))
        _messageManager.GetPenultimateMessage()->message = QString::number(pos.x(),'f',1) + ", " + QString::number(imHeight()-pos.y(),'f',1) + " px";
}

void GLWidget::mouseMoveEvent(QMouseEvent *event)
{
    if (hasDataLoaded())
    {
        _parentSet->setCurrentWidgetIdx(_widgetId);

#if QT_VERSION >= 5
        QPointF pos = m_bDisplayMode2D ?  _matrixManager.WindowToImage(event->localPos(), _params.m_zoom) : event->localPos();
#else
        QPointF pos = m_bDisplayMode2D ?  _matrixManager.WindowToImage(event->posF(), _params.m_zoom) : event->posF();
#endif

        if (m_bDisplayMode2D)

            refreshPositionMessage(m_lastMoveImage = pos);

        if (m_bDisplayMode2D || (m_interactionMode == SELECTION))
        {

            if(polygon().isSelected())                    // MOVE POLYGON

                polygon().translate(pos - _matrixManager.WindowToImage(m_lastPosWindow, _params.m_zoom));

            else                                                        // REFRESH HELPER POLYGON

                polygon().refreshHelper(pos,(event->modifiers() & Qt::ShiftModifier));
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
                else if ( event->buttons() == Qt::MiddleButton ){

                    if (event->modifiers() & Qt::ShiftModifier)         // ZOOM VIEW

                        _params.changeZoom(dPWin.y());

                    else if( vpWidth() || vpHeight())                   // TRANSLATION VIEW
                    {
                        QPointF dp = m_bDisplayMode2D ? pos - m_lastPosImage : QPointF(dPWin .x(),-dPWin .y()) * m_GLData->getBBoxMaxSize();
                        _matrixManager.translate(dp.x()/vpWidth(),dp.y()/vpHeight(),0.0,_params.m_speed);
                    }
                }
                else if (event->buttons() == Qt::RightButton)           // ROTATION Z
                    r.z = (float)dPWin.x() / vpWidth();

                _matrixManager.rotate(r.x, r.y, r.z, 50.0f *_params.m_speed);
            }
        }

        m_lastPosWindow = event->pos();

        update();
    }
}

void GLWidget::mouseDoubleClickEvent(QMouseEvent *event)
{
    if (hasDataLoaded() && m_GLData->Clouds.size())
    {
#if QT_VERSION >= 5
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

void GLWidget::setOption(QFlags<cGLData::Option> option,bool show)
{
    if (hasDataLoaded()) m_GLData->setOption(option,show);

    if( option & cGLData::OpShow_Mess)_messageManager.constructMessagesList(show,m_interactionMode,m_bDisplayMode2D,hasDataLoaded());

    update();
}

void GLWidget::reset()
{
    _params.reset();
    _historyManager.reset();

    m_bFirstAction = true;

    m_GLData = NULL;

    resetView();
}

void GLWidget::resetView(bool zoomfit, bool showMessage,bool resetMatrix)
{
    if (resetMatrix)
        _matrixManager.resetAllMatrix( hasDataLoaded() ? m_GLData->getBBoxCenter() :  Pt3dr(0.f,0.f,0.f) );

    if (hasDataLoaded()) m_GLData->clearPolygon();

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
//    else
//        refreshPositionMessage(m_lastPosImage); //TODO: debugger

    if (zoomfit) zoomFit();

    update();
}

void GLWidget::createContexMenuActions()
{
    QString IconFolder = QString(MMDir().c_str()) + "data/ico/";

    /*_rename    = new QAction(tr("Rename"), this);
    _showNames = new QAction(tr("Show names") , this);

    _AllW      = new QAction(QIcon(IconFolder + "AllW.ico"),            tr("AllW") , this);
    _ThisW     = new QAction(QIcon(IconFolder + "ThisW.ico"),           tr("ThisW"), this);
    _ThisP     = new QAction(QIcon(IconFolder + "ThisP.ico"),           tr("ThisP"), this);*/

    _validate  = new QAction(QIcon(IconFolder + "smile.ico"),           tr("Validate"), this);
    _dubious   = new QAction(QIcon(IconFolder + "interrogation.ico"),   tr("Dubious") , this);
    _refuted   = new QAction(QIcon(IconFolder + "refuted.ico"),         tr("Refuted") , this);
    _highLight = new QAction(QIcon(IconFolder + "HL.ico"),              tr("Highlight"), this);
    _noSaisie  = new QAction(QIcon(IconFolder + "vide.ico"),            tr("Not captured"), this);

    /*connect(_rename,		    SIGNAL(triggered()),   _signalMapper, SLOT(rename()));
    connect(_showNames,		    SIGNAL(triggered()),   _signalMapper, SLOT(showNames()));

    connect(_AllW,      	    SIGNAL(triggered()),   _signalMapper, SLOT(AllW()));
    connect(_ThisW,             SIGNAL(triggered()),   _signalMapper, SLOT(ThisW()));
    connect(_ThisP,             SIGNAL(triggered()),   _signalMapper, SLOT(ThisP()));*/

    _signalMapper = new QSignalMapper (this);

    connect(_validate,		    SIGNAL(triggered()),   _signalMapper, SLOT(map()));
    connect(_dubious,		    SIGNAL(triggered()),   _signalMapper, SLOT(map()));
    connect(_refuted,		    SIGNAL(triggered()),   _signalMapper, SLOT(map()));
    connect(_noSaisie,		    SIGNAL(triggered()),   _signalMapper, SLOT(map()));

    connect(_highLight,		    SIGNAL(triggered()),   _signalMapper, SLOT(map()));

    _signalMapper->setMapping (_validate,  NS_SaisiePts::eEPI_Valide);
    _signalMapper->setMapping (_dubious,   NS_SaisiePts::eEPI_Douteux);
    _signalMapper->setMapping (_refuted,   NS_SaisiePts::eEPI_Refute);
    _signalMapper->setMapping (_noSaisie,  NS_SaisiePts::eEPI_NonSaisi);
    _signalMapper->setMapping (_highLight, -1);

    connect (_signalMapper, SIGNAL(mapped(int)), this, SLOT(setPointState(int)));
}

void GLWidget::contextMenuEvent(QContextMenuEvent * event)
{
    QMenu menu(this);

    if ((event->modifiers() & Qt::ShiftModifier))
    {
        menu.addAction(_rename);
        menu.addAction(_showNames);
    }
    else if ((event->modifiers() & Qt::ControlModifier))
    {
        menu.addAction(_AllW);
        menu.addAction(_ThisW);
        menu.addAction(_ThisP);
    }
    else
    {
        menu.addAction(_validate);
        menu.addAction(_dubious);
        menu.addAction(_refuted);
        menu.addAction(_noSaisie);
        menu.addAction(_highLight);
    }

    menu.exec(event->globalPos());
}

void GLWidget::setPointState(int state)
{
    polygon().setNearestPointState(m_lastPosImage, state);
}
