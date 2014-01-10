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
  , _idx(idx)
  , _parentSet(theSet)
{
    _matrixManager.resetAllMatrix();

    _time.start();

    setFocusPolicy(Qt::StrongFocus);

    setAcceptDrops(true);           //drag & drop handling

    setMouseTracking(true);

    constructMessagesList(true);

    _painter = new QPainter();

    QGLFormat tformGL(QGL::SampleBuffers);
    tformGL.setSamples(16);
    setFormat(tformGL);
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
    m_GLData = aData;

    m_GLData->setPainter(_painter);

    clearPolyline();

    m_bDisplayMode2D = !m_GLData->isImgEmpty();
    m_bFirstAction = m_GLData->isNewMask();

    resetView(showMessage, doZoom);
}

void GLWidget::paintGL()
{
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    //gradient color background
    cImageGL::drawGradientBackground(_matrixManager.vpWidth(), _matrixManager.vpHeight(), _BGColor0, _BGColor1);

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

    if (hasDataLoaded()&&(m_bDisplayMode2D || (m_interactionMode == SELECTION))) drawPolygon();
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
        switch(event->key())
        {
        case Qt::Key_Escape:
            clearPolyline();
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
        default:
            event->ignore();
            break;
        }
    }
    update();
}

void GLWidget::keyReleaseEvent(QKeyEvent* event)
{
    if ((event->key() == Qt::Key_Shift) && hasDataLoaded())
    {
        m_GLData->m_polygon.helper()->clear();
        m_GLData->m_polygon.resetSelectedPoint();
    }
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

void GLWidget::drawPolygon()
{
    _matrixManager.orthoProjection();

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    _painter->begin(this);

    if (m_bDisplayMode2D) // TODO pas beau !!!
    {
        _matrixManager.PolygonImageToWindow(m_GLData->m_polygon, _params.m_zoom).draw();
        _matrixManager.PolygonImageToWindow(*(m_GLData->m_polygon.helper()), _params.m_zoom).draw();
    }
    else
    {
        m_GLData->m_polygon.draw();
        m_GLData->m_polygon.helper()->draw();
    }

    _painter->setRenderHint(QPainter::Antialiasing,false);
    _painter->end();
}

void GLWidget::setInteractionMode(int mode, bool showmessage)
{
    m_interactionMode = mode;

    switch (mode)
    {
    case TRANSFORM_CAMERA:
        clearPolyline();
        break;
    case SELECTION:
    {
        if(hasDataLoaded() && !m_bDisplayMode2D) //3D
            _matrixManager.setMatrices();
    }
        break;
    }

    showBall(mode ? TRANSFORM_CAMERA : SELECTION && hasDataLoaded());
    showAxis(false);

    if (mode == SELECTION)
    {
        showCams(false);
        showBBox(false);
    }

    constructMessagesList(showmessage);

    update();
}

void GLWidget::setView(VIEW_ORIENTATION orientation)
{
    if (hasDataLoaded())
    {
        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();

        switch (orientation)
        {
        case TOP_VIEW:
            glRotatef(90.0f,1.0f,0.0f,0.0f);
            break;
        case BOTTOM_VIEW:
            glRotatef(-90.0f,1.0f,0.0f,0.0f);
            break;
        case FRONT_VIEW:
            glRotatef(0.0,1.0f,0.0f,0.0f);
            break;
        case BACK_VIEW:
            glRotatef(180.0f,0.0f,1.0f,0.0f);
            break;
        case LEFT_VIEW:
            glRotatef(90.0f,0.0f,1.0f,0.0f);
            break;
        case RIGHT_VIEW:
            glRotatef(-90.0f,0.0f,1.0f,0.0f);
        }

        glGetDoublev(GL_MODELVIEW_MATRIX, _matrixManager.m_rotationMatrix);

        _matrixManager.resetTranslationMatrix(m_GLData->getBBoxCenter());
    }
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
            float rw = (float)m_GLData->glMaskedImage._m_image->width()  / (float) _matrixManager.vpWidth();
            float rh = (float)m_GLData->glMaskedImage._m_image->height() / (float) _matrixManager.vpHeight();

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

#if QT_VER==5
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
                cPolygon &polygon = m_GLData->m_polygon;

                if(!polygon.isClosed())             // ADD POINT

                    polygon.addPoint(m_lastPosImage);

                else if (event->modifiers() & Qt::ShiftModifier) // INSERT POINT

                    polygon.insertPoint();

                else if (polygon.idx() != -1)

                    polygon.setPointSelected();
            }
        }
        else if (event->button() == Qt::RightButton)

            if (event->modifiers() & Qt::ControlModifier)

                m_GLData->m_polygon.removeLastPoint();

            else

                m_GLData->m_polygon.removeNearestOrClose(m_lastPosImage);

        else if (event->button() == Qt::MiddleButton)

            m_lastClickZoom = m_lastPosWindow;
    }
}

void GLWidget::mouseReleaseEvent(QMouseEvent *event)
{
    if ( event->button() == Qt::LeftButton && hasDataLoaded() )
    {
        m_GLData->m_polygon.finalMovePoint(); //ne pas factoriser

        m_GLData->m_polygon.findNearestPoint(m_lastPosImage);

        update();
    }
}

void GLWidget::mouseMoveEvent(QMouseEvent *event)
{
    if (hasDataLoaded())
    {
        _parentSet->setCurrentWidgetIdx(_idx);

#if QT_VER == 5
        QPointF pos = m_bDisplayMode2D ?  _matrixManager.WindowToImage(event->localPos(), _params.m_zoom) : event->localPos();
#else
        QPointF pos = m_bDisplayMode2D ?  _matrixManager.WindowToImage(event->posF(), _params.m_zoom) : event->posF();
#endif

        if (m_bDisplayMode2D)
        {
            m_lastMoveImage = pos;
            if (_messageManager.DrawMessages())
            {
                float w  = m_GLData->glMaskedImage._m_image->width();
                float h  = m_GLData->glMaskedImage._m_image->height();
                if  ((pos.x()>=0.f)&&(pos.y()>=0.f)&&(pos.x()<w)&&(pos.y()<h))
                    _messageManager.GetPenultimateMessage()->message = QString::number(pos.x(),'f',1) + ", " + QString::number(h-pos.y(),'f',1) + " px";
            }
        }

        if (m_bDisplayMode2D || (m_interactionMode == SELECTION))

            m_GLData->m_polygon.refreshHelper(pos,(event->modifiers() & Qt::ShiftModifier));

        if (m_interactionMode == TRANSFORM_CAMERA)
        {
            QPoint dPWin = event->pos() - m_lastPosWindow;

            if ( event->buttons())
            {
                float rX,rY,rZ;
                rX = rY = rZ = 0;
                if ( event->buttons() == Qt::LeftButton ) // rotation autour de X et Y
                {
                    rX = (float)dPWin.y() / _matrixManager.vpWidth();
                    rY = (float)dPWin.x() / _matrixManager.vpHeight();
                }
                else if ( event->buttons() == Qt::MiddleButton )
                {
                    if (event->modifiers() & Qt::ShiftModifier)         // ZOOM VIEW
                    {
                        if (dPWin.y() > 0) _params.m_zoom *= pow(2.f, ((float)dPWin.y()) *.05f);
                        else if (dPWin.y() < 0) _params.m_zoom /= pow(2.f, -((float)dPWin.y()) *.05f);
                    }
                    else if((_matrixManager.vpWidth()!=0.f) || (_matrixManager.vpHeight()!=0.f)) // TRANSLATION VIEW
                    {
                            QPointF dp = m_bDisplayMode2D ? pos - m_lastPosImage : QPointF(dPWin.x(),-dPWin.y())*m_GLData->getBBoxMaxSize();

                            _matrixManager.m_translationMatrix[0] += _params.m_speed * dp.x()/_matrixManager.vpWidth();
                            _matrixManager.m_translationMatrix[1] += _params.m_speed * dp.y()/_matrixManager.vpHeight();
                    }
                }
                else if (event->buttons() == Qt::RightButton)           // rotation autour de Z
                    rZ = (float)dPWin.x() / _matrixManager.vpWidth();

                _matrixManager.rotateMatrix(rX, rY, rZ, 50.0f *_params.m_speed);
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
#if QT_VER == 5
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

        cPolygon polyg = m_GLData->m_polygon;

        if(mode == ADD || mode == SUB)
        {
            if ((polyg.size() < 3) || (!polyg.isClosed()))
                return;

            if (!m_bDisplayMode2D)
                for (int aK=0; aK < polyg.size(); ++aK)
                    polyg[aK].setY((float)_matrixManager.vpHeight() - polyg[aK].y());
        }

        if (m_bDisplayMode2D)
            m_GLData->editImageMask(mode,polyg,m_bFirstAction);
        else
            m_GLData->editCloudMask(mode,polyg,m_bFirstAction,_matrixManager);

        if (((mode == ADD)||(mode == SUB)) && (m_bFirstAction)) m_bFirstAction = false;

        if (saveInfos) // TODO A deplacer
        {
            selectInfos info;
            info.poly   = m_GLData->m_polygon.getVector();
            info.selection_mode   = mode;

            _matrixManager.exportMatrices(info);

            _infos.push_back(info);
        }

        clearPolyline();

        update();
    }
}

void GLWidget::clearPolyline()
{
    if (hasDataLoaded())
        m_GLData->m_polygon.clear();
}

void GLWidget::undo() // TODO A deplacer
{
    if (_infos.size() && hasDataLoaded())
    {
        if ((!m_bDisplayMode2D) || (_infos.size() == 1))
            Select(ALL, false);

        for (int aK = 0; aK < _infos.size()-1; ++aK)
        {
            selectInfos &infos = _infos[aK];

            cPolygon Polygon;
            Polygon.setClosed(true);
            //Polygon.setVector(infos.poly); //TODO
            m_GLData->setPolygon(Polygon);

            if (!m_bDisplayMode2D)
            {
                _matrixManager.importMatrices(infos);
                m_bFirstAction = (aK==0);
            }

            Select(infos.selection_mode, false);
        }

        _infos.pop_back();
    }
}

void GLWidget::showAxis(bool show)
{
    if (hasDataLoaded())
    {
        m_GLData->showOption(cGLData::OpShow_Axis);
        m_GLData->pAxis->setVisible(show);
    }
    update();
}

void GLWidget::showBall(bool show)
{
    if (hasDataLoaded())
    {

        m_GLData->showOption(cGLData::OpShow_Ball);
        m_GLData->pBall->setVisible(show);
    }
    update();
}

void GLWidget::showCams(bool show)
{
    if (hasDataLoaded())
    {
        for (int i=0; i < m_GLData->Cams.size();i++)
            m_GLData->Cams[i]->setVisible(show);
    }

    update();
}

void GLWidget::showBBox(bool show)
{
    if (hasDataLoaded())
    {
        m_GLData->showOption(cGLData::OpShow_BBox);
        m_GLData->pBbox->setVisible(show);
    }

    update();
}

void GLWidget::constructMessagesList(bool show)
{
    _messageManager.constructMessagesList(show,m_interactionMode,m_bDisplayMode2D,hasDataLoaded());

    update();
}

void GLWidget::reset()
{
    clearPolyline();

    _params.reset();

    m_bFirstAction = true;

    m_GLData = NULL; //  TODO le m_GLData est il bien delete?

    resetView();
}

void GLWidget::resetView(bool zoomfit, bool showMessage)
{
    _matrixManager.resetAllMatrix( hasDataLoaded() ? m_GLData->getBBoxCenter() :  Pt3dr(0.f,0.f,0.f) );

    if (!m_bDisplayMode2D)
    {

        showBall(hasDataLoaded());
        showAxis(false);
        showBBox(false);
        showCams(false);
    }

    constructMessagesList(showMessage);

    if (zoomfit) zoomFit();

    update();
}
