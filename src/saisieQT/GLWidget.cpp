#include "GLWidget.h"

//Min and max zoom ratio (relative)
const float GL_MAX_ZOOM = 50.f;
const float GL_MIN_ZOOM = 0.01f;

GLWidget::GLWidget(int idx, GLWidgetSet *theSet, const QGLWidget *shared) : QGLWidget(NULL,shared)
  , m_bDrawMessages(true)
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
    _matrixManager.resetRotationMatrix();

    _time.start();

    setFocusPolicy(Qt::StrongFocus);

    //drag & drop handling
    setAcceptDrops(true);

    setMouseTracking(true);

    constructMessagesList(m_bDrawMessages);
}

void GLWidget::resizeGL(int width, int height)
{
    if (width==0 || height==0) return;

    m_glRatio  = (float) width/height;

    glViewport( 0, 0, width, height );
    glGetIntegerv (GL_VIEWPORT, _matrixManager.getGLViewport());

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

    // TODO a simplifier /////////////////////////////

    clearPolyline();

    if (m_GLData->is3D())
    {
        m_bDisplayMode2D = false;

        _matrixManager.resetRotationMatrix();
        _matrixManager.resetTranslationMatrix(m_GLData->getBBoxCenter());
    }

    if (!m_GLData->isImgEmpty())
    {
        m_bDisplayMode2D = true;
        resetProjectionMatrice();
        m_bFirstAction = m_GLData->glMaskedImage._m_newMask;
    }

    if (doZoom) zoomFit();

    constructMessagesList(showMessage);

    //  //////////////////////////////////////////////////////////////////////////
    update();
}

void GLWidget::resetProjectionMatrice()
{
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glGetDoublev (GL_MODELVIEW_MATRIX, _matrixManager.getModelViewMatrix());
}

void GLWidget::setBackgroundColors(const QColor &col0, const QColor &col1)
{
    _BGColor0 = col0;
    _BGColor1 = col1;
}

void GLWidget::paintGL()
{
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    //gradient color background
    cImageGL::drawGradientBackground(_matrixManager.vpWidth(), _matrixManager.vpHeight(), _BGColor0, _BGColor1);

    glClear(GL_DEPTH_BUFFER_BIT);

    if (hasDataLoaded())
    {
        if (!m_GLData->isImgEmpty())
        {
            _matrixManager.doProjection(m_lastClickZoom, _params.m_zoom);

            m_GLData->glMaskedImage.draw();

            glPopMatrix();
        }
        else if(m_GLData->is3D())
        {
            zoom();

            _matrixManager.applyTransfo();

            m_GLData->draw();

            glPopMatrix();
        }

        if (m_bDisplayMode2D || (m_interactionMode == SELECTION)) drawPolygon();

        if (m_bDrawMessages && _messageManager.size())
        {
            if (m_bDisplayMode2D)
            {
                _messageManager.GetLastMessage()->message = QString::number(_params.m_zoom*100,'f',1) + "%";

                float px = m_lastMoveImage.x();
                float py = m_lastMoveImage.y();
                float w  = m_GLData->glMaskedImage._m_image->width();
                float h  = m_GLData->glMaskedImage._m_image->height();

                if  ((px>=0.f)&&(py>=0.f)&&(px<w)&&(py<h))
                    _messageManager.GetPenultimateMessage()->message = QString::number(px,'f',1) + ", " + QString::number(h-py,'f',1) + " px";
            }
            else
                computeFPS(_messageManager.LastMessage());
        }
    }

    _messageManager.draw();
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

bool GLWidget::hasDataLoaded()
{
    return (m_GLData == NULL) ? false : true;
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

    if (mimeData->hasFormat("text/uri-list"))
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

#ifdef _DEBUG
            QString formatedMessage = QString("File dropped: %1").arg(fileNames[i]);
            printf(" %s\n",qPrintable(formatedMessage));
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

    if (m_bDisplayMode2D)
    {
        _matrixManager.PolygonImageToWindow(m_GLData->m_polygon, _params.m_zoom).draw();
        _matrixManager.PolygonImageToWindow(*(m_GLData->m_polygon.helper()), _params.m_zoom).draw();
    }
    else if (m_GLData->is3D())
    {
        m_GLData->m_polygon.draw();
        m_GLData->m_polygon.helper()->draw();
    }
}

// zoom in 3D mode
void GLWidget::zoom()
{
    if (m_GLData != NULL)
    {
        GLdouble zoom = (GLdouble) _params.m_zoom;
        GLdouble far  = (GLdouble) 2.f*m_GLData->getBBoxMaxSize();

        MatrixManager::mglOrtho(-zoom*m_glRatio,zoom*m_glRatio,-zoom, zoom,-far, far);
    }
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
        if(hasDataLoaded() && m_GLData->is3D()) //3D
            _matrixManager.setMatrices();
    }
        break;
    default:
        break;
    }

    constructMessagesList(showmessage);
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

    update();
}

void GLWidget::zoomFit()
{
    if (hasDataLoaded())
    {
        if(!m_GLData->isImgEmpty())
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
    else if (hasDataLoaded() && m_GLData->is3D())
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
        m_GLData->m_polygon.finalMovePoint(m_lastPosImage); //ne pas factoriser

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

        if (m_bDisplayMode2D)  m_lastMoveImage = pos;

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
                        if (m_bDisplayMode2D)
                        {
                            QPointF dp = pos - m_lastPosImage;

                            _matrixManager.m_glPosition[0] += _params.m_speed * dp.x()/_matrixManager.vpWidth();
                            _matrixManager.m_glPosition[1] += _params.m_speed * dp.y()/_matrixManager.vpHeight();
                        }
                        else
                        {
                            _matrixManager.m_translationMatrix[0] += _params.m_speed*dPWin.x()*m_GLData->getBBoxMaxSize()/_matrixManager.vpWidth();
                            _matrixManager.m_translationMatrix[1] -= _params.m_speed*dPWin.y()*m_GLData->getBBoxMaxSize()/_matrixManager.vpHeight();
                        }
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

        _matrixManager.setMatrices();

        int idx1 = -1;
        int idx2;

        pos.setY(_matrixManager.vpHeight() - pos.y());

        for (int aK=0; aK < m_GLData->Clouds.size();++aK)
        {
            float sqrD;
            float dist = FLT_MAX;
            idx2 = -1; // TODO a verifier, pourquoi init a -1 , probleme si plus 2 nuages...
            QPointF proj;

            GlCloud *a_cloud = m_GLData->Clouds[aK];

            for (int bK=0; bK < a_cloud->size();++bK)
            {
                _matrixManager.getProjection(proj, a_cloud->getVertex( bK ).getPosition());

                sqrD = (proj.x()-pos.x())*(proj.x()-pos.x()) + (proj.y()-pos.y())*(proj.y()-pos.y());

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
            GlCloud *a_cloud = m_GLData->Clouds[idx1];
            Pt3dr Pt = a_cloud->getVertex( idx2 ).getPosition();

            m_GLData->setGlobalCenter(Pt);

            _matrixManager.resetTranslationMatrix(Pt);

            update();
        }
    }
}

void GLWidget::Select(int mode, bool saveInfos)
{
    if (hasDataLoaded())
    {
        QPointF P2D;
        bool pointInside;
        cPolygon polyg;

        if(mode == ADD || mode == SUB)
        {
            cPolygon &polygon = m_GLData->m_polygon;

            if ((polygon.size() < 3) || (!polygon.isClosed()))
                return;

            if (!m_bDisplayMode2D)
            {
                for (int aK=0; aK < polygon.size(); ++aK)
                {
                    polyg.add(QPointF(polygon[aK].x(), (float)_matrixManager.vpHeight() - polygon[aK].y()));
                }
            }
            else
                polyg = polygon;
        }

        if (m_bDisplayMode2D)
        {
            QPainter    p;
            QBrush SBrush(Qt::white);
            QBrush NSBrush(Qt::black);

            p.begin(m_GLData->getMask());
            p.setCompositionMode(QPainter::CompositionMode_Source);
            p.setPen(Qt::NoPen);

            if(mode == ADD)
            {
                if (m_bFirstAction)
                {
                    p.fillRect(m_GLData->getMask()->rect(), Qt::black);
                }
                p.setBrush(SBrush);
                p.drawPolygon(polyg.getVector().data(),polyg.size());
            }
            else if(mode == SUB)
            {
                p.setBrush(NSBrush);
                p.drawPolygon(polyg.getVector().data(),polyg.size());
            }
            else if(mode == ALL)
            {
                p.fillRect(m_GLData->getMask()->rect(), Qt::white);
            }
            else if(mode == NONE)
            {
                p.fillRect(m_GLData->getMask()->rect(), Qt::black);
            }
            p.end();

            if(mode == INVERT)
                m_GLData->getMask()->invertPixels(QImage::InvertRgb);

            m_GLData->glMaskedImage._m_mask->ImageToTexture(m_GLData->getMask());
        }
        else
        {
            _matrixManager.setModelViewMatrix();

            for (int aK=0; aK < m_GLData->Clouds.size(); ++aK)
            {
                GlCloud *a_cloud = m_GLData->Clouds[aK];

                for (uint bK=0; bK < (uint) a_cloud->size();++bK)
                {
                    GlVertex &P  = a_cloud->getVertex( bK );
                    Pt3dr  Pt = P.getPosition();

                    switch (mode)
                    {
                    case ADD:
                        _matrixManager.getProjection(P2D, Pt);
                        pointInside = polyg.isPointInsidePoly(P2D);
                        if (m_bFirstAction)
                            P.setVisible(pointInside);
                        else
                            P.setVisible(pointInside||P.isVisible());
                        break;
                    case SUB:
                        if (P.isVisible())
                        {
                            _matrixManager.getProjection(P2D, Pt);
                            pointInside = polyg.isPointInsidePoly(P2D);
                            P.setVisible(!pointInside);
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

        if (((mode == ADD)||(mode == SUB)) && (m_bFirstAction)) m_bFirstAction = false;

        if (saveInfos)
        {
            selectInfos info;
            info.poly   = m_GLData->m_polygon.getVector();
            info.selection_mode   = mode;

            _matrixManager.exportMatrices(info);

            _infos.push_back(info);
        }

        clearPolyline();
    }
}

void GLWidget::clearPolyline()
{
    if (hasDataLoaded())
    {
        cPolygon &poly = m_GLData->m_polygon;

        poly.clear();
        poly.setClosed(false);
        poly.helper()->clear();
    }

    update();
}

void GLWidget::undo()
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
            Polygon.setVector(infos.poly);
            m_GLData->setPolygon(Polygon);

            if (!m_bDisplayMode2D)
            {
                _matrixManager.importMatrices(infos);

                if (aK==0) m_bFirstAction = true;
                else m_bFirstAction = false;
            }

            Select(infos.selection_mode, false);
        }

        _infos.pop_back();
    }
}

void GLWidget::showAxis(bool show)
{
    if (hasDataLoaded())
        m_GLData->pAxis->setVisible(show);
    update();
}

void GLWidget::showBall(bool show)
{
    if (hasDataLoaded())
        m_GLData->pBall->setVisible(show);
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
        m_GLData->pBbox->setVisible(show);

    update();
}

void GLWidget::constructMessagesList(bool show)
{
    m_bDrawMessages = show;

    _messageManager.constructMessagesList(show,m_interactionMode,m_bDisplayMode2D,hasDataLoaded());

    update();
}

void GLWidget::reset()
{
    if (!m_bDisplayMode2D)
    {
        _matrixManager.resetRotationMatrix();
        if (hasDataLoaded()) _matrixManager.resetTranslationMatrix(m_GLData->getBBoxCenter());
        _matrixManager.resetPosition();
    }

    clearPolyline();

    _params.reset();

    m_bFirstAction = true;

    m_GLData = NULL;

    resetView();
}

void GLWidget::resetView()
{
    constructMessagesList(true);

    if (m_bDisplayMode2D)
        zoomFit();
    else
    {
        _matrixManager.resetRotationMatrix();

        if (hasDataLoaded())
        {
            setZoom(m_GLData->getBBoxMaxSize());
            _matrixManager.resetTranslationMatrix(m_GLData->getBBoxCenter());
        }

        showBall(hasDataLoaded());
        showAxis(false);
        showBBox(false);
        showCams(false);
    }

    update();
}
