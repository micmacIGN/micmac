#include "GLWidget.h"

#include "GLWidgetSet.h"

//Min and max zoom ratio (relative)
const float GL_MAX_ZOOM = 50.f;
const float GL_MIN_ZOOM = 0.01f;

GLWidget::GLWidget(int idx, GLWidgetSet *theSet, const QGLWidget *shared) : QGLWidget(NULL,shared)
  , m_font(font())
  , m_bDrawMessages(true)
  , m_interactionMode(TRANSFORM_CAMERA)
  , m_bFirstAction(true)  
  , m_GLData(NULL)
  , m_bDisplayMode2D(false)
  , _params(ViewportParameters())
  , _frameCount(0)
  , _previousTime(0)
  , _currentTime(0)
  , _idx(idx)
  , _parentSet(theSet)
{
    resetRotationMatrix();

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
    glGetIntegerv (GL_VIEWPORT, _g_Cam.getGLViewport());

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

    if (hasDataLoaded())
    {
        clearPolyline();

        if (m_GLData->is3D())
        {
            m_bDisplayMode2D = false;

            if (doZoom) setZoom(m_GLData->getBBoxMaxSize());

            resetRotationMatrix();
            resetTranslationMatrix();
        }

        if (!m_GLData->isImgEmpty())
        {
            m_bDisplayMode2D = true;

            if (doZoom) zoomFit();

            //position de l'image dans la vue gl
            glMatrixMode(GL_MODELVIEW);
            glLoadIdentity();
            glGetDoublev (GL_MODELVIEW_MATRIX, _g_Cam.getModelViewMatrix());

            m_bFirstAction = m_GLData->glMaskedImage._m_newMask;
        }

        glGetIntegerv (GL_VIEWPORT, _g_Cam.getGLViewport());

        constructMessagesList(showMessage);

        update();
    }
}

void GLWidget::setBackgroundColors(const QColor &col0, const QColor &col1)
{
    _BGColor0 = col0;
    _BGColor1 = col1;
}

int GLWidget::renderTextLine(MessageToDisplay messageTD, int x, int y, int sizeFont)
{
    qglColor(messageTD.color);

    m_font.setPointSize(sizeFont);

    renderText(x, y, messageTD.message,m_font);

    return (QFontMetrics(m_font).boundingRect(messageTD.message).height()*5)/4;
}

std::list<MessageToDisplay>::iterator GLWidget::GetLastMessage()
{
    std::list<MessageToDisplay>::iterator it = --m_messagesToDisplay.end();

    return it;
}

void GLWidget::paintGL()
{
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    //gradient color background
    drawGradientBackground();
    //we clear background
    glClear(GL_DEPTH_BUFFER_BIT);

    if (hasDataLoaded())
    {
        if (!m_GLData->isImgEmpty())
        {
            _g_Cam.doProjection(m_lastClickZoom, _params.m_zoom);

            m_GLData->glMaskedImage.draw();

            glPopMatrix();
        }
        else if(m_GLData->is3D())
        {

            // CAMERA BEGIN ===================
            zoom();            
            // CAMERA END ===================

            glMatrixMode(GL_MODELVIEW);
            glLoadIdentity();
            glPushMatrix();

            glMultMatrixf(_rotationMatrix);
            glTranslatef(_params.m_translationMatrix[0],_params.m_translationMatrix[1],_params.m_translationMatrix[2]);

            m_GLData->draw();

            glPopMatrix();
        }

        if (m_bDisplayMode2D || (m_interactionMode == SELECTION)) drawPolygon();

        if (m_bDrawMessages && m_messagesToDisplay.size())
        {
            if (m_bDisplayMode2D)
            {
                GetLastMessage()->message = QString::number(_params.m_zoom*100,'f',1) + "%";

                float px = m_lastMoveImage.x();
                float py = m_lastMoveImage.y();
                float w  = m_GLData->glMaskedImage._m_image->width();
                float h  = m_GLData->glMaskedImage._m_image->height();

                if  ((px>=0.f)&&(py>=0.f)&&(px<w)&&(py<h))
                    (--GetLastMessage())->message = QString::number(px,'f',1) + ", " + QString::number(h-py,'f',1) + " px";
            }
            else
                computeFPS(m_messagesToDisplay.back());
        }
    }

    if (!m_messagesToDisplay.empty())
    {
        int _glViewport2 = (int) _g_Cam.ViewPort(2);
        int _glViewport3 = (int) _g_Cam.ViewPort(3);

        int ll_curHeight, lr_curHeight, lc_curHeight; //lower left, lower right and lower center y position
        ll_curHeight = lr_curHeight = lc_curHeight = _glViewport3 - m_font.pointSize()*m_messagesToDisplay.size();
        int uc_curHeight = 10;            //upper center

        std::list<MessageToDisplay>::iterator it = m_messagesToDisplay.begin();
        while (it != m_messagesToDisplay.end())
        {
            QRect rect = QFontMetrics(m_font).boundingRect(it->message);
            switch(it->position)
            {
            case LOWER_LEFT_MESSAGE:
                ll_curHeight -= renderTextLine(*it, 10, ll_curHeight);
                break;
            case LOWER_RIGHT_MESSAGE:
                lr_curHeight -= renderTextLine(*it, _glViewport2 - 120, lr_curHeight);
                break;
            case LOWER_CENTER_MESSAGE:
                lc_curHeight -= renderTextLine(*it,(_glViewport2-rect.width())/2, lc_curHeight);
                break;
            case UPPER_CENTER_MESSAGE:
                uc_curHeight += renderTextLine(*it,(_glViewport2-rect.width())/2, uc_curHeight+rect.height());
                break;
            case SCREEN_CENTER_MESSAGE:
                renderTextLine(*it,(_glViewport2-rect.width())/2, (_glViewport3-rect.height())/2,12);
            }
            ++it;
        }
    }
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

void GLWidget::displayNewMessage(const QString& message,
                                 MessagePosition pos,
                                 QColor color)
{
    if (message.isEmpty())
    {
        m_messagesToDisplay.clear();

        return;
    }

    MessageToDisplay mess;
    mess.message = message;
    mess.position = pos;
    mess.color = color;
    m_messagesToDisplay.push_back(mess);
}

void GLWidget::drawGradientBackground()
{
    glEnable(GL_BLEND);
    glBlendFunc(GL_ONE,GL_ZERO);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();

    int w = (_g_Cam.ViewPort(2)>>1)+1;
    int h = (_g_Cam.ViewPort(3)>>1)+1;
    glOrtho(-w,w,-h,h,-2.f, 2.f);

    const uchar BkgColor[3] = {(uchar) _BGColor0.red(),(uchar) _BGColor0.green(), (uchar) _BGColor0.blue()};
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    //Gradient "texture" drawing
    glBegin(GL_QUADS);
    //user-defined background color for gradient start
    glColor3ubv(BkgColor);
    glVertex2f(-w,h);
    glVertex2f(w,h);
    //and the inverse of points color for gradient end
    glColor3ub(_BGColor1.red(),_BGColor1.green(),_BGColor1.blue());
    glVertex2f(w,-h);
    glVertex2f(-w,-h);
    glEnd();

    glDisable(GL_BLEND);
}

void GLWidget::drawPolygon()
{
    _g_Cam.orthoProjection();

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    if (m_bDisplayMode2D)
    {
        _g_Cam.PolygonImageToWindow(m_GLData->m_polygon, _params.m_zoom).draw();
        _g_Cam.PolygonImageToWindow(*(m_GLData->m_polygon.helper()), _params.m_zoom).draw();
    }
    else if (m_GLData->is3D())
    {
        m_GLData->m_polygon.draw();
        m_GLData->m_polygon.helper()->draw();
    }
}

void mglOrtho( GLdouble left, GLdouble right,
               GLdouble bottom, GLdouble top,
               GLdouble near_val, GLdouble far_val )
{
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(left, right, bottom, top, near_val, far_val);

}

// zoom in 3D mode
void GLWidget::zoom()
{
    if (m_GLData != NULL)
    {
        GLdouble zoom = (GLdouble) _params.m_zoom;
        GLdouble far  = (GLdouble) 2.f*m_GLData->getBBoxMaxSize();

        mglOrtho(-zoom*m_glRatio,zoom*m_glRatio,-zoom, zoom,-far, far);
    }
}

void GLWidget::setInteractionMode(INTERACTION_MODE mode, bool showmessage)
{
    m_interactionMode = mode;

    switch (mode)
    {
        case TRANSFORM_CAMERA:
            clearPolyline();
            break;
        case SELECTION:
        {
            if(m_GLData->is3D()) //3D
                _g_Cam.setMatrices();
        }
            break;
        default:
            break;
    }

    constructMessagesList(showmessage);
}

void GLWidget::setView(VIEW_ORIENTATION orientation)
{
    makeCurrent();

    GLdouble eye[3] = {0.0, 0.0, 0.0};
    GLdouble top[3] = {0.0, 0.0, 0.0};
    GLdouble s[3]   = {0.0, 0.0, 0.0};
    GLdouble u[3]   = {0.0, 0.0, 0.0};

    switch (orientation)
    {
    case TOP_VIEW:
        eye[2] = -1.0;
        top[1] =  1.0;
        break;
    case BOTTOM_VIEW:
        eye[2] =  1.0;
        top[1] = -1.0;
        break;
    case FRONT_VIEW:
        eye[1] = 1.0;
        top[2] = 1.0;
        break;
    case BACK_VIEW:
        eye[1] = -1.0;
        top[2] =  1.0;
        break;
    case LEFT_VIEW:
        eye[0] = 1.0;
        top[2] = 1.0;
        break;
    case RIGHT_VIEW:
        eye[0] = -1.0;
        top[2] =  1.0;
    }

    crossprod(eye, top, s);
    crossprod(s, eye, u);

//    _g_rotationMatrix[0] = s[0];
//    _g_rotationMatrix[1] = s[1];
//    _g_rotationMatrix[2] = s[2];

//    _g_rotationMatrix[3] = u[0];
//    _g_rotationMatrix[4] = u[1];
//    _g_rotationMatrix[5] = u[2];

//    _g_rotationMatrix[6] = -eye[0];
//    _g_rotationMatrix[7] = -eye[1];
//    _g_rotationMatrix[8] = -eye[2];

    resetTranslationMatrix();
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
    if (hasDataLoaded() && !m_GLData->isImgEmpty())
    {
        float rw = (float)m_GLData->glMaskedImage._m_image->width()  / (float) _g_Cam.vpWidth();
        float rh = (float)m_GLData->glMaskedImage._m_image->height() / (float) _g_Cam.vpHeight();

        if(rw>rh)
            setZoom(1.f/rw); //orientation landscape
        else
            setZoom(1.f/rh); //orientation portrait

        _g_Cam.scaleAndTranslate(-rw, -rh, _params.m_zoom);

        m_GLData->glMaskedImage.setDimensions(2.f*rh,2.f*rw);
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

        m_lastPosImage =  m_bDisplayMode2D ? _g_Cam.WindowToImage(m_lastPosWindow, _params.m_zoom) : m_lastPosWindow;

        if ( event->button() == Qt::LeftButton )
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

            m_GLData->m_polygon.removeClosestPoint(m_lastPosImage);

        else if (event->button() == Qt::MiddleButton)

            m_lastClickZoom = m_lastPosWindow;
    }
}

void GLWidget::mouseReleaseEvent(QMouseEvent *event)
{
    if ( event->button() == Qt::LeftButton && hasDataLoaded() )
    {
        m_GLData->m_polygon.finalMovePoint(m_lastPosImage); //ne pas factoriser

        m_GLData->m_polygon.findClosestPoint(m_lastPosImage);

        update();
    }
}

void GLWidget::mouseMoveEvent(QMouseEvent *event)
{
    if (hasDataLoaded())
    {
        _parentSet->setCurrentWidgetIdx(_idx);
	
	#if QT_VER == 5
	    QPointF pos = m_bDisplayMode2D ?  _g_Cam.WindowToImage(event->localPos(), _params.m_zoom) : event->localPos();
	#else
	    QPointF pos = m_bDisplayMode2D ?  _g_Cam.WindowToImage(event->posF(), _params.m_zoom) : event->posF();
	#endif

        if (m_bDisplayMode2D)  m_lastMoveImage = pos;

        if (m_bDisplayMode2D || (m_interactionMode == SELECTION))

            m_GLData->m_polygon.refreshHelper(pos,(event->modifiers() & Qt::ShiftModifier));

        if (m_interactionMode == TRANSFORM_CAMERA)
        {
            QPoint dPWin = event->pos() - m_lastPosWindow;

            float _glViewport2 = (float) _g_Cam.ViewPort(2);
            float _glViewport3 = (float) _g_Cam.ViewPort(3);


            if ( event->buttons())
            {
                float rX,rY,rZ;
                rX = rY = rZ = 0;
                if ( event->buttons() == Qt::LeftButton ) // rotation autour de X et Y
                {
                    rX = 50.0f * _params.m_speed * dPWin.y() / _g_Cam.vpWidth();
                    rY = 50.0f * _params.m_speed * dPWin.x() / _g_Cam.vpHeight();
                }
                else if ( event->buttons() == Qt::MiddleButton )
                {
                    if (event->modifiers() & Qt::ShiftModifier)         // ZOOM VIEW
                    {
                        if (dPWin.y() > 0) _params.m_zoom *= pow(2.f, ((float)dPWin.y()) *.05f);
                        else if (dPWin.y() < 0) _params.m_zoom /= pow(2.f, -((float)dPWin.y()) *.05f);
                    }
                    else if((_glViewport2!=0.f) || (_glViewport3!=0.f)) // TRANSLATION VIEW
                    {
                        if (m_bDisplayMode2D)
                        {
                            QPointF dp = pos - m_lastPosImage;

                            _g_Cam.m_glPosition[0] += _params.m_speed * dp.x()/_glViewport2;
                            _g_Cam.m_glPosition[1] += _params.m_speed * dp.y()/_glViewport3;
                        }
                        else
                        {
                            _params.m_translationMatrix[0] += _params.m_speed*dPWin.x()*m_GLData->getBBoxMaxSize()/_glViewport2;
                            _params.m_translationMatrix[1] -= _params.m_speed*dPWin.y()*m_GLData->getBBoxMaxSize()/_glViewport3;
                        }
                    }
                }
                else if (event->buttons() == Qt::RightButton)           // rotation autour de Z
                    rZ = 50.0f * _params.m_speed * dPWin.x() / _glViewport2;

                glMatrixMode(GL_MODELVIEW);
                glLoadIdentity();
                glMultMatrixf(_rotationMatrix);

                glRotatef(rX,1.0,0.0,0.0);
                glRotatef(rY,0.0,1.0,0.0);
                glRotatef(rZ,0.0,0.0,1.0);
                glGetFloatv(GL_MODELVIEW_MATRIX, _rotationMatrix);
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

        _g_Cam.setMatrices();

        int idx1 = -1;
        int idx2;

        pos.setY(_g_Cam.ViewPort(3) - pos.y());

        for (int aK=0; aK < m_GLData->Clouds.size();++aK)
        {
            float sqrD;
            float dist = FLT_MAX;
            idx2 = -1;
            QPointF proj;

            Cloud *a_cloud = m_GLData->Clouds[aK];

            for (int bK=0; bK < a_cloud->size();++bK)
            {
                _g_Cam.getProjection(proj, a_cloud->getVertex( bK ).getPosition());

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
            Cloud *a_cloud = m_GLData->Clouds[idx1];
            Pt3dr Pt = a_cloud->getVertex( idx2 ).getPosition();

            m_GLData->setBBoxCenter(Pt);

            m_GLData->pBall->setPosition(Pt);
            m_GLData->pAxis->setPosition(Pt);
            m_GLData->pBbox->setPosition(Pt);

            for (int aK=0; aK < m_GLData->Clouds.size();++aK)
            {
                m_GLData->Clouds[aK]->setPosition(Pt);
            }

            resetTranslationMatrix();

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
                    polyg.add(QPointF(polygon[aK].x(), (float)_g_Cam.ViewPort(3) - polygon[aK].y()));
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
            for (int aK=0; aK < m_GLData->Clouds.size(); ++aK)
            {
                Cloud *a_cloud = m_GLData->Clouds[aK];

                for (uint bK=0; bK < (uint) a_cloud->size();++bK)
                {
                    Vertex &P  = a_cloud->getVertex( bK );
                    Pt3dr  Pt = P.getPosition();

                    switch (mode)
                    {
                    case ADD:
                        _g_Cam.getProjection(P2D, Pt);
                        pointInside = polyg.isPointInsidePoly(P2D);
                        if (m_bFirstAction)
                            P.setVisible(pointInside);
                        else
                            P.setVisible(pointInside||P.isVisible());
                        break;
                    case SUB:
                        if (P.isVisible())
                        {
                            _g_Cam.getProjection(P2D, Pt);
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

            for (int aK=0; aK<4; ++aK)
                info.glViewport[aK] = _g_Cam.ViewPort(aK);
            for (int aK=0; aK<16; ++aK)
            {
                // TODO faire plus simple
                info.mvmatrix[aK]   = _g_Cam.mvMatrix(aK);
                info.projmatrix[aK] = _g_Cam.projMatrix(aK);
            }

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
                for (int bK=0; bK<16;++bK)
                {
                    _g_Cam.getModelViewMatrix()[bK]  = infos.mvmatrix[bK];
                    _g_Cam.getProjectionMatrix()[bK] = infos.projmatrix[bK];
                }
                for (int bK=0; bK<4;++bK)  _g_Cam.getGLViewport()[bK] = infos.glViewport[bK];

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

    displayNewMessage(QString());

    if (m_bDrawMessages)
    {
        if(hasDataLoaded())
        {
            if(m_bDisplayMode2D)
            {
                displayNewMessage(tr("POSITION PIXEL"),LOWER_RIGHT_MESSAGE, Qt::lightGray);
                displayNewMessage(tr("ZOOM"),LOWER_LEFT_MESSAGE, Qt::lightGray);
            }
            else
            {
                if (m_interactionMode == TRANSFORM_CAMERA)
                {
                    displayNewMessage(tr("Move mode"),UPPER_CENTER_MESSAGE);
                    displayNewMessage(tr("Left click: rotate viewpoint / Right click: translate viewpoint"),LOWER_CENTER_MESSAGE);
                }
                else if (m_interactionMode == SELECTION)
                {
                    displayNewMessage(tr("Selection mode"),UPPER_CENTER_MESSAGE);
                    displayNewMessage(tr("Left click: add contour point / Right click: close"),LOWER_CENTER_MESSAGE);
                    displayNewMessage(tr("Space: add / Suppr: delete"),LOWER_CENTER_MESSAGE);
                }

                displayNewMessage(tr("0 Fps"), LOWER_LEFT_MESSAGE, Qt::lightGray);
            }
        }
        else
            displayNewMessage(tr("Drag & drop images or ply files"));
    }

    update();
}

void GLWidget::reset()
{
    resetRotationMatrix();
    resetTranslationMatrix();

    _g_Cam.resetPosition();

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
        resetRotationMatrix();
        resetTranslationMatrix();

        if (hasDataLoaded())
            setZoom(m_GLData->getBBoxMaxSize());

        showBall(hasDataLoaded());
        showAxis(false);
        showBBox(false);
        showCams(false);
    }

    update();
}

void GLWidget::resetRotationMatrix()
{
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glGetFloatv(GL_MODELVIEW_MATRIX, _rotationMatrix);
}

void GLWidget::resetTranslationMatrix()
{
    if (hasDataLoaded())
    {
        Pt3dr center = m_GLData->getBBoxCenter();

        _params.m_translationMatrix[0] = -center.x;
        _params.m_translationMatrix[1] = -center.y;
        _params.m_translationMatrix[2] = -center.z;

    }
}

//------------------------------------------------------------------------

c3DCamera::c3DCamera()
{
    _mvMatrix   = new GLdouble[16];
    _projMatrix = new GLdouble[16];
    _glViewport = new GLint[4];

    m_glPosition[0] = m_glPosition[1] = 0.f;
}

c3DCamera::~c3DCamera()
{
    delete [] _mvMatrix;
    delete [] _projMatrix;
    delete [] _glViewport;
}

void c3DCamera::doProjection(QPointF point, float zoom)
{
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();

    glPushMatrix();
    glMultMatrixd(_projMatrix);

    if(_projMatrix[0] != zoom)
    {
        GLint recal;
        GLdouble wx, wy, wz;

        recal = _glViewport[3] - (GLint) point.y() - 1.f;

        gluUnProject ((GLdouble) point.x(), (GLdouble) recal, 1.f,
                      _mvMatrix, _projMatrix, _glViewport, &wx, &wy, &wz);

        glTranslatef(wx,wy,0);
        glScalef(zoom/_projMatrix[0], zoom/_projMatrix[0], 1.f);
        glTranslatef(-wx,-wy,0);
    }

    glTranslatef(m_glPosition[0],m_glPosition[1],0.f);

    m_glPosition[0] = m_glPosition[1] = 0.f;

    glGetDoublev (GL_PROJECTION_MATRIX, _projMatrix);
}

void c3DCamera::orthoProjection()
{
    mglOrtho(0,_glViewport[2],_glViewport[3],0,-1,1);
}

void c3DCamera::scaleAndTranslate(float x, float y, float zoom)
{
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glPushMatrix();
    glScalef(zoom, zoom, 1.f);
    glTranslatef(x,y,0.f);
    glGetDoublev (GL_PROJECTION_MATRIX, _projMatrix);
    glPopMatrix();

    m_glPosition[0] = m_glPosition[1] = 0.f;
}

void c3DCamera::setMatrices()
{
    glMatrixMode(GL_MODELVIEW);
    glGetDoublev(GL_MODELVIEW_MATRIX, _mvMatrix);

    glMatrixMode(GL_PROJECTION);
    glGetDoublev(GL_PROJECTION_MATRIX, _projMatrix);

    glGetIntegerv(GL_VIEWPORT, _glViewport);
}

void c3DCamera::getProjection(QPointF &P2D, Pt3dr P)
{
    GLdouble xp,yp,zp;
    gluProject(P.x,P.y,P.z,_mvMatrix,_projMatrix,_glViewport,&xp,&yp,&zp);
    P2D = QPointF(xp,yp);
}

QPointF c3DCamera::WindowToImage(QPointF const &pt, float zoom)
{
    QPointF res( pt.x()         - .5f*_glViewport[2]*(1.f+ _projMatrix[12]),
                -pt.y()  -1.f   + .5f*_glViewport[3]*(1.f- _projMatrix[13]));

    res /= zoom;

    return res;
}

QPointF c3DCamera::ImageToWindow(QPointF const &im, float zoom)
{
    return QPointF (im.x()*zoom + .5f*_glViewport[2]*(1.f + _projMatrix[12]),
            - 1.f - im.y()*zoom + .5f*_glViewport[3]*(1.f - _projMatrix[13]));
}

cPolygon c3DCamera::PolygonImageToWindow(cPolygon polygon, float zoom)
{
    cPolygon poly = polygon;
    poly.clearPoints();
    for (int aK = 0;aK < polygon.size(); ++aK)
        poly.add(ImageToWindow(polygon[aK],zoom));

    return poly;
}


