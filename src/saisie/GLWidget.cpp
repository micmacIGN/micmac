#include "GLWidget.h"

//Min and max zoom ratio (relative)
const float GL_MAX_ZOOM = 50.f;
const float GL_MIN_ZOOM = 0.01f;

//invalid GL list index
const GLuint GL_INVALID_LIST_ID = (~0);

using namespace Cloud_;
using namespace std;

GLWidget::GLWidget(QWidget *parent, cData *data) : QGLWidget(parent)
  , m_rw(1.f)
  , m_rh(1.f)
  , m_font(font())
  , m_bDrawMessages(true)
  , m_bObjectCenteredView(true)
  , m_interactionMode(TRANSFORM_CAMERA)
  , m_bFirstAction(true)
  , m_textureImage(GL_INVALID_LIST_ID)
  , m_textureMask(GL_INVALID_LIST_ID)
  , m_params(ViewportParameters())
  , m_Data(data)
  , m_speed(0.4f)
  , m_bDisplayMode2D(false)
  , m_Click(0)
  , _vertexbuffer(QGLBuffer::VertexBuffer)
  , _frameCount(0)
  , _previousTime(0)
  , _currentTime(0)
  , _fps(0.0f)
  , _g_mouseLeftDown(false)
  , _g_mouseMiddleDown(false)
  , _g_mouseRightDown(false)
  , _mask(NULL)
{
    resetRotationMatrix();

    _time.start();

    setFocusPolicy(Qt::StrongFocus);

    //drag & drop handling
    setAcceptDrops(true);

    m_glPosition[0] = m_glPosition[1] = 0.f;

    _mvmatrix   = new GLdouble[16];
    _projmatrix = new GLdouble[16];
    _glViewport = new GLint[4];

    m_font.setPointSize(10);

    _theBall = new cBall();
    _theAxis = new cAxis();
    _theBBox = new cBBox();

    installEventFilter(this);
    setMouseTracking(true);
}

GLWidget::~GLWidget()
{
    if (m_textureImage != GL_INVALID_LIST_ID)
    {
        glDeleteLists(m_textureImage,1);
        m_textureImage = GL_INVALID_LIST_ID;
    }

    if (m_textureMask != GL_INVALID_LIST_ID)
    {
        glDeleteLists(m_textureMask,1);
        m_textureMask = GL_INVALID_LIST_ID;
    }

    if(_mask)
        delete _mask;

    delete [] _mvmatrix;
    delete [] _projmatrix;
    delete [] _glViewport;

    delete _theBall;
    delete _theAxis;
    delete _theBBox;
}

bool GLWidget::eventFilter(QObject* object,QEvent* event)
{
    QMouseEvent *mouseEvent = static_cast<QMouseEvent*>(event);

    if(event->type() == QEvent::MouseMove)
    {
        QPointF pos = mouseEvent->localPos();
        if (m_bDisplayMode2D)
        {
            m_lastClickZoom = pos;
            pos = WindowToImage(mouseEvent->localPos());
            m_lastMoveImg = pos;
        }

        if (m_bDisplayMode2D || (m_interactionMode == SELECTION))
        {
            int sz = m_polygon.size();

            if(!m_polygon.isClosed())
            {
                if (sz == 1)     // add current mouse position to polygon (dynamic display)
                    m_polygon.add(pos);
                else if (sz > 1) //replace last point by the current one
                    m_polygon[sz-1] = pos;
            }
            else
            {
                if(sz)           // move vertex or insert vertex (dynamic display)
                {
                    QKeyEvent *keyEvent = static_cast<QKeyEvent*>(event);
                    if (keyEvent->modifiers().testFlag(Qt::ShiftModifier))
                    {
                        m_polygon.fillDihedron(pos, m_dihedron);
                    }
                    else
                    {
                        if (m_Click == 1)
                            m_polygon.fillDihedron2(pos, m_dihedron);
                        else
                            m_polygon.findClosestPoint(pos);
                    }
                }
            }
        }

        if (m_bDisplayMode2D || (m_interactionMode == TRANSFORM_CAMERA))
        {
            QPointF dp = pos - m_lastPos;

            if ( _g_mouseLeftDown ) // rotation autour de X et Y
            {
                float d_angleX = m_speed * dp.y() / (float) _glViewport[3];
                float d_angleY = m_speed * dp.x() / (float) _glViewport[2];

                m_params.angleX += d_angleX;
                m_params.angleY += d_angleY;

                setRotateOx_m33( d_angleX, _g_rotationOx );
                setRotateOy_m33( d_angleY, _g_rotationOy );

                mult_m33( _g_rotationOx, _g_rotationMatrix, _g_tmpoMatrix );
                mult_m33( _g_rotationOy, _g_tmpoMatrix, _g_rotationMatrix );
            }
            else if ( _g_mouseMiddleDown )
            {
                if (mouseEvent->modifiers() & Qt::ShiftModifier) // zoom
                {
                    m_lastClickZoom =  m_lastClickWin;

                    float dy = (mouseEvent->pos().y() - m_lastClickWin.y())*0.001f;

                    if (dy > 0.f) m_params.zoom *= pow(2.f, dy);
                    else  m_params.zoom /= pow(2.f, -dy);

                    if (m_params.zoom < GL_MIN_ZOOM) m_params.zoom = GL_MIN_ZOOM;
                    else if (m_params.zoom > GL_MAX_ZOOM) m_params.zoom = GL_MAX_ZOOM;
                }
                else if((_glViewport[2]!=0) || (_glViewport[3]!=0)) // translation
                {
                    if (m_Data->NbImages())
                    {
                        m_glPosition[0] += 2.f * dp.x()/_glViewport[2];
                        m_glPosition[1] += 2.f * dp.y()/_glViewport[3];
                    }
                    else
                    {
                        m_bObjectCenteredView = false;
                        m_params.m_translationMatrix[0] += m_speed * dp.x()*m_Data->m_diam/_glViewport[2];
                        m_params.m_translationMatrix[1] -= m_speed * dp.y()*m_Data->m_diam/_glViewport[3];
                    }
                }
            }
            else if ( _g_mouseRightDown ) // rotation autour de Z
            {
                float d_angleZ =  m_speed * dp.x() / (float) _glViewport[2];

                m_params.angleZ += d_angleZ;

                setRotateOz_m33( d_angleZ, _g_rotationOz );

                mult_m33( _g_rotationOz, _g_rotationMatrix, _g_tmpoMatrix );

                for (int i = 0; i < 9; ++i) _g_rotationMatrix[i] = _g_tmpoMatrix[i];
            }
        }

        update();
        return true;
    }  
    else if (event->type() == QEvent::MouseButtonPress)
    {
       if (m_bDisplayMode2D)
           m_lastPos = WindowToImage(mouseEvent->pos());
       else
           m_lastPos = mouseEvent->pos();

       m_lastClickWin = mouseEvent->pos();

       if ( mouseEvent->button() == Qt::LeftButton )
       {
           _g_mouseLeftDown = true;

           if (m_bDisplayMode2D || (m_interactionMode == SELECTION))
           {
               if (hasDataLoaded())
               {
                   if(!m_polygon.isClosed())        // add point to polygon
                   {
                       if (m_polygon.size() >= 1)
                           m_polygon[m_polygon.size()-1] = m_lastPos;

                       m_polygon.add(m_lastPos);
                   }
                   else // modify polygon (insert or move vertex)
                   {
                       if (mouseEvent->modifiers().testFlag(Qt::ShiftModifier))
                       {
                           if ((m_polygon.size() >=2) && m_dihedron.size() && m_polygon.isClosed())
                           {
                               int idx = -1;

                               for (int i=0;i<m_polygon.size();++i)
                               {
                                   if (m_polygon[i] == m_dihedron[0]) idx = i;
                               }

                               if (idx >=0) m_polygon.insert(idx+1, m_dihedron[1]);
                           }

                           m_dihedron.clear();
                       }
                       else if (m_polygon.idx() != -1)
                           m_Click++;
                   }
               }
           }
       }
       else if (mouseEvent->button() == Qt::RightButton)
       {
           _g_mouseRightDown = true; // for rotation around Z (in 3D)

           int idx = m_polygon.idx();
           if ((idx >=0)&&(idx<m_polygon.size())&&m_polygon.isClosed())
           {
               m_polygon.remove(idx);   // remove closest point

               m_polygon.findClosestPoint(m_lastPos);

               if (m_polygon.size() < 2) m_polygon.setClosed(false);
           }
           else // close polygon
           {
               m_polygon.close();
               m_Click = 0;
           }
       }
       else if (mouseEvent->button() == Qt::MiddleButton)
       {
           if (m_bDisplayMode2D || (m_interactionMode == TRANSFORM_CAMERA))
               _g_mouseMiddleDown = true;
       }
       update();
       return true;
    }
    else if (event->type() == QEvent::MouseButtonRelease)
    {
        if ( mouseEvent->button() == Qt::LeftButton )
        {
            _g_mouseLeftDown = false;

            int idx = m_polygon.idx();
            if ((m_Click >=1) && (idx>=0) && m_dihedron.size())
            {
                m_polygon[idx] = m_dihedron[1];

                m_dihedron.clear();
                m_Click = 0;
            }

            if ((m_Click >=1) && m_polygon.isClosed())
            {
                m_polygon.findClosestPoint(m_lastPos);
            }
        }
        if ( mouseEvent->button() == Qt::RightButton  )
        {
            _g_mouseRightDown = false;
        }
        if ( mouseEvent->button() == Qt::MiddleButton  )
        {
            _g_mouseMiddleDown = false;
        }

        update();

        return true;
    }
    else
    {
        return QObject::eventFilter(object,event);
    }
}

void GLWidget::resizeGL(int width, int height)
{
    if (width==0 || height==0) return;

    m_glRatio  = (float) width/height;

    glViewport( 0, 0, width, height );
    glGetIntegerv (GL_VIEWPORT, _glViewport);

    if (m_Data->NbImages())
        zoomFit();
}

//-------------------------------------------------------------------------
// Computes the frames rate
//-------------------------------------------------------------------------
void GLWidget::computeFPS()
{
    //  Increase frame count
    _frameCount++;

    _currentTime = _time.elapsed();

    //  Compute elapsed time
    int deltaTime = _currentTime - _previousTime;

    if(deltaTime > 1000)
    {
        //  compute the number of frames per second
        _fps = _frameCount / (deltaTime / 1000.0f);

        //  Set time
        _previousTime = _currentTime;

        //  Reset frame count
        _frameCount = 0;

        if (_fps > 1e-3)
        {
            m_messageFPS = "fps: " + QString::number(_fps);
        }
    }
}

void GLWidget::drawQuad(GLfloat originX, GLfloat originY, GLfloat glh, GLfloat glw)
{
    glBegin(GL_QUADS);
    {
        glTexCoord2f(0.0f, 0.0f);
        glVertex2f(originX, originY);
        glTexCoord2f(1.0f, 0.0f);
        glVertex2f(originX+glw, originY);
        glTexCoord2f(1.0f, 1.0f);
        glVertex2f(originX+glw, originY+glh);
        glTexCoord2f(0.0f, 1.0f);
        glVertex2f(originX, originY+glh);
    }
    glEnd();
}

void GLWidget::drawQuad(GLfloat originX, GLfloat originY, GLfloat glh, GLfloat glw, QColor color)
{
    glColor4f(color.redF(),color.greenF(),color.blueF(),color.alphaF());
    drawQuad(originX,originY,glh,glw);
}

void GLWidget::drawQuad(GLfloat originX, GLfloat originY, GLfloat glh, GLfloat glw, GLuint idTexture)
{
    glEnable(GL_TEXTURE_2D);
    glBindTexture( GL_TEXTURE_2D, idTexture );
    drawQuad(originX,originY,glh,glw);
    glBindTexture( GL_TEXTURE_2D, 0);
    glDisable(GL_TEXTURE_2D);
}

void GLWidget::enableOptionLine()
{
    glDisable(GL_DEPTH_TEST);
    glEnable (GL_LINE_SMOOTH);
    glEnable (GL_BLEND);
    glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glHint (GL_LINE_SMOOTH_HINT, GL_DONT_CARE);
}

void GLWidget::disableOptionLine()
{
    glDisable(GL_BLEND);
    glDisable (GL_LINE_SMOOTH);
    glEnable(GL_DEPTH_TEST);
}

void GLWidget::paintGL()
{
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glEnable(GL_BLEND);
    glBlendFunc(GL_ONE,GL_ZERO);

    //gradient color background
    drawGradientBackground();
    //we clear background
    glClear(GL_DEPTH_BUFFER_BIT);

    glDisable(GL_BLEND);

    if (m_Data->NbImages())
    {
        glEnable(GL_BLEND);
        glBlendFunc(GL_ONE,GL_ZERO);

        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();

        glDisable(GL_ALPHA_TEST);
        glDisable(GL_DEPTH_TEST);

        GLfloat glw = 2.f*m_rw;
        GLfloat glh = 2.f*m_rh;

        glPushMatrix();
        glMultMatrixd(_projmatrix);

        if(_projmatrix[0] != m_params.zoom)
        {
            GLint recal;
            GLdouble wx, wy, wz;

            recal = _glViewport[3] - (GLint) m_lastClickZoom.y() - 1.f;

            gluUnProject ((GLdouble) m_lastClickZoom.x(), (GLdouble) recal, 1.f,
                          _mvmatrix, _projmatrix, _glViewport, &wx, &wy, &wz);

            glTranslatef(wx,wy,0);
            glScalef(m_params.zoom/_projmatrix[0], m_params.zoom/_projmatrix[0], 1.f);
            glTranslatef(-wx,-wy,0);
        }

        glTranslatef(m_glPosition[0],m_glPosition[1],0.f);

        m_glPosition[0] = m_glPosition[1] = 0.f;

        glGetDoublev (GL_PROJECTION_MATRIX, _projmatrix);
        drawQuad(0, 0, glh, glw,QColor(255,255,255));

        if(_mask != NULL && !_g_mouseMiddleDown)
        {
            drawQuad(0,0,glh,glw,m_textureMask);
            glBlendFunc(GL_ONE,GL_ONE);

            drawQuad(0, 0, glh, glw,QColor(128,128,128));
            glBlendFunc(GL_DST_COLOR,GL_SRC_COLOR);
        }

        drawQuad(0,0,glh,glw,m_textureImage);

        glPopMatrix();

        glDisable(GL_BLEND);
        glEnable(GL_DEPTH_TEST);
        glEnable(GL_ALPHA_TEST);
        glMatrixMode(GL_MODELVIEW);

        //Affichage du zoom et des coordonnées image
        if (m_bDrawMessages)
        {
            glColor3f(1.f,1.f,1.f);

            renderText(10, _glViewport[3] - m_font.pointSize(), QString::number(m_params.zoom*100,'f',1) + "%", m_font);

            if  ((m_lastMoveImg.x()>=0)&&(m_lastMoveImg.y()>=0)&&(m_lastMoveImg.x()<_glImg.width())&&(m_lastMoveImg.y()<_glImg.height()))
                renderText(_glViewport[2] - 120, _glViewport[3] - m_font.pointSize(), QString::number(m_lastMoveImg.x(),'f',1) + ", " + QString::number(_glImg.height()-m_lastMoveImg.y(),'f',1) + " px", m_font);
        }
    }
    else
    {
        zoom();

        static GLfloat trans44[16], rot44[16], tmp[16];
        m33_to_m44( _g_rotationMatrix, rot44 );
        setTranslate_m3(  m_params.m_translationMatrix, trans44 );

        //mult( trans44, rot44, tmp );
        mult( rot44, trans44, tmp );
        transpose( tmp, _g_glMatrix );
        glLoadMatrixf( _g_glMatrix );

        if (m_Data->NbClouds())
        {
            glEnable(GL_DEPTH_TEST);

            glEnableClientState(GL_VERTEX_ARRAY);
            glEnableClientState(GL_COLOR_ARRAY);

            _vertexbuffer.bind();
            glVertexPointer(3, GL_FLOAT, 0, NULL);
            _vertexbuffer.release();

            _vertexColor.bind();
            glColorPointer(3, GL_FLOAT, 0, NULL);
            _vertexColor.release();

            glDrawArrays( GL_POINTS, 0, m_Data->getCloud(0)->size()*3 );

            glDisableClientState(GL_VERTEX_ARRAY);
            glDisableClientState(GL_COLOR_ARRAY);

            glDisable(GL_DEPTH_TEST);
        }

        enableOptionLine();

        if (_theBall->isVisible())
            _theBall->draw();
        else if (_theAxis->isVisible())
            _theAxis->draw();

        _theBBox->draw();

        //cameras
        for (int i=0; i<_pCams.size();i++) _pCams[i]->draw();

        disableOptionLine();

        if (m_Data->NbClouds()&& m_bDrawMessages)
        {
            computeFPS();

            glColor4f(0.8f,0.9f,1.0f,0.9f);

            renderText(10, _glViewport[3]- m_font.pointSize(), m_messageFPS, m_font);
        }
    }

    if (m_bDisplayMode2D || (m_interactionMode == SELECTION))
    {
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        glOrtho(0,_glViewport[2],_glViewport[3],0,-1,1);
        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();

        enableOptionLine();

        drawPolygon();

        disableOptionLine();
    }

    //current messages (if valid)
    if (!m_messagesToDisplay.empty())
    {
        glColor3f(1.f,1.f,1.f);

        int lc_currentHeight = _glViewport[3] - m_font.pointSize()*m_messagesToDisplay.size(); //lower center
        int uc_currentHeight = 10;            //upper center

        std::list<MessageToDisplay>::iterator it = m_messagesToDisplay.begin();
        while (it != m_messagesToDisplay.end())
        {
            switch(it->position)
            {
            case LOWER_LEFT_MESSAGE:
            {
                renderText(10, lc_currentHeight, it->message,m_font);
                int messageHeight = QFontMetrics(m_font).height();
                lc_currentHeight -= (messageHeight*5)/4; //add a 25% margin
            }
                break;
            case LOWER_CENTER_MESSAGE:
            {
                QRect rect = QFontMetrics(m_font).boundingRect(it->message);
                renderText((_glViewport[2]-rect.width())/2, lc_currentHeight, it->message,m_font);
                int messageHeight = QFontMetrics(m_font).height();
                lc_currentHeight += (messageHeight*5)/4; //add a 25% margin
            }
                break;
            case UPPER_CENTER_MESSAGE:
            {
                QRect rect = QFontMetrics(m_font).boundingRect(it->message);
                renderText((_glViewport[2]-rect.width())/2, uc_currentHeight+rect.height(), it->message,m_font);
                uc_currentHeight += (rect.height()*5)/4; //add a 25% margin
            }
                break;
            case SCREEN_CENTER_MESSAGE:
            {
                m_font.setPointSize(12);
                QRect rect = QFontMetrics(m_font).boundingRect(it->message);
                renderText((_glViewport[2]-rect.width())/2, (_glViewport[3]-rect.height())/2, it->message,m_font);
                m_font.setPointSize(10);
            }
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
        case Qt::Key_Plus:
            if (m_bDisplayMode2D)
                setZoom(m_params.zoom*1.5f);
            else
                ptSizeUp(true);
            break;
        case Qt::Key_Minus:
            if (m_bDisplayMode2D)
                setZoom(m_params.zoom/1.5f);
            else
                ptSizeUp(false);
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
    if  (event->key() == Qt::Key_Shift)
    {
        m_dihedron.clear();
        m_Click = 0;
    }
}

void GLWidget::setBufferGl(bool onlyColor)
{
    if(_vertexbuffer.isCreated() && !onlyColor)
        _vertexbuffer.destroy();
    if(_vertexColor.isCreated())
        _vertexColor.destroy();

    int sizeClouds = m_Data->getSizeClouds();

    if (sizeClouds == 0) return;

    GLfloat* vertices = NULL, *colors = NULL;

    if(!onlyColor)
        vertices = new GLfloat[sizeClouds*3];

    colors     = new GLfloat[sizeClouds*3];
    int pitchV = 0;

    for (int aK=0; aK < m_Data->NbClouds();++aK){

        Cloud* pcloud = m_Data->getCloud(aK);
        uint sizeCloud = pcloud->size();

        for(int bK=0; bK< (int)sizeCloud; bK++)
        {
            Vertex vert = pcloud->getVertex(bK);
            QColor colo = vert.getColor();
            if(!onlyColor)
            {
                vertices[pitchV+bK*3 + 0 ] = vert.x();
                vertices[pitchV+bK*3 + 1 ] = vert.y();
                vertices[pitchV+bK*3 + 2 ] = vert.z();
            }
            if(vert.isVisible())
            {
                colors[pitchV+bK*3 + 0 ]   = colo.redF();
                colors[pitchV+bK*3 + 1 ]   = colo.greenF();
                colors[pitchV+bK*3 + 2 ]   = colo.blueF();
            }
            else
            {
                colors[pitchV+bK*3 + 0 ]   = colo.redF()   *0.7;
                colors[pitchV+bK*3 + 1 ]   = colo.greenF() *0.6;
                colors[pitchV+bK*3 + 2 ]   = colo.blueF()  *0.8;
            }
        }

        pitchV += sizeCloud;
    }

    if(!onlyColor)
    {
        _vertexbuffer.create();
        _vertexbuffer.setUsagePattern(QGLBuffer::StaticDraw);
        _vertexbuffer.bind();
        _vertexbuffer.allocate(vertices, sizeClouds* 3 * sizeof(GLfloat));
        _vertexbuffer.release();
    }

    _vertexColor.create();
    _vertexColor.setUsagePattern(QGLBuffer::StaticDraw);
    _vertexColor.bind();
    _vertexColor.allocate(colors, sizeClouds* 3 * sizeof(GLfloat));
    _vertexColor.release();

    if(!onlyColor)
        delete [] vertices;
    delete [] colors;
}

void GLWidget::setData(cData *data)
{
    clearPolyline();
    m_Data = data;

    if ((m_Data->NbClouds())||(m_Data->NbCameras()))
    {
        m_bDisplayMode2D = false;

        float scale = m_Data->m_diam / 1.5f;

        if (m_Data->NbClouds())  setBufferGl();

        if (m_Data->NbCameras())
        {
            for (int i=0; i<m_Data->NbCameras();i++)
            {
                cCam *pCam = new cCam(m_Data->getCamera(i));

                pCam->setScale(scale);
                pCam->setVisible(true);

                _pCams.push_back(pCam);
            }
        }

        setZoom(m_Data->getScale());

        _theBall->setPosition(m_Data->getCenter());
        _theBall->setScale(scale);
        _theBall->setVisible(true);

        _theAxis->setPosition(m_Data->getCenter());
        _theAxis->setScale(scale);

        _theBBox->setPosition(m_Data->getCenter());
        _theBBox->set(m_Data->m_minX,m_Data->m_minY,m_Data->m_minZ,m_Data->m_maxX,m_Data->m_maxY,m_Data->m_maxZ);

        resetTranslationMatrix();
    }

    if (m_Data->NbImages())
    {
        m_bDisplayMode2D = true;

        _glImg = QGLWidget::convertToGLFormat( *m_Data->getCurImage() );

        applyGamma(m_params.getGamma());

        zoomFit();

        //position de l'image dans la vue gl             
        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();

        glGetDoublev (GL_MODELVIEW_MATRIX, _mvmatrix);
        glGenTextures(1, &m_textureImage );

        ImageToTexture(m_textureImage, &_glImg);

        if(_mask)
            delete _mask;
        else
            glGenTextures(1, &m_textureMask );

        _mask = new QImage(_glImg.size(),_glImg.format());

        if (m_Data->NbMasks())
        {
            *_mask = QGLWidget::convertToGLFormat( *m_Data->getCurMask() );
            m_bFirstAction = false;
        }
        else
        {
            QGLWidget::convertToGLFormat(*_mask);
            _mask->fill(Qt::white);
            m_bFirstAction = true;
        }

        ImageToTexture(m_textureMask, _mask);
    }

    glGetIntegerv (GL_VIEWPORT, _glViewport);

    update();
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
                                 MessagePosition pos)
{
    if (message.isEmpty())
    {
        m_messagesToDisplay.clear();

        return;
    }

    MessageToDisplay mess;
    mess.message = message;
    mess.position = pos;
    m_messagesToDisplay.push_back(mess);
}

void GLWidget::drawGradientBackground()
{
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();

    int w = (_glViewport[2]>>1)+1;
    int h = (_glViewport[3]>>1)+1;
    glOrtho(-w,w,-h,h,-2.f, 2.f);

    const uchar BkgColor[3] = {(uchar) colorBG0.red(),(uchar) colorBG0.green(), (uchar) colorBG0.blue()};
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    //Gradient "texture" drawing
    glBegin(GL_QUADS);
    //user-defined background color for gradient start
    glColor3ubv(BkgColor);
    glVertex2f(-w,h);
    glVertex2f(w,h);
    //and the inverse of points color for gradient end
    glColor3ub(colorBG1.red(),colorBG1.green(),colorBG1.blue());
    glVertex2f(w,-h);
    glVertex2f(-w,-h);
    glEnd();
}

void GLWidget::drawPolygon()
{
    if (m_Data->NbImages())
    {
        cPolygon poly = m_polygon;
        poly.clear();
        for (int aK = 0;aK < m_polygon.size(); ++aK)
        {
            poly.add(ImageToWindow(m_polygon[aK]));
        }

        poly.draw();

        poly.clear();
        for (int aK = 0;aK < m_dihedron.size(); ++aK)
        {
            poly.add(ImageToWindow(m_dihedron[aK]));
        }

        poly.drawDihedron();
    }
    else
    {
        m_polygon.draw();
        m_dihedron.drawDihedron();
    }
}

// zoom in 3D mode
void GLWidget::zoom()
{
    GLdouble zoom = m_params.zoom;

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();

    glOrtho(-zoom*m_glRatio,zoom*m_glRatio,-zoom, zoom,-2.f*m_Data->m_diam, 2.f*m_Data->m_diam);

    glMatrixMode(GL_MODELVIEW);
}

void GLWidget::setInteractionMode(INTERACTION_MODE mode)
{
    m_interactionMode = mode;

    if(!m_bDisplayMode2D)
    {
        switch (mode)
        {
            case TRANSFORM_CAMERA:
            {
                if (hasDataLoaded() && showMessages())
                {
                    clearPolyline();
                    displayMoveMessages();
                }
                showBall(true);
                showAxis(false);

                emit(interactionMode(false));
            }
                break;
            case SELECTION:
            {
                if(!m_Data->NbImages())
                    setProjectionMatrix();

                if (hasDataLoaded() && showMessages())
                {
                   displaySelectionMessages();
                }
                showBall(false);
                showCams(false);
                showAxis(false);
                showBBox(false);

                emit(interactionMode(true));
            }
                break;
            default:
                break;
        }
    }
}

void GLWidget::setView(VIEW_ORIENTATION orientation)
{
    makeCurrent();

    GLdouble eye[3] = {0.0, 0.0, 0.0};
    GLdouble top[3] = {0.0, 0.0, 0.0};
    GLdouble s[3]   = {0.0, 0.0, 0.0};
    GLdouble u[3]   = {0.0, 0.0, 0.0};

    switch (orientation)
    {glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
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

    _g_rotationMatrix[0] = s[0];
    _g_rotationMatrix[1] = s[1];
    _g_rotationMatrix[2] = s[2];

    _g_rotationMatrix[3] = u[0];
    _g_rotationMatrix[4] = u[1];
    _g_rotationMatrix[5] = u[2];

    _g_rotationMatrix[6] = -eye[0];
    _g_rotationMatrix[7] = -eye[1];
    _g_rotationMatrix[8] = -eye[2];

//    m_params.m_translationMatrix[0] = m_Data->m_cX;
//    m_params.m_translationMatrix[1] = m_Data->m_cY;
//    m_params.m_translationMatrix[2] = m_Data->m_cZ;

    resetTranslationMatrix();
}

void GLWidget::onWheelEvent(float wheelDelta_deg)
{
    //convert degrees in zoom 'power'
    float zoomFactor = pow(1.1f,wheelDelta_deg *.05f);

    setZoom(m_params.zoom*zoomFactor);
}

void GLWidget::setZoom(float value)
{
    if (value < GL_MIN_ZOOM)
        value = GL_MIN_ZOOM;
    else if (value > GL_MAX_ZOOM)
        value = GL_MAX_ZOOM;

    m_params.zoom = value;   

    update();
}

void GLWidget::zoomFit()
{
    //width and height ratio between viewport and image
    m_rw = (float)_glImg.width()/ _glViewport[2];
    m_rh = (float)_glImg.height()/_glViewport[3];

    if(m_rw>m_rh)
        setZoom(1.f/m_rw); //orientation landscape
    else
        setZoom(1.f/m_rh); //orientation portrait

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glPushMatrix();
    glScalef(m_params.zoom, m_params.zoom, 1.0);
    glTranslatef(-m_rw,-m_rh,0);
    glGetDoublev (GL_PROJECTION_MATRIX, _projmatrix);
    glPopMatrix();

    m_glPosition[0] = 0;
    m_glPosition[1] = 0;
}

void GLWidget::zoomFactor(int percent)
{
    if (m_bDisplayMode2D)
        setZoom((float) percent / 100.f);
    else
        setZoom(m_Data->getScale() / (float) percent * 100.f);
}

void GLWidget::wheelEvent(QWheelEvent* event)
{
    if ((m_interactionMode == SELECTION)&&(!m_bDisplayMode2D))
    {
        event->ignore();
        return;
    }

    //see QWheelEvent documentation ("distance that the wheel is rotated, in eighths of a degree")
    float wheelDelta_deg = event->angleDelta().y() / 8.f;

    m_lastClickZoom = event->posF();

    onWheelEvent(wheelDelta_deg);
}

void GLWidget::ImageToTexture(GLuint idTexture, QImage *image)
{
    glBindTexture( GL_TEXTURE_2D, idTexture );
    glTexImage2D( GL_TEXTURE_2D, 0, 4, image->width(), image->height(), 0, GL_RGBA, GL_UNSIGNED_BYTE, image->bits());
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST );
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST );
    glBindTexture( GL_TEXTURE_2D, 0);
    glDisable(GL_TEXTURE_2D);
}

void GLWidget::mouseDoubleClickEvent(QMouseEvent *event)
{
    if (m_Data->NbClouds())
    {
        QPointF pos = event->localPos();

        setProjectionMatrix();

        int idx1 = -1;
        int idx2;

        pos.setY(_glViewport[3] - pos.y());

        for (int aK=0; aK < m_Data->NbClouds();++aK)
        {
            float sqrD;
            float dist = FLT_MAX;
            idx2 = -1;
            QPointF proj;

            Cloud *a_cloud = m_Data->getCloud(aK);

            for (int bK=0; bK < a_cloud->size();++bK)
            {
                getProjection(proj, a_cloud->getVertex( bK ));

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
            Cloud *a_cloud = m_Data->getCloud(idx1);
            Vertex &P = a_cloud->getVertex( idx2 );

            //m_Data->m_cX = P.x();
            //m_Data->m_cY = P.y();
            //m_Data->m_cZ = P.z();

            m_Data->setCenter(P.getCoord());

            update();
        }
    }
}

bool isPointInsidePoly(const QPointF& P, const QVector< QPointF> &poly)
{
    int vertices=poly.size();
    if (vertices<3)
        return false;

    bool inside = false;

    QPointF A = poly[0];
    QPointF B;

    for (int i=1;i<=vertices;++i)
    {
        B = poly[i%vertices];

        //Point Inclusion in Polygon Test (inspired from W. Randolph Franklin - WRF)
        if (((B.y() <= P.y()) && (P.y()<A.y())) ||
                ((A.y() <= P.y()) && (P.y()<B.y())))
        {
            float ABy = A.y()-B.y();
            float t = (P.x()-B.x())*ABy-(A.x()-B.x())*(P.y()-B.y());
            if (ABy<0)
                t=-t;

            if (t<0)
                inside = !inside;
        }

        A=B;
    }

    return inside;
}

void GLWidget::setProjectionMatrix()
{
    glMatrixMode(GL_MODELVIEW);
    glGetDoublev(GL_MODELVIEW_MATRIX, _mvmatrix);

    glMatrixMode(GL_PROJECTION);
    glGetDoublev(GL_PROJECTION_MATRIX, _projmatrix);

    glGetIntegerv(GL_VIEWPORT, _glViewport);
}

void GLWidget::getProjection(QPointF &P2D, Vertex P)
{
    GLdouble xp,yp,zp;
    gluProject(P.x(),P.y(),P.z(),_mvmatrix,_projmatrix,_glViewport,&xp,&yp,&zp);
    P2D = QPointF(xp,yp);
}

QPointF GLWidget::WindowToImage(QPointF const &pt)
{
    QPointF res;

    res.setX(( pt.x()         - _glViewport[2]*.5f ) - _projmatrix[12]*_glViewport[2]*.5f);
    res.setY((-pt.y()  -1.f   + _glViewport[3]*.5f ) - _projmatrix[13]*_glViewport[3]*.5f);

    res /= m_params.zoom;

    return res;
}

QPointF GLWidget::ImageToWindow(QPointF const &im)
{
    QPointF res;

    res.setX(im.x()*m_params.zoom + _glViewport[2]*.5f + _projmatrix[12]*_glViewport[2]*.5f);
    res.setY(- 1.f - im.y()*m_params.zoom + _glViewport[3]*.5f - _projmatrix[13]*_glViewport[3]*.5f);

    return res;
}

void GLWidget::Select(int mode)
{
    QPointF P2D;
    bool pointInside;
    QVector< QPointF> polyg;

    if(mode == ADD || mode == SUB)
    {
        if ((m_polygon.size() < 3) || (!m_polygon.isClosed()))
            return;

        if (!m_bDisplayMode2D)
        {
            for (int aK=0; aK < m_polygon.size(); ++aK)
            {
               polyg.push_back(QPointF(m_polygon[aK].x(), _glViewport[3] - m_polygon[aK].y()));
            }
        }
        else
            polyg = m_polygon.getVector();
    }

    if (m_bDisplayMode2D)
    {
         QPainter    p;
         QBrush SBrush(Qt::white);
         QBrush NSBrush(Qt::black);

         p.begin(_mask);
         p.setCompositionMode(QPainter::CompositionMode_Source);
         p.setPen(Qt::NoPen);

         if(mode == ADD)
         {
             if (m_bFirstAction)
             {
                 p.fillRect(_mask->rect(), Qt::black);
             }
             p.setBrush(SBrush);
             p.drawPolygon(polyg.data(),polyg.size());
         }
         else if(mode == SUB)
         {
             p.setBrush(NSBrush);
             p.drawPolygon(polyg.data(),polyg.size());
         }
         else if(mode == ALL)
         {
             p.fillRect(_mask->rect(), Qt::white);
         }
         else if(mode == NONE)
         {
             p.fillRect(_mask->rect(), Qt::black);
         }
         p.end();

         if(mode == INVERT)         
            _mask->invertPixels(QImage::InvertRgb);

         ImageToTexture(m_textureMask, _mask);
    }
    else
    {
        for (uint aK=0; aK < (uint) m_Data->NbClouds(); ++aK)
        {
            Cloud *a_cloud = m_Data->getCloud(aK);

            for (uint bK=0; bK < (uint) a_cloud->size();++bK)
            {
                Vertex P = a_cloud->getVertex( bK );

                switch (mode)
                {
                case ADD:
                    getProjection(P2D, P);
                    pointInside = isPointInsidePoly(P2D,polyg);
                    if (m_bFirstAction)
                        emit selectedPoint(aK,bK,pointInside);
                    else
                        emit selectedPoint(aK,bK,pointInside||P.isVisible());
                    break;
                case SUB:
                    if (P.isVisible())
                    {
                        getProjection(P2D, P);
                        pointInside = isPointInsidePoly(P2D,polyg);
                        emit selectedPoint(aK,bK,!pointInside);
                    }
                    break;
                case INVERT:
                    //if (m_previousAction == NONE)  m_bFirstAction = true;
                    emit selectedPoint(aK,bK,!P.isVisible());
                    break;
                case ALL:
                    m_bFirstAction = true;
                    emit selectedPoint(aK,bK, true);
                    break;
                case NONE:
                    emit selectedPoint(aK,bK,false);
                    break;
                }
            }

            setBufferGl(true);
        }
    }

    if (((mode == ADD)||(mode == SUB)) && (m_bFirstAction)) m_bFirstAction = false;

    selectInfos info;
    info.params = m_params;
    info.poly   = m_polygon.getVector();
    info.selection_mode   = mode;

    m_infos.push_back(info);

    clearPolyline();
}

void GLWidget::clearPolyline()
{
    m_polygon.clear();
    m_polygon.setClosed(false);
    m_dihedron.clear();
    m_Click = 0;

    update();
}

void GLWidget::ptSizeUp(bool up)
{
    if (up)
        m_params.PointSize++;
    else
        m_params.PointSize--;

    if (m_params.PointSize == 0)
        m_params.PointSize = 1;

    glPointSize(m_params.PointSize);

    update();
}

void GLWidget::showAxis(bool show)
{
    _theAxis->setVisible(show);

    update();
}

void GLWidget::showBall(bool show)
{
    _theBall->setVisible(show);

    update();
}

void GLWidget::showCams(bool show)
{
    for (int i=0; i < _pCams.size();i++)
        _pCams[i]->setVisible(show);

    update();
}

void GLWidget::showBBox(bool show)
{
    _theBBox->setVisible(show);

    update();
}

void GLWidget::showMessages(bool show)
{
    m_bDrawMessages = show;

    if ((m_bDrawMessages)&&(!m_bDisplayMode2D))
    {
        if (m_interactionMode == TRANSFORM_CAMERA)
            displayMoveMessages();
        else if (m_interactionMode == SELECTION)
            displaySelectionMessages();
    }
    else
        displayNewMessage(QString());

    update();
}

void GLWidget::displaySelectionMessages()
{
    displayNewMessage(QString());
    displayNewMessage(tr("Selection mode"),UPPER_CENTER_MESSAGE);
    displayNewMessage(tr("Left click: add contour point / Right click: close"),LOWER_CENTER_MESSAGE);
    displayNewMessage(tr("Space: add / Suppr: delete"),LOWER_CENTER_MESSAGE);
}

void GLWidget::displayMoveMessages()
{
    displayNewMessage(QString());
    displayNewMessage(tr("Move mode"),UPPER_CENTER_MESSAGE);
    displayNewMessage(tr("Left click: rotate viewpoint / Right click: translate viewpoint"),LOWER_CENTER_MESSAGE);
}

void GLWidget::reset()
{
    resetRotationMatrix();
    resetTranslationMatrix();

    m_glPosition[0] = m_glPosition[1] = 0.f;

    clearPolyline();

    m_params.reset();
    m_Data->clearClouds();
    m_Data->clearCameras();
    m_Data->clearImages();
    m_Data->clearMasks();
    m_Data->reset();

    m_bFirstAction = true;
}

void GLWidget::resetView()
{
    if (m_bDisplayMode2D)
    {
        zoomFit();

        update();
    }
    else
    {
        resetRotationMatrix();
        resetTranslationMatrix();

        setZoom(m_Data->getScale());

        m_bObjectCenteredView = true;

        showBall(true);
        showAxis(false);
        showBBox(false);
        showCams(false);

        emit(interactionMode(false));
    }
}

void GLWidget::resetRotationMatrix()
{
    _g_rotationMatrix[0] = _g_rotationMatrix[4] = _g_rotationMatrix[8] = 1;
    _g_rotationMatrix[1] = _g_rotationMatrix[2] = _g_rotationMatrix[3] = 0;
    _g_rotationMatrix[5] = _g_rotationMatrix[6] = _g_rotationMatrix[7] = 0;
}

void GLWidget::resetTranslationMatrix()
{
    Pt3dr center = m_Data->getCenter();

    m_params.m_translationMatrix[0] = -center.x;
    m_params.m_translationMatrix[1] = -center.y;
    m_params.m_translationMatrix[2] = -center.z;
}

void GLWidget::applyGamma(float aGamma)
{
    if (aGamma == 1.f) return;

    QRgb  pixel;
    int r,g,b;

    float _gamma = 1.f / aGamma;

    for(int i=0; i< _glImg.width();++i)
        for(int j=0; j<_glImg.height();++j)
        {
            pixel = _glImg.pixel(i,j);

            r = 255*pow((float) qRed(pixel)  / 255.f, _gamma);
            g = 255*pow((float) qGreen(pixel)/ 255.f, _gamma);
            b = 255*pow((float) qBlue(pixel) / 255.f, _gamma);

            if (r>255) r = 255;
            if (g>255) g = 255;
            if (b>255) b = 255;

            _glImg.setPixel(i,j, qRgb(r,g,b) );
        }
}
