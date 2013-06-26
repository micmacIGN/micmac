#include "GLWidget.h"

//Min and max zoom ratio (relative)
const float GL_MAX_ZOOM_RATIO = 1.0e6f;
const float GL_MIN_ZOOM_RATIO = 1.0e-6f;

//invalid GL list index
const GLuint GL_INVALID_LIST_ID = (~0);

using namespace Cloud_;
using namespace std;

bool g_mouseLeftDown = false;
bool g_mouseMiddleDown = false;
bool g_mouseRightDown = false;

GLfloat g_tmpMatrix[9],
g_rotationOx[9],
g_rotationOy[9],
g_rotationOz[9],
g_rotationMatrix[9] = { 1, 0, 0,
                        0, 1, 0,
                        0, 0, 1 },
g_glMatrix[16];

inline void setRotateOx_m33( const float i_angle, GLfloat o_m[9] )
{
    GLfloat co = (GLfloat)cos( i_angle ),
            si = (GLfloat)sin( i_angle );
    o_m[0]=1.f;		o_m[1]=0.f;		o_m[2]=0.f;
    o_m[3]=0.f;		o_m[4]=co;		o_m[5]=-si;
    o_m[6]=0.f;		o_m[7]=si;		o_m[8]=co;
}

inline void setRotateOy_m33( const float i_angle, GLfloat o_m[9] )
{
    GLfloat co = (GLfloat)cos( i_angle ),
            si = (GLfloat)sin( i_angle );
    o_m[0]=co;		o_m[1]=0;		o_m[2]=si;
    o_m[3]=0;		o_m[4]=1;		o_m[5]=0;
    o_m[6]=-si;		o_m[7]=0;		o_m[8]=co;
}

inline void setRotateOz_m33( const float i_angle, GLfloat o_m[9] )
{
    GLfloat co = (GLfloat)cos( i_angle ),
            si = (GLfloat)sin( i_angle );
    o_m[0]=co;		o_m[1]=si;		o_m[2]=0;
    o_m[3]=-si;		o_m[4]=co;		o_m[5]=0;
    o_m[6]=0;		o_m[7]=0;		o_m[8]=1;
}

inline void setTranslate_m3( const GLfloat *i_a, GLfloat o_m[16] )
{
    o_m[0] =1.;	 o_m[1] =0.f;	o_m[2] =0.f;	 o_m[3] =i_a[0];
    o_m[4] =0.f;	 o_m[5] =1.f;	o_m[6] =0.f;	 o_m[7] =i_a[1];
    o_m[8] =0.f;	 o_m[9] =0.f;	o_m[10]=1.f;	 o_m[11]=i_a[2];
    o_m[12]=0.f;	 o_m[13]=0.f;	o_m[14]=0.f;	 o_m[15]=1.f;
}

inline void mult( const GLfloat i_a[16], const GLfloat i_b[16], GLfloat o_m[16] )
{
    o_m[0] =i_a[0]*i_b[0]+i_a[1]*i_b[4]+i_a[2]*i_b[8]+i_a[3]*i_b[12];     o_m[1] =i_a[0]*i_b[1]+i_a[1]*i_b[5]+i_a[2]*i_b[9]+i_a[3]*i_b[13];     o_m[2] =i_a[0]*i_b[2]+i_a[1]*i_b[6]+i_a[2]*i_b[10]+i_a[3]*i_b[14];     o_m[3] =i_a[0]*i_b[3]+i_a[1]*i_b[7]+i_a[2]*i_b[11]+i_a[3]*i_b[15];
    o_m[4] =i_a[4]*i_b[0]+i_a[5]*i_b[4]+i_a[6]*i_b[8]+i_a[7]*i_b[12];     o_m[5] =i_a[4]*i_b[1]+i_a[5]*i_b[5]+i_a[6]*i_b[9]+i_a[7]*i_b[13];     o_m[6] =i_a[4]*i_b[2]+i_a[5]*i_b[6]+i_a[6]*i_b[10]+i_a[7]*i_b[14];     o_m[7] =i_a[4]*i_b[3]+i_a[5]*i_b[7]+i_a[6]*i_b[11]+i_a[7]*i_b[15];
    o_m[8] =i_a[8]*i_b[0]+i_a[9]*i_b[4]+i_a[10]*i_b[8]+i_a[11]*i_b[12];   o_m[9] =i_a[8]*i_b[1]+i_a[8]*i_b[5]+i_a[10]*i_b[9]+i_a[11]*i_b[13];   o_m[10]=i_a[8]*i_b[2]+i_a[9]*i_b[6]+i_a[10]*i_b[10]+i_a[11]*i_b[14];   o_m[11]=i_a[8]*i_b[3]+i_a[9]*i_b[7]+i_a[10]*i_b[11]+i_a[11]*i_b[15];
    o_m[12]=i_a[12]*i_b[0]+i_a[13]*i_b[4]+i_a[14]*i_b[8]+i_a[15]*i_b[12]; o_m[13]=i_a[12]*i_b[1]+i_a[13]*i_b[5]+i_a[14]*i_b[9]+i_a[15]*i_b[13]; o_m[14]=i_a[12]*i_b[2]+i_a[13]*i_b[6]+i_a[14]*i_b[10]+i_a[15]*i_b[14]; o_m[15]=i_a[12]*i_b[3]+i_a[13]*i_b[7]+i_a[14]*i_b[11]+i_a[15]*i_b[15];
}

inline void mult_m33( const GLfloat i_a[9], const GLfloat i_b[9], GLfloat o_m[9] )
{
    o_m[0]=i_a[0]*i_b[0]+i_a[1]*i_b[3]+i_a[2]*i_b[6];		o_m[1]=i_a[0]*i_b[1]+i_a[1]*i_b[4]+i_a[2]*i_b[7];		o_m[2]=i_a[0]*i_b[2]+i_a[1]*i_b[5]+i_a[2]*i_b[8];
    o_m[3]=i_a[3]*i_b[0]+i_a[4]*i_b[3]+i_a[5]*i_b[6];		o_m[4]=i_a[3]*i_b[1]+i_a[4]*i_b[4]+i_a[5]*i_b[7];		o_m[5]=i_a[3]*i_b[2]+i_a[4]*i_b[5]+i_a[5]*i_b[8];
    o_m[6]=i_a[6]*i_b[0]+i_a[7]*i_b[3]+i_a[8]*i_b[6];		o_m[7]=i_a[6]*i_b[1]+i_a[7]*i_b[4]+i_a[8]*i_b[7];		o_m[8]=i_a[6]*i_b[2]+i_a[7]*i_b[5]+i_a[8]*i_b[8];
}

inline void m33_to_m44( const GLfloat i_m[9], GLfloat o_m[16] )
{
    o_m[0]=i_m[0];		o_m[4]=i_m[3];		o_m[8] =i_m[6];		o_m[12]=0.f;
    o_m[1]=i_m[1];		o_m[5]=i_m[4];		o_m[9] =i_m[7];		o_m[13]=0.f;
    o_m[2]=i_m[2];		o_m[6]=i_m[5];		o_m[10]=i_m[8];		o_m[14]=0.f;
    o_m[3]=0.f;			o_m[7]=0.f;			o_m[11]=0.f;		o_m[15]=1.f;
}

inline void transpose( const GLfloat *i_a, GLfloat *o_m )
{
    o_m[0]=i_a[0];		o_m[4]=i_a[1];		o_m[8]=i_a[2];		o_m[12]=i_a[3];
    o_m[1]=i_a[4];		o_m[5]=i_a[5];		o_m[9]=i_a[6];		o_m[13]=i_a[7];
    o_m[2]=i_a[8];		o_m[6]=i_a[9];		o_m[10]=i_a[10];	o_m[14]=i_a[11];
    o_m[3]=i_a[12];		o_m[7]=i_a[13];		o_m[11]=i_a[14];	o_m[15]=i_a[15];
}

inline void crossprod( const GLdouble u[3], const GLdouble v[3], GLdouble o_m[3] )
{
    o_m[0] = u[1]*v[2] - u[2]*v[1];
    o_m[1] = u[2]*v[0] - u[0]*v[2];
    o_m[2] = u[0]*v[1] - u[1]*v[0];
}

inline void normalize( GLdouble o_m[3] )
{
    GLdouble norm = sqrt((double) (o_m[0]*o_m[0] + o_m[1]*o_m[1] + o_m[2]*o_m[2]));

    o_m[0] = o_m[0]/norm;
    o_m[1] = o_m[1]/norm;
    o_m[2] = o_m[2]/norm;
}

GLWidget::GLWidget(QWidget *parent, cData *data) : QGLWidget(parent)
  , m_font(font())
  , m_bCloudLoaded(false)
  , m_bCameraLoaded(false)
  , m_bDrawAxis(false)
  , m_bDrawBall(true)
  , m_bDrawCams(true)
  , m_bDrawMessages(true)
  , m_bDrawBbox(false)
  , m_bObjectCenteredView(true)
  , m_bPolyIsClosed(false)
  , m_interactionMode(TRANSFORM_CAMERA)
  , m_bFirstAdd(true)
  , m_previousAction(NONE)
  , m_trihedronGLList(GL_INVALID_LIST_ID)
  , m_ballGLList(GL_INVALID_LIST_ID)
  , m_nbGLLists(0)
  , m_params(ViewportParameters())
  , m_Data(data)
  , m_speed(2.5f)
  , m_vertexbuffer(QGLBuffer::VertexBuffer)  
  , _frameCount(0)
  , _previousTime(0)
  , _currentTime(0)
  , _fps(0.0f)
  , m_selection_mode(NONE)
{
    //setMouseTracking(true);

    //drag & drop handling
    setAcceptDrops(true);
}

GLWidget::~GLWidget()
{
    if (m_trihedronGLList != GL_INVALID_LIST_ID)
    {
        glDeleteLists(m_trihedronGLList,1);
        m_trihedronGLList = GL_INVALID_LIST_ID;
    }
    if (m_ballGLList != GL_INVALID_LIST_ID)
    {
        glDeleteLists(m_ballGLList,1);
        m_ballGLList = GL_INVALID_LIST_ID;
    }
}

void GLWidget::initializeGL()
{
    if (m_bInitialized)
        return;

    //    glClearDepth( 100.f );
    glEnable( GL_DEPTH_TEST );

    m_bInitialized = true;
}

void GLWidget::resizeGL(int width, int height)
{
    if (width==0 || height==0) return;

    m_glWidth  = (float)width;
    m_glHeight = (float)height;

    glViewport( 0, 0, width, height );
}

//-------------------------------------------------------------------------
// Calculates the frames per second
//-------------------------------------------------------------------------
void GLWidget::calculateFPS()
{
    //  Increase frame count
    _frameCount++;

    //  Get the number of milliseconds since glutInit called
    //  (or first call to glutGet(GLUT ELAPSED TIME)).
#ifndef WIN32
    _currentTime = glutGet(GLUT_ELAPSED_TIME);
#else
    _currentTime = GetTickCount();
#endif

    //  Calculate time passed
    int deltaTime = _currentTime - _previousTime;

    if(deltaTime > 1000)
    {
        //  calculate the number of frames per second
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

void GLWidget::paintGL()
{
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glMatrixMode(GL_MODELVIEW);

    draw3D();

    if (hasCloudLoaded())
    {

        glEnableClientState(GL_VERTEX_ARRAY);
        glEnableClientState(GL_COLOR_ARRAY);

        m_vertexbuffer.bind();
        glVertexPointer(3, GL_FLOAT, 0, NULL);
        m_vertexbuffer.release();

        m_vertexColor.bind();
        glColorPointer(3, GL_FLOAT, 0, NULL);
        m_vertexColor.release();

        glDrawArrays( GL_POINTS, 0, m_Data->getCloud(0)->size()*3 );

        glDisableClientState(GL_VERTEX_ARRAY);
        glDisableClientState(GL_COLOR_ARRAY);
    }

    //current messages (if valid)
    if (!m_messagesToDisplay.empty())
    {
        //Some versions of Qt seem to need glColorf instead of glColorub! (see https://bugreports.qt-project.org/browse/QTBUG-6217)
        glColor3f(1.f,1.f,1.f);

        int fontSize = 10;
        int lc_currentHeight = m_glHeight- fontSize*m_messagesToDisplay.size(); //lower center
        int uc_currentHeight = 10;            //upper center

        std::list<MessageToDisplay>::iterator it = m_messagesToDisplay.begin();
        while (it != m_messagesToDisplay.end())
        {
            m_font.setPointSize(fontSize);

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
                renderText((m_glWidth-rect.width())/2, lc_currentHeight, it->message,m_font);
                int messageHeight = QFontMetrics(m_font).height();
                lc_currentHeight += (messageHeight*5)/4; //add a 25% margin
            }
                break;
            case UPPER_CENTER_MESSAGE:
            {
                QRect rect = QFontMetrics(m_font).boundingRect(it->message);
                renderText((m_glWidth-rect.width())/2, uc_currentHeight+rect.height(), it->message,m_font);
                uc_currentHeight += (rect.height()*5)/4; //add a 25% margin
            }
                break;
            case SCREEN_CENTER_MESSAGE:
            {
                m_font.setPointSize(12);
                QRect rect = QFontMetrics(m_font).boundingRect(it->message);
                renderText((m_glWidth-rect.width())/2, (m_glHeight-rect.height())/2, it->message,m_font);
            }
            }

            ++it;
        }
    }

    if (m_messagesToDisplay.begin()->position != SCREEN_CENTER_MESSAGE)
    {
        if (m_bDrawBall) drawBall();
        else if (m_bDrawAxis) drawAxis();
    }

    if (m_bDrawCams) drawCams();

    if (m_bDrawBbox) drawBbox();

    if (m_interactionMode == SEGMENT_POINTS)
    {
        glMatrixMode(GL_PROJECTION);
        glPushMatrix();
        glLoadIdentity();
        glOrtho(0,m_glWidth,m_glHeight,0,-1,1);
        glMatrixMode(GL_MODELVIEW);
        glPushMatrix();
        glLoadIdentity();
        glPushAttrib(GL_ENABLE_BIT);
        glDisable(GL_DEPTH_TEST);
        glDisable(GL_LIGHTING);
        glDisable(GL_TEXTURE_2D);
        glColor3f(0,1,0);

        glBegin(m_bPolyIsClosed ? GL_LINE_LOOP : GL_LINE_STRIP);
        for (int aK = 0;aK < (int) m_polygon.size(); ++aK)
        {
            glVertex2f(m_polygon[aK].x, m_polygon[aK].y);
        }
        glEnd();

        // Closing 2D
        glPopAttrib();
        glPopMatrix(); // restore modelview
        glMatrixMode(GL_PROJECTION);
        glPopMatrix();
        glMatrixMode(GL_MODELVIEW);
    }

    if ((m_messagesToDisplay.begin()->position != SCREEN_CENTER_MESSAGE) && m_bDrawMessages)
    {
        calculateFPS();

        glColor4f(0.0f,0.7f,1.0f,0.6f);
        int fontSize = 10;
        m_font.setPointSize(fontSize);
        renderText(10,  m_glHeight- fontSize, m_messageFPS,m_font);
    }
}

void GLWidget::mousePressEvent(QMouseEvent *event)
{
    m_lastPos = Pt2df(event->pos().x(),event->pos().y());

    if ( event->buttons()&Qt::LeftButton )
    {
        g_mouseLeftDown = true;

        if (m_interactionMode == SEGMENT_POINTS)
        {
            if (!m_bPolyIsClosed)
            {
                if (m_polygon.size() < 2)
                    m_polygon.push_back(m_lastPos);
                else
                {
                    m_polygon[m_polygon.size()-1] = m_lastPos;
                    m_polygon.push_back(m_lastPos);
                }
            }
            else
            {
                clearPolyline();
                m_polygon.push_back(m_lastPos);
            }
        }
    }
    else if (event->buttons()&Qt::RightButton)
    {
        if (m_interactionMode == TRANSFORM_CAMERA)
            g_mouseRightDown = true;
        else
        {
            closePolyline();
            update();
        }
    }
    else if (event->buttons()&Qt::MiddleButton)
    {
        if (m_interactionMode == TRANSFORM_CAMERA)
            g_mouseMiddleDown = true;
    }
}

void GLWidget::mouseReleaseEvent(QMouseEvent *event)
{
    if ( !( event->buttons()&Qt::LeftButton ) )
        g_mouseLeftDown = false;
    if ( !( event->buttons()&Qt::RightButton ) )
        g_mouseRightDown = false;
    if ( !( event->buttons()&Qt::MiddleButton ) )
         g_mouseMiddleDown = false;
}

void GLWidget::keyPressEvent(QKeyEvent* event)
{
    switch(event->key())
    {
    case Qt::Key_Escape:
        clearPolyline();
        break;
    case Qt::Key_Plus:
        ptSizeUp(true);
        break;
    case Qt::Key_Minus:
        ptSizeUp(false);
        break;
    default:
    {
        event->ignore();
        return;
    }
    }
    update();
}

void GLWidget::setBufferGl(bool onlyColor)
{

    if(m_vertexbuffer.isCreated() && !onlyColor)
        m_vertexbuffer.destroy();
    if(m_vertexColor.isCreated())
        m_vertexColor.destroy();

    int sizeClouds = m_Data->getSizeClouds();

    if (sizeClouds == 0) return;

    GLfloat* vertices = NULL, *colors = NULL;

    if(!onlyColor)
        vertices   = new GLfloat[sizeClouds*3];

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
                colors[pitchV+bK*3 + 0 ]   = colo.redF()*1.5;
                colors[pitchV+bK*3 + 1 ]   = colo.greenF()*0.6;
                colors[pitchV+bK*3 + 2 ]   = colo.blueF()*0.6;
            }
        }

        pitchV += sizeCloud;
    }

    if(!onlyColor)
    {
        m_vertexbuffer.create();
        m_vertexbuffer.setUsagePattern(QGLBuffer::StaticDraw);
        m_vertexbuffer.bind();
        m_vertexbuffer.allocate(vertices, sizeClouds* 3 * sizeof(GLfloat));
        m_vertexbuffer.release();
    }

    m_vertexColor.create();
    m_vertexColor.setUsagePattern(QGLBuffer::StaticDraw);
    m_vertexColor.bind();
    m_vertexColor.allocate(colors, sizeClouds* 3 * sizeof(GLfloat));
    m_vertexColor.release();

    if(!onlyColor)
        delete [] vertices;
    delete [] colors;
}

void GLWidget::setData(cData *data)
{
    m_Data = data;

    setBufferGl();

    if (m_Data->NbClouds())
    {
        setCloudLoaded(true);
        setZoom(m_Data->getCloud(0)->getScale());

        m_params.m_translationMatrix[0] = -m_Data->m_cX;
        m_params.m_translationMatrix[1] = -m_Data->m_cY;
        m_params.m_translationMatrix[2] = -m_Data->m_cZ;
    }

    if (m_Data->NbCameras())
    {
        setCameraLoaded(true);
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
    int w = (m_glWidth>>1)+1;
    int h = (m_glHeight>>1)+1;

    QSettings settings;
    settings.beginGroup("OpenGL");

    const unsigned char BkgColor[3]		=   {10,102,151};

    const unsigned char* bkgCol = BkgColor;

    //Gradient "texture" drawing
    glBegin(GL_QUADS);
    //user-defined background color for gradient start
    glColor3ubv(bkgCol);
    glVertex2f(-w,h);
    glVertex2f(w,h);
    //and the inverse of points color for gradient end
    glColor3ub(0,0,0);
    glVertex2f(w,-h);
    glVertex2f(-w,-h);
    glEnd();
}

void GLWidget::draw3D()
{
    makeCurrent();

    setStandardOrthoCenter();
    //glEnable(GL_DEPTH_TEST);

    //gradient color background
    drawGradientBackground();
    //we clear background
    glClear(GL_DEPTH_BUFFER_BIT);

    zoom();

    //then, the modelview matrix
    glMatrixMode(GL_MODELVIEW);

    static GLfloat trans44[16], rot44[16], tmp[16];
    m33_to_m44( g_rotationMatrix, rot44 );
    setTranslate_m3(  m_params.m_translationMatrix, trans44 );

    //mult( trans44, rot44, tmp );
    mult( rot44, trans44, tmp );
    transpose( tmp, g_glMatrix );
    glLoadMatrixf( g_glMatrix );

}

void GLWidget::setStandardOrthoCenter()
{
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    float halfW = float(m_glWidth)*0.5;
    float halfH = float(m_glHeight)*0.5;
    glOrtho(-halfW,halfW,-halfH,halfH,-100.0f, 100.0f);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
}

void GLWidget::zoom()
{
    GLdouble zoom = m_params.zoom;
    GLdouble fAspect = (GLdouble) m_glWidth/ m_glHeight;

    GLdouble left   = -zoom*fAspect;
    GLdouble right  =  zoom*fAspect;

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();

    glOrtho(left, right, -zoom, zoom, -100.0f, 100.0f);

}

void GLWidget::setInteractionMode(INTERACTION_MODE mode)
{
    m_interactionMode = mode;

    switch (mode) {
    case TRANSFORM_CAMERA:
        setMouseTracking(false);
        break;
    case SEGMENT_POINTS:
        setProjectionMatrix();
        setMouseTracking(true);
        break;
    default:
        break;
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

    g_rotationMatrix[0] = s[0];
    g_rotationMatrix[1] = s[1];
    g_rotationMatrix[2] = s[2];

    g_rotationMatrix[3] = u[0];
    g_rotationMatrix[4] = u[1];
    g_rotationMatrix[5] = u[2];

    g_rotationMatrix[6] = -eye[0];
    g_rotationMatrix[7] = -eye[1];
    g_rotationMatrix[8] = -eye[2];

    m_params.m_translationMatrix[0] = m_Data->m_cX;
    m_params.m_translationMatrix[1] = m_Data->m_cY;
    m_params.m_translationMatrix[2] = m_Data->m_cZ;
}

void GLWidget::onWheelEvent(float wheelDelta_deg)
{
    //convert degrees in zoom 'power'
    float zoomFactor = pow(1.1f,wheelDelta_deg *.05f);
    setZoom(m_params.zoom*zoomFactor);

    update();
}

void GLWidget::setZoom(float value)
{
    if (value < GL_MIN_ZOOM_RATIO)
        value = GL_MIN_ZOOM_RATIO;
    else if (value > GL_MAX_ZOOM_RATIO)
        value = GL_MAX_ZOOM_RATIO;

    m_params.zoom = value;
}

void GLWidget::wheelEvent(QWheelEvent* event)
{
    if (m_interactionMode == SEGMENT_POINTS)
    {
        event->ignore();
        return;
    }

    //see QWheelEvent documentation ("distance that the wheel is rotated, in eighths of a degree")
    float wheelDelta_deg = (float)event->delta() / 8.0f;

    onWheelEvent(wheelDelta_deg);

    emit mouseWheelRotated(wheelDelta_deg);
}

void GLWidget::mouseMoveEvent(QMouseEvent *event)
{
    if (event->x()<0 || event->y()<0 || event->x()>width() || event->y()>height())
        return;

    Pt2df pos = Pt2df(event->pos().x(),event->pos().y());

    if ((m_interactionMode == SEGMENT_POINTS) )
    {
        if(!m_bPolyIsClosed)
        {
            int sz = m_polygon.size();

            if (sz == 0)
                return;
            else if (sz == 1)
                m_polygon.push_back(pos);
            else
                //replace last point by the current one
                m_polygon[sz-1] = pos;

            update();
        }

        event->ignore();
    }
    else if (g_mouseLeftDown || g_mouseMiddleDown|| g_mouseRightDown)
    {
        Pt2df dp = pos-m_lastPos;

        if ( g_mouseLeftDown ) // rotation autour de X et Y
        {
            float angleX =  m_speed * dp.y / (float) m_glHeight;
            float angleY =  m_speed * dp.x / (float) m_glWidth;

            setAngles(angleX, angleY, m_params.angleZ);

            setRotateOx_m33( angleX, g_rotationOx );
            setRotateOy_m33( angleY, g_rotationOy );

            mult_m33( g_rotationOx, g_rotationMatrix, g_tmpMatrix );
            mult_m33( g_rotationOy, g_tmpMatrix, g_rotationMatrix );
        }
        else if ( g_mouseMiddleDown ) // translation
        {
            m_bObjectCenteredView = false;
            m_params.m_translationMatrix[0] += m_speed * dp.x*m_Data->m_diam/(float)m_glWidth;
            m_params.m_translationMatrix[1] -= m_speed * dp.y*m_Data->m_diam/(float)m_glHeight;
        }
        else if ( g_mouseRightDown ) // rotation autour de Z
        {
            float angleZ =  m_speed * dp.x / (float) m_glWidth;

            setAngles( m_params.angleX,  m_params.angleY, angleZ);

            setRotateOz_m33( angleZ, g_rotationOz );

            mult_m33( g_rotationOz, g_rotationMatrix, g_tmpMatrix );

            for (int i = 0; i < 9; ++i) g_rotationMatrix[i] = g_tmpMatrix[i];
        }

        update();
    }

    m_lastPos = pos;
}

bool isPointInsidePoly(const Pt2df& P, const std::vector< Pt2df > poly)
{
    unsigned vertices=poly.size();
    if (vertices<3)
        return false;

    bool inside = false;

    Pt2df A = poly[0];
    for (unsigned i=1;i<=vertices;++i)
    {
        Pt2df B = poly[i%vertices];

        //Point Inclusion in Polygon Test (inspired from W. Randolph Franklin - WRF)
        if (((B.y <= P.y) && (P.y<A.y)) ||
                ((A.y <= P.y) && (P.y<B.y)))
        {
            float ABy = A.y-B.y;
            float t = (P.x-B.x)*ABy-(A.x-B.x)*(P.y-B.y);
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
    glGetDoublev(GL_MODELVIEW_MATRIX, (GLdouble*) &_MM);

    glMatrixMode(GL_PROJECTION);
    glGetDoublev(GL_PROJECTION_MATRIX, (GLdouble*) &_MP);

    glGetIntegerv(GL_VIEWPORT, _VP);
}

void GLWidget::getProjection(Pt2df &P2D, Vertex P)
{
    GLdouble xp,yp,zp;
    gluProject(P.x(),P.y(),P.z(),_MM,_MP,_VP,&xp,&yp,&zp);
    P2D = Pt2df(xp,yp);
}

void GLWidget::Select(int mode)
{
    Pt2df P2D;
    bool pointInside;
    std::vector < Pt2df > polyg;

    if(mode == ADD || mode == SUB)
    {
        if ((m_polygon.size() < 3) || (!m_bPolyIsClosed))
            return;

        for (int aK=0; aK < (int) m_polygon.size(); ++aK)
            polyg.push_back(Pt2df(m_polygon[aK].x, m_glHeight - m_polygon[aK].y));
    }

    for (int aK=0; aK < m_Data->NbClouds(); ++aK)
    {
        Cloud *a_cloud = m_Data->getCloud(aK);

        for (int bK=0; bK < a_cloud->size();++bK)
        {
            Vertex &P = a_cloud->getVertex( bK );
            switch (mode)
            {
            case ADD:
                getProjection(P2D, P);
                pointInside = isPointInsidePoly(P2D,polyg);
                if (m_bFirstAdd)
                    emit selectedPoint((uint)aK,(uint)bK,pointInside);
                else
                    emit selectedPoint((uint)aK,(uint)bK,pointInside||P.isVisible());
                break;
            case SUB:
                if (P.isVisible())
                {
                    getProjection(P2D, P);
                    pointInside = isPointInsidePoly(P2D,polyg);
                    emit selectedPoint((uint)aK,(uint)bK,!pointInside);
                }
                break;
            case INVERT:
                if (m_previousAction == NONE)  m_bFirstAdd = true;
                emit selectedPoint((uint)aK,(uint)bK,!P.isVisible());
                break;
            case ALL:
                m_bFirstAdd = true;
                emit selectedPoint((uint)aK,(uint)bK, true);
                break;
            case NONE:
                emit selectedPoint((uint)aK,(uint)bK,false);
                break;
            }
        }

        setBufferGl(true);
    }

    if ((mode == ADD) && (m_bFirstAdd)) m_bFirstAdd = false;

    m_previousAction = mode;
    //m_infos.push_back(cSaisieInfos(m_params, m_polygon, selection_mode));
}

void GLWidget::deletePolylinePoint()
{
    float dist2 = FLT_MAX;
    int dx, dy, d2;
    int idx = -1;

    for (int aK =0; aK < (int) m_polygon.size();++aK)
    {
        dx = m_polygon[aK].x-m_lastPos.x;
        dy = m_polygon[aK].y-m_lastPos.y;
        d2 = dx*dx + dy*dy;

        if (d2 < dist2)
        {
            dist2 = d2;
            idx = aK;
        }
    }

    if (idx != -1)
    {
        for (int aK =idx; aK < (int)m_polygon.size()-1;++aK)
        {
            m_polygon[aK] = m_polygon[aK+1];
        }

        m_polygon.pop_back();
    }
}

void GLWidget::clearPolyline()
{
    m_polygon.clear();
    m_bPolyIsClosed = false;
}

void GLWidget::closePolyline()
{
    if (!m_bPolyIsClosed)
    {
        //remove last point if needed
        int sz = m_polygon.size();
        if ((sz > 2) && (m_polygon[sz-1] == m_polygon[sz-2]))
            m_polygon.resize(sz-1);

        sz = m_polygon.size();
        if (sz > 2) m_polygon.resize(sz-1);

        m_bPolyIsClosed = true;
    }
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

void GLWidget::drawAxis()
{
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();

    if (m_trihedronGLList == GL_INVALID_LIST_ID)
    {
        m_trihedronGLList = glGenLists(1);
        glNewList(m_trihedronGLList, GL_COMPILE);

        glPushAttrib(GL_LINE_BIT | GL_DEPTH_BUFFER_BIT);
        glEnable(GL_LINE_SMOOTH);
        glEnable(GL_DEPTH_TEST);

        //trihedra OpenGL drawing
        glBegin(GL_LINES);
        glColor3f(1.0f,0.0f,0.0f);
        glVertex3f(0.0f,0.0f,0.0f);
        glVertex3f(0.4f,0.0f,0.0f);
        glColor3f(0.0f,1.0f,0.0f);
        glVertex3f(0.0f,0.0f,0.0f);
        glVertex3f(0.0f,0.4f,0.0f);
        glColor3f(0.0f,0.7f,1.0f);
        glVertex3f(0.0f,0.0f,0.0f);
        glVertex3f(0.0f,0.0f,0.4f);
        glEnd();

        glPopAttrib();

        glEndList();
    }

    glTranslatef(m_Data->m_cX,m_Data->m_cY,m_Data->m_cZ);

    glCallList(m_trihedronGLList);

    glPopMatrix();
}

//draw a unit circle in a given plane (0=YZ, 1 = XZ, 2=XY)
void glDrawUnitCircle(unsigned char dim, int steps = 64)
{
    float thetaStep = static_cast<float>(2.0*PI/(double)steps);
    float theta = 0.0f;
    unsigned char dimX = (dim<2 ? dim+1 : 0);
    unsigned char dimY = (dimX<2 ? dimX+1 : 0);

    GLfloat P[4];

    for (int i=0;i<4;++i) P[i] = 0.0f;

    glBegin(GL_LINE_LOOP);
    for (int i=0;i<steps;++i)
    {
        P[dimX] = cos(theta);
        P[dimY] = sin(theta);
        glVertex3fv(P);
        theta += thetaStep;
    }
    glEnd();
}

void GLWidget::drawBall()
{
    if (!m_bObjectCenteredView) return;

    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();

    // ball radius
    //float scale = 0.05f * (float) m_glWidth/ (float) m_glHeight;
    float scale = m_Data->m_diam / 1.5f;

    if (m_ballGLList == GL_INVALID_LIST_ID)
    {
        incrNbGLLists();
        m_ballGLList = getNbGLLists();
        glNewList(m_ballGLList, GL_COMPILE);

        //draw 3 circles
        glPushAttrib(GL_LINE_BIT);
        glEnable(GL_LINE_SMOOTH);
        glPushAttrib(GL_COLOR_BUFFER_BIT);
        glEnable(GL_BLEND);
        const float c_alpha = 0.6f;
        glLineWidth(1.0f);

        glColor4f(1.0f,0.0f,0.0f,c_alpha);
        glDrawUnitCircle(0);
        glBegin(GL_LINES);
        glVertex3f(-1.0f,0.0f,0.0f);
        glVertex3f( 1.0f,0.0f,0.0f);
        glEnd();

        glColor4f(0.0f,1.0f,0.0f,c_alpha);
        glDrawUnitCircle(1);
        glBegin(GL_LINES);
        glVertex3f(0.0f,-1.0f,0.0f);
        glVertex3f(0.0f, 1.0f,0.0f);
        glEnd();

        glColor4f(0.0f,0.7f,1.0f,c_alpha);
        glDrawUnitCircle(2);
        glBegin(GL_LINES);
        glVertex3f(0.0f,0.0f,-1.0f);
        glVertex3f(0.0f,0.0f, 1.0f);
        glEnd();

        glPopAttrib();

        glEndList();
    }

    glTranslatef(m_Data->m_cX,m_Data->m_cY,m_Data->m_cZ);
    glScalef(scale,scale,scale);

    glCallList(m_ballGLList);

    glPopMatrix();
}

void GLWidget::drawCams()
{
    float scale = 1.f;

    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();

    incrNbGLLists();
    GLuint list = getNbGLLists();
    glNewList(list, GL_COMPILE);

    glLineWidth(2);
    glPointSize(7);
    if (m_Data->NbClouds())
    {
        scale = 0.0005*m_Data->getCloud(0)->getScale();
    }
    for (int i=0; i<m_Data->NbCameras();i++)
    {
        CamStenope * pCam = m_Data->getCamera(i);

        REAL f = pCam->Focale();
        Pt3dr C  = pCam->VraiOpticalCenter();
        Pt3dr P1 = pCam->ImEtProf2Terrain(Pt2dr(0,0),scale*f);
        Pt3dr P2 = pCam->ImEtProf2Terrain(Pt2dr(pCam->Sz().x,0),scale*f);
        Pt3dr P3 = pCam->ImEtProf2Terrain(Pt2dr(0,pCam->Sz().y),scale*f);
        Pt3dr P4 = pCam->ImEtProf2Terrain(Pt2dr(pCam->Sz().x,pCam->Sz().y),scale*f);

        //translation
        if (m_Data->NbClouds())
        {
            Pt3dr translation = m_Data->getCloud(0)->getTranslation();

            C = C + translation;
            P1 = P1 + translation;
            P2 = P2 + translation;
            P3 = P3 + translation;
            P4 = P4 + translation;
        }

        glBegin(GL_LINES);
            //perspective cone
            qglColor(QColor(0,0,0));
            glVertex3d(C.x, C.y, C.z);
            glVertex3d(P1.x, P1.y, P1.z);

            glVertex3d(C.x, C.y, C.z);
            glVertex3d(P2.x, P2.y, P2.z);

            glVertex3d(C.x, C.y, C.z);
            glVertex3d(P3.x, P3.y, P3.z);

            glVertex3d(C.x, C.y, C.z);
            glVertex3d(P4.x, P4.y, P4.z);

            //Image
            qglColor(QColor(255,0,0));
            glVertex3d(P1.x, P1.y, P1.z);
            glVertex3d(P2.x, P2.y, P2.z);

            glVertex3d(P4.x, P4.y, P4.z);
            glVertex3d(P2.x, P2.y, P2.z);

            glVertex3d(P3.x, P3.y, P3.z);
            glVertex3d(P1.x, P1.y, P1.z);

            glVertex3d(P4.x, P4.y, P4.z);
            glVertex3d(P3.x, P3.y, P3.z);
        glEnd();

        glBegin(GL_POINTS);
            glVertex3d(C.x, C.y, C.z);
        glEnd();
    }

    glEndList();

    glCallList(list);

    glPointSize(m_params.PointSize);
    glLineWidth(m_params.LineWidth);
    glPopMatrix();
}

void GLWidget::drawBbox()
{
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();

    incrNbGLLists();
    GLuint list = getNbGLLists();
    glNewList(list, GL_COMPILE);

    glLineWidth(1);

    if (m_Data->NbClouds())
    {
        float minX, minY, minZ, maxX, maxY, maxZ;

        minX = m_Data->m_minX;
        minY = m_Data->m_minY;
        minZ = m_Data->m_minZ;
        maxX = m_Data->m_maxX;
        maxY = m_Data->m_maxY;
        maxZ = m_Data->m_maxZ;

        Pt3dr P1(minX, minY, minZ);
        Pt3dr P2(minX, minY, maxZ);
        Pt3dr P3(minX, maxY, maxZ);
        Pt3dr P4(minX, maxY, minZ);
        Pt3dr P5(maxX, minY, minZ);
        Pt3dr P6(maxX, maxY, minZ);
        Pt3dr P7(maxX, maxY, maxZ);
        Pt3dr P8(maxX, minY, maxZ);

        glBegin(GL_LINES);

        qglColor(QColor("orange"));

            glVertex3d(P1.x, P1.y, P1.z);
            glVertex3d(P2.x, P2.y, P2.z);

            glVertex3d(P3.x, P3.y, P3.z);
            glVertex3d(P2.x, P2.y, P2.z);

            glVertex3d(P1.x, P1.y, P1.z);
            glVertex3d(P4.x, P4.y, P4.z);

            glVertex3d(P1.x, P1.y, P1.z);
            glVertex3d(P5.x, P5.y, P5.z);

            glVertex3d(P7.x, P7.y, P7.z);
            glVertex3d(P3.x, P3.y, P3.z);

            glVertex3d(P7.x, P7.y, P7.z);
            glVertex3d(P6.x, P6.y, P6.z);

            glVertex3d(P8.x, P8.y, P8.z);
            glVertex3d(P5.x, P5.y, P5.z);

            glVertex3d(P7.x, P7.y, P7.z);
            glVertex3d(P8.x, P8.y, P8.z);

            glVertex3d(P5.x, P5.y, P5.z);
            glVertex3d(P6.x, P6.y, P6.z);

            glVertex3d(P4.x, P4.y, P4.z);
            glVertex3d(P6.x, P6.y, P6.z);

            glVertex3d(P8.x, P8.y, P8.z);
            glVertex3d(P2.x, P2.y, P2.z);

            glVertex3d(P4.x, P4.y, P4.z);
            glVertex3d(P3.x, P3.y, P3.z);


        glEnd();

        glPopAttrib();

        glEndList();
    }

    glCallList(list);

    glPointSize(m_params.PointSize);
    glLineWidth(m_params.LineWidth);
    glPopMatrix();
}

void GLWidget::showAxis(bool show)
{
    m_bDrawAxis = show;
    if (m_bDrawAxis) m_bDrawBall = false;

    update();
}

void GLWidget::showBall(bool show)
{
    m_bDrawBall = show;
    if (m_bDrawBall) m_bDrawAxis = false;

    update();
}

void GLWidget::showCams(bool show)
{
    m_bDrawCams = show;

    update();
}

void GLWidget::showBBox(bool show)
{
    m_bDrawBbox = show;

    update();
}

void GLWidget::showMessages(bool show)
{
    m_bDrawMessages = show;

    if (show)
    {
        if (m_interactionMode == TRANSFORM_CAMERA) showMoveMessages();
        else showSelectionMessages();
    }
    else displayNewMessage(QString());

    update();
}

bool GLWidget::showMessages(){return m_bDrawMessages;}

void GLWidget::showSelectionMessages()
{
    displayNewMessage(QString());
    displayNewMessage("Selection mode",UPPER_CENTER_MESSAGE);
    displayNewMessage("Left click: add contour point / Right click: close / Echap: delete polyline",LOWER_CENTER_MESSAGE);
    displayNewMessage("Space: add points inside polyline / Suppr: delete points inside polyline",LOWER_CENTER_MESSAGE);
}

void GLWidget::showMoveMessages()
{
    displayNewMessage(QString());
    displayNewMessage("Move mode",UPPER_CENTER_MESSAGE);
    displayNewMessage("Left click: rotate viewpoint / Right click: translate viewpoint",LOWER_CENTER_MESSAGE);
}

void GLWidget::setAngles(float angleX, float angleY, float angleZ)
{
    m_params.angleX = angleX;
    m_params.angleY = angleY;
    m_params.angleZ = angleZ;
}
