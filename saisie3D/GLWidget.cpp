#include <QtGui/QMouseEvent>
#include <QSettings>
#include <QMessageBox>
#include "GLWidget.h"
#include "GL/glu.h"

#include "Cloud.h"

#include <cmath>
#include <limits>
#include <iostream>
#include <algorithm>

const GLfloat g_trackballScale = 1.f;

//Min and max zoom ratio (relative)
const float GL_MAX_ZOOM_RATIO = 1.0e6f;
const float GL_MIN_ZOOM_RATIO = 1.0e-6f;

ViewportParameters::ViewportParameters()
    : pixelSize(1.0f)
    , zoom(1.0f)
    , defaultPointSize(1.0f)
    , defaultLineWidth(1.0f)
{
    //viewMat.toIdentity();
}

ViewportParameters::ViewportParameters(const ViewportParameters& params)
    : pixelSize(params.pixelSize)
    , zoom(params.zoom)
    , defaultPointSize(params.defaultPointSize)
    , defaultLineWidth(params.defaultLineWidth)
{
}

int g_mouseOldX, g_mouseOldY;
bool g_mouseLeftDown = false;
GLfloat g_tmpMatix[9],
        g_rotationOx[9],
        g_rotationOy[9],
        g_rotationMatrix[9] = { 1, 0, 0,
                               0, 1, 0,
                               0, 0, 1 },
        g_inverseRotationMatrix[9] = { 1, 0, 0,
                                       0, 1, 0,
                                       0, 0, 1 },
        g_glMatrix[16];

const GLfloat g_u[] = { 1.f, 0.f, 0.f },
              g_v[] = { 0.f, 1.f, 0.f };
GLfloat g_uu[3], g_vv[3];

GLfloat g_angleOx = 0.f,
        g_angleOy  = 0.f;

using namespace std;

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

inline void setRotate_m33( const float i_angle, const float i_u[3], GLfloat o_m[9] )
{
    GLfloat co = (GLfloat)cos( i_angle ),
            si = (GLfloat)sin( i_angle );
    o_m[0]=i_u[0]*i_u[0]+(1-i_u[0]*i_u[0])*co;		o_m[1]=i_u[0]*i_u[1]*(1-co)-i_u[0]*si;			o_m[2]=i_u[0]*i_u[2]*(1-co)+i_u[1]*si;
    o_m[3]=i_u[0]*i_u[1]*(1-co)+i_u[2]*si;			o_m[4]=i_u[1]*i_u[1]+(1-i_u[1]*i_u[1])*co;		o_m[5]=i_u[1]*i_u[2]*(1-co)-i_u[0]*si;
    o_m[6]=i_u[0]*i_u[2]*(1-co)-i_u[1]*si;			o_m[7]=i_u[1]*i_u[2]*(1-co)+i_u[0]*si;			o_m[8]=i_u[2]*i_u[2]+(1-i_u[2]*i_u[2])*co;
}

inline void setRotate_m33( const float i_angle, const float i_x, const float i_y, const float i_z, GLfloat o_m[9] )
{
    GLfloat co = (GLfloat)cos( i_angle ),
            si = (GLfloat)sin( i_angle );
    o_m[0]=i_x*i_x+(1-i_x*i_x)*co;		o_m[1]=i_x*i_y*(1-co)-i_x*si;		o_m[2]=i_x*i_z*(1-co)+i_z*si;
    o_m[3]=i_x*i_y*(1-co)+i_z*si;		o_m[4]=i_y*i_y+(1-i_y*i_y)*co;		o_m[5]=i_y*i_z*(1-co)-i_x*si;
    o_m[6]=i_x*i_z*(1-co)-i_y*si;		o_m[7]=i_y*i_z*(1-co)+i_x*si;		o_m[8]=i_z*i_z+(1-i_z*i_z)*co;
}

inline void setTranslate( const float i_x, const float i_y, const float i_z, GLfloat o_m[16] )
{
     o_m[0] =1.;	 o_m[1] =0.f;	o_m[2] =0.f;	 o_m[3] =i_x;
     o_m[4] =0.f;	 o_m[5] =1.f;	o_m[6] =0.f;	 o_m[7] =i_y;
     o_m[8] =0.f;	 o_m[9] =0.f;	o_m[10]=1.f;	 o_m[11]=i_z;
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

inline bool inverse_m33( const GLfloat i_a[9], GLfloat i_b[9] )
{
    bool invertible = true;

    i_b[0] = i_a[4]*i_a[8]-i_a[5]*i_a[7];
    i_b[3] = i_a[5]*i_a[6]-i_a[3]*i_a[8];
    i_b[6] = i_a[3]*i_a[7]-i_a[4]*i_a[6];

    GLfloat det = i_a[0]*i_b[0]+i_a[1]*i_b[3]+i_a[2]*i_b[6];
    if ( det<numeric_limits<GLfloat>::epsilon() ){
        det = 1.f;
        invertible = false;
    }

    i_b[1]=( i_a[2]*i_a[7]-i_a[1]*i_a[8] )/det; i_b[2]=( i_a[1]*i_a[5]-i_a[2]*i_a[4] )/det;
    i_b[4]=( i_a[0]*i_a[8]-i_a[2]*i_a[6] )/det; i_b[5]=( i_a[2]*i_a[3]-i_a[0]*i_a[5] )/det;
    i_b[7]=( i_a[1]*i_a[6]-i_a[0]*i_a[7] )/det; i_b[8]=( i_a[0]*i_a[4]-i_a[1]*i_a[3] )/det;
    i_b[0]/=det; i_b[3]/=det; i_b[6]/=det;

    return invertible;
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

inline void transpose_m33( const GLfloat i_a[9], GLfloat o_m[9] )
{
    o_m[0]=i_a[0];	o_m[3]=i_a[1];	o_m[6]=i_a[2];
    o_m[1]=i_a[3];	o_m[4]=i_a[4];	o_m[7]=i_a[5];
    o_m[2]=i_a[6];	o_m[5]=i_a[7];	o_m[8]=i_a[8];
}

GLWidget::GLWidget(QWidget *parent) : QGLWidget(parent)
      , m_interactionMode(TRANSFORM_CAMERA)
      , m_font(font())
      , m_bCloudLoaded(false)
      , m_params(ViewportParameters())
{
    m_minX = m_minY = m_minZ = FLT_MAX;
    m_maxX = m_maxY = m_maxZ = FLT_MIN;
    m_cX = m_cY = m_cZ = m_diam = 0.f;

    setMouseTracking(true);

    //drag & drop handling
    setAcceptDrops(true);
}

void GLWidget::initializeGL()
{
    if (m_bInitialized)
        return;

    glShadeModel( GL_SMOOTH );

    glClearDepth( 100.f );
    glEnable( GL_DEPTH_TEST );
    glDepthFunc( GL_LEQUAL );

    //transparency off by default
    glDisable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    glHint( GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST );

    m_bInitialized = true;
}

void GLWidget::resizeGL(int width, int height)
{
    if (width==0 || height==0) return;

    m_glWidth  = (float)width;
    m_glHeight = (float)height;

    glViewport( 0, 0, width, height );
}

/*void GLWidget::getContext(glDrawContext& context)
{
    //display size
    context.glW = m_glWidth;
    context.glH = m_glHeight;
    context.flags = 0;

    const mmGui::ParamStruct& guiParams = mmGui::Parameters();

    //decimation options
    context.decimateCloudOnMove = guiParams.decimateCloudOnMove;

    //text display
    context.dispNumberPrecision = guiParams.displayedNumPrecision;
    context.labelsTransparency = guiParams.labelsTransparency;

    memcpy(context.pointsDefaultCol,guiParams.pointsDefaultCol,sizeof(unsigned char)*3);
    memcpy(context.textDefaultCol,guiParams.textDefaultCol,sizeof(unsigned char)*3);
    memcpy(context.labelDefaultCol,guiParams.labelCol,sizeof(unsigned char)*3);
    memcpy(context.bbDefaultCol,guiParams.bbDefaultCol,sizeof(unsigned char)*3);

    //default font size
    setFontPointSize(1);
}*/

void GLWidget::paintGL()
{
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glMatrixMode(GL_MODELVIEW);

    draw3D();

    glPointSize(m_params.defaultPointSize);

    glBegin(GL_POINTS);
    for(int aK=0; aK< m_ply.size(); aK++)
    {
        for(int bK=0; bK< m_ply[aK].getVertexNumber(); bK++)
        {
            Cloud_::Vertex vert = m_ply[aK].getVertex(bK);
            qglColor( vert.m_color );
            //glColor3ub( vert.m_color.red(), vert.m_color.green(), vert.m_color.blue() );
            glVertex3f( (vert.x() - m_cX)/m_diam, (vert.y() - m_cY)/m_diam, (vert.z()-m_cZ)/m_diam );
        }
    }
    glEnd();

    //current messages (if valid)
    if (!m_messagesToDisplay.empty())
    {
        //Some versions of Qt seem to need glColorf instead of glColorub! (see https://bugreports.qt-project.org/browse/QTBUG-6217)
        glColor3f(1.f,1.f,1.f);

        std::list<MessageToDisplay>::iterator it = m_messagesToDisplay.begin();
        while (it != m_messagesToDisplay.end())
        {
            QFont newFont(m_font);
            newFont.setPointSize(12);
            QRect rect = QFontMetrics(newFont).boundingRect(it->message);
            //only one message supported in the screen center (for the moment ;)
            renderText((m_glWidth-rect.width())/2, (m_glHeight-rect.height())/2, it->message,newFont);

            ++it;
        }
    }

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

        glBegin(GL_LINE_LOOP);
        for (int aK = 0;aK < m_polygon.size(); ++aK)
        {
            glVertex2f(m_polygon[aK].x(), m_polygon[aK].y());
        }
        glEnd();

        // Closing 2D
        glPopAttrib();
        glPopMatrix(); // restore modelview
        glMatrixMode(GL_PROJECTION);
        glPopMatrix();
        glMatrixMode(GL_MODELVIEW);
    }
}

void GLWidget::mousePressEvent(QMouseEvent *event)
{
    if ( event->buttons()&Qt::LeftButton )
    {
        g_mouseLeftDown = true;
        g_mouseOldX = event->x();
        g_mouseOldY = event->y();

        if (m_interactionMode == SEGMENT_POINTS)
            m_polygon.push_back(event->pos());
    }
    else if ( (event->buttons()&Qt::RightButton)&&(m_interactionMode == SEGMENT_POINTS) )
    {
        m_polygon.clear();
    }

    lastPos = event->pos();
}

void GLWidget::mouseReleaseEvent(QMouseEvent *event)
{
    if ( !( event->buttons()&Qt::LeftButton ) )
        g_mouseLeftDown = false;
}

void GLWidget::keyPressEvent(QKeyEvent* event)
{
    switch(event->key())
    {
    case Qt::Key_Escape:
        close();
        break;
    case Qt::Key_Space:
        //to do: segment point cloud
        segment(true);
        break;
    default:
        event->ignore();
        break;
    }
}

void GLWidget::addPly( const QString &i_ply_file ) {

    Cloud_::Cloud a_ply;
    a_ply.loadPly( i_ply_file.toStdString() );
    m_ply.push_back(a_ply);
    if (!hasCloudLoaded()) setCloudLoaded(true);

    //compute bounding box

    int nbPts = a_ply.getVertexNumber();
    for (int aK=0; aK < nbPts; ++aK)
    {
        Cloud_::Vertex vert = a_ply.getVertex(aK);

        if (vert.x() > m_maxX) m_maxX = vert.x();
        if (vert.x() < m_minX) m_minX = vert.x();
        if (vert.y() > m_maxY) m_maxY = vert.y();
        if (vert.y() < m_minY) m_minY = vert.y();
        if (vert.z() > m_maxZ) m_maxZ = vert.z();
        if (vert.z() < m_minZ) m_minZ = vert.z();
    }

    m_cX = (m_minX + m_maxX) * .5f;
    m_cY = (m_minY + m_maxY) * .5f;
    m_cZ = (m_minZ + m_maxZ) * .5f;

    //cout << "center " << m_cX <<" "<< m_cY <<" "<< m_cZ << endl;

    m_diam = max(m_maxX-m_minX, max(m_maxY-m_minY, m_maxZ-m_minZ));

    updateGL();
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
            //fileNames[i] = QUrl(fileNames[i].trimmed()).toLocalFile(); //toLocalFile removes the end of filenames sometimes!
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
                                 MessageType type/*=CUSTOM_MESSAGE*/)
{
    if (message.isEmpty())
    {

        std::list<MessageToDisplay>::iterator it = m_messagesToDisplay.begin();
        while (it != m_messagesToDisplay.end())
        {
            //same position? we remove the message
            if (it->position == pos)
                it = m_messagesToDisplay.erase(it);
            else
                ++it;
        }

        return;
    }

    MessageToDisplay mess;
    mess.message = message;
    mess.messageValidity_sec = 5; //TODO: Timer::Sec()+displayMaxDelay_sec;
    mess.position = pos;
    mess.type = type;
    m_messagesToDisplay.push_back(mess);
}

void GLWidget::drawGradientBackground()
{
    int w = (m_glWidth>>1)+1;
    int h = (m_glHeight>>1)+1;

    QSettings settings;
    settings.beginGroup("OpenGL");

    const unsigned char* bkgCol = mmColor::defaultBkgColor;
    const unsigned char* forCol = mmColor::white;

    //Gradient "texture" drawing
    glBegin(GL_QUADS);
    //we the user-defined background color for gradient start
    glColor3ubv(bkgCol);
    glVertex2f(-w,h);
    glVertex2f(w,h);
    //and the inverse of points color for gradient end
    glColor3ub(255-forCol[0],255-forCol[1],255-forCol[2]);
    glVertex2f(w,-h);
    glVertex2f(-w,-h);
    glEnd();
}

void GLWidget::draw3D()
{
    makeCurrent();

    setStandardOrthoCenter();
    glEnable(GL_DEPTH_TEST);

    glPointSize(m_params.defaultPointSize);
    glLineWidth(m_params.defaultLineWidth);

    //gradient color background
    if (mmGui::Parameters().drawBackgroundGradient)
    {
        drawGradientBackground();
        //we clear background
        glClear(GL_DEPTH_BUFFER_BIT);
    }
    else
    {
        const unsigned char* bkgCol = mmGui::Parameters().backgroundCol;
        glClearColor(	(float)bkgCol[0] / 255.0f,
                        (float)bkgCol[1] / 255.0f,
                        (float)bkgCol[2] / 255.0f,
                        1.0f);

        //we clear background
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    }

    zoom();

    //then, the modelview matrix
    glMatrixMode(GL_MODELVIEW);

    static GLfloat trans44[16], rot44[16], tmp[16];
    m33_to_m44( g_rotationMatrix, rot44 );
    setTranslate( 0.f, 0.f, 0.f, trans44 );

    mult( trans44, rot44, tmp );
    transpose( tmp, g_glMatrix );
    glLoadMatrixf( g_glMatrix );

    update();
}

void GLWidget::setStandardOrthoCenter()
{
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    float halfW = float(m_glWidth)*0.5;
    float halfH = float(m_glHeight)*0.5;
    float maxS = ElMax(halfW,halfH);
    glOrtho(-halfW,halfW,-halfH,halfH,-maxS,maxS);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
}

void GLWidget::setFontPointSize(int pixelSize)
{
    m_font.setPointSize(pixelSize);
}

int GLWidget::getFontPointSize() const
{
    return m_font.pointSize();
}

void GLWidget::zoom()
{
    GLdouble zoom = m_params.zoom;
    GLdouble fAspect = (GLdouble) m_glWidth/ m_glHeight;

    GLdouble left   = -zoom*fAspect;
    GLdouble right  =  zoom*fAspect;

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();

    glOrtho(left, right, -zoom, zoom, -zoom, zoom);
}

void GLWidget::setInteractionMode(INTERACTION_MODE mode)
{
    m_interactionMode = mode;
}

void GLWidget::setView(MM_VIEW_ORIENTATION orientation)
{
    makeCurrent();

    GLdouble eye[3] = {0.0, 0.0, 0.0};
    GLdouble top[3] = {0.0, 0.0, 0.0};

    //we look at (0,0,0) by default
    switch (orientation)
    {
    case MM_TOP_VIEW:
        eye[2] = 1.0;
        top[1] = 1.0;
        break;
    case MM_BOTTOM_VIEW:
        eye[2] = -1.0;
        top[1] = -1.0;
        break;
    case MM_FRONT_VIEW:
        eye[1] = -1.0;
        top[2] = 1.0;
        break;
    case MM_BACK_VIEW:
        eye[1] = 1.0;
        top[2] = 1.0;
        break;
    case MM_LEFT_VIEW:
        eye[0] = -1.0;
        top[2] = 1.0;
        break;
    case MM_RIGHT_VIEW:
        eye[0] = 1.0;
        top[2] = 1.0;
        break;
    case MM_ISO_VIEW_1:
        eye[0] = -1.0;
        eye[1] = -1.0;
        eye[2] = 1.0;
        top[0] = 1.0;
        top[1] = 1.0;
        top[2] = 1.0;
        break;
    case MM_ISO_VIEW_2:
        eye[0] = 1.0;
        eye[1] = 1.0;
        eye[2] = 1.0;
        top[0] = -1.0;
        top[1] = -1.0;
        top[2] = 1.0;
        break;
    }

    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();
    gluLookAt(eye[0],eye[1],eye[2],0.0,0.0,0.0,top[0],top[1],top[2]);
    glPopMatrix();

    updateGL();
}

void GLWidget::onWheelEvent(float wheelDelta_deg)
{
    //convert degrees in zoom 'power'
    static const float c_defaultDeg2Zoom = 20.0f;
    float zoomFactor = pow(1.1f,wheelDelta_deg / c_defaultDeg2Zoom);
    updateZoom(zoomFactor);

    updateGL();
}

void GLWidget::updateZoom(float zoomFactor)
{
    if (zoomFactor>0.0 && zoomFactor!=1.0)
        setZoom(m_params.zoom*zoomFactor);
}

void GLWidget::setZoom(float value)
{
    if (value < GL_MIN_ZOOM_RATIO)
        value = GL_MIN_ZOOM_RATIO;
    else if (value > GL_MAX_ZOOM_RATIO)
        value = GL_MAX_ZOOM_RATIO;

    if (m_params.zoom != value)
    {
        m_params.zoom = value;
    }
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

    if (m_interactionMode == SEGMENT_POINTS)
    {
        int sz = m_polygon.size();

        if (sz <2)
           return;
        else if (sz == 2)
            m_polygon.push_back(event->pos());
        else
            //we replace last point by the current one
            m_polygon[sz-1] = event->pos();

        updateGL();

        //event->ignore();
        return;
    }

    if ( g_mouseLeftDown )
    {
        int dx = event->x()-g_mouseOldX,
            dy = event->y()-g_mouseOldY;

        g_mouseOldX = event->x();
        g_mouseOldY = event->y();

        setRotateOx_m33( ( g_trackballScale*dy )/m_glHeight, g_rotationOx );
        setRotateOy_m33( ( g_trackballScale*dx )/m_glWidth, g_rotationOy );
        mult_m33( g_rotationOx, g_rotationMatrix, g_tmpMatix );
        mult_m33( g_rotationOy, g_tmpMatix, g_rotationMatrix );
        inverse_m33( g_rotationMatrix, g_inverseRotationMatrix );

        GLfloat minv[9];
        mult_m33( g_rotationMatrix, g_inverseRotationMatrix, minv );

        updateGL();
    }
}

//Polyline objects are considered as 2D polylines !
bool isPointInsidePoly(const QPoint& P, const QVector < QPoint > poly)
{
    //nombre de sommets
    unsigned vertices=poly.size();
    if (vertices<2)
        return false;

    bool inside = false;

    QPoint A = poly[0];
    for (unsigned i=1;i<=vertices;++i)
    {
        QPoint B = poly[i%vertices];

        //Point Inclusion in Polygon Test (inspired from W. Randolph Franklin - WRF)
        if (((B.y()<=P.y()) && (P.y()<A.y())) ||
                ((A.y()<=P.y()) && (P.y()<B.y())))
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

void GLWidget::segment(bool inside)
{
    if (m_polygon.size() < 2)
    {
        //displayNewMessage("No polyline defined!",UPPER_CENTER_MESSAGE, MANUAL_SEGMENTATION_MESSAGE);
        return;
    }

    //viewing parameters
    double MM[16], MP[16];

    glMatrixMode(GL_MODELVIEW);
    glGetDoublev(GL_PROJECTION_MATRIX, (GLdouble*) &MM);

    glMatrixMode(GL_PROJECTION);
    glGetDoublev(GL_MODELVIEW_MATRIX, (GLdouble*) &MP);

    const float half_w = (float)m_glWidth * 0.5f;
    const float half_h = (float)m_glHeight * 0.5f;

    int VP[4];
    makeCurrent();
    glGetIntegerv(GL_VIEWPORT, VP);

    QPoint P2D;
    bool pointInside;

    int cpt_inside = 0;
    int cpt_outside = 0;
    int cpt_total = 0;
    for (int aK=0; aK < m_ply.size(); ++aK)
    {
        Cloud_::Cloud a_cloud = m_ply[aK];

        cpt_total += a_cloud.getVertexNumber();

        for (int bK=0; bK < a_cloud.getVertexNumber();++bK)
        {
            Cloud_::Vertex P = a_cloud.getVertex( bK );  //attention constructeur par copie absent ?

            GLdouble xp,yp,zp;
            gluProject(P.x(),P.y(),P.z(),MM,MP,VP,&xp,&yp,&zp);
            P2D.setX(xp - half_w);
            P2D.setY(yp - half_h);

            pointInside = isPointInsidePoly(P2D,m_polygon);

            if ((inside && !pointInside)||(!inside && pointInside))
                cpt_inside++;
            else
                cpt_outside++;

        }
    }

    cout << " nombre de points a l'interieur: " << cpt_inside <<endl;
    cout << " nombre de points a l'exterieur: " << cpt_outside <<endl;
    cout << " nombre de points total: " << cpt_inside <<endl;

   /* const double* MM = m_associatedWin->getModelViewMatd(); //viewMat
    const double* MP = m_associatedWin->getProjectionMatd(); //projMat
    const float half_w = (float)m_associatedWin->width() * 0.5f;
    const float half_h = (float)m_associatedWin->height() * 0.5f;

    int VP[4];
    m_associatedWin->getViewportArray(VP);

    CCVector3 P;
    CCVector2 P2D;
    bool pointInside;

    //for each selected entity
    for (ccHObject::Container::iterator p=m_toSegment.begin();p!=m_toSegment.end();++p)
    {
        ccGenericPointCloud* cloud = ccHObjectCaster::ToGenericPointCloud(*p);
        assert(cloud);

        ccGenericPointCloud::VisibilityTableType* vis = cloud->getTheVisibilityArray();
        assert(vis);

        unsigned i,cloudSize = cloud->size();

        //we project each point and we check if it falls inside the segmentation polyline
        for (i=0;i<cloudSize;++i)
        {
            cloud->getPoint(i,P);

            GLdouble xp,yp,zp;
            gluProject(P.x,P.y,P.z,MM,MP,VP,&xp,&yp,&zp);
            P2D.x = xp - half_w;
            P2D.y = yp - half_h;

            pointInside = CCLib::ManualSegmentationTools::isPointInsidePoly(P2D,m_segmentationPoly);

            if ((inside && !pointInside)||(!inside && pointInside))
                vis->setValue(i,0); //hiddenValue=0
        }
    }

    m_somethingHasChanged = true;
    validButton->setEnabled(true);
    validAndDeleteButton->setEnabled(true);
    razButton->setEnabled(true);
    pauseSegmentationMode(true); */
}
