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

#include "../../include/StdAfx.h"

const GLfloat g_trackballScale = 1.f;

//Min and max zoom ratio (relative)
const float GL_MAX_ZOOM_RATIO = 1.0e6f;
const float GL_MIN_ZOOM_RATIO = 1.0e-6f;

using namespace Cloud_;

ViewportParameters::ViewportParameters()
    : zoom(1.0f)
    , PointSize(1.0f)
    , LineWidth(1.0f)
{}

ViewportParameters::ViewportParameters(const ViewportParameters& params)
    : zoom(params.zoom)
    , PointSize(params.PointSize)
    , LineWidth(params.LineWidth)
{}

bool g_mouseLeftDown = false;
bool g_mouseRightDown = false;
GLfloat g_tmpMatrix[9],
        g_rotationOx[9],
        g_rotationOy[9],
        g_rotationMatrix[9] = { 1, 0, 0,
                               0, 1, 0,
                               0, 0, 1 },
        g_translationMatrix[3] = { 0, 0, 0 },
       /* g_inverseRotationMatrix[9] = { 1, 0, 0,
                                       0, 1, 0,
                                       0, 0, 1 },*/
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

GLWidget::GLWidget(QWidget *parent) : QGLWidget(parent)
      , m_interactionMode(TRANSFORM_CAMERA)
      , m_font(font())
      , m_bCloudLoaded(false)
      , m_params(ViewportParameters())
      , m_bPolyIsClosed(false)
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

    //glHint( GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST );
    glHint( GL_PERSPECTIVE_CORRECTION_HINT, GL_FASTEST );

    m_bInitialized = true;
}

void GLWidget::resizeGL(int width, int height)
{
    if (width==0 || height==0) return;

    m_glWidth  = (float)width;
    m_glHeight = (float)height;

    glViewport( 0, 0, width, height );
}

void GLWidget::paintGL()
{
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glMatrixMode(GL_MODELVIEW);

    draw3D();

    glPointSize(m_params.PointSize);

    glBegin(GL_POINTS);
    for(int aK=0; aK< m_ply.size(); aK++)
    {
        for(int bK=0; bK< m_ply[aK].size(); bK++)
        {
            Vertex vert = m_ply[aK].getVertex(bK);
            if (vert.isVisible())
            {
                qglColor( vert.getColor() );
                glVertex3f( vert.x(), vert.y(), vert.z() );
            }
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

        glBegin(m_bPolyIsClosed ? GL_LINE_LOOP : GL_LINE_STRIP);
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
    m_lastPos = event->pos();

    if ( event->buttons()&Qt::LeftButton )
    {
        g_mouseLeftDown = true;

        if ((m_interactionMode == SEGMENT_POINTS) && !m_bPolyIsClosed )
        {
            if (m_polygon.size() < 2)
                m_polygon.push_back(m_lastPos);
            else
            {
                m_polygon[m_polygon.size()-1] = m_lastPos;
                m_polygon.push_back(m_lastPos);
            }
        }
    }
    else if (event->buttons()&Qt::RightButton)
    {
        if (m_interactionMode == TRANSFORM_CAMERA)
            g_mouseRightDown = true;
        else
            closePolyline();
    }
}

void GLWidget::mouseReleaseEvent(QMouseEvent *event)
{
    if ( !( event->buttons()&Qt::LeftButton ) )
        g_mouseLeftDown = false;
    if ( !( event->buttons()&Qt::RightButton ) )
        g_mouseRightDown = false;
}

void GLWidget::keyPressEvent(QKeyEvent* event)
{
    switch(event->key())
    {
    case Qt::Key_Escape:
        clearPolyline();
        break;
    case Qt::Key_Space:
        segment(true);
        break;
    case Qt::Key_Delete:
        segment(false);
        break;
    case Qt::Key_Plus:
        ptSizeUp(true);
        break;
    case Qt::Key_Minus:
        ptSizeUp(false);
        break;
    case Qt::Key_F5:
        clearPolyline();
        break;
    default:
        event->ignore();
        break;
    }
}

void GLWidget::addPly( const QString &i_ply_file )
{
    Cloud a_ply, a_res;
    a_ply.loadPly( i_ply_file.toStdString() );

    //compute bounding box
    int nbPts = a_ply.size();
    for (int aK=0; aK < nbPts; ++aK)
    {
        Vertex vert = a_ply.getVertex(aK);

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

    m_diam = max(m_maxX-m_minX, max(m_maxY-m_minY, m_maxZ-m_minZ));

    //center and scale cloud
    Vector3 pt3d;
    for (int aK=0; aK < nbPts; ++aK)
    {
        Vertex vert = a_ply.getVertex(aK);
        Vertex vert_res = vert;

        pt3d.x = (vert.x() - m_cX) / m_diam;
        pt3d.y = (vert.y() - m_cY) / m_diam;
        pt3d.z = (vert.z() - m_cZ) / m_diam;

        vert_res.setCoord(pt3d);
        vert_res.setColor(vert.getColor());

        a_res.addVertex(vert_res);
    }

    a_res.setTranslation(Vector3(m_cX, m_cY, m_cZ));
    a_res.setScale((float) m_diam);

    //translate and scale back clouds if needed
    for (int aK=0; aK< m_ply.size();++aK)
    {
        if (m_ply[aK].getScale()) //cloud has been scaled
        {
            Vector3 translation = m_ply[aK].getTranslation();
            float scale = m_ply[aK].getScale();

            for (int bK=0; bK < m_ply[aK].size();++bK)
            {
                Vertex vert = m_ply[aK].getVertex(bK);

                pt3d.x = ((vert.x() * scale + translation.x) - m_cX) / m_diam;
                pt3d.y = ((vert.y() * scale + translation.y) - m_cY) / m_diam;
                pt3d.z = ((vert.z() * scale + translation.z) - m_cZ) / m_diam;

                vert.setCoord(pt3d);

                m_ply[aK].setVertex(bK, vert);
            }

            m_ply[aK].setTranslation(Vector3(m_cX, m_cY, m_cZ));
            m_ply[aK].setScale((float) m_diam);
        }
    }

    m_ply.push_back(a_res);

    if (!hasCloudLoaded()) setCloudLoaded(true);

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
    //user-defined background color for gradient start
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

    glPointSize(m_params.PointSize);
    glLineWidth(m_params.LineWidth);

    //gradient color background
    drawGradientBackground();
    //we clear background
    glClear(GL_DEPTH_BUFFER_BIT);

    zoom();

    //then, the modelview matrix
    glMatrixMode(GL_MODELVIEW);

    static GLfloat trans44[16], rot44[16], tmp[16];
    m33_to_m44( g_rotationMatrix, rot44 );
    setTranslate_m3( g_translationMatrix, trans44 );

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
    GLdouble s[3]   = {0.0, 0.0, 0.0};
    GLdouble u[3]   = {0.0, 0.0, 0.0};

    switch (orientation)
    {
    case MM_TOP_VIEW:
        eye[2] = -1.0;
        top[1] =  1.0;
        break;
    case MM_BOTTOM_VIEW:
        eye[2] =  1.0;
        top[1] = -1.0;
        break;
    case MM_FRONT_VIEW:
        eye[1] = 1.0;
        top[2] = 1.0;
        break;
    case MM_BACK_VIEW:
        eye[1] = -1.0;
        top[2] =  1.0;
        break;
    case MM_LEFT_VIEW:
        eye[0] = 1.0;
        top[2] = 1.0;
        break;
    case MM_RIGHT_VIEW:
        eye[0] = -1.0;
        top[2] =  1.0;
        break;
    case MM_ISO_VIEW_1:
        eye[0] =  1.0;
        eye[1] =  1.0;
        eye[2] = -1.0;
        top[0] =  1.0;
        top[1] =  1.0;
        top[2] =  1.0;
        normalize(eye);
        normalize(top);
        break;
    case MM_ISO_VIEW_2:
        eye[0] = -1.0;
        eye[1] = -1.0;
        eye[2] = -1.0;
        top[0] = -1.0;
        top[1] = -1.0;
        top[2] =  1.0;
        normalize(eye);
        normalize(top);
        break;
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


    g_translationMatrix[0] = m_cX;
    g_translationMatrix[1] = m_cY;
    g_translationMatrix[2] = m_cZ;

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

    if ((m_interactionMode == SEGMENT_POINTS) )
    {
        if(!m_bPolyIsClosed)
        {
            int sz = m_polygon.size();

            if (sz == 0)
               return;
            else if (sz == 1)
                m_polygon.push_back(event->pos());
            else
                //we replace last point by the current one
                m_polygon[sz-1] = event->pos();

            updateGL();
        }

        event->ignore();
        return;
    }
    else
    {
        QPoint dp = event->pos()-m_lastPos;

        m_lastPos = event->pos();

        if ( g_mouseLeftDown )
        {
            setRotateOx_m33( ( g_trackballScale*dp.y() )/m_glHeight, g_rotationOx );
            setRotateOy_m33( ( g_trackballScale*dp.x() )/m_glWidth, g_rotationOy );

            mult_m33( g_rotationOx, g_rotationMatrix, g_tmpMatrix );
            mult_m33( g_rotationOy, g_tmpMatrix, g_rotationMatrix );
        }
        else if ( g_mouseRightDown )
        {
           g_translationMatrix[0] = m_cX - dp.x()/m_glWidth;
           g_translationMatrix[1] = m_cY - dp.y()/m_glHeight;
        }

        updateGL();
    }
}

bool isPointInsidePoly(const QPoint& P, const QVector < QPoint > poly)
{
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
        return;

    //viewing parameters
    double MM[16], MP[16];
    int VP[4];

    glMatrixMode(GL_MODELVIEW);
    glGetDoublev(GL_MODELVIEW_MATRIX, (GLdouble*) &MM);

    glMatrixMode(GL_PROJECTION);
    glGetDoublev(GL_PROJECTION_MATRIX, (GLdouble*) &MP);

    //makeCurrent();
    glGetIntegerv(GL_VIEWPORT, VP);

    QPoint P2D;
    bool pointInside;

    QVector < QPoint > polyg;
    for (int aK=0; aK < m_polygon.size(); ++aK)
    {
        polyg.push_back(QPoint(m_polygon[aK].x(), m_glHeight - m_polygon[aK].y()));
    }

    for (int aK=0; aK < m_ply.size(); ++aK)
    {
        Cloud_::Cloud a_cloud = m_ply[aK];

        for (int bK=0; bK < a_cloud.size();++bK)
        {
            Cloud_::Vertex P = a_cloud.getVertex( bK );

            if (P.isVisible())
            {
                GLdouble xp,yp,zp;
                gluProject(P.x(),P.y(),P.z(),MM,MP,VP,&xp,&yp,&zp);

                P2D.setX(xp);
                P2D.setY(yp);

                pointInside = isPointInsidePoly(P2D,polyg);

                if ((inside && !pointInside)||(!inside && pointInside))
                    m_ply[aK].getVertex(bK).setVisible(false);
            }
        }
    }
}

void GLWidget::clearPolyline()
{
    m_polygon.clear();
    m_bPolyIsClosed = false;
}

void GLWidget::closePolyline()
{
    //remove last point if needed
    int sz = m_polygon.size();
    if ((sz > 2) &&  (m_polygon[sz-1] == m_polygon[sz-2]))
        m_polygon.resize(sz-1);

    m_bPolyIsClosed = true;
}

void GLWidget::undoAll()
{
    clearPolyline();

    for (int aK=0; aK < m_ply.size(); ++aK)
    {
        for (int bK=0; bK < m_ply[aK].size();++bK)
        {
            m_ply[aK].getVertex(bK).setVisible(true);
        }
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
}
