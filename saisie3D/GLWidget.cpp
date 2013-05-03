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
    , defaultPointSize(1)
    , defaultLineWidth(1)
    , objectCenteredView(true)
    , pivotPoint(0.0f)
    , m_rot(0)
    , m_trans(QVector<GLdouble>(3,0))
    , m_scale(1)

{
    //viewMat.toIdentity();
}

ViewportParameters::ViewportParameters(const ViewportParameters& params)
    : pixelSize(params.pixelSize)
    , zoom(params.zoom)
    //, viewMat(params.viewMat)
    , defaultPointSize(params.defaultPointSize)
    , defaultLineWidth(params.defaultLineWidth)
    , objectCenteredView(params.objectCenteredView)
    , pivotPoint(params.pivotPoint)
    , m_rot(params.m_rot)
    , m_trans(params.m_trans)
    , m_scale(params.m_scale)
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

inline void m33_to_gl( const float i_m[9], GLfloat o_m[16] )
{
    o_m[0]=i_m[0];		o_m[1]=i_m[3];		o_m[2]=i_m[6];		o_m[3]=0.f;
    o_m[4]=i_m[1];		o_m[5]=i_m[4];		o_m[6]=i_m[7];		o_m[7]=0.f;
    o_m[8]=i_m[2];		o_m[9]=i_m[5];		o_m[10]=i_m[8];		o_m[11]=0.f;
    o_m[12]=0.f;		o_m[13]=0.f;		o_m[14]=0.f;		o_m[15]=1.f;
}

inline void setRotateOx( const float i_angle, GLfloat *o_m )
{
    GLfloat co = (GLfloat)cos( i_angle ),
            si = (GLfloat)sin( i_angle );
    o_m[0] =1.f;		o_m[1] =0;		o_m[2] =0;		o_m[3] =0;
    o_m[4] =0.f;		o_m[5] =co;		o_m[6] =-si;	o_m[7] =0;
    o_m[8] =0.f;		o_m[9] =si;		o_m[10]=co;		o_m[11]=0;
    o_m[12]=0.f;		o_m[13]=0;		o_m[14]=0;		o_m[15]=1;
}

inline void setIdentity( GLfloat o_m[16] )
{
    o_m[0] =1.f;	o_m[1] =0.f;	o_m[2] =0.f;	o_m[3] =0.f;
    o_m[4] =0.f;	o_m[5] =1.f;	o_m[6] =0.f;	o_m[7] =0.f;
    o_m[8] =0.f;	o_m[9] =0.f;	o_m[10]=1.f;	o_m[11]=0.f;
    o_m[12]=0.f;	o_m[13]=0.f;	o_m[14]=0.f;	o_m[15]=1.f;
}

inline void setRotateOx_m33( const float i_angle, GLfloat o_m[9] )
{
    GLfloat co = (GLfloat)cos( i_angle ),
            si = (GLfloat)sin( i_angle );
    o_m[0]=1.f;		o_m[1]=0.f;		o_m[2]=0.f;
    o_m[3]=0.f;		o_m[4]=co;		o_m[5]=-si;
    o_m[6]=0.f;		o_m[7]=si;		o_m[8]=co;
}

inline void setRotateOx_t( const float i_angle, GLfloat *o_m )
{
    GLfloat co = (GLfloat)cos( i_angle ),
            si = (GLfloat)sin( i_angle );
    o_m[0]=0;		o_m[4]=0;		o_m[8]=0;		o_m[12]=0;
    o_m[1]=0;		o_m[5]=co;		o_m[9]=-si;		o_m[13]=0;
    o_m[2]=0;		o_m[6]=si;		o_m[10]=co;		o_m[14]=0;
    o_m[3]=0;		o_m[7]=0;		o_m[11]=0;		o_m[15]=1;
}

inline void setRotateOy( const float i_angle, GLfloat *o_m )
{
    GLfloat co = (GLfloat)cos( i_angle ),
            si = (GLfloat)sin( i_angle );
    o_m[0]=co;		o_m[1]=0;		o_m[2]=si;		o_m[3]=0;
    o_m[4]=0;		o_m[5]=1;		o_m[6]=0;		o_m[7]=0;
    o_m[8]=-si;		o_m[9]=0;		o_m[10]=co;		o_m[11]=0;
    o_m[12]=0;		o_m[13]=0;		o_m[14]=0;		o_m[15]=1;
}

inline void setRotateOz( const float i_angle, GLfloat *o_m )
{
    GLfloat co = (GLfloat)cos( i_angle ),
            si = (GLfloat)sin( i_angle );
    o_m[0] =co;		o_m[1] =-si;	o_m[2] =0.f;	o_m[3] =0.f;
    o_m[4] =si;		o_m[5] =co;		o_m[6] =0.f;	o_m[7] =0.f;
    o_m[8] =0.f;	o_m[9] =0;		o_m[10]=1.f;	o_m[11]=0.f;
    o_m[12]=0.f;	o_m[13]=0;		o_m[14]=0.f;	o_m[15]=1.f;
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
    o_m[0]=co;		o_m[1]=-si;		o_m[2]=0;
    o_m[3]=si;		o_m[4]=co;		o_m[5]=0;
    o_m[6]=0;		o_m[7]=0;		o_m[8]=1.f;
}

inline void setRotateOy_t( const float i_angle, GLfloat *o_m )
{
    GLfloat co = (GLfloat)cos( i_angle ),
            si = (GLfloat)sin( i_angle );
    o_m[0]=co;		o_m[4]=0;		o_m[8]=si;		o_m[12]=0;
    o_m[1]=0;		o_m[5]=1;		o_m[9]=0;		o_m[13]=0;
    o_m[2]=-si;		o_m[6]=0;		o_m[10]=co;		o_m[14]=0;
    o_m[3]=0;		o_m[7]=0;		o_m[11]=0;		o_m[15]=1;
}


inline void setRotate( const float i_angle, const float i_x, const float i_y, const float i_z, GLfloat o_m[16] )
{
    GLfloat co = (GLfloat)cos( i_angle ),
            si = (GLfloat)sin( i_angle );
    o_m[0]=i_x*i_x+(1-i_x*i_x)*co;		o_m[1]=i_x*i_y*(1-co)-i_x*si;		o_m[2]=i_x*i_z*(1-co)+i_z*si;		o_m[3]=0;
    o_m[4]=i_x*i_y*(1-co)+i_z*si;		o_m[5]=i_y*i_y+(1-i_y*i_y)*co;		o_m[6]=i_y*i_z*(1-co)-i_x*si;		o_m[7]=0;
    o_m[8]=i_x*i_z*(1-co)-i_y*si;		o_m[9]=i_y*i_z*(1-co)+i_x*si;		o_m[10]=i_z*i_z+(1-i_z*i_z)*co;		o_m[11]=0;
    o_m[12]=0;							o_m[13]=0;							o_m[14]=0;							o_m[15]=1;
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

inline void mult_m44_v( const GLfloat i_m[16], const GLfloat i_v[4], GLfloat o_v[4] )
 {
     o_v[0] = i_m[0]*i_v[0]+i_m[1]*i_v[1]+i_m[2]*i_v[2]+i_m[3]*i_v[3];
     o_v[1] = i_m[4]*i_v[0]+i_m[5]*i_v[1]+i_m[5]*i_v[2]+i_m[7]*i_v[3];
     o_v[2] = i_m[8]*i_v[0]+i_m[9]*i_v[1]+i_m[10]*i_v[2]+i_m[11]*i_v[3];
     o_v[3] = i_m[12]*i_v[0]+i_m[13]*i_v[1]+i_m[14]*i_v[2]+i_m[15]*i_v[3];
 }

inline void mult_m33( const GLfloat i_a[9], const GLfloat i_b[9], GLfloat o_m[9] )
 {
     o_m[0]=i_a[0]*i_b[0]+i_a[1]*i_b[3]+i_a[2]*i_b[6];		o_m[1]=i_a[0]*i_b[1]+i_a[1]*i_b[4]+i_a[2]*i_b[7];		o_m[2]=i_a[0]*i_b[2]+i_a[1]*i_b[5]+i_a[2]*i_b[8];
     o_m[3]=i_a[3]*i_b[0]+i_a[4]*i_b[3]+i_a[5]*i_b[6];		o_m[4]=i_a[3]*i_b[1]+i_a[4]*i_b[4]+i_a[5]*i_b[7];		o_m[5]=i_a[3]*i_b[2]+i_a[4]*i_b[5]+i_a[5]*i_b[8];
     o_m[6]=i_a[6]*i_b[0]+i_a[7]*i_b[3]+i_a[8]*i_b[6];		o_m[7]=i_a[6]*i_b[1]+i_a[7]*i_b[4]+i_a[8]*i_b[7];		o_m[8]=i_a[6]*i_b[2]+i_a[7]*i_b[5]+i_a[8]*i_b[8];
 }

inline void mult_m33_v( const GLfloat i_m[9], const GLfloat i_v[3], GLfloat o_vv[3] )
{
    o_vv[0]=i_m[0]*i_v[0]+i_m[1]*i_v[1]+i_m[2]*i_v[2];
    o_vv[1]=i_m[3]*i_v[0]+i_m[4]*i_v[1]+i_m[5]*i_v[2];
    o_vv[2]=i_m[6]*i_v[0]+i_m[7]*i_v[1]+i_m[8]*i_v[2];
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
     // , m_glWidth(0)
     // , m_glHeight(0)
      , m_interactionMode(TRANSFORM_CAMERA)
      , m_font(font())
      , m_bCloudLoaded(false)
      , m_validModelviewMatrix(false)
      , m_updateFBO(true)
      , m_bFitCloud(true)
      , m_params(ViewportParameters())
{
    m_minX = m_minY = m_minZ = FLT_MAX;
    m_maxX = m_maxY = m_maxZ = FLT_MIN;
    m_cX = m_cY = m_cZ = m_diam = 0.f;

    setCloudLoaded(false);
    setMouseTracking(true);

    //drag & drop handling
    setAcceptDrops(true);
}

void GLWidget::initializeGL()
{
    if (m_initialized)
        return;

    glShadeModel( GL_SMOOTH );
   /* glClearColor( mmColor::defaultBkgColor[0] / 255.0f,
                  mmColor::defaultBkgColor[1] / 255.0f,
                  mmColor::defaultBkgColor[2] / 255.0f,
                  1.f );*/

    if (m_params.getRot()==0)
    {
        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();
        m_params.modifRot() = new GLdouble[16];
        glGetDoublev(GL_MODELVIEW_MATRIX, m_params.modifRot());
    }

    m_boule = makeBoule();

    //glClearDepth( 1.f );
    glEnable( GL_DEPTH_TEST );
    glDepthFunc( GL_LEQUAL );

    //transparency off by default
    glDisable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    glHint( GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST );



    m_initialized = true;
}

void GLWidget::resizeGL(int width, int height)
{
    if (width==0 || height==0) return;

    m_glWidth  = (float)width;
    m_glHeight = (float)height;

    glViewport( 0, 0, width, height );

    int h = qMin(double(height), double(width)*(m_maxY-m_minY)/(m_maxX-m_minX));
    int w = h*(m_maxX-m_minX)/(m_maxY-m_minY);

    //glViewport((width - w) / 2, (height - h) / 2, w, h);

    winZ = w * m_minZ / (m_maxX-m_minX);

   /* glMatrixMode( GL_PROJECTION );
    glLoadIdentity();
    gluPerspective( 45.f, (GLfloat)w/(GLfloat)h, 0.1f, 100.f );*/
}

void GLWidget::getContext(glDrawContext& context)
{
    //display size
    context.glW = m_glWidth;
    context.glH = m_glHeight;
    //context._win = this;
    context.flags = 0;

    const mmGui::ParamStruct& guiParams = mmGui::Parameters();

    //decimation options
    context.decimateCloudOnMove = guiParams.decimateCloudOnMove;

    //scalar field colorbar
    //context.sfColorScaleToDisplay = 0;

    //text display
    context.dispNumberPrecision = guiParams.displayedNumPrecision;
    context.labelsTransparency = guiParams.labelsTransparency;

    //default materials
    /*context.defaultMat.name = "default";
    memcpy(context.defaultMat.diffuseFront,guiParams.meshFrontDiff,sizeof(float)*4);
    memcpy(context.defaultMat.diffuseBack,guiParams.meshBackDiff,sizeof(float)*4);
    memcpy(context.defaultMat.ambient,ccColor::bright,sizeof(float)*4);
    memcpy(context.defaultMat.specular,guiParams.meshSpecular,sizeof(float)*4);
    memcpy(context.defaultMat.emission,ccColor::night,sizeof(float)*4);
    context.defaultMat.shininessFront = 30;
    context.defaultMat.shininessBack = 50;*/
    //default colors
    memcpy(context.pointsDefaultCol,guiParams.pointsDefaultCol,sizeof(unsigned char)*3);
    memcpy(context.textDefaultCol,guiParams.textDefaultCol,sizeof(unsigned char)*3);
    memcpy(context.labelDefaultCol,guiParams.labelCol,sizeof(unsigned char)*3);
    memcpy(context.bbDefaultCol,guiParams.bbDefaultCol,sizeof(unsigned char)*3);

    //default font size
    setFontPointSize(1);

    //VBO
    //context.vbo = m_vbo;
}

void GLWidget::paintGL() {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glMatrixMode(GL_MODELVIEW);

    //if (m_updateFBO)
    //{
        draw3D();
        m_updateFBO=false;
    //}

    glPointSize(2);

    glBegin(GL_POINTS);
    for(int aK=0; aK< m_ply.size(); aK++)
    {
        for(int bK=0; bK< m_ply[aK].getVertexNumber(); bK++)
        {
            Cloud_::Vertex vert = m_ply[aK].getVertex(bK);
            qglColor( vert.m_color );
            //glColor3ub( vert.m_color.red(), vert.m_color.green(), vert.m_color.blue() );
            glVertex3f( vert.x(), vert.y(), vert.z() );
        }
    }
    glEnd();



    //current messages (if valid)
    if (!m_messagesToDisplay.empty())
    {
        //int currentTime_sec = ccTimer::Sec();
        //ccLog::Print(QString("[paintGL] Current time: %1 s.").arg(currentTime_sec));

        //if fbo --> override color
        //Some versions of Qt seem to need glColorf instead of glColorub! (see https://bugreports.qt-project.org/browse/QTBUG-6217)
        //glColor3ubv(m_fbo && m_activeGLFilter ? ccColor::black : textCol);
        //unsigned char* col =(m_fbo && m_activeGLFilter ? ccColor::black : textCol);
        //glColor3f((float)col[0]/(float)MAX_COLOR_COMP,(float)col[1]/(float)MAX_COLOR_COMP,(float)col[2]/(float)MAX_COLOR_COMP);
        glColor3f(1.f,1.f,1.f);

        std::list<MessageToDisplay>::iterator it = m_messagesToDisplay.begin();
        while (it != m_messagesToDisplay.end())
        {
            //no more valid? we delete the message
            /*if (it->messageValidity_sec < currentTime_sec)
            {
                it = m_messagesToDisplay.erase(it);
            }
            else
            {*/

                QFont newFont(m_font);
                newFont.setPointSize(12);
                QRect rect = QFontMetrics(newFont).boundingRect(it->message);
                //only one message supported in the screen center (for the moment ;)
                renderText((m_glWidth-rect.width())/2, (m_glHeight-rect.height())/2, it->message,newFont);

                ++it;
            //}
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
        glColor3f(1,0,0);

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
    }

    lastPos = event->pos();
}

void GLWidget::mouseReleaseEvent(QMouseEvent *event)
{
    if ( !( event->buttons()&Qt::LeftButton ) )
        g_mouseLeftDown = false;
}

/*void GLWidget::keyPressEvent(QKeyEvent* event) {
    switch(event->key()) {
    case Qt::Key_Escape:
        close();
        break;
    default:
        event->ignore();
        break;
    }
}*/

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

    m_diam = sqrt((m_cX-m_minX)*(m_cX-m_minX)+(m_cY-m_minY)*(m_cY-m_minY)+(m_cZ-m_minZ)*(m_cZ-m_minZ));
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

    /*QString filename("none");
    if (event->mimeData()->hasFormat("FileNameW"))
    {
    QByteArray data = event->mimeData()->data("FileNameW");
    filename = QString::fromUtf16((ushort*)data.data(), data.size() / 2);
    event->acceptProposedAction();
    }
    else if (event->mimeData()->hasFormat("FileName"))
    {
    filename = event->mimeData()->data("FileNameW");
    event->acceptProposedAction();
    }

    std::cout << QString("Drop file(s): %1").arg(filename));
    //*/

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
    mess.messageValidity_sec = 500; //TODO: Timer::Sec()+displayMaxDelay_sec;
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

    //if a FBO is activated
   // TODO
    /* if (fbo)
    {
        fbo->start();
        ccGLUtils::CatchGLError("glWidget::paintGL/FBO start");
    }*/

    setStandardOrthoCenter();
    glDisable(GL_DEPTH_TEST);

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

    /****************************************/
    /****  PASS: 2D/BACKGROUND/NO LIGHT  ****/
    /****************************************/
    /*context.flags = CC_DRAW_2D;
    if (m_interactionMode == TRANSFORM_ENTITY)
    context.flags |= CC_VIRTUAL_TRANS_ENABLED;

    //we draw 2D entities
    if (m_globalDBRoot)
    m_globalDBRoot->draw(context);
    m_winDBRoot->draw(context);
    //*/

    /****************************************/
    /****  PASS: 3D/BACKGROUND/NO LIGHT  ****/
    /****************************************/
    // TODO: context.flags = CC_DRAW_3D | CC_DRAW_FOREGROUND;
    // TODO:if (m_interactionMode == TRANSFORM_ENTITY)
    // TODO:    context.flags |= CC_VIRTUAL_TRANS_ENABLED;

    //glEnable(GL_DEPTH_TEST);

    /****************************************/
    /****    PASS: 3D/FOREGROUND/LIGHT   ****/
    /****************************************/
    //TODO: if (m_sunLightEnabled)
    //TODO:     context.flags |= CC_LIGHT_ENABLED;
    //TODO: if (m_lodActivated)
     //TODO:    context.flags |= CC_LOD_ACTIVATED;

    //we enable absolute sun light (if activated)
   //TODO:  if (m_sunLightEnabled)
        //glEnableSunLight();

    //we setup projection matrix
    glMatrixMode(GL_PROJECTION);
    glLoadMatrixd(getProjectionMatd());




    //then, the modelview matrix
    glMatrixMode(GL_MODELVIEW);

    static GLfloat trans44[16], rot44[16], tmp[16];
    m33_to_m44( g_rotationMatrix, rot44 );
    setTranslate( 0.f, 0.f, -7.f, trans44 );

    mult( trans44, rot44, tmp );
    transpose( tmp, g_glMatrix );
    glLoadMatrixf( g_glMatrix );

   //TODO: glLoadMatrixd(getModelViewMatd());



    //we draw 3D entities
    //TOTOTODODODO
    /*if (m_globalDBRoot)
    {
        m_globalDBRoot->draw(context);

    }
    m_winDBRoot->draw(context);*/

    if (m_bFitCloud && m_bCloudLoaded)
        zoomGlobal();
    else
        zoom();

    //for connected items
    emit drawing3D();

    //we disable shader
    //if (m_activeShader)
    //    m_activeShader->stop();

    //we disable lights
    //if (m_sunLightEnabled)
    glDisable(GL_LIGHT0);

    //we disable fbo
   /* TODO if (fbo)
    {
        fbo->stop();
        ccGLUtils::CatchGLError("glWidget::paintGL/FBO stop");
    } */

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

void GLWidget::glEnableSunLight()
{
    //glLightfv(GL_LIGHT0,GL_DIFFUSE,ccGui::Parameters().lightDiffuseColor);
    glLightfv(GL_LIGHT0, GL_DIFFUSE,  mmColor::dark);
    glLightfv(GL_LIGHT0, GL_AMBIENT,  mmColor::night);
    glLightfv(GL_LIGHT0, GL_SPECULAR, mmColor::middle);
    glLightfv(GL_LIGHT0, GL_POSITION, m_sunLightPos);
    glLightModelf(GL_LIGHT_MODEL_TWO_SIDE,GL_TRUE);
    glEnable(GL_LIGHT0);
}

void GLWidget::setFontPointSize(int pixelSize)
{
    m_font.setPointSize(pixelSize);
}

int GLWidget::getFontPointSize() const
{
    return m_font.pointSize();
}

void GLWidget::redraw()
{
    m_updateFBO=true;
    updateGL();
}

void GLWidget::zoomGlobal()
{
    GLdouble fAspect = (GLdouble) m_glWidth/ m_glHeight;

    GLdouble left   = m_cX - m_diam*fAspect;
    GLdouble right  = m_cX + m_diam*fAspect;
    GLdouble bottom = m_cY - m_diam;
    GLdouble top    = m_cY + m_diam;

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(left, right, bottom, top, -1.0, 1.0);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    /*static int prevent_loop = 0 ;

    if (glGetError() != GL_NO_ERROR)
    {
       if (!prevent_loop) {
           QMessageBox msgBox;
           QString text = "OpenGL error detected.\n\n";
           msgBox.setText(text);
           msgBox.exec();

          prevent_loop = 1 ;
       }
    }*/
}

void GLWidget::zoom()
{
    GLdouble zoom = m_params.zoom;
    GLdouble fAspect = (GLdouble) m_glWidth/ m_glHeight;

    GLdouble left   = m_cX - m_diam*fAspect*zoom;
    GLdouble right  = m_cX + m_diam*fAspect*zoom;
    GLdouble bottom = m_cY - m_diam*zoom;
    GLdouble top    = m_cY + m_diam*zoom;

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();

    glOrtho(left, right, bottom, top, -zoom, zoom);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
}

void GLWidget::setView(MM_VIEW_ORIENTATION orientation, bool forceRedraw/*=true*/)
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

    //ccGLMatrix result;

    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();
    gluLookAt(eye[0],eye[1],eye[2],0.0,0.0,0.0,top[0],top[1],top[2]);
    //glGetFloatv(GL_MODELVIEW_MATRIX, result.data());
    //result.data()[14] = 0.0; //annoying value (?!)
    glPopMatrix();

    //return result;


    invalidateVisualization();

    //we emit the 'baseViewMatChanged' signal
    //emit baseViewMatChanged();

    if (forceRedraw)
        redraw();
}

void GLWidget::invalidateVisualization()
{
    m_validModelviewMatrix=false;
    m_updateFBO=true;
}

/*const double* GLWidget::getModelViewMatd()
{
    if (!m_validModelviewMatrix)
        recalcModelViewMatrix();

    return m_viewMatd;
}*/

void GLWidget::onWheelEvent(float wheelDelta_deg)
{
    m_bFitCloud = false;

    //convert degrees in zoom 'power'
    static const float c_defaultDeg2Zoom = 20.0f;
    float zoomFactor = pow(1.1f,wheelDelta_deg / c_defaultDeg2Zoom);
    updateZoom(zoomFactor);

    redraw();
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
        invalidateViewport();
        invalidateVisualization();
    }
}

void GLWidget::invalidateViewport()
{
    //m_validProjectionMatrix=false;
    m_updateFBO=true;
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

const double* GLWidget::getProjectionMatd()
{
    if (!m_validProjectionMatrix)
        recalcProjectionMatrix();

    return m_projMatd;
}

void GLWidget::recalcModelViewMatrix()
{
    makeCurrent();

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    //apply zoom
    float totalZoom = m_params.zoom / m_params.pixelSize;
    glScalef(totalZoom,totalZoom,totalZoom);

 /*   if (m_params.objectCenteredView)
    {
        //place origin on camera center
        glTranslatef(-m_params.cameraCenter.x, -m_params.cameraCenter.y, -m_params.cameraCenter.z);

        //go back to initial origin
        glTranslatef(m_params.pivotPoint.x, m_params.pivotPoint.y, m_params.pivotPoint.z);

        //rotation (viewMat is simply a rotation matrix around the pivot here!)
        glMultMatrixf(m_params.viewMat.data());

        //place origin on pivot point
        glTranslatef(-m_params.pivotPoint.x, -m_params.pivotPoint.y, -m_params.pivotPoint.z);
    }
    else
    {
        //rotation (viewMat is the rotation around the camera center here - no pivot)
        glMultMatrixf(m_params.viewMat.data());

        //place origin on camera center
        glTranslatef(-m_params.cameraCenter.x, -m_params.cameraCenter.y, -m_params.cameraCenter.z);
    }*/

    //we save visualization matrix
    //glGetFloatv(GL_MODELVIEW_MATRIX, m_viewMat.data());
    glGetDoublev(GL_MODELVIEW_MATRIX, m_viewMatd);

    m_validModelviewMatrix=true;
}

void GLWidget::recalcProjectionMatrix()
{
    makeCurrent();

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();

    float bbHalfDiag = 1.0f;
    Vector3 bbCenter(0.0f);

    //compute center of viewed objects constellation

    //TODO !!!
    /*
    if (m_globalDBRoot)
    {
        //get whole bounding-box
        ccBBox box = m_globalDBRoot->getBB(true, true, this);
        if (box.isValid())
        {
            //get bbox center
            bbCenter = box.getCenter();
            //get half bbox diagonal length
            bbHalfDiag = box.getDiagNorm()*0.5;
        }
    }*/

    //virtual pivot point (i.e. to handle viewer-based mode smoothly)
    Vector3 pivotPoint = (m_params.objectCenteredView ? m_params.pivotPoint : bbCenter);

    //distance between camera and pivot point
    float CP = pivotPoint.norm();
    //distance between pivot point and DB farthest point
    float MP = (bbCenter-pivotPoint).norm() + bbHalfDiag;

    //max distance (camera to 'farthest' point)
    float maxDist = CP + MP;

    float maxDist_pix = maxDist / m_params.pixelSize * m_params.zoom;
    maxDist_pix = std::max<float>(maxDist_pix,1.0f);

    float halfW = static_cast<float>(m_glWidth)*0.5f;
    float halfH = static_cast<float>(m_glHeight)*0.5f;

    glOrtho(-halfW,halfW,-halfH,halfH,-maxDist_pix,maxDist_pix);

    //we save projection matrix
    //glGetFloatv(GL_PROJECTION_MATRIX, m_projMat.data());
    glGetDoublev(GL_PROJECTION_MATRIX, m_projMatd);

    m_validProjectionMatrix = true;
}

GLdouble*& ViewportParameters::modifRot() { return m_rot; }
QVector<GLdouble>& ViewportParameters::modifTrans() { return m_trans; }

const GLdouble* ViewportParameters::getRot() const { return m_rot; }
const QVector<GLdouble>& ViewportParameters::getTrans() const { return m_trans; }
const GLdouble& ViewportParameters::getScale() const { return m_scale; }

void GLWidget::glCircle3i(GLint radius, GLdouble * m)
{
    //dessine un cercle de rayon 1 et d'axe z
    //m : orientation finale du cercle pour dessiner l'arrière-plan en plus foncé
    //recherche des angles limites d'arrière-plan
    int precision = 100;
    double i1 = 0;
    double i2 = 0;
    double lim = (m[6]!=0)? -atan(-m[2]/m[6]) : PI/2.0;	//R31/R32
    if (m[6]>0)
    {
        i1 = lim*precision/2/PI;
        i2 = lim*precision/2/PI + 50;
    }
    else if (m[6]<0)
    {
        i1 = lim*precision/2/PI - 50;
        i2 = lim*precision/2/PI;
    }
    else if (m[6]==0 && -m[2]>0)
    {
        i1 = 25;
        i2 = 75;
    }
    else if (m[6]==0 && -m[2]<0)
    {
        i1 = -25;
        i2 = 25;
    }
    else if (m[6]==0 && m[2]==0)
    {
        i1 = 0;
        i2 = precision;
    }

    //dessin
    glDisable(GL_TEXTURE_2D);
    glLineWidth(3);
    glBegin(GL_LINE_LOOP);
    double angle;
    qglColor(QColor(155,200,255));
    for (int i=i1; i <i2; i++)
    {
        angle = i*2*PI/precision;
        glVertex2f(cos(angle)*radius, sin(angle)*radius);
    }
    if (m[6]!=0 || m[2]!=0)
    {	//(m[6]==0 && m[2]==0) => cercle d'axe z : tous les points sont visibles
        qglColor(QColor(78,100,128));
        if (i2>=100) i2 -= 100;
        else i1 += 100;	//i1<0
        for (int i=i2; i <i1; i++)
        {
            angle = i*2*PI/precision;
            glVertex2f(cos(angle)*radius, sin(angle)*radius);
        }
    }
    glEnd();
    glLineWidth(1);
}

void multTransposeMatrix(const GLdouble* m)
{
    GLdouble* m2 = new GLdouble[16];
    for (int i=0; i<4; ++i)
    for (int j=0; j<4; ++j)
        m2[4*i+j] = m[4*j+i];
    glMultMatrixd(m2);
    delete m2;
}

//récupération des signaux de la souris/////////////////////////////////////////
pair<QVector<double>,QVector<double> > GLWidget::getMouseDirection (const QPoint& P, GLdouble * matrice) const {
    //récupère la direction et le point d'origine d'un clic souris P en coordonnées chantier (selon matrice)
        //vecteur clic en coordonnées observateur
    GLint * view = new GLint[4];
    glGetIntegerv(GL_VIEWPORT, view);
    double taille = min(view[2], view[3]);
    double window_x = P.x()-view[0] - double(view[2])/2.0;
    double window_y = (view[3] - P.y()+view[1]) - double(view[3])/2.0;
        double x = m_maxX * window_x / (taille/2.0);
        double y = m_maxY * window_y / (taille/2.0);
        QVector<double> ray_pnt(4,0);
            ray_pnt[3] = 1;
        QVector<double> ray_vec(4,0);
            ray_vec[0] = x;
            ray_vec[1] = y;
            ray_vec[2] =  -m_minZ;
    delete view;

        QVector<double> ray_pnt2(4,0);
        QVector<double> ray_vec2(4,0);
        for (int i=0; i<4; i++) {
        for (int j=0; j<4; j++) {
                        ray_pnt2[i] += matrice[j*4+i] * ray_pnt.at(j);	//P2=n*P, n est selon les colonnes
                        ray_vec2[i] += matrice[j*4+i] * ray_vec.at(j);
        }
    }
    double norm_vec2 = 0;
    for (int i=0; i<3; i++)
                norm_vec2 += ray_vec2.at(i) * ray_vec2.at(i);
    norm_vec2 = sqrt(norm_vec2);
    for (int i=0; i<3; i++)
        ray_vec2[i] /= norm_vec2;

        return pair<QVector<double>,QVector<double> >(ray_pnt2,ray_vec2);
}

QVector<double> GLWidget::getSpherePoint(const QPoint& mouse) const
{
    //point sur la sphère pointé par la souris (ref de la sphère)
        //direction du pointeur
    GLdouble * m = new GLdouble[16];
    GLdouble sc = min(m_maxX-m_minX,m_maxY-m_minY)/m_minZ*(m_maxZ+m_minZ)/8.0;
    glPushMatrix();
    glLoadIdentity();
    glScaled(1.0/sc,1.0/sc,1.0/sc);
    multTransposeMatrix(m_params.getRot());
    glTranslated(0, 0, +(m_minZ+m_maxZ)/2);
    glGetDoublev(GL_MODELVIEW_MATRIX, m);
    glPopMatrix();
    pair<QVector<double>,QVector<double> > pa = getMouseDirection (mouse, m);
    QVector<double> P = pa.first;
    QVector<double> V = pa.second;
    //point sur la sphère
    QVector<double> C(3,0);
    QVector<double> diff(3);
    for (int i=0; i<3; i++)
                diff[i] = P.at(i) - C.at(i);	//diff = P-C
    double scal1 = 0;	//scal1 = <V;P-C>
    double D1 = 0;	//D1 = |P-C|²
    double D2 = 0;	//D2 = |V|²
    for (int i=0; i<3; i++)
    {
        scal1 += V.at(i) * diff.at(i);
        D1 += diff.at(i) * diff.at(i);
        D2 += V.at(i) * V.at(i);
    }
    double delta = scal1*scal1 - (D1-1.0f)*D2;
    /*if (delta<0) {
        return getPlanPoint(pa);
    }*/
    delete [] m;
    delta = sqrt(delta);
    double lambda1 = (-scal1 - delta)/D2;
    double lambda2 = (-scal1 + delta)/D2;	//point de l'autre côté de la sphère
    double lambda = (abs(lambda1)<abs(lambda2)) ? lambda1 : lambda2;	//c'est le plus proche de P

    QVector<double> M(3);
    for (int i=0; i<3; i++)
                M[i] = P.at(i) + lambda*V.at(i);	//M = P + lambda*C
    return M;
}

GLuint GLWidget::makeBoule()
{
    //dessin des cercles de la boule
    GLuint list = glGenLists(1);
    //parametres->incrNbGLLists();
    //GLuint list = parametres->getNbGLLists();
    glNewList(list, GL_COMPILE);
    glMatrixMode(GL_MODELVIEW);

    glPushMatrix();
    glCircle3i(1, m_params.modifRot());

    GLdouble * m = new GLdouble[16];	//pb avec glGet
    for (int i=0; i<4; i++)
    {
        for (int j=0; j<4; j++)
        {
            if (j==1) m[i+4*j] = m_params.getRot()[i+4*(j+1)];
            else if (j==2) m[i+4*j] = -m_params.getRot()[i+4*(j-1)];
            else m[i+4*j] = m_params.getRot()[i+4*j];
        }
    }
    glRotated(90, 1, 0, 0);
    glCircle3i(1, m);

    GLdouble * n = new GLdouble[16];
    for (int i=0; i<4; i++)
    {
        for (int j=0; j<4; j++)
        {
            if (j==0) n[i+4*j] = -m[i+4*(j+2)];
            else if (j==2) n[i+4*j] = m[i+4*(j-2)];
            else n[i+4*j] = m[i+4*j];
        }
    }
    glRotated(90, 0, 1, 0);
    glCircle3i(1, n);
    glPopMatrix();

    delete [] m;
    delete [] n;
    glEndList();
    return list;
}

void GLWidget::setRotation(GLdouble* R) {
        m_params.modifRot() = R;
    //boule = makeBoule();
    update();
}

void GLWidget::setTranslation(const QVector<GLdouble>& T)
{
    if (T.at(0)==m_params.getTrans().at(0) &&
        T.at(1)==m_params.getTrans().at(1) &&
        T.at(2)==m_params.getTrans().at(2))
        return;

    //si la translation est > à la moitié de la fenêtre + la moitié de l'objet, ça ne sert à rien
    QVector<GLdouble> Tverif(3);
    glPushMatrix();
    glLoadIdentity();
    glMultMatrixd(m_params.getRot());
    GLdouble * m = new GLdouble[16];
    glGetDoublev(GL_MODELVIEW_MATRIX, m);
    for (int i=0; i<3; i++) {
    for (int j=0; j<3; j++) {
                Tverif[i] = m[j*4+i] * T.at(j);	//m'
    }}
    glPopMatrix();
    /*double l = max(max(zoneChantierEtCam[1]-zoneChantierEtCam[0], zoneChantierEtCam[3]-zoneChantierEtCam[2]), zoneChantierEtCam[5]-zoneChantierEtCam[4])*sqrt(2)/2.0;
    for (int i=0; i<3; i++) {
        double tmax = l/2.0 + (espace[2*i+1] - espace[2*i])/2.0;
        if (abs(Tverif[i])>tmax) return;
    }*/

    for (int i=0; i<3; i++)
        m_params.modifTrans()[i] = T.at(i);
    update();
}

void GLWidget::convertRotation(int direction, const GLdouble& R, bool anti)
{	//0 à 2
    //!anti : rotation selon la sphère, anti : rotation selon l'écran
    GLdouble* Rf = new GLdouble[16];
    QVector<GLdouble> axe(3,0);
    axe[direction] = 1;

    //la rotation dépend de la rotation courante
    glPushMatrix();
    glLoadIdentity();
    if (anti)
        glRotated(R, axe.at(0), axe.at(1), axe.at(2));
    glMultMatrixd(m_params.getRot());
    if (!anti)
        glRotated(R, axe.at(0), axe.at(1), axe.at(2));
    glGetDoublev(GL_MODELVIEW_MATRIX, Rf);
    glPopMatrix();

    setRotation(Rf);
}

void GLWidget::convertTranslation(int direction, const GLdouble& T)
{	//0 à 2
    QVector<GLdouble> Tf(3);

    //la direction finale de la translation dépend de la rotation courante
    for (int i=0; i<3; i++)
    {
        Tf[i] = T * m_params.getRot()[i*4+direction] / m_params.getScale();
    }
    glPopMatrix();

    for (int i=0; i<3; i++)
        Tf[i] = m_params.getTrans().at(i) + Tf.at(i);

    setTranslation(Tf);
}

void GLWidget::mouseMoveEvent(QMouseEvent *event)
{
    /*if (event->x()<0 || event->y()<0 || event->x()>width() || event->y()>height())
        return;
    if (event->buttons() & Qt::LeftButton)
    {
        //point sur la sphère
        QVector<double> C(3,0);
        QVector<double> M = getSpherePoint(event->pos());

        if (M.count()==0) return;

        //nouvel angle
        int coordy = posSphere.first - 1;
        if (coordy<0) coordy += 3;
        int coordx = coordy - 1;
        if (coordx<0) coordx += 3;
        double angle = atan2(M.at(coordy)-C.at(coordy), M.at(coordx)-C.at(coordx))*180/PI;
        convertRotation(posSphere.first, angle - posSphere.second, false);	//rotation suivant la sphère
    }
    else if (event->buttons() & Qt::RightButton)
    {
        double dx = event->x() - lastPos.x();
        double dy = event->y() - lastPos.y();
        GLdouble coeff = (-m_params.getTrans().at(2) + (m_minZ+m_maxZ)/2.0) / winZ;
        convertTranslation(0, dx*coeff);
        convertTranslation(1, -dy*coeff);
    }
    lastPos = event->pos();*/

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

        this->updateGL();
    }
}
