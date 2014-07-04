#include "MatrixManager.h"

MatrixManager::MatrixManager()
{
    _mvMatrix   = new GLdouble[16];
    _projMatrix = new GLdouble[16];
    _glViewport = new GLint[4];

    _rX = PI;
    _rY = 0.0;
    _distance = 10.f;

    _upY = 1.f;

    resetAllMatrix();
}

MatrixManager::~MatrixManager()
{
    delete [] _mvMatrix;
    delete [] _projMatrix;
    delete [] _glViewport;
}

void MatrixManager::setGLViewport(GLint x, GLint y, GLsizei width, GLsizei height)
{
    m_glRatio  = (float) width/height;

    glViewport( 0, 0, width, height );
    glGetIntegerv (GL_VIEWPORT, getGLViewport());

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();

    glGetDoublev (GL_PROJECTION_MATRIX, _projMatrix);
}

void MatrixManager::doProjection(QPointF point, float zoom)
{    
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glMultMatrixd(_projMatrix);

    zoom *= 2.f/_glViewport[2];

    if(_projMatrix[0] != zoom)
    {
        GLdouble wx, wy, wz;

        GLint recal = _glViewport[3] - (GLint) point.y() /*- 1*/;

        //from viewport to world coordinates
        //TODO peut etre simplifier!
        mmUnProject ((GLdouble) point.x(), (GLdouble) recal, 1.f,
                      _mvMatrix, _projMatrix, _glViewport, &wx, &wy, &wz);

        GLfloat scale = zoom/_projMatrix[0];

        glTranslatef(wx,wy,0);
        glScalef(scale,scale, 1.f);
        glTranslatef(-wx,-wy,0);
    }

    glTranslatef(m_translationMatrix[0],m_translationMatrix[1],0.f);

    m_translationMatrix[0] = m_translationMatrix[1] = 0.f;

    glGetDoublev (GL_PROJECTION_MATRIX, _projMatrix);

}

GLdouble MatrixManager::rY() const
{
    return _rY;
}

void MatrixManager::setRY(const GLdouble &rY)
{
    _rY = rY;
}

void MatrixManager::resetMatrixProjection(float x, float y)
{

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();

    glScalef(_glViewport[3]/2.f,_glViewport[2]/2.f,1.f);
    glTranslatef(x,y,0.f);
    glGetDoublev (GL_PROJECTION_MATRIX, _projMatrix);

    m_translationMatrix[0] = m_translationMatrix[1] = 0.f;
}

void MatrixManager::translate(float tX, float tY, float tZ)
{    
    float inverMat[4][4];

    float translation[3];
    translation[0] = tX;
    translation[1] = tY;
    translation[2] = tZ;

    MatrixInverse(_mvMatrix, inverMat,translation); // on se place dans le repere

    m_translationMatrix[0] += translation[0];
    m_translationMatrix[1] += translation[1];
    m_translationMatrix[2] += translation[2];
}

void MatrixManager::setMatrices()
{
    glMatrixMode(GL_MODELVIEW);
    glGetDoublev(GL_MODELVIEW_MATRIX, _mvMatrix);

    glMatrixMode(GL_PROJECTION);
    glGetDoublev(GL_PROJECTION_MATRIX, _projMatrix);

    glGetIntegerv(GL_VIEWPORT, _glViewport);
}

void MatrixManager::importMatrices(selectInfos &infos)
{
    for (int aK=0; aK<4; ++aK)
        _glViewport[aK] = infos.glViewport[aK];
    for (int aK=0; aK<16; ++aK)
    {
        _mvMatrix[aK] = infos.mvmatrix[aK];
        _projMatrix[aK] = infos.projmatrix[aK];
    }
}

void MatrixManager::exportMatrices(selectInfos &infos)
{
    for (int aK=0; aK<4; ++aK)
        infos.glViewport[aK] = _glViewport[aK];
    for (int aK=0; aK<16; ++aK)
    {
        infos.mvmatrix[aK]   = _mvMatrix[aK];
        infos.projmatrix[aK] = _projMatrix[aK];
    }
}

void MatrixManager::getProjection(QPointF &P2D, Pt3dr P)
{
    GLdouble xp,yp,zp;
    mmProject(P.x,P.y,P.z,_mvMatrix,_projMatrix,_glViewport,&xp,&yp,&zp);
    P2D = QPointF(xp,yp);
}

QPointF MatrixManager::WindowToImage(QPointF const &winPt, float zoom)
{
    QPointF res( winPt.x()         - .5f*_glViewport[2]*(1.f + _projMatrix[12]),
                -winPt.y()  -1.f   + .5f*_glViewport[3]*(1.f - _projMatrix[13]));

    res /= zoom;

    return res;
}

QPointF MatrixManager::ImageToWindow(QPointF const &imPt, float zoom)
{
    return QPointF (imPt.x()*zoom + .5f*_glViewport[2]*(1.f + _projMatrix[12]),
            - 1.f - imPt.y()*zoom + .5f*_glViewport[3]*(1.f - _projMatrix[13]));
}

void MatrixManager::mglOrtho( GLdouble left, GLdouble right,
               GLdouble bottom, GLdouble top,
               GLdouble near_val, GLdouble far_val )
{
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(left, right, bottom, top, near_val, far_val);
}

void MatrixManager::resetRotationMatrix()
{
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glGetDoublev(GL_MODELVIEW_MATRIX, m_rotationMatrix);
}

void MatrixManager::resetTranslationMatrix(Pt3dr center)
{
    m_translationMatrix[0] = -center.x;
    m_translationMatrix[1] = -center.y;
    m_translationMatrix[2] = -center.z;
}

void MatrixManager::resetAllMatrix(Pt3d<double> center)
{
    _targetCamera.x = 0;
    _targetCamera.y = 0;
    _targetCamera.z = 0;

    _distance = 10.f;
    _upY = 1.f;

    resetRotationMatrix();

    resetModelViewMatrix();

    resetTranslationMatrix(center);
}

void MatrixManager::resetModelViewMatrix()
{
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glGetDoublev (GL_MODELVIEW_MATRIX, getModelViewMatrix());
}

void MatrixManager::setModelViewMatrix()
{
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glMultMatrixd(m_rotationMatrix);
    glTranslated(m_translationMatrix[0],m_translationMatrix[1],m_translationMatrix[2]);

    glGetDoublev (GL_MODELVIEW_MATRIX, _mvMatrix);
}

void MatrixManager::glOrthoZoom(float zoom, float farr)
{
    MatrixManager::mglOrtho(
        (GLdouble)( -zoom*getGlRatio() ),
        (GLdouble)( zoom*getGlRatio() ),
        (GLdouble)( -zoom ),
        (GLdouble)zoom,
        (GLdouble)( -farr ),
        (GLdouble)farr);
}

void MatrixManager::setView(VIEW_ORIENTATION orientation, Pt3d<double> centerScene)
{
    resetAllMatrix(centerScene);

    switch (orientation)
    {
    case TOP_VIEW:
        _rX = PI;
        _rY = PI/2.f;
        break;
    case BOTTOM_VIEW:
        _rX = 0.0;
        _rY = 0.0;
        break;
    case FRONT_VIEW:
        _rX = PI;
        _rY = 0.0;
        break;
    case BACK_VIEW:
        _rX = PI;
        _rY = -PI/2.f;
        break;
    case LEFT_VIEW:
        _rX = PI/2.f;
        _rY = 0.0;
        break;
    case RIGHT_VIEW:

        _rX = -PI/2.f;
        _rY = 0.0;
    }
}

void MatrixManager::SetArcBallCamera(float zoom)
{
    setDistance(zoom);
    glOrthoZoom(zoom,zoom + _diameterScene);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    Pt3d<double> camPos;

    _targetCamera.x -= m_translationMatrix[0];
    _targetCamera.y -= m_translationMatrix[1];
    _targetCamera.z -= m_translationMatrix[2];

    camPos.x = _targetCamera.x +  _distance * -sinf(_rX) * cosf(_rY);
    camPos.y = _targetCamera.y +  _distance * -sinf(_rY);
    camPos.z = _targetCamera.z + -_distance *  cosf(_rX) * cosf(_rY);

    // Set the camera position and lookat point
    mmLookAt(camPos.x,camPos.y,camPos.z,      // Camera position
                  _targetCamera.x, _targetCamera.y, _targetCamera.z,    // Look at point
                  0.0, _upY, 0.0);

    resetTranslationMatrix();

    glGetDoublev(GL_MODELVIEW_MATRIX,  _mvMatrix);
    glGetDoublev(GL_PROJECTION_MATRIX, _projMatrix); // TODO a placer pour le realiser une seule fois
}

void MatrixManager::handleRotation(QPointF clicPosMouse)
{

    QPointF centerProj;

    getProjection(centerProj,centerScene());

    QPointF projMouse(clicPosMouse.x(), vpHeight() - clicPosMouse.y());

    _lR = (projMouse.x() < centerProj.x()) ? -1 : 1;
    _uD = (projMouse.y() > centerProj.y()) ? -1 : 1;


    if(rY()>0)
        _uD = -_uD;

    if((rY()< 0 && rY() < - PI) ||
       (rY()> PI && rY() < 2.f * PI))
         _uD = - _uD;

    if((abs(rY()) < HANGLE))
        _uD = 1;
    else if (((abs(rY()) < PI + HANGLE ) &&
            (abs(rY()) > PI - HANGLE)))
        _uD = -1;

}

void MatrixManager::setMatrixDrawViewPort()
{
    glPushMatrix();
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();    
    glTranslatef(-1.f,-1.f,0.f);
    glScalef(2.f/(float)_glViewport[2],2.f/(float)_glViewport[3],1.f);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
}

void MatrixManager::applyAllTransformation(bool mode2D,QPoint pt,float zoom)
{
    if (mode2D)
        doProjection(pt, zoom);
    else
        SetArcBallCamera(zoom);
}

void MatrixManager::rotateArcBall(float rX, float rY, float rZ, float factor)
{

    rX = _uD*rX;
    float ry = _rY;
    int sR = -1;

    if(abs(_rY)>= 0 && abs(_rY)<= 2.f * PI)
        sR = 1;

    _rX -= rX * factor * sR;
    _rY -= rY * factor;
    _rX = fmod(_rX,2*PI);
    _rY = fmod(_rY,2*PI);

    if(
            (abs(ry)<PI/2.f && abs(_rY)>PI/2.f) ||
            (abs(ry)>PI/2.f && abs(_rY)<PI/2.f) ||
            (abs(ry)< 3*PI/2.f && abs(_rY)> 3*PI/2.f)||
            (abs(ry)> 3*PI/2.f && abs(_rY)< 3*PI/2.f)
            )
    {
        if((abs(ry)< 2.f*PI - PI/4.f))

            _upY = -_upY;        
    }
}

void MatrixManager::MatrixInverse(GLdouble OpenGLmatIn[], float matOut[][4],float* vec)
{
    float matIn[4][4];
    // OpenGL matrix is column major matrix in 1x16 array. Convert it to row major 4x4 matrix
    for(int m=0, k=0; m<=3 && k<16; m++)
      for(int n=0;n<=3;n++)
      {
        matIn[m][n] = OpenGLmatIn[k];
        k++;
      }
    // 3x3 rotation Matrix Transpose ( it is equal to invering rotations) . Since rotation matrix is anti-symmetric matrix, transpose is equal to Inverse.
    for(int i=0 ; i<3; i++){
    for(int j=0; j<3; j++){
      matOut[j][i] = matIn[i][j];
     }
    }
    // Negate the translations ( equal to inversing translations)
    float vTmp[3];

    vTmp[0] = -matIn[3][0];
    vTmp[1] = -matIn[3][1];
    vTmp[2] = -matIn[3][2];
    // Roatate this vector using the above newly constructed rotation matrix
    matOut[3][0] = vTmp[0]*matOut[0][0] + vTmp[1]*matOut[1][0] + vTmp[2]*matOut[2][0];
    matOut[3][1] = vTmp[0]*matOut[0][1] + vTmp[1]*matOut[1][1] + vTmp[2]*matOut[2][1];
    matOut[3][2] = vTmp[0]*matOut[0][2] + vTmp[1]*matOut[1][2] + vTmp[2]*matOut[2][2];

    // Take care of the unused part of the OpenGL 4x4 matrix
    matOut[0][3] = matOut[1][3] = matOut[2][3] = 0.0f;
    matOut[3][3] = 1.0f;

    float inVec[3];

    inVec[0] = vec[0] * matOut[0][0] + vec[1]* matOut[1][0] + vec[2]* matOut[2][0];// + matOut[0][3];
    inVec[1] = vec[0] * matOut[0][1] + vec[1]* matOut[1][1] + vec[2]* matOut[2][1];// + matOut[1][3];
    inVec[2] = vec[0] * matOut[0][2] + vec[1]* matOut[1][2] + vec[2]* matOut[2][2];// + matOut[2][3];

    vec[0] = inVec[0];
    vec[1] = inVec[1];
    vec[2] = inVec[2];
}

Pt3d<double> MatrixManager::centerScene() const
{
    return _centerScene;
}

void MatrixManager::setCenterScene(const Pt3d<double> &centerScene)
{
    _centerScene = centerScene;
}

void MatrixManager::setSceneTopo(const Pt3d<double> &centerScene,float diametre)
{
    _centerScene    = centerScene;
    _diameterScene  = diametre;
}

QPointF MatrixManager::screen2TransABall(QPointF ptScreen)
{
    return QPointF(ptScreen .x()/((float)vpWidth()*_projMatrix[0]),-ptScreen .y()/((float)vpHeight()*_projMatrix[5]))*2.f;
}

GLdouble MatrixManager::distance() const
{
    return _distance;
}

void MatrixManager::setDistance(const GLdouble &distance)
{
    _distance = distance;
}
