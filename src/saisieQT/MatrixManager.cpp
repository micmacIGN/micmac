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
}



void MatrixManager::doProjection(QPointF point, float zoom)
{
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();

    glPushMatrix();
    glMultMatrixd(_projMatrix);

    if(_projMatrix[0] != zoom)
    {
        GLdouble wx, wy, wz;

        GLint recal = _glViewport[3] - (GLint) point.y() - 1;

         //from viewport to world coordinates
        #if defined Q_WS_MAC
            #if MAC_OS_X_VERSION_MAX_ALLOWED >= MAC_OS_X_VERSION_10_9
                    bool result;
                    GLKVector3 window_coord = GLKVector3Make(point.x(), recal, 1.f);
                    GLKVector3 wPt = GLKMathUnproject(window_coord, (GLKMatrix4) _mvMatrix, (GLKMatrix4) _projMatrix, _glViewport, &result);
                    wx = wPt.x;
                    wy = wPt.y;
                    wz = 0.f;
            #else
                    gluUnProject ((GLdouble) point.x(), (GLdouble) recal, 1.f,
                    _mvMatrix, _projMatrix, _glViewport, &wx, &wy, &wz);
            #endif
        #else
                gluUnProject ((GLdouble) point.x(), (GLdouble) recal, 1.f,
                              _mvMatrix, _projMatrix, _glViewport, &wx, &wy, &wz);
        #endif

        glTranslatef(wx,wy,0);
        glScalef(zoom/_projMatrix[0], zoom/_projMatrix[0], 1.f);
        glTranslatef(-wx,-wy,0);
    }

    glTranslatef(m_translationMatrix[0],m_translationMatrix[1],0.f);

    m_translationMatrix[0] = m_translationMatrix[1] = 0.f;

    glGetDoublev (GL_PROJECTION_MATRIX, _projMatrix);
}

void MatrixManager::getProjection3D(QPointF &P2D, Pt3dr &P)
{
    GLint recal = _glViewport[3] - (GLint) P2D.y() - 1;

    #if defined Q_WS_MAC
        #if MAC_OS_X_VERSION_MAX_ALLOWED >= MAC_OS_X_VERSION_10_9
                bool result;
                GLKVector3 window_coord = GLKVector3Make(P2D.x(), recal, 1.f);
                GLKVector3 wPt = GLKMathUnproject(window_coord, (GLKMatrix4) _mvMatrix, (GLKMatrix4) _projMatrix, _glViewport, &result);
                P = Pt3dr(wPt.x, wPt.y, wPt.z)
        #else
                GLdouble xp,yp,zp;
                gluUnProject ((GLdouble) P2D.x(), (GLdouble) recal, 1.f,_mvMatrix, _projMatrix, _glViewport, &xp, &yp, &zp);

                P = Pt3dr(xp,yp,zp);
        #endif
    #else
        GLdouble xp,yp,zp;
        gluUnProject((GLdouble) P2D.x(),(GLdouble) recal, 1.f,_mvMatrix,_projMatrix,_glViewport,&xp,&yp,&zp);

        P = Pt3dr(xp,yp,zp);
#endif
}

GLdouble MatrixManager::rY() const
{
    return _rY;
}

void MatrixManager::setRY(const GLdouble &rY)
{
    _rY = rY;
}


void MatrixManager::translate(float x, float y)
{
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glPushMatrix();
    glTranslatef(x,y,0.f);
    glGetDoublev (GL_PROJECTION_MATRIX, _projMatrix);
    glPopMatrix();

    m_translationMatrix[0] = m_translationMatrix[1] = 0.f;
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
    #if defined Q_WS_MAC
        #if MAC_OS_X_VERSION_MAX_ALLOWED >= MAC_OS_X_VERSION_10_9
                bool result;
                GLKVector3 Pt = GLKVector3Make(P.x,P.y,P.z);
                GLKVector3 wPt = GLKMathproject(Pt, (GLKMatrix4) _mvMatrix, (GLKMatrix4) _projMatrix, _glViewport);
                P2D = QPointF(wPt.x,wPt.y);
        #else
                GLdouble xp,yp,zp;
                gluProject(P.x,P.y,P.z,_mvMatrix,_projMatrix,_glViewport,&xp,&yp,&zp);

                P2D = QPointF(xp,yp);
        #endif
    #else
        GLdouble xp,yp,zp;
        gluProject(P.x,P.y,P.z,_mvMatrix,_projMatrix,_glViewport,&xp,&yp,&zp);
        P2D = QPointF(xp,yp);
    #endif
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

void MatrixManager::applyTransfo()
{
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glPushMatrix();

    glMultMatrixd(m_rotationMatrix);
    //glTranslated(m_translationMatrix[0],m_translationMatrix[1],m_translationMatrix[2]);
}

void MatrixManager::setModelViewMatrix()
{
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glMultMatrixd(m_rotationMatrix);
    glTranslated(m_translationMatrix[0],m_translationMatrix[1],m_translationMatrix[2]);

    glGetDoublev (GL_MODELVIEW_MATRIX, _mvMatrix);
}

void MatrixManager::zoom(float zoom, float farr)
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
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    _centerScene = centerScene;

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

    glGetDoublev(GL_MODELVIEW_MATRIX, m_rotationMatrix);

    resetTranslationMatrix(centerScene);
}

void MatrixManager::rotate(GLdouble* matrix, float rX, float rY, float rZ, float factor)
{
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glMultMatrixd(matrix);

    glRotated(rX * factor,1.0,0.0,0.0);
    glRotated(rY * factor,0.0,1.0,0.0);
    glRotated(rZ * factor,0.0,0.0,1.0);
    glGetDoublev(GL_MODELVIEW_MATRIX, matrix);
}

void MatrixManager::rotate(float rX, float rY, float rZ, float factor)
{
    rotate(m_rotationMatrix, rX, rY, rZ, factor);
}

void MatrixManager::arcBall()
{
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    Pt3d<double> camPos;

    Pt3d<double> target ;//=  _centerScene;

    target.x = -m_translationMatrix[0];
    target.y = -m_translationMatrix[1];
    target.z = -m_translationMatrix[2];

    camPos.x = target.x +  _distance * -sinf(_rX) * cosf(_rY);
    camPos.y = target.y +  _distance * -sinf(_rY);
    camPos.z = target.z + -_distance * cosf(_rX) * cosf(_rY);

    // Set the camera position and lookat point
    #if defined Q_WS_MAC
        #if MAC_OS_X_VERSION_MAX_ALLOWED >= MAC_OS_X_VERSION_10_9
            GLKMatrix4MakeLookAt(camPos.x,camPos.y,camPos.z,
                                 target.x, target.y, target.z,
                                 0.0, _upY, 0.0);
        #else
            gluLookAt(camPos.x,camPos.y,camPos.z,
                  target.x, target.y, target.z,
                  0.0, _upY, 0.0);
        #endif
    #else
        gluLookAt(camPos.x,camPos.y,camPos.z,      // Camera position
                  target.x, target.y, target.z,    // Look at point
                  0.0, _upY, 0.0);
    #endif

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

    float hAngle = PI / 24;


    if(rY()>0)
        _uD = -_uD;

    if((rY()< 0 && rY() < - PI) ||
       (rY()> PI && rY() < 2.f * PI))
         _uD = - _uD;

    if((abs(rY()) < hAngle))
        _uD = 1;
    else if (((abs(rY()) < PI + hAngle) &&
            (abs(rY()) > PI - hAngle)))
        _uD = -1;



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
            //(abs(ry) > (2.f * PI -1.f) && abs(_rY)< 0.5) ||
            (abs(ry)<PI/2.f && abs(_rY)>PI/2.f) ||
            (abs(ry)>PI/2.f && abs(_rY)<PI/2.f) ||
            (abs(ry)< 3*PI/2.f && abs(_rY)> 3*PI/2.f)||
            (abs(ry)> 3*PI/2.f && abs(_rY)< 3*PI/2.f)
            )
    {
        if((abs(ry)< 2.f*PI - PI/4.f))
        {
            //printf("FLIP\n");
            _upY = -_upY;
        }
    }

    //printf("x %f\n",_rX);
    //printf("y %f\n",_rY);
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

void MatrixManager::translate(float tX, float tY, float tZ, float factor)
{
    float inverMat[4][4];

    float translation[3];
    translation[0] = factor * tX;
    translation[1] = factor * tY;
    translation[2] = factor * tZ;

    MatrixInverse(_mvMatrix, inverMat,translation);

    m_translationMatrix[0] += translation[0];
    m_translationMatrix[1] += translation[1];
    m_translationMatrix[2] += translation[2];
}

GLdouble MatrixManager::distance() const
{
    return _distance;
}

void MatrixManager::setDistance(const GLdouble &distance)
{
    _distance = distance;
}
