#include "MatrixManager.h"

MatrixManager::MatrixManager(eNavigationType nav):
_eNavigation(nav),
 _factor(1.0)
{

//	_MatrixPassageCamera	= new GLdouble[16];
//	_MatrixPassageCameraInv	= new GLdouble[16];
//	_positionCamera	= new GLdouble[4];


    loadIdentity(_mvMatrixOld);
//	loadIdentity(_mvMatrixOldInv);
//	loadIdentity(_MatrixPassageCamera);

    _rX  = 0.0;
    _rY	 = 0.0;
    _rZ	 = 0.0;
    _upY = 1.f;

    _distance = 10.f;

}

MatrixManager::~MatrixManager()
{
//    delete [] _mvMatrix;
//    delete [] _projMatrix;
//    delete [] _glViewport;
//	delete [] _MatrixPassageCamera;
//	delete [] _MatrixPassageCameraInv;
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

void MatrixManager::resetViewPort()
{
    glGetIntegerv (GL_VIEWPORT, getGLViewport());
}

void MatrixManager::translate(float tX, float tY, float tZ)
{
    //float inverMat[4][4];

    GLdouble translation[3];
    translation[0] = tX;
    translation[1] = tY;
    translation[2] = tZ;

    MatrixInverse(_mvMatrix, NULL,translation); // on se place dans le repere

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

void MatrixManager::importMatrices(const selectInfos &infos)
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

void MatrixManager::getProjection(QPointF &P2D, QVector3D P)
{
    GLdouble xp,yp,zp;
    mmProject(P.x(),P.y(),P.z(),_mvMatrix,_projMatrix,_glViewport,&xp,&yp,&zp);
    P2D = QPointF(xp,yp);
}

void MatrixManager::getInverseProjection(QVector3D &P, QPointF P2D, float dist)
{
    GLdouble xp,yp,zp;
    mmUnProject(P2D.x(), P2D.y(), dist,_mvMatrix,_projMatrix,_glViewport,&xp,&yp,&zp);

    P.setX(xp);
    P.setY(yp);
    P.setZ(zp);
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
    return QPointF ((imPt.x()*zoom + .5f*_glViewport[2]*(1.f + _projMatrix[12])),
            (- 1.f - imPt.y()*zoom + .5f*_glViewport[3]*(1.f - _projMatrix[13])));
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

void MatrixManager::resetTranslationMatrix(QVector3D center)
{
    m_translationMatrix[0] = -center.x();
    m_translationMatrix[1] = -center.y();
    m_translationMatrix[2] = -center.z();
}

void MatrixManager::resetAllMatrix(QVector3D center, bool resetALL)
{
    _targetCamera.setX(0);
    _targetCamera.setY(0);
    _targetCamera.setZ(0);

    _distance = 10.f;

    if(resetALL)
    {
        loadIdentity(_mvMatrixOld);

        _rX  = 0;
        _rY	 = 0;
        _rZ	 = 0.0;
        _upY = 1;
        _lR	 = 1;
        _uD  = 1;
    }

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

void MatrixManager::glOrthoZoom(float zoom, float farr)
{

    MatrixManager::mglOrtho(
        (GLdouble)( -zoom*getGlRatio() ),
        (GLdouble)( zoom*getGlRatio() ),
        (GLdouble)( -zoom ),
        (GLdouble)zoom,
        (GLdouble)( farr),
        (GLdouble)0);
}

void MatrixManager::setView(VIEW_ORIENTATION orientation, QVector3D centerScene)
{
    resetAllMatrix(centerScene);

    switch (orientation)
    {
    case TOP_VIEW:
        _rX = M_PI;
        _rY = M_PI_2;
        break;
    case BOTTOM_VIEW:
        _rX = 0.0;
        _rY = 0.0;
        break;
    case FRONT_VIEW:
        _rX = M_PI;
        _rY = 0.0;
        break;
    case BACK_VIEW:
        _rX = M_PI;
        _rY = -M_PI_2;
        break;
    case LEFT_VIEW:
        _rX = M_PI_2;
        _rY = 0.0;
        break;
    case RIGHT_VIEW:

        _rX = -M_PI_2;
        _rY = 0.0;
    }
}

void MatrixManager::setArcBallCamera(float aDistance)
{

    setDistance((aDistance+ _diameterScene)*2.0);
    glOrthoZoom(aDistance,(aDistance + _diameterScene)*4.0);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    _targetCamera.setX( _targetCamera.x() - m_translationMatrix[0]);
    _targetCamera.setY( _targetCamera.y()- m_translationMatrix[1]);
    _targetCamera.setZ( _targetCamera.z()- m_translationMatrix[2]);

    _cX = cosf(_rX);
    _cY = cosf(_rY);
    _sX = sinf(_rX);
    _sY = sinf(_rY);

    GLdouble  posCamera[4];
    GLdouble  up[4] = {_upY*sin(_rZ),_upY*cos(_rZ),0.0,0.0};

    posCamera[0] = distance() * -_sX * _cY;
    posCamera[1] = distance() * -_sY;
    posCamera[2] = distance() *  _cX * _cY;

    if(isBallNavigation())
    {
        MatrixInverse(_mvMatrixOld,NULL,posCamera);
        MatrixInverse(_mvMatrixOld,NULL,up);
    }

    _camPos.setX(_targetCamera.x() + posCamera[0]);
    _camPos.setY(_targetCamera.y() + posCamera[1]);
    _camPos.setZ( _targetCamera.z() + posCamera[2]);

    mmLookAt(_camPos.x(),_camPos.y(),_camPos.z(),								// Camera position
                  _targetCamera.x(), _targetCamera.y(), _targetCamera.z(),    // Look at point
                  up[0],up[1] ,up[2]);									// up

    resetTranslationMatrix();

    glGetDoublev(GL_MODELVIEW_MATRIX,  _mvMatrix);
    glGetDoublev(GL_PROJECTION_MATRIX, _projMatrix); // TODO a placer pour le realiser une seule fois
}

void MatrixManager::printVecteur(GLdouble* posCameraOut, const char* nameVariable)
{
    printf("%s : [%f,%f,%f]\n",nameVariable,posCameraOut[0],posCameraOut[1],posCameraOut[2]);
}

QPointF MatrixManager::centerVP()
{
    QPointF centerViewPort((((float)vpWidth())/2.0),(((float)vpHeight())/2.0));

    return centerViewPort;
}

QRectF MatrixManager::getRectViewportToImage(float zoom)
{
    QPointF c0(0.f,0.f);
    QPointF c3(vpWidth(),vpHeight());

    QPointF p0Img = WindowToImage(c0, zoom);
    QPointF p1Img = WindowToImage(c3, zoom);

    return QRectF(p0Img ,p1Img);
}

void MatrixManager::handleRotation(QPointF clicPosMouse)
{

    {
        /*
    GLdouble translation[3]		=  {_targetCamera.x,_targetCamera.y,_targetCamera.z};
        GLdouble posCamera[4] = {0.0,0.0,_distance,1.0};
        GLdouble axeY[3] = {0,1.0,0};
        GLdouble axeX1[3] = {cos(-_rX),0,-sin(-_rX)};
        GLdouble rotationY[16];
        GLdouble rotationX1[16];

        matriceRotation(axeY,rotationY,-_rX);
        matriceRotation(axeX1,rotationX1,_rY);
        multiplicationMat(rotationX1,rotationY,_MatrixPassageCamera);
        addTranslationToMat(_MatrixPassageCamera,translation);
        GLdouble posCameraOut[4];
        multiplication(posCamera,posCameraOut,_MatrixPassageCamera);
        GLdouble posInit[4];
        gluInvertMatrix(_MatrixPassageCamera,_MatrixPassageCameraInv);
        multiplication(posCameraOut,posInit,_MatrixPassageCameraInv);

        GLdouble posCAMO[4] = {0,0,_distance,0};
        GLdouble posCAMI[4];

        MatrixInverse(_mvMatrixOld,_mvMatrixOldInv,posCAMO);

        multiplication(posCAMO,posCAMI,_mvMatrixOldInv);

        posCAMO[0] += translation[0];
        posCAMO[1] += translation[1];
        posCAMO[2] += translation[2];

    */
    }

    if(isBallNavigation())
    {

        QPointF centerViewPort = centerVP();

        QLineF vectorR(clicPosMouse,centerViewPort);

        float rayon		= ((float)vpHeight()/3.0);
        float length	= vectorR.length();

        if(eNavigation() == eNavig_Ball_OneTouch)
            _factor = length / (rayon);

        CopyMatrix(getModelViewMatrix(),_mvMatrixOld);
        _rX  = 0;
        _rY	 = 0;
        _rZ	 = 0;
        _upY = 1;
        _lR	 = 1;
        _uD  = 1;
    }
}

void MatrixManager::rotateArcBall(float rX, float rY, float rZ, float factor)
{

    rX = _uD*rX;
    float ry = _rY;
    int sR = -1;

    if(abs(_rY)>= 0 && abs(_rY)<= M_2PI)
        sR = 1;

    if(eNavigation() == eNavig_Ball_OneTouch)
    {
        float factorZ	= min(1.0,max(0.0,_factor - 1.0));
        float factorXY	= 1.0 - factorZ;

        _rX += rX * factor * sR * factorXY;
        _rY -= rY * factor		* factorXY;
        _rZ += rZ 				* factorZ;
    }
    else
    {
        _rX += rX * factor * sR;
        _rY -= rY * factor;
        _rZ += rZ;
    }

    _rX = fmod(_rX,M_2PI);
    _rY = fmod(_rY,M_2PI);
    _rZ = fmod(_rZ,M_2PI);

    if(
            (abs(ry)<M_PI_2 && abs(_rY)>M_PI_2) ||
            (abs(ry)>M_PI_2 && abs(_rY)<M_PI/2.f) ||
            (abs(ry)< 3*M_PI_2 && abs(_rY)> 3*M_PI_2)||
            (abs(ry)> 3*M_PI_2 && abs(_rY)< 3*M_PI_2)
            )
    {
        if((abs(ry)< M_2PI - M_PI_4))

            _upY = -_upY;
    }
}

void MatrixManager::setMatrixDrawViewPort()
{
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
        setArcBallCamera(zoom);
}

void MatrixManager::MatrixInverse(GLdouble* OpenGLmatIn, GLdouble *matOutGL,GLdouble* vec)
{
    float matIn[4][4];
    float matOut[4][4];
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

    if(matOutGL)
    {
        for(int m=0, k=0; m<=3 && k<16; m++)
          for(int n=0;n<=3;n++)
          {
            matOutGL[k] = matOut[m][n];
            k++;
          }
    }

    if(vec)
    {
        float inVec[3];

        inVec[0] = vec[0] * matOut[0][0] + vec[1]* matOut[1][0] + vec[2]* matOut[2][0];// + matOut[0][3];
        inVec[1] = vec[0] * matOut[0][1] + vec[1]* matOut[1][1] + vec[2]* matOut[2][1];// + matOut[1][3];
        inVec[2] = vec[0] * matOut[0][2] + vec[1]* matOut[1][2] + vec[2]* matOut[2][2];// + matOut[2][3];

        vec[0] = inVec[0];
        vec[1] = inVec[1];
        vec[2] = inVec[2];
    }
}

QVector3D MatrixManager::centerScene() const
{
    return _centerScene;
}

void MatrixManager::setCenterScene(const QVector3D &centerScene)
{
    _centerScene = centerScene;
}

void MatrixManager::setSceneTopo(const QVector3D &centerScene,float diametre)
{
    _centerScene    = centerScene;
    _diameterScene  = diametre;
}

QPointF MatrixManager::screen2TransABall(QPointF ptScreen)
{
    return QPointF(ptScreen .x()/((float)vpWidth()*_projMatrix[0]),-ptScreen .y()/((float)vpHeight()*_projMatrix[5]))*2.f;
}

void MatrixManager::multiplication(GLdouble* posIn, GLdouble* posOut, GLdouble* mat)
{
    for (int i = 0; i < 4; ++i,mat += 4)
    {
        posOut[i] = mat[0]*posIn[0] + mat[1]*posIn[1] + mat[2]*posIn[2] + mat[3]*posIn[3];
    }
}

void MatrixManager::matriceRotation(GLdouble* axe, GLdouble* matRot,GLdouble angle)
{

        GLdouble c = cos(angle);
        GLdouble s = sin(angle);

        GLdouble ux = axe[0];
        GLdouble uy = axe[1];
        GLdouble uz = axe[2];

        //GLdouble L = (ux*ux + uy * uy + uz * uz);
        //angle = angle * M_PI / 180.0; //converting to radian value
        GLdouble ux2 = ux * ux;
        GLdouble uy2 = uy * uy;
        GLdouble uz2 = uz * uz;

//	    matRot[0]	=	(ux2 + (uy2 + uz2) * c) / L;
//	    matRot[1]	=	(ux * uy * (1 - c) - uz * sqrt(L) * s) / L;
//	    matRot[2]	=	(ux * uz * (1 - c) + uy * sqrt(L) * s) / L;
//	    matRot[3]	=	0.0;

//	    matRot[4]	=	(ux * uy * (1 - c) + uz * sqrt(L) * s) / L;
//	    matRot[5]	=	(uy2 + (ux2 + uz2) * c) / L;
//	    matRot[6]	=	(uy * uz * (1 - c) - ux * sqrt(L) * s) / L;
//	    matRot[7]	=	0.0;

//	    matRot[8]  =	(ux * uz * (1 - c) - uy * sqrt(L) * s) / L;
//	    matRot[9]  =	(uy * uz * (1 - c) + ux * sqrt(L) * s) / L;
//	    matRot[10] =	(uz2 + (ux2 + uy2) * c) / L;
//	    matRot[11] =	0.0;

//	    matRot[12] =	0.0;
//	    matRot[13] =	0.0;
//	    matRot[14] =	0.0;
//	    matRot[15] =	1.0;

        matRot[0]	=	(ux2 + (1 - ux2) * c);
        matRot[1]	=	(ux * uy * (1 - c) - uz  * s) ;
        matRot[2]	=	(ux * uz * (1 - c) + uy  * s) ;
        matRot[3]	=	0.0;

        matRot[4]	=	(ux * uy * (1 - c) + uz  * s) ;
        matRot[5]	=	(uy2 + (1 - uy2) * c) ;
        matRot[6]	=	(uy * uz * (1 - c) - ux  * s) ;
        matRot[7]	=	0.0;

        matRot[8]  =	(ux * uz * (1 - c) - uy  * s) ;
        matRot[9]  =	(uy * uz * (1 - c) + ux  * s) ;
        matRot[10] =	(uz2 + (1- uz2) * c) ;
        matRot[11] =	0.0;

        matRot[12] =	0.0;
        matRot[13] =	0.0;
        matRot[14] =	0.0;
        matRot[15] =	1.0;
}


void MatrixManager::loadIdentity(GLdouble* matOut)
{
    matOut[M11] = 1.0;    // Column 1
    matOut[M12] = 0.0;    // Column 2
    matOut[M13] = 0.0;    // Column 3
    matOut[M14] = 0.0;    // Column 4

    // Row 2
    matOut[M21] = 0.0;    // Column 1
    matOut[M22] = 1.0;    // Column 2
    matOut[M23] = 0.0;    // Column 3
    matOut[M24] = 0.0;    // Column 4

    // Row 3
    matOut[M31] = 0.0;    // Column 1
    matOut[M32] = 0.0;    // Column 2
    matOut[M33] = 1.0;    // Column 3
    matOut[M34] = 0.0;    // Column 4

    // Row 4
    matOut[M41] = 0.0;    // Column 1
    matOut[M42] = 0.0;    // Column 2
    matOut[M43] = 0.0;    // Column 3
    matOut[M44] = 1.0;    // Column 3
}
eNavigationType MatrixManager::eNavigation() const
{
    return _eNavigation;
}

void MatrixManager::setENavigation(const eNavigationType& eNavigation)
{
    _eNavigation = eNavigation;

    resetAllMatrix();

}

bool MatrixManager::isBallNavigation()
{
    return eNavigation() == eNavig_Ball || eNavigation() == eNavig_Ball_OneTouch;
}

void MatrixManager::multiplicationMat(GLdouble* mat1, GLdouble* mat2, GLdouble* matOut)
{
    matOut[M11] = mat1[M11] * mat2[M11]   +   mat1[M12] * mat2[M21]   +   mat1[M13] * mat2[M31]   +   mat1[M14] * mat2[M41];    // Column 1
    matOut[M12] = mat1[M11] * mat2[M12]   +   mat1[M12] * mat2[M22]   +   mat1[M13] * mat2[M32]   +   mat1[M14] * mat2[M42];    // Column 2
    matOut[M13] = mat1[M11] * mat2[M13]   +   mat1[M12] * mat2[M23]   +   mat1[M13] * mat2[M33]   +   mat1[M14] * mat2[M43];    // Column 3
    matOut[M14] = mat1[M11] * mat2[M14]   +   mat1[M12] * mat2[M24]   +   mat1[M13] * mat2[M34]   +   mat1[M14] * mat2[M44];    // Column 4

    // Row 2
    matOut[M21] = mat1[M21] * mat2[M11]   +   mat1[M22] * mat2[M21]   +   mat1[M23] * mat2[M31]   +   mat1[M24] * mat2[M41];    // Column 1
    matOut[M22] = mat1[M21] * mat2[M12]   +   mat1[M22] * mat2[M22]   +   mat1[M23] * mat2[M32]   +   mat1[M24] * mat2[M42];    // Column 2
    matOut[M23] = mat1[M21] * mat2[M13]   +   mat1[M22] * mat2[M23]   +   mat1[M23] * mat2[M33]   +   mat1[M24] * mat2[M43];    // Column 3
    matOut[M24] = mat1[M21] * mat2[M14]   +   mat1[M22] * mat2[M24]   +   mat1[M23] * mat2[M34]   +   mat1[M24] * mat2[M44];    // Column 4

    // Row 3
    matOut[M31] = mat1[M31] * mat2[M11]   +   mat1[M32] * mat2[M21]   +   mat1[M33] * mat2[M31]   +   mat1[M34] * mat2[M41];    // Column 1
    matOut[M32] = mat1[M31] * mat2[M12]   +   mat1[M32] * mat2[M22]   +   mat1[M33] * mat2[M32]   +   mat1[M34] * mat2[M42];    // Column 2
    matOut[M33] = mat1[M31] * mat2[M13]   +   mat1[M32] * mat2[M23]   +   mat1[M33] * mat2[M33]   +   mat1[M34] * mat2[M43];    // Column 3
    matOut[M34] = mat1[M31] * mat2[M14]   +   mat1[M32] * mat2[M24]   +   mat1[M33] * mat2[M34]   +   mat1[M34] * mat2[M44];    // Column 4

    // Row 4
    matOut[M41] = mat1[M41] * mat2[M11]   +   mat1[M42] * mat2[M21]   +   mat1[M43] * mat2[M31]   +   mat1[M44] * mat2[M41];    // Column 1
    matOut[M42] = mat1[M41] * mat2[M12]   +   mat1[M42] * mat2[M22]   +   mat1[M43] * mat2[M32]   +   mat1[M44] * mat2[M42];    // Column 2
    matOut[M43] = mat1[M41] * mat2[M13]   +   mat1[M42] * mat2[M23]   +   mat1[M43] * mat2[M33]   +   mat1[M44] * mat2[M43];    // Column 3
    matOut[M44] = mat1[M41] * mat2[M14]   +   mat1[M42] * mat2[M24]   +   mat1[M43] * mat2[M34]   +   mat1[M44] * mat2[M44];    // Column 3
}

void MatrixManager::addTranslationToMat(GLdouble* mat, GLdouble* translation)
{
    mat[M14] += translation[0];
    mat[M24] += translation[1];
    mat[M34] += translation[2];
}

bool MatrixManager::gluInvertMatrix(const GLdouble m[], GLdouble invOut[])
{
    double inv[16], det;
    int i;

    inv[0] = m[5]  * m[10] * m[15] -
             m[5]  * m[11] * m[14] -
             m[9]  * m[6]  * m[15] +
             m[9]  * m[7]  * m[14] +
             m[13] * m[6]  * m[11] -
             m[13] * m[7]  * m[10];

    inv[4] = -m[4]  * m[10] * m[15] +
             m[4]  * m[11] * m[14] +
             m[8]  * m[6]  * m[15] -
             m[8]  * m[7]  * m[14] -
             m[12] * m[6]  * m[11] +
             m[12] * m[7]  * m[10];

    inv[8] = m[4]  * m[9] * m[15] -
             m[4]  * m[11] * m[13] -
             m[8]  * m[5] * m[15] +
             m[8]  * m[7] * m[13] +
             m[12] * m[5] * m[11] -
             m[12] * m[7] * m[9];

    inv[12] = -m[4]  * m[9] * m[14] +
              m[4]  * m[10] * m[13] +
              m[8]  * m[5] * m[14] -
              m[8]  * m[6] * m[13] -
              m[12] * m[5] * m[10] +
              m[12] * m[6] * m[9];

    inv[1] = -m[1]  * m[10] * m[15] +
             m[1]  * m[11] * m[14] +
             m[9]  * m[2] * m[15] -
             m[9]  * m[3] * m[14] -
             m[13] * m[2] * m[11] +
             m[13] * m[3] * m[10];

    inv[5] = m[0]  * m[10] * m[15] -
             m[0]  * m[11] * m[14] -
             m[8]  * m[2] * m[15] +
             m[8]  * m[3] * m[14] +
             m[12] * m[2] * m[11] -
             m[12] * m[3] * m[10];

    inv[9] = -m[0]  * m[9] * m[15] +
             m[0]  * m[11] * m[13] +
             m[8]  * m[1] * m[15] -
             m[8]  * m[3] * m[13] -
             m[12] * m[1] * m[11] +
             m[12] * m[3] * m[9];

    inv[13] = m[0]  * m[9] * m[14] -
              m[0]  * m[10] * m[13] -
              m[8]  * m[1] * m[14] +
              m[8]  * m[2] * m[13] +
              m[12] * m[1] * m[10] -
              m[12] * m[2] * m[9];

    inv[2] = m[1]  * m[6] * m[15] -
             m[1]  * m[7] * m[14] -
             m[5]  * m[2] * m[15] +
             m[5]  * m[3] * m[14] +
             m[13] * m[2] * m[7] -
             m[13] * m[3] * m[6];

    inv[6] = -m[0]  * m[6] * m[15] +
             m[0]  * m[7] * m[14] +
             m[4]  * m[2] * m[15] -
             m[4]  * m[3] * m[14] -
             m[12] * m[2] * m[7] +
             m[12] * m[3] * m[6];

    inv[10] = m[0]  * m[5] * m[15] -
              m[0]  * m[7] * m[13] -
              m[4]  * m[1] * m[15] +
              m[4]  * m[3] * m[13] +
              m[12] * m[1] * m[7] -
              m[12] * m[3] * m[5];

    inv[14] = -m[0]  * m[5] * m[14] +
              m[0]  * m[6] * m[13] +
              m[4]  * m[1] * m[14] -
              m[4]  * m[2] * m[13] -
              m[12] * m[1] * m[6] +
              m[12] * m[2] * m[5];

    inv[3] = -m[1] * m[6] * m[11] +
             m[1] * m[7] * m[10] +
             m[5] * m[2] * m[11] -
             m[5] * m[3] * m[10] -
             m[9] * m[2] * m[7] +
             m[9] * m[3] * m[6];

    inv[7] = m[0] * m[6] * m[11] -
             m[0] * m[7] * m[10] -
             m[4] * m[2] * m[11] +
             m[4] * m[3] * m[10] +
             m[8] * m[2] * m[7] -
             m[8] * m[3] * m[6];

    inv[11] = -m[0] * m[5] * m[11] +
              m[0] * m[7] * m[9] +
              m[4] * m[1] * m[11] -
              m[4] * m[3] * m[9] -
              m[8] * m[1] * m[7] +
              m[8] * m[3] * m[5];

    inv[15] = m[0] * m[5] * m[10] -
              m[0] * m[6] * m[9] -
              m[4] * m[1] * m[10] +
              m[4] * m[2] * m[9] +
              m[8] * m[1] * m[6] -
              m[8] * m[2] * m[5];

    det = m[0] * inv[0] + m[1] * inv[4] + m[2] * inv[8] + m[3] * inv[12];

    if (det == 0)
        return false;

    det = 1.0 / det;

    for (i = 0; i < 16; i++)
        invOut[i] = inv[i] * det;

    return true;
}

void MatrixManager::CopyMatrix(GLdouble* source, GLdouble* in)
{
    in[M11] = source[M11];    // Column 1
    in[M12] = source[M12];    // Column 2
    in[M13] = source[M13];    // Column 3
    in[M14] = source[M14];    // Column 4

    // Row 2
    in[M21] = source[M21];    // Column 1
    in[M22] = source[M22];    // Column 2
    in[M23] = source[M23];    // Column 3
    in[M24] = source[M24];    // Column 4

    // Row 3
    in[M31] = source[M31];    // Column 1
    in[M32] = source[M32];    // Column 2
    in[M33] = source[M33];    // Column 3
    in[M34] = source[M34];    // Column 4

    // Row 4
    in[M41] = source[M41];    // Column 1
    in[M42] = source[M42];    // Column 2
    in[M43] = source[M43];    // Column 3
    in[M44] = source[M44];    // Column 3
}

GLdouble MatrixManager::distance() const
{
    return _distance;
}

void MatrixManager::setDistance(const GLdouble &distance)
{
    _distance = distance;
}
