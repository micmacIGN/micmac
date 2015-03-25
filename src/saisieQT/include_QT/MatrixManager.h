#ifndef __MATRIXMANAGER__
#define __MATRIXMANAGER__

#include "mmglu.h"
#include "HistoryManager.h"
#include "Settings.h"

//! View orientation
enum VIEW_ORIENTATION {  TOP_VIEW,      /**< Top view (eye: +Z) **/
                         BOTTOM_VIEW,	/**< Bottom view **/
                         FRONT_VIEW,	/**< Front view **/
                         BACK_VIEW,     /**< Back view **/
                         LEFT_VIEW,     /**< Left view **/
                         RIGHT_VIEW     /**< Right view **/
};

#define HANGLE (PI/24)

#define DUMPV(varname)   printVecteur(varname,#varname);


enum matID
{
    M11,
    M12,
    M13,
    M14,
    M21,
    M22,
    M23,
    M24,
    M31,
    M32,
    M33,
    M34,
    M41,
    M42,
    M43,
    M44
    };

class MatrixManager
{
public:
    MatrixManager(eNavigationType nav = eNavig_Ball);
    ~MatrixManager();

    GLdouble*   getModelViewMatrix(){return _mvMatrix;}

    GLdouble*   getProjectionMatrix(){return _projMatrix;}

    GLint*      getGLViewport(){return _glViewport;}

    void        setGLViewport(GLint x, GLint y,GLsizei width, GLsizei height);

    void        doProjection(QPointF point, float glOrthoZoom);

    void        resetMatrixProjection(float x, float y);

    GLdouble    mvMatrix(int i)     { return _mvMatrix[i];   }

    GLdouble    projMatrix(int i)   { return _projMatrix[i]; }

    GLint       Viewport(int i)     { return _glViewport[i]; }

    GLint       vpWidth()     { return _glViewport[2]; }

    GLint       vpHeight()    { return _glViewport[3]; }

    void        setMatrices();

    void        rotateArcBall(float rX, float rY, float rZ, float factor);

    void        importMatrices(const selectInfos &infos);
    void        exportMatrices(selectInfos &infos);

    //! 3D point projection in viewport
    void        getProjection(QPointF &P2D, QVector3D P);

    //! projection from viewport to world coordinates
    void        getInverseProjection(QVector3D &P, QPointF P2D, float dist);

    //! Project a point from window to image
    QPointF     WindowToImage(const QPointF &winPt, float glOrthoZoom);

    //! Project a point from image to window
    QPointF     ImageToWindow(const QPointF &imPt, float glOrthoZoom);

    static void mglOrtho( GLdouble left, GLdouble right, GLdouble bottom, GLdouble top, GLdouble near_val, GLdouble far_val );

    //! Reset rotation matrix
    void        resetRotationMatrix();

    void        resetModelViewMatrix();

    //! Reset translation matrix
    void        resetTranslationMatrix(QVector3D center = QVector3D(0.f,0.f,0.f));

    void        resetViewPort();

    void        resetAllMatrix(QVector3D center = QVector3D(0.f,0.f,0.f), bool resetALL = true);

//    void        setModelViewMatrix();

    void        glOrthoZoom(float glOrthoZoom, float far);

    float       getGlRatio(){return m_glRatio;}

    void        setView(VIEW_ORIENTATION orientation, QVector3D centerScene);

    GLdouble    m_rotationMatrix[16];
    GLdouble    m_translationMatrix[3];

    void        translate(float tX, float tY, float tZ);
    GLdouble    distance() const;
    void        setDistance(const GLdouble &distance);

    void        setArcBallCamera(float distance);
    QVector3D       centerScene() const;
    void        setCenterScene(const QVector3D &centerScene);

    void        MatrixInverse(GLdouble* OpenGLmatIn, GLdouble* matOutGL = NULL, GLdouble* vec = NULL);

    void        handleRotation(QPointF clicPosMouse);

    void        setMatrixDrawViewPort();

    void        applyAllTransformation(bool mode2D, QPoint pt, float zoom);

    GLdouble    rY() const;
    void        setRY(const GLdouble &rY);

    void        setSceneTopo(const QVector3D& centerScene, float diametre);

    QPointF     screen2TransABall(QPointF ptScreen);

    void		printVecteur(GLdouble* posCameraOut,const char* nameVariable);

    eNavigationType eNavigation() const;
    void		setENavigation(const eNavigationType& eNavigation);

    bool		isBallNavigation();

    QPointF		centerVP();

    QRectF      getRectViewportToImage(float zoom);

private:
    //! GL context aspect ratio (width/height)

    void        multiplication(GLdouble* posIn, GLdouble* posOut, GLdouble* mat);

    void		matriceRotation(GLdouble* axe, GLdouble* matRot, GLdouble angle);

    void        multiplicationMat(GLdouble* mat1, GLdouble* mat2,GLdouble* matOut);

    void		addTranslationToMat(GLdouble* mat, GLdouble* translation);

    bool		gluInvertMatrix(const GLdouble m[16], GLdouble invOut[16]);

    void		CopyMatrix(GLdouble * source,GLdouble * in);

    float       m_glRatio;

    GLdouble    _mvMatrix[16];
    GLdouble    _projMatrix[16];
    GLint       _glViewport[4];


    GLdouble    _rX;
    GLdouble    _rY;
    GLdouble    _rZ;

    GLdouble    _upY;

    GLdouble    _distance;
    QVector3D   _centerScene;

    float       _diameterScene;

    int         _lR;
    int         _uD;

    QVector3D	_targetCamera;
    QVector3D	_camPos;

    GLdouble		_cX;
    GLdouble		_cY;
    GLdouble		_sX;
    GLdouble		_sY;

    GLdouble	  _mvMatrixOld[16];
//	GLdouble	  _mvMatrixOldInv[16];

//	GLdouble    *_MatrixPassageCamera;
//	GLdouble    *_MatrixPassageCameraInv;
//	GLdouble    *_positionCamera;
    void loadIdentity(GLdouble* matOut);

    eNavigationType _eNavigation;

    float		_factor;
};

#endif
