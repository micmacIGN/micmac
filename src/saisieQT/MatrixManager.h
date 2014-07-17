#ifndef __MATRIXMANAGER__
#define __MATRIXMANAGER__

#include "Engine.h"
#include "mmglu.h"

//! View orientation
enum VIEW_ORIENTATION {  TOP_VIEW,      /**< Top view (eye: +Z) **/
                         BOTTOM_VIEW,	/**< Bottom view **/
                         FRONT_VIEW,	/**< Front view **/
                         BACK_VIEW,     /**< Back view **/
                         LEFT_VIEW,     /**< Left view **/
                         RIGHT_VIEW     /**< Right view **/
};

#define HANGLE (PI/24)

class selectInfos;

class MatrixManager
{
public:
    MatrixManager();
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

    void        importMatrices(selectInfos &infos);
    void        exportMatrices(selectInfos &infos);

    //! 3D point projection in viewport
    void        getProjection(QPointF &P2D, Pt3dr P);

    //! Project a point from window to image
    QPointF     WindowToImage(const QPointF &winPt, float glOrthoZoom);

    //! Project a point from image to window
    QPointF     ImageToWindow(const QPointF &imPt, float glOrthoZoom);

    static void mglOrtho( GLdouble left, GLdouble right, GLdouble bottom, GLdouble top, GLdouble near_val, GLdouble far_val );

    //! Reset rotation matrix
    void        resetRotationMatrix();

    void        resetModelViewMatrix();

    //! Reset translation matrix
    void        resetTranslationMatrix(Pt3dr center = Pt3dr(0.f,0.f,0.f));

    void        resetAllMatrix(Pt3dr center = Pt3dr(0.f,0.f,0.f));

    void        setModelViewMatrix();

    void        glOrthoZoom(float glOrthoZoom, float far);

    float       getGlRatio(){return m_glRatio;}

    void        setView(VIEW_ORIENTATION orientation, Pt3dr centerScene);

    GLdouble    m_rotationMatrix[16];
    GLdouble    m_translationMatrix[3];

    void        translate(float tX, float tY, float tZ);
    GLdouble    distance() const;
    void        setDistance(const GLdouble &distance);

    void        SetArcBallCamera(float zoom);
    Pt3dr       centerScene() const;
    void        setCenterScene(const Pt3dr &centerScene);

    void        MatrixInverse(GLdouble OpenGLmatIn[], float matOut[][4], float *vec);   

    void        handleRotation(QPointF clicPosMouse);

    void        setMatrixDrawViewPort();

    void        applyAllTransformation(bool mode2D, QPoint pt, float zoom);

    GLdouble rY() const;
    void setRY(const GLdouble &rY);

    void setSceneTopo(const Pt3d<double> &centerScene, float diametre);

    QPointF     screen2TransABall(QPointF ptScreen);

private:
    //! GL context aspect ratio (width/height)
    float       m_glRatio;

    GLdouble    *_mvMatrix;
    GLdouble    *_projMatrix;
    GLint       *_glViewport;
    GLdouble    _rX;
    GLdouble    _rY;
    GLdouble    _upY;

    GLdouble    _distance;
    Pt3dr       _centerScene;

    float       _diameterScene;

    int         _lR;
    int         _uD;

    Pt3d<double> _targetCamera;

};

#endif
