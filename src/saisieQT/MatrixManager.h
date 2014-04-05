#ifndef __MATRIXMANAGER__
#define __MATRIXMANAGER__

#include "Engine.h"

//! View orientation
enum VIEW_ORIENTATION {  TOP_VIEW,      /**< Top view (eye: +Z) **/
                         BOTTOM_VIEW,	/**< Bottom view **/
                         FRONT_VIEW,	/**< Front view **/
                         BACK_VIEW,     /**< Back view **/
                         LEFT_VIEW,     /**< Left view **/
                         RIGHT_VIEW     /**< Right view **/
};

class selectInfos;

class MatrixManager
{
public:
    MatrixManager();
    ~MatrixManager();

    GLdouble*   getModelViewMatrix(){return _mvMatrix;}

    GLdouble*   getProjectionMatrix(){return _projMatrix;}

    QPointF     translateImgToWin(float zoom);

    GLint*      getGLViewport(){return _glViewport;}

    void        setGLViewport(GLint x, GLint y,GLsizei width, GLsizei height);

    void        doProjection(QPointF point, float zoom);

    //void        orthoProjection();

    void        translate(float x, float y);

    GLdouble    mvMatrix(int i)     { return _mvMatrix[i];   }

    GLdouble    projMatrix(int i)   { return _projMatrix[i]; }

    GLint       Viewport(int i)     { return _glViewport[i]; }

    GLint       vpWidth()     { return _glViewport[2]; }

    GLint       vpHeight()    { return _glViewport[3]; }

    void        setMatrices();

    void        rotate(GLdouble* matrix, float rX, float rY, float rZ, float factor);
    void        rotate(float rX, float rY, float rZ, float factor);

    void        rotateArcBall(float rX, float rY, float rZ, float factor);

    void        importMatrices(selectInfos &infos);
    void        exportMatrices(selectInfos &infos);

    //! 3D point projection in viewport
    void        getProjection(QPointF &P2D, Pt3dr P);

    //! Project a point from window to image
    QPointF     WindowToImage(const QPointF &winPt, float zoom);

    //! Project a point from image to window
    QPointF     ImageToWindow(const QPointF &imPt, float zoom);

    static void mglOrtho( GLdouble left, GLdouble right, GLdouble bottom, GLdouble top, GLdouble near_val, GLdouble far_val );

    //! Reset rotation matrix
    void        resetRotationMatrix();

    void        resetModelViewMatrix();

    //! Reset translation matrix
    void        resetTranslationMatrix(Pt3dr center = Pt3dr(0.f,0.f,0.f));

    void        resetAllMatrix(Pt3dr center = Pt3dr(0.f,0.f,0.f));

    void        applyTransfo();

    void        setModelViewMatrix();

    void        zoom(float zoom, float far);

    float       getGlRatio(){return m_glRatio;}

    void        setView(VIEW_ORIENTATION orientation, Pt3d<double> centerScene);

    GLdouble    m_rotationMatrix[16];
    GLdouble    m_translationMatrix[3];

    void        translate(float tX, float tY, float tZ, float factor);
    GLdouble    distance() const;
    void        setDistance(const GLdouble &distance);

    void        arcBall();
    Pt3d<double> centerScene() const;
    void        setCenterScene(const Pt3d<double> &centerScene);

    void        MatrixInverse(GLdouble OpenGLmatIn[], float matOut[][4], float *vec);

private:
    //! GL context aspect ratio (width/height)
    float       m_glRatio;

    GLdouble    *_mvMatrix;
    GLdouble    *_projMatrix;
    GLint       *_glViewport;
    GLdouble    _rX;
    GLdouble    _rY;
    GLdouble    _distance;
    Pt3d<double> _centerScene;

};

#endif
