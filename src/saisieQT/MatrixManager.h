#ifndef __MATRIXMANAGER__
#define __MATRIXMANAGER__

#include "3DObject.h"
#include "Engine.h"

class MatrixManager
{
public:
    MatrixManager();
    ~MatrixManager();

    GLdouble*   getModelViewMatrix(){return _mvMatrix;}
    GLdouble*   getProjectionMatrix(){return _projMatrix;}
    GLint*      getGLViewport(){return _glViewport;}

    void        doProjection(QPointF point, float zoom);

    void        orthoProjection();

    void        scaleAndTranslate(float x, float y, float zoom);

    GLdouble    mvMatrix(int i)     { return _mvMatrix[i];   }
    GLdouble    projMatrix(int i)   { return _projMatrix[i]; }

    GLint       Viewport(int i)     { return _glViewport[i]; }

    GLint       vpWidth()     { return _glViewport[2]; }
    GLint       vpHeight()    { return _glViewport[3]; }

    void        setMatrices();

    void        rotateMatrix(GLdouble* matrix, float rX, float rY, float rZ, float factor);
    void        rotateMatrix(float rX, float rY, float rZ, float factor);

    void        importMatrices(selectInfos &infos);
    void        exportMatrices(selectInfos &infos);

    void        resetPosition(){m_glPosition[0] = m_glPosition[1] = 0.f;}

    //! 3D point projection in viewport
    void        getProjection(QPointF &P2D, Pt3dr P);

    //! Project a point from window to image
    QPointF     WindowToImage(const QPointF &pt, float zoom);

    //! Project a point from image to window
    QPointF     ImageToWindow(const QPointF &im, float zoom);

    cPolygon    PolygonImageToWindow(cPolygon polygon, float zoom);

    static void mglOrtho( GLdouble left, GLdouble right, GLdouble bottom, GLdouble top, GLdouble near_val, GLdouble far_val );

    GLfloat     m_glPosition[2];

    //! Reset rotation matrix
    void        resetRotationMatrix();

    void        resetModelViewMatrix();

    //! Reset translation matrix
    void        resetTranslationMatrix(Pt3dr center = Pt3dr(0.f,0.f,0.f));

    void        resetAllMatrix(Pt3dr center = Pt3dr(0.f,0.f,0.f))
    {
                resetRotationMatrix();

                resetModelViewMatrix();

                resetTranslationMatrix(center);
    }

    void        applyTransfo();

    void        setModelViewMatrix();

    void        zoom(float zoom, float far,float m_glRatio);

    GLdouble    m_rotationMatrix[16];
    GLdouble    m_translationMatrix[3];

private:
    GLdouble    *_mvMatrix;
    GLdouble    *_projMatrix;
    GLint       *_glViewport;    
};

#endif
