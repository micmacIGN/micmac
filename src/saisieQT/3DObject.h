#ifndef __3DOBJECT__
#define __3DOBJECT__

#include "StdAfx.h"

#ifdef Int
    #undef Int
#endif
#include <QColor>
#include <QGLWidget>

#include "GL/glu.h"

#define QMaskedImage cMaskedImage<QImage>

enum LINE_STYLE
{
    LINE_NOSTIPPLE,
    LINE_STIPPLE
};

class cObject
{
    public:
        cObject();
        cObject(Pt3dr pt, QColor col);
        virtual ~cObject();


        Pt3dr   getPosition()   { return _position; }
        QColor  getColor()      { return _color;    }
        float   getScale()      { return _scale;    }
        bool    isVisible()     { return _bVisible; }

        void    setPosition(Pt3dr const &aPt)  { _position = aPt;  }
        void    setColor(QColor const &aCol)   { _color = aCol;    }
        void    setVisible(bool aVis)          { _bVisible = aVis; }
        void    setScale(float aScale)         { _scale = aScale;  }

        cObject & operator = (const cObject &);

    protected:

        Pt3dr   _position;
        QColor  _color;
        float   _scale;

        float   _alpha;
        bool    _bVisible;
};

class cObjectGL : public cObject
{
   public:
        cObjectGL(){}
        virtual ~cObjectGL(){}

        virtual void draw()=0;

protected:

        void enableOptionLine();

        void disableOptionLine();

};

class cCircle : public cObjectGL
{
    public:
        cCircle(Pt3dr, QColor, float, float, bool, int dim);
        cCircle(int dim);

        void    draw();

        void    setLineWidth(float width){_lineWidth = width;}

    private:
        float   _lineWidth;
        int     _dim;
};

class cCross : public cObjectGL
{
    public:
        cCross(Pt3dr, QColor, float, float, bool, int dim);

        void    draw();

        void    setLineWidth(float width){_lineWidth = width;}

    private:
        float   _lineWidth;
        int     _dim;
};

class cBall : public cObjectGL
{
    public:

        cBall(Pt3dr pt = Pt3dr(0.f,0.f,0.f), float scale = 1.f, bool isVis = true, float lineWidth = 1.f);
        ~cBall();

        void    setPosition(Pt3dr const &aPt);
        Pt3dr   getPosition();
        void    setColor(QColor const &aCol);
        void    setVisible(bool aVis);
        void    setScale(float aScale);

        void    draw();

        void    setLineWidth(float width);

    private:
        float   _lineWidth;

        cCircle *_cl0;
        cCircle *_cl1;
        cCircle *_cl2;

        cCross  *_cr0;
        cCross  *_cr1;
        cCross  *_cr2;
};

class cAxis : public cObjectGL
{
    public:
        cAxis(Pt3dr pt, float scale = 1.f);

        void    draw();

        void    setLineWidth(float width){_lineWidth = width;}

    private:
        float   _lineWidth;
};

class cBBox : public cObjectGL
{
    public:
        cBBox(Pt3dr pt, float scale, Pt3dr min, Pt3dr max);

        void    draw();

        void    setLineWidth(float width){_lineWidth = width;}

        void set(Pt3d<double> min, Pt3d<double> max);
private:
        float   _lineWidth;
        Pt3dr   _min;
        Pt3dr   _max;
};

class cCam : public cObjectGL
{
    public:
        cCam(CamStenope *pCam, float scale, bool isVisible = true);

        void    draw();

        void    setLineWidth(float width){_lineWidth = width;}
        void    setpointSize(float size) {_pointSize = size;}

    private:
        float   _lineWidth;
        float   _pointSize;

        CamStenope *_Cam;
};


class cPolygonHelper;

class cPolygon : public cObjectGL
{
    public:

        cPolygon(float lineWidth = 1.0f, QColor lineColor = Qt::green,  QColor pointColor = Qt::red,int style = LINE_NOSTIPPLE);

        void    draw();

        void    close();

        bool    isPointInsidePoly(const QPointF& P);

        void    findClosestPoint(const QPointF &pos);

        void    removeClosestPoint(QPointF pos);

        void    setLineWidth(float width){_lineWidth = width;}
        void    setpointSize(float size) {_pointSize = size;}

        void    add(QPointF const &pt){_points.push_back(pt);}
        void    addPoint(QPointF const &pt);

        void    clear();
        void    clearPoints() {_points.clear();}

        void    setClosed(bool aBool){ _bPolyIsClosed = aBool; }
        bool    isClosed(){ return _bPolyIsClosed;}

        int     size(){ return _points.size(); }

        QPointF & operator[](int ak){ return _points[ak]; }
        const QPointF & operator[](int ak) const { return _points[ak]; }

        cPolygon & operator = (const cPolygon &);

        void    insertPoint( int i, const QPointF & value );

        void    insertPoint();

        void    removePoint( int i );

        QVector <QPointF> const getVector(){ return _points; }
        void setVector(QVector <QPointF> const &aPts){ _points = aPts; }

        int     idx(){return _idx;}

        void    setPointSelected(){_bSelectedPoint = true;}
        bool    isPointSelected(){return _bSelectedPoint;}
        void    resetSelectedPoint(){_bSelectedPoint = false;}

        cPolygonHelper* helper() {return _helper;}

        void    refreshHelper(QPointF pos, bool insertMode);

        void    finalMovePoint(QPointF pos);

protected:
        cPolygon(float lineWidth, QColor lineColor,  QColor pointColor, bool withHelper, int style = LINE_STIPPLE);

        QVector <QPointF>   _points;
        cPolygonHelper*     _helper;
        float               _lineWidth;
        QColor              _lineColor;
        int                 _idx;

private:
        float               _pointSize;
        float               _sqr_radius;

        //!states if polygon is closed
        bool                _bPolyIsClosed;

        //!states if point with index _idx is selected
        bool                _bSelectedPoint;

        int                 _style;
};

class cPolygonHelper : public cPolygon
{

public:

    cPolygonHelper(cPolygon* polygon,float lineWidth, QColor lineColor = Qt::blue,  QColor pointColor = Qt::blue);

    void   fill(const QPointF &pos);

    void   fill2(const QPointF &pos);

    void   SetPoints(QPointF p1, QPointF p2, QPointF p3);

private:

    cPolygon* _polygon;
};
#include <QGLShaderProgram>

class cImageGL : public cObjectGL
{
    public:
        cImageGL(float gamma = 1.0f);
        ~cImageGL();

        void    draw(QColor color);

        void    drawQuad();

        void    draw();

        void    setPosition(GLfloat originX, GLfloat originY);
        void    setDimensions(GLfloat glh, GLfloat glw);

        void    PrepareTexture(QImage *pImg);

        void    ImageToTexture(QImage *pImg);

        GLuint* getTexture(){return &_texture;}

        //height and width of original data
        int     width()  {return _size.width();}
        int     height() {return _size.height();}

        void    setGamma(float gamma){_gamma = (gamma >= 0) ? gamma : 0;}

        float   getGamma(){ return _gamma;}

        void    incGamma(float dgamma){setGamma(_gamma + dgamma);}

private:

        QGLShaderProgram _program;

        int     _matrixLocation;
        int     _texLocation  ;
        int     _gammaLocation;

        GLfloat _originX;
        GLfloat _originY;
        GLfloat _glh;
        GLfloat _glw;

        QSize   _size;

        //! Texture image
        GLuint  _texture;
        float   _gamma;
        GLfloat _pmat[16];

};

template<class T>
class cMaskedImage
{

public:

    cMaskedImage(float gamma = 1.0f):
        _m_image(NULL),
        _m_mask(NULL),
        _m_newMask(true),
        _gamma(gamma)
    {}

    ~cMaskedImage()
    {}

    void deallocImages()
    {
        if(_m_image != NULL) delete _m_image;
        if(_m_mask != NULL) delete _m_mask;
    }

    T           *_m_image;
    T           *_m_mask;

    bool        _m_newMask;
    float       _gamma;

};

class cMaskedImageGL : public cMaskedImage<cImageGL>, public cObjectGL
{

public:

    cMaskedImageGL(){}

    cMaskedImageGL(QMaskedImage &qMaskedImage);

    void SetDimensions(float h,float w)
    {
        _m_image->setDimensions(h,w);
        _m_mask->setDimensions(h,w);
    }

    void draw();

};

#endif //__3DObject__
