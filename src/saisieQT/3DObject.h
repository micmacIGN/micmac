#ifndef __3DOBJECT__
#define __3DOBJECT__

#include "StdAfx.h"

#ifdef Int
    #undef Int
#endif
#include <QColor>
#include <QGLWidget>

#include "GL/glu.h"

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

        cBall(Pt3dr pt = Pt3dr(0.f,0.f,0.f), float scale = 1.f, float lineWidth = 1.f, bool isVis = false);
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
        cAxis();

        void    draw();

        void    setLineWidth(float width){_lineWidth = width;}

    private:
        float   _lineWidth;
};

class cBBox : public cObjectGL
{
    public:
        cBBox();

        void    set(float minX, float minY, float minZ, float maxX, float maxY, float maxZ);

        void    draw();

        void    setLineWidth(float width){_lineWidth = width;}

    private:
        float   _lineWidth;
        float   _minX;
        float   _minY;
        float   _minZ;
        float   _maxX;
        float   _maxY;
        float   _maxZ;
};

class cCam : public cObjectGL
{
    public:
        cCam();
        cCam(CamStenope *pCam);

        void    draw();

        void    setLineWidth(float width){_lineWidth = width;}
        void    setpointSize(float size) {_pointSize = size;}

    private:
        float   _lineWidth;
        float   _pointSize;

        CamStenope *_Cam;
};

class cPolygon : public cObjectGL
{
    public:
        cPolygon();
        cPolygon(const cPolygon&);

        void    draw();
        void    drawDihedron();

        void    close();

        bool    isPointInsidePoly(const QPointF& P);

        //!used for point insertion
        void    fillDihedron(const QPointF &pos, cPolygon &dihedron);

        //!used for point moving
        void    fillDihedron2(const QPointF &pos, cPolygon &dihedron);

        void    findClosestPoint(const QPointF &pos);

        void    setLineWidth(float width){_lineWidth = width;}
        void    setpointSize(float size) {_pointSize = size;}

        void    add(QPointF const &pt){ _points.push_back(pt); }

        void    clear();
        void    clearPoints() {_points.clear();}

        void    setClosed(bool aBool){ _bPolyIsClosed = aBool; }
        bool    isClosed(){ return _bPolyIsClosed;}

        int     size(){ return _points.size(); }

        QPointF & operator[](int ak){ return _points[ak]; }
        const QPointF & operator[](int ak) const { return _points[ak]; }

        cPolygon & operator = (const cPolygon &);

        void    insert( int i, const QPointF & value );

        void    remove ( int i );

        QVector <QPointF> const getVector(){ return _points; }
        void setVector(QVector <QPointF> const &aPts){ _points = aPts; }

        int     idx(){return _idx;}

        void    clicked(){_click++;}
        int     click(){return _click;}
        void    resetClick(){_click=0;}

private:
        float               _lineWidth;
        float               _pointSize;
        float               _sqr_radius;

        bool                _bPolyIsClosed;

        int                 _idx;

        int                 _click;

        QVector <QPointF>   _points;
};

class cImageGL : public cObjectGL
{
    public:
        cImageGL();
        ~cImageGL();

        void    draw(QColor color);

        void    drawQuad();

        void    draw();

        void    setPosition(GLfloat originX, GLfloat originY);
        void    setDimensions(GLfloat glh, GLfloat glw);

        void    setDimensions(GLfloat originX, GLfloat originY, GLfloat glh, GLfloat glw);

        void    PrepareTexture(QImage *pImg);

        void    ImageToTexture(QImage *pImg);

        GLuint* getTexture(){return &_texture;}

        //height and width of original data
        int     width()  {return _size.width();}
        int     height() {return _size.height();}

private:
        GLfloat _originX;
        GLfloat _originY;
        GLfloat _glh;
        GLfloat _glw;

        QSize   _size;

        //! Texture image
        GLuint  _texture;
};

#endif //__3DObject__
