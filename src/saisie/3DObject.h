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
        virtual ~cObject();


        Pt3dr   getPosition()   { return _position;  }
        QColor  getColor()      { return _color;     }
        float   getScale()      { return _scale;     }
        bool    isVisible()     { return _bVisible;  }

        void    setPosition(Pt3dr const &aPt)  { _position = aPt;   }
        void    setColor(QColor const &aCol)   { _color = aCol;     }
        void    setVisible(bool aVis)          { _bVisible = aVis;  }
        void    setScale(float aScale)         { _scale = aScale;   }

        virtual void draw()=0;


        cObject & operator = (const cObject &);

    protected:

        Pt3dr   _position;
        QColor  _color;
        float   _scale;

        float   _alpha;
        bool    _bVisible;
};

class cCircle : public cObject
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

class cCross : public cObject
{
    public:
        cCross(Pt3dr, QColor, float, float, bool, int dim);

        void    draw();

        void    setLineWidth(float width){_lineWidth = width;}

    private:
        float   _lineWidth;
        int     _dim;
};

class cBall : public cObject
{
    public:

        cBall(Pt3dr pt = Pt3dr(0.f,0.f,0.f), float scale = 1.f, float lineWidth = 1.f, bool isVis = false);
        ~cBall();

        void    setPosition(Pt3dr const &aPt);
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

class cAxis : public cObject
{
    public:
        cAxis();

        void    draw();

        void    setLineWidth(float width){_lineWidth = width;}

    private:
        float   _lineWidth;
};

class cBBox : public cObject
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

class cCam : public cObject
{
    public:
        cCam(CamStenope *pCam);

        void    draw();

        void    setLineWidth(float width){_lineWidth = width;}
        void    setpointSize(float size) {_pointSize = size;}

    private:
        float   _lineWidth;
        float   _pointSize;

        CamStenope *_Cam;
};

class cPolygon : public cObject
{
    public:
        cPolygon();

        void    draw();

        void    setLineWidth(float width){_lineWidth = width;}
        void    setpointSize(float size) {_pointSize = size;}

        void    add(QPointF const &pt){ _Poly.push_back(pt); }

        void    clear(){ _Poly.clear(); }

        void    setClosed(bool aBool){ _bPolyIsClosed = aBool; }
        bool    isClosed(){ return _bPolyIsClosed;}

        int     size(){ return _Poly.size(); }

        QPointF & operator[](int ak){ return _Poly[ak]; }
        const QPointF & operator[](int ak) const { return _Poly[ak]; }

        cPolygon & operator = (const cPolygon &);

        void    insert( int i, const QPointF & value ){ _Poly.insert(i,value); }

        void    remove ( int i ){ _Poly.remove(i);}

        QVector <QPointF> getVector(){ return _Poly; }

        void    close();

        void    setIdx(int idx){_idx =idx;}

    private:
        float               _lineWidth;
        float               _pointSize;

        bool                _bPolyIsClosed;

        int                 _idx;

        QVector <QPointF>   _Poly;
};

void glDrawUnitCircle(uchar dim, float cx, float cy, float r, int steps = 64);

#endif //__3DObject__
