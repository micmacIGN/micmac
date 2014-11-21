#ifndef __CLOUD__
#define __CLOUD__

#include "3DObject.h"

class GlVertex : public cObjectGL
{
public:
    GlVertex(Pt3dr pos = Pt3dr(0.f,0.f,0.f), QColor color_default = Qt::white, Pt3dr nrm = Pt3dr(0.f,0.f,0.f)) :
        cObjectGL(pos, color_default),
        _nrm(nrm)
    {}
    void draw(){}

    Pt3dr getNormal() { return _nrm; }

    Pt3dr _nrm;
};

class GlCloud : public cObjectGL
{
public:
    GlCloud(){}
    GlCloud(vector<GlVertex> const &, int type=1);

    static GlCloud* loadPly(string,  int *incre = NULL);

    void    addVertex( const GlVertex & vertex);
    GlVertex& getVertex( uint );
    int     size();
    int     type() { return _type; }

    void    clear();

    void    setBufferGl(bool onlyColor=false);

    Pt3dr   getSum() { return _sum; }

    void    draw();

private:
    vector<GlVertex> _vertices;

    QGLBuffer   _vertexbuffer;
    QGLBuffer   _vertexColor;

    int         _type;  //data stored (0: xyz, 1:xyzrgb, 2: xyzrgba 3:xyznxnynz 4:xyznxnynzrgb 5:xyznxnynzrgba)

    Pt3dr       _sum;  //coordinate sums to compute scene center
};


#endif
