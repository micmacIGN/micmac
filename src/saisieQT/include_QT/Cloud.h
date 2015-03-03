#ifndef __CLOUD__
#define __CLOUD__

#include "3DObject.h"
#include "general/ply_struct.h"

class GlVertex : public cObjectGL
{
public:
	GlVertex(QVector3D pos = QVector3D(0.f,0.f,0.f), QColor color_default = Qt::white, QVector3D nrm = QVector3D(0.f,0.f,0.f)) :
        cObjectGL(pos, color_default),
        _nrm(nrm)
    {}
    void draw(){}

	QVector3D getNormal() { return _nrm; }

	QVector3D _nrm;
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

	QVector3D   getSum() { return _sum; }

    void    draw();

private:
    vector<GlVertex> _vertices;

    QGLBuffer   _vertexbuffer;
    QGLBuffer   _vertexColor;

    int         _type;  //data stored (0: xyz, 1:xyzrgb, 2: xyzrgba 3:xyznxnynz 4:xyznxnynzrgb 5:xyznxnynzrgba)

	QVector3D       _sum;  //coordinate sums to compute scene center
};


#endif
