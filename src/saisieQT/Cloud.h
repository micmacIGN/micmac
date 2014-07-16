#ifndef __CLOUD__
#define __CLOUD__

#include <vector>
#include <string>

#include <fstream>
#include <iostream>

#ifdef Int
    #undef Int
#endif

#include "3DObject.h"

#include <QtOpenGL/QGLBuffer>



class GlVertex : public cObjectGL
{
public:
    GlVertex(Pt3dr pos = Pt3dr(0.f,0.f,0.f), QColor color_default = Qt::white) :
        cObjectGL(pos, color_default)
    {}
    void draw(){}
};

class GlCloud : public cObjectGL
{
public:
    GlCloud(){}
    GlCloud(vector<GlVertex> const &);

    static GlCloud* loadPly(string,  int *incre = NULL);

    void    addVertex( const GlVertex & vertex);
    GlVertex& getVertex( uint );
    int     size();

    void    clear();

    void    setBufferGl(bool onlyColor=false);

    void    draw();

private:
    vector<GlVertex> _vertices;

    QGLBuffer   _vertexbuffer;
    QGLBuffer   _vertexColor;
};


#endif
