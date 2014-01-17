#ifndef __CLOUD__
#define __CLOUD__

#include <vector>
#include <string>

#include <fstream>
#include <iostream>

#ifdef Int
    #undef Int
#endif

#include <QtOpenGL/QGLBuffer>

#include "3DObject.h"

#define GlVertex cObject

class GlCloud : public cObjectGL
{
public:
    GlCloud(){}
    GlCloud(vector<GlVertex> const &);

    static GlCloud* loadPly(string,  int *incre = NULL);

    void    addVertex( const GlVertex & );
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
