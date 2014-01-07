#ifndef __CLOUD__
#define __CLOUD__

#include <vector>
#include <string>

#ifdef Int
    #undef Int
#endif

#include <QtOpenGL/QGLBuffer>

#include "3DObject.h"

namespace Cloud_
{
    #define Vertex cObject

    class Cloud : public cObjectGL
    {
        public:
            Cloud(){}
            Cloud(vector<Vertex> const &);

            static Cloud* loadPly(string,  int *incre = NULL);

            void    addVertex( const Vertex & );
            Vertex& getVertex( uint );
            int     size();

            void    clear();

            void    setBufferGl(bool onlyColor=false);

            void    draw();

        private:
            vector<Vertex> _vertices;

            QGLBuffer   _vertexbuffer;
            QGLBuffer   _vertexColor;
    };
}

#endif
