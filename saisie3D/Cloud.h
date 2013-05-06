#ifndef __CLOUD__
#define __CLOUD__

#include <vector>
#include <string>

#include <QColor>

namespace Cloud_
{
    class Pt3D
    {
        public:
            Pt3D();
            Pt3D(float x, float y, float z);

            float x() {return m_x;}
            float y() {return m_y;}
            float z() {return m_z;}

            Pt3D &operator=(const Pt3D &);

        private:
            float m_x;
            float m_y;
            float m_z;
    };

    class Vertex
    {
        public:
            Vertex(Pt3D, QColor);

            float x() {return m_position.x();}
            float y() {return m_position.y();}
            float z() {return m_position.z();}

            Pt3D m_position;
            QColor m_color;

    };

    class Cloud
    {
        private:
            std::vector<Vertex> m_vertices;

        public:
            Cloud(){}

            // renvoie true si le fichier a pu être lu
            bool    loadPly( const std::string & );

            void    addVertex( const Vertex & );
            Vertex  getVertex( unsigned int );
            int     getVertexNumber();
    };

}

#endif
