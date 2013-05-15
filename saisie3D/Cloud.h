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

            void setX(float aX){m_x = aX;}
            void setY(float aY){m_y = aY;}
            void setZ(float aZ){m_z = aZ;}

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

            Pt3D   getCoord()   {return m_position;}
            QColor getColor()   {return m_color;}
            bool   isVisible()  {return m_bVisible;}

            void setCoord(Pt3D const &aPt)      {m_position = aPt;}
            void setColor(QColor const &aCol)   {m_color = aCol;}
            void setVisible(bool aVis)          {m_bVisible = aVis;}

        private:
            Pt3D    m_position;
            QColor  m_color;
            bool    m_bVisible;

    };

    class Cloud
    {
        public:
            Cloud();

            // renvoie true si le fichier a pu être lu
            bool    loadPly( const std::string & );

            void    addVertex( const Vertex & );
            Vertex& getVertex( unsigned int );
            void    setVertex( unsigned int, Vertex const & );
            int     size();

            void    setTranslation( const Pt3D & aPt ) {m_translation = aPt;}
            Pt3D    getTranslation(){return m_translation;}

            void    setScale( const float & aS ) {m_scale = aS;}
            float   getScale(){return m_scale;}

            void    clear();

        private:
            std::vector<Vertex> m_vertices;
            Pt3D m_translation;
            float m_scale;
    };

}

#endif
