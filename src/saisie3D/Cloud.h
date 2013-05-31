#ifndef __CLOUD__
#define __CLOUD__

#include <vector>
#include <string>

#include <QColor>
#include "mmVector3.h"

namespace Cloud_
{
    class Vertex
    {
        public:
            Vertex(Vector3, QColor);

            float x() {return m_position.x;}
            float y() {return m_position.y;}
            float z() {return m_position.z;}

            Vector3 getCoord()   {return m_position;}
            QColor  getColor()   {return m_color;}
            bool    isVisible()  {return m_bVisible;}

            void setCoord(Vector3 const &aPt)   {m_position = aPt;}
            void setColor(QColor const &aCol)   {m_color = aCol;}
            void setVisible(bool aVis)          {m_bVisible = aVis;}

        private:
            Vector3 m_position;
            QColor  m_color;
            bool    m_bVisible;

    };

    class Cloud
    {
        public:
            Cloud();
            Cloud(std::vector<Vertex> const &);

            // renvoie true si le fichier a pu être lu
            static Cloud* loadPly( std::string );

            void    addVertex( const Vertex & );
            Vertex& getVertex( unsigned int );
            void    setVertex( unsigned int, Vertex const & );
            int     size();

            void    setTranslation( const Vector3 & aPt ) {m_translation = aPt;}
            Vector3 getTranslation(){return m_translation;}

            void    setScale( const float & aS ) {m_scale = aS;}
            float   getScale(){return m_scale;}

            void    clear();

        private:
            std::vector<Vertex> m_vertices;
            Vector3 m_translation;
            float m_scale;
    };

}

#endif
