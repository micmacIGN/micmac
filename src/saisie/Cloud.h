#ifndef __CLOUD__
#define __CLOUD__

#include <vector>
#include <string>

#include "StdAfx.h"

#ifdef Int
    #undef Int
#endif
#include <QColor>

namespace Cloud_
{
    class Vertex
    {
        public:
            Vertex();
            Vertex(Pt3dr, QColor);

            float   x() {return _position.x;}
            float   y() {return _position.y;}
            float   z() {return _position.z;}

            Pt3dr   getCoord()   {return _position;}
            QColor  getColor()   {return _color;}

            //! States if a point is visible (ie: selected)
            bool    isVisible()  {return _bVisible;}

            void    setCoord(Pt3dr const &aPt)     {_position = aPt;}
            void    setColor(QColor const &aCol)   {_color = aCol;}
            void    setVisible(bool aVis)          {_bVisible = aVis;}

        private:
            Pt3dr   _position;
            QColor  _color;
            bool    _bVisible;

    };

    class Cloud
    {
        public:
            Cloud();
            Cloud(std::vector<Vertex> const &);

            static Cloud* loadPly(std::string,  int *incre = NULL);

            void    addVertex( const Vertex & );
            Vertex& getVertex( uint );
            int     size();

            void    setTranslation( const Pt3dr & aPt ) {_translation = aPt;}
            Pt3dr   getTranslation(){return _translation;}

            void    setScale( const float & aS ) {_scale = aS;}
            float   getScale(){return _scale;}

            void    clear();

        private:
            std::vector<Vertex> _vertices;
            Pt3dr _translation;
            float _scale;
    };

}

#endif
