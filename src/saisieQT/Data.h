#ifndef DATA_H
#define DATA_H


#include "StdAfx.h"
#include "Cloud.h"
#include <QImage>

using namespace Cloud_;

//invalid GL list index
const GLuint GL_INVALID_LIST_ID = (~0);

class cData
{
    public:

        cData();
        ~cData();

        void addCamera(CamStenope *);
        void addCloud(Cloud *);
        void addImage(QImage *);
        void addMask(QImage *);

        void clearCameras();
        void clearClouds();
        void clearImages();
        void clearMasks();

        bool isDataLoaded(){return getNbClouds()||getNbCameras() ||getNbImages();}

        int getNbCameras() {return _Cameras.size();}
        int getNbClouds()  {return _Clouds.size(); }
        int getNbImages()  {return _Images.size(); }
        int getNbMasks()   {return _Masks.size();  }

        CamStenope * & getCamera(int aK) {return _Cameras[aK];}
        Cloud * &      getCloud(int aK)  {return _Clouds[aK]; }
        QImage * &     getImage(int aK)  {return _Images[aK]; }
        QImage * &     getMask(int aK)   {return _Masks[aK];  }
        QImage * &     getCurImage()     {return _Images[_curImgIdx];}
        QImage * &     getCurMask()      {return _Masks[_curImgIdx];}

        void    setCurImageIdx(int idx)     {_curImgIdx = idx;}
        int     getCurImageIdx()            {return _curImgIdx;}

        void    fillCurMask(){getCurMask()->fill(Qt::white);}

        void    deleteCurMask();

        void    getBB();

        int     getSizeClouds();

        void    setCenter(Pt3dr const &pt){_center = pt;}
        Pt3dr   getCenter(){return _center;}

        float   getScale(){return m_diam;}

        void    reset();

        void    applyGamma(float aGamma);
        void    applyGammaToImage(int aK, float aGamma);


        void    drawQuad(GLfloat originX, GLfloat originY, GLfloat glh, GLfloat glw);

        void    drawQuad(GLfloat originX, GLfloat originY, GLfloat glh, GLfloat glw, QColor color);

        void    drawQuad(GLfloat originX, GLfloat originY, GLfloat glh, GLfloat glw, GLuint idTexture);

        //!Bounding box and diameter of all clouds
        float   m_minX, m_maxX, m_minY, m_maxY, m_minZ, m_maxZ, m_diam;

        //! Texture image
        GLuint  m_textureImage;

   private:

        vector <CamStenope *> _Cameras;
        vector <Cloud *>      _Clouds;
        vector <QImage *>     _Images;
        vector <QImage *>     _Masks;

        int                   _curImgIdx;

        float                 _gamma;

        Pt3dr                 _center;  // center of all clouds
};
#endif // DATA_H
