#ifndef DATA_H
#define DATA_H


#include "StdAfx.h"
#include "Cloud.h"
#include <QImage>

using namespace Cloud_;

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

        int NbCameras() {return _Cameras.size();}
        int NbClouds()  {return _Clouds.size();}
        int NbImages()  {return _Images.size();}
        int NbMasks()   {return _Masks.size();}

        CamStenope * & getCamera(int aK) {return _Cameras[aK];}
        Cloud * &      getCloud(int aK)  {return _Clouds[aK];}
        QImage * &     getImage(int aK)  {return _Images[aK];}
        QImage * &     getMask(int aK)   {return _Masks[aK];}
        QImage * &     getCurImage()     {return _Images[_curImgIdx];}
        QImage * &     getCurMask()      {return _Masks[_curImgIdx];}

        void    setCurImage(int idx)     {_curImgIdx = idx;}

        void    getBB(Cloud *);

        int     getSizeClouds();

        void    reset();

        //Bounding box, center and diameter of all clouds
        float m_minX, m_maxX, m_minY, m_maxY, m_minZ, m_maxZ, m_cX, m_cY, m_cZ, m_diam;

   private:

        vector <CamStenope *> _Cameras;
        vector <Cloud *>      _Clouds;
        vector <QImage *>     _Images;
        vector <QImage *>     _Masks;

        int                   _curImgIdx;

        float                 _gamma;
};
#endif // DATA_H
