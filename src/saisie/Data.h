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

        bool isDataLoaded(){return NbClouds()||NbCameras() ||NbImages();}

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

        void    getBB();

        int     getSizeClouds();

        void    setCenter(Pt3dr const &pt){_center = pt;}
        Pt3dr   getCenter(){return _center;}

        float   getScale(){return m_diam;}

        void    reset();

        //Bounding box, center and diameter of all clouds
        float m_minX, m_maxX, m_minY, m_maxY, m_minZ, m_maxZ, m_diam;



   private:

        vector <CamStenope *> _Cameras;
        vector <Cloud *>      _Clouds;
        vector <QImage *>     _Images;
        vector <QImage *>     _Masks;

        int                   _curImgIdx;

        float                 _gamma;

        Pt3dr                 _center;
};
#endif // DATA_H
