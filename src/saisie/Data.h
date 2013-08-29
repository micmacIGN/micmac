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

        int NbCameras(){return m_Cameras.size();}
        int NbClouds(){return m_Clouds.size();}
        int NbImages(){return m_Images.size();}
        int NbMasks(){return  m_Masks.size();}

        CamStenope * & getCamera(int aK) {return m_Cameras[aK];}
        Cloud * &      getCloud(int aK)  {return m_Clouds[aK];}
        QImage * &     getImage(int aK)  {return m_Images[aK];}
        QImage * &     getMask(int aK)   {return m_Masks[aK];}
        QImage * &     getCurImage()     {return m_Images[m_curImgIdx];}
        QImage * &     getCurMask()      {return m_Masks[m_curImgIdx];}

        void    setCurImage(int idx)     {m_curImgIdx = idx;}

        void    getBB(Cloud *);

        int     getSizeClouds();

        void    reset();

        //Bounding box, center and diameter of all clouds
        float m_minX, m_maxX, m_minY, m_maxY, m_minZ, m_maxZ, m_cX, m_cY, m_cZ, m_diam;

   private:

        vector <CamStenope *> m_Cameras;
        vector <Cloud *>      m_Clouds;
        vector <QImage *>     m_Images;
        vector <QImage *>     m_Masks;

        int                   m_curImgIdx;
};
#endif // DATA_H
