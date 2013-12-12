#ifndef DATA_H
#define DATA_H

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

        bool isDataLoaded(){return getNbClouds()||getNbCameras() ||getNbImages();}
        bool is3D(){return getNbClouds()||getNbCameras();}

        int getNbCameras() {return _Cameras.size();}
        int getNbClouds()  {return _Clouds.size(); }
        int getNbImages()  {return _Images.size(); }
        int getNbMasks()   {return _Masks.size();  }

        CamStenope *   getCamera(int aK) {return aK < (int)_Cameras.size() ? _Cameras[aK] : NULL;}
        Cloud *        getCloud(int aK)  {return aK < (int)_Clouds.size() ? _Clouds[aK] : NULL;  }
        QImage *       getImage(int aK)  {return aK < (int)_Images.size() ? _Images[aK] : NULL;  }
        QImage *       getMask(int aK)   {return aK < (int)_Masks.size() ? _Masks[aK] : NULL;    }
        QImage *       getCurMask()      {return _Masks[getNbMasks()-1];}

        void    fillMask(int aK){getMask(aK)->fill(Qt::white);}

        void    getBB();

        int     getSizeClouds();

        Pt3dr   getCenter(){return _center;}

        float   getScale(){return m_diam;}

        void    reset();

        void    applyGamma(float aGamma);
        void    applyGammaToImage(int aK, float aGamma);

        //!Bounding box and diameter of all clouds
        float   m_minX, m_maxX, m_minY, m_maxY, m_minZ, m_maxZ, m_diam;

   private:

        vector <CamStenope *> _Cameras;
        vector <Cloud *>      _Clouds;
        vector <QImage *>     _Images;
        vector <QImage *>     _Masks;

        float                 _gamma;

        Pt3dr                 _center;  // center of all clouds
};
#endif // DATA_H
