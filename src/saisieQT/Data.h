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

        void PushBackMaskedImage(QMaskedImage maskedImage);

        void clearCameras();
        void clearClouds();
        void clearImages();

        void clearAll();

        bool isDataLoaded(){return getNbClouds()||getNbCameras() ||getNbImages();}
        bool is3D(){return getNbClouds()||getNbCameras();}

        int getNbCameras() {return _Cameras.size();}
        int getNbClouds()  {return _Clouds.size(); }
        int getNbImages()  {return _MaskedImages.size(); }

        CamStenope *   getCamera(int aK) {return aK < (int)_Cameras.size() ? _Cameras[aK] : NULL;}
        Cloud *        getCloud(int aK)  {return aK < (int)_Clouds.size() ? _Clouds[aK] : NULL;  }
        QImage *       getImage(int aK)  {return aK < (int)_MaskedImages.size() ? ((QMaskedImage)_MaskedImages[aK])._m_image : NULL;  }
        QImage *       getMask(int aK)   {return aK < (int)_MaskedImages.size() ? ((QMaskedImage)_MaskedImages[aK])._m_mask  : NULL;    }

        QMaskedImage&  getMaskedImage(int aK)   {return _MaskedImages[aK];}

        void    getBB();

        int     getSizeClouds();

        Pt3dr   getCenter(){return _center;}

        float   getScale(){return m_diam;}

        void    reset();

        //!Bounding box and diameter of all clouds
        Pt3dr   m_min, m_max;
        float   m_diam;

   private:

        vector <CamStenope *> _Cameras;
        vector <Cloud *>      _Clouds;
        vector<QMaskedImage>  _MaskedImages;

        float                 _gamma;

        Pt3dr                 _center;  // center of all clouds
};
#endif // DATA_H
