#ifndef DATA_H
#define DATA_H

#include <QImage>
#include "Cloud.h"

class GlCloud;

class cData
{
    public:

        cData();
        ~cData();

        void addCamera(CamStenope *);
        void addCloud(GlCloud *);

        void pushBackMaskedImage(QMaskedImage maskedImage);

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
        GlCloud *      getCloud(int aK)  {return aK < (int)_Clouds.size() ? _Clouds[aK] : NULL;  }
        QImage *       getImage(int aK)  {return aK < (int)_MaskedImages.size() ? ((QMaskedImage)_MaskedImages[aK])._m_image : NULL;  }
        QImage *       getMask(int aK)   {return aK < (int)_MaskedImages.size() ? ((QMaskedImage)_MaskedImages[aK])._m_mask  : NULL;    }

        QMaskedImage&  getMaskedImage(int aK)   {return _MaskedImages[aK];}

        void    computeBBox();

        int     getCloudsSize();

        Pt3dr   getBBoxCenter();
        Pt3dr   getMin(){return _min;}
        Pt3dr   getMax(){return _max;}

        float   getBBoxMaxSize();

        void    reset();

   private:

        vector <CamStenope *> _Cameras;
        vector <GlCloud *>    _Clouds;
        vector <QMaskedImage> _MaskedImages;

        //!Bounding box of all data
        Pt3dr   _min, _max;
};
#endif // DATA_H
