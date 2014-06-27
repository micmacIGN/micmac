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
        void addObject(cObject *);

        void replaceCloud(GlCloud * cloud, int id = 0);

        void pushBackMaskedImage(QMaskedImage maskedImage);

        void clearCameras();
        void clearClouds();
        void clearImages();
        void clearObjects();

        void clearAll();

        void clear(int aK);

        bool isDataLoaded() { return getNbClouds()||getNbCameras()||getNbImages(); }
        bool is3D()         { return getNbClouds()||getNbCameras(); }

        int getNbCameras()  { return _Cameras.size(); }
        int getNbClouds()   { return _Clouds.size();  }
        int getNbImages()   { return _MaskedImages.size(); }
        int getNbPolygons() { return _vPolygons.size(); }

        CamStenope *   getCamera(int aK) { return aK < _Cameras.size() ? _Cameras[aK] : NULL; }
        GlCloud *      getCloud(int aK)  { return aK < _Clouds.size() ? _Clouds[aK] : NULL;   }
        QImage *       getImage(int aK)  { return aK < _MaskedImages.size() ? ((QMaskedImage)_MaskedImages[aK])._m_image : NULL; }
        QImage *       getMask(int aK)   { return aK < _MaskedImages.size() ? ((QMaskedImage)_MaskedImages[aK])._m_mask  : NULL; }
        cPolygon*      getPolygon(int aK){ return _vPolygons[aK]; }

        int            idPolygon(cPolygon* polygon);

        QMaskedImage&  getMaskedImage(int aK)   { return _MaskedImages[aK]; }

        void    getMinMax(Pt3dr);
        void    computeBBox(int idCloud = -1);

        int     getCloudsSize();

        Pt3dr   getBBoxCenter();
        Pt3dr   getMin(){ return _min; }
        Pt3dr   getMax(){ return _max; }

        float   getBBoxMaxSize();

        void    reset();

        void    cleanCameras();

        void    deleteCloud(int idCloud);

        void    addReplaceCloud(GlCloud *cloud, int id = 0);
private:

        QVector <CamStenope *> _Cameras;
        QVector <GlCloud *>    _Clouds;
        QVector <QMaskedImage> _MaskedImages;

        //! list of polygons
        QVector<cPolygon*>     _vPolygons;

        //!Bounding box of all data
        Pt3dr   _min, _max;
};
#endif // DATA_H
