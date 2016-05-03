#ifndef DATA_H
#define DATA_H

#include "Cloud.h"

class GlCloud;

#ifdef USE_MIPMAP_HANDLER
	typedef std::pair<MipmapHandler::Mipmap *, MipmapHandler::Mipmap *> MaskedImage;
#endif

class cData
{
    public:

        cData();
        ~cData();

        void addCamera(cCamHandler*);
        void addCloud(GlCloud *);
        void addObject(cObject *);

        void replaceCloud(GlCloud * cloud, int id = 0);

	#ifdef USE_MIPMAP_HANDLER
		void addImage( const MaskedImage &aMaskedImage );

		int getNbImages() { return (int)_maskedImages.size(); }

		MaskedImage & getMaskedImage(int aK) { return _maskedImages[aK]; }
	#else
        void pushBackMaskedImage(QMaskedImage *maskedImage);

        int getNbImages()   { return _MaskedImages.size(); }

        QMaskedImage*  getMaskedImage(int aK)   { return _MaskedImages[aK]; }
	#endif

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
        int getNbPolygons() { return _vPolygons.size(); }

        cCamHandler *   getCamera(int aK) { return aK < _Cameras.size() ? _Cameras[aK] : NULL; }
        GlCloud *      getCloud(int aK)  { return aK < _Clouds.size() ? _Clouds[aK] : NULL;   }
        //QImage *       getImage(int aK)  { return aK < _MaskedImages.size() ? ((QMaskedImage)_MaskedImages[aK])._m_image : NULL; }
        //QImage *       getMask(int aK)   { return aK < _MaskedImages.size() ? ((QMaskedImage)_MaskedImages[aK])._m_mask  : NULL; }
        cPolygon*      getPolygon(int aK){ return _vPolygons[aK]; }

        int            idPolygon(cPolygon* polygon);

        void    getMinMax(QVector3D);
        void    computeCenterAndBBox(int idCloud = -1); //compute clouds bounding box and center

        int     getCloudsSize();

        QVector3D   getBBoxCenter();
        QVector3D   getCloudsCenter(){ return _centroid; }
        QVector3D   getMin(){ return _min; }
        QVector3D   getMax(){ return _max; }

        float   getBBoxMaxSize();

        void    reset();

        void    cleanCameras();

        void    deleteCloud(int idCloud);

        void    addReplaceCloud(GlCloud *cloud, int id = 0);
private:

        QVector <cCamHandler *> _Cameras;
        QVector <GlCloud *>    _Clouds;
	#ifdef USE_MIPMAP_HANDLER
		std::vector<MaskedImage> _maskedImages;
	#else
		QVector <QMaskedImage*> _MaskedImages;
	#endif

        //! list of polygons
        QVector<cPolygon*>     _vPolygons;

        //!Bounding box of all data
        QVector3D   _min, _max;

        //!data centroid
        QVector3D   _centroid;
};

#endif // DATA_H
