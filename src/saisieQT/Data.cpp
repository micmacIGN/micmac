#include "Data.h"
#include <limits>

cData::cData()
{
    reset();
}

cData::~cData()
{
    clearAll();
}

void cData::addCloud(GlCloud * aCloud)
{
    _Clouds.push_back(aCloud);
}

void cData::addObject(cObject * aObj)
{
    _vPolygons.push_back((cPolygon*) aObj);
}

void cData::replaceCloud(GlCloud *cloud, int id)
{
    if(id < _Clouds.size())
    {
        _Clouds[id] = cloud;
        computeCenterAndBBox(id);
    }
}

void cData::addReplaceCloud(GlCloud *cloud, int id)
{
    if(id < _Clouds.size())
        _Clouds[id] = cloud;
    else
        addCloud(cloud);

    computeCenterAndBBox(id);
}

void cData::addCamera(cCamHandler * aCam)
{
    _Cameras.push_back(aCam);
}

#ifdef USE_MIPMAP_HANDLER
	void cData::addImage( const MaskedImage &aMaskedImage )
	{
		_maskedImages.push_back(aMaskedImage);
	}
#else
	void cData::pushBackMaskedImage(QMaskedImage *maskedImage)
	{

		_MaskedImages.push_back(maskedImage);

	}
#endif

void cData::clearClouds()
{
    qDeleteAll(_Clouds);
    _Clouds.clear();

    reset();
}

void cData::clearCameras()
{
    qDeleteAll(_Cameras);

    _Cameras.clear();

    reset();
}

#ifdef USE_MIPMAP_HANDLER
	void cData::clearImages()
	{
		_maskedImages.clear();
		reset();
	}
#else
	void cData::clearImages()
	{
		//qDeleteAll(_MaskedImages);

		for (int idQMImg = 0; idQMImg < _MaskedImages.size(); ++idQMImg)
		{
		    if(_MaskedImages[idQMImg])
		        delete _MaskedImages[idQMImg];
		    _MaskedImages[idQMImg] = NULL;
		}

		_MaskedImages.clear();
		reset();
	}
#endif

void cData::clearObjects()
{
    for (int idpoly = 0; idpoly < _vPolygons.size(); ++idpoly)
    {
        if(_vPolygons[idpoly])
            delete _vPolygons[idpoly];
        _vPolygons[idpoly] = NULL;
    }

    _vPolygons.clear();

    reset();
}

void cData::clearAll()
{
    clearClouds();
    clearCameras();
    clearImages();
    clearObjects();
}

void cData::clear(int aK)
{
    if (_Clouds.size())
    {
        if(_Clouds[aK])
        {
            delete _Clouds[aK];
            _Clouds[aK] = NULL;
        }
    }
    if (_Cameras.size())
    {
        if(_Cameras[aK])
        {
            delete _Cameras[aK];
            _Cameras[aK] = NULL;
        }
    }

	#ifndef USE_MIPMAP_HANDLER
		if (aK >= 0 && aK < _MaskedImages.size()) delete _MaskedImages[aK];
	#endif
}

int cData::idPolygon(cPolygon *polygon)
{
    return _vPolygons.indexOf(polygon);
}

void cData::reset()
{

    _min.setX(std::numeric_limits<float>::max());
    _min.setY(std::numeric_limits<float>::max());
    _min.setZ(std::numeric_limits<float>::max());
    _max.setX(-std::numeric_limits<float>::max());
    _max.setY(-std::numeric_limits<float>::max());
    _max.setZ(-std::numeric_limits<float>::max());

//    _min.x = _min.y = _min.z =  FLT_MAX;
//    _max.x = _max.y = _max.z = -FLT_MAX;
}

void cData::cleanCameras()
{
    _Cameras.clear();
}

void cData::deleteCloud(int idCloud)
{
   GlCloud* cloud = getCloud(idCloud);
   if(cloud)
       delete cloud;
}

int cData::getCloudsSize()
{
    int sizeClouds = 0;
    for (int aK=0; aK < _Clouds.size();++aK)
        sizeClouds += _Clouds[aK]->size();

    return sizeClouds;
}

void cData::getMinMax(QVector3D pt)
{
    if (pt.x() > _max.x()) _max.setX(pt.x());
    if (pt.x() < _min.x()) _min.setX(pt.x());
    if (pt.y() > _max.y()) _max.setY(pt.y());
    if (pt.y() < _min.y()) _min.setY(pt.y());
    if (pt.z() > _max.z()) _max.setZ(pt.z());
    if (pt.z() < _min.z()) _min.setZ(pt.z());
}

//compute bounding box
void cData::computeCenterAndBBox(int idCloud)
{
    QVector3D sum(0.,0.,0.);
    int cpt = 0;

    for (int bK=0; bK < _Clouds.size();++bK)
    {
        if(idCloud == -1 || bK == idCloud)
        {
            GlCloud * aCloud = _Clouds[bK];

            sum = sum + aCloud->getSum();
            cpt += aCloud->size();

            for (int aK=0; aK < aCloud->size(); ++aK)
            {
                getMinMax(aCloud->getVertex(aK).getPosition());
            }
        }
    }

    if(idCloud == -1)
    for (int  cK=0; cK < _Cameras.size();++cK)
    {
        cCamHandler * aCam = _Cameras[cK];

        QVector <QVector3D> vert;
        QVector3D c1, c2, c3, c4;

        aCam->getCoins(c1,c2,c3,c4,1.f);
        vert.push_back(aCam->getCenter());
        vert.push_back(c1);
        vert.push_back(c2);
        vert.push_back(c3);
        vert.push_back(c4);

        for (int aK=0; aK < vert.size(); ++aK)
        {
            getMinMax(vert[aK]);
        }

        sum = sum + aCam->getCenter();
        sum = sum + c1;
        sum = sum + c2;
        sum = sum + c3;
        sum = sum + c4;

        cpt += 5;
    }

    _centroid = sum / cpt;
}

// compute BBox center
QVector3D cData::getBBoxCenter()
{
    return QVector3D((_min.x() + _max.x()) * .5f, (_min.y() + _max.y()) * .5f, (_min.z() + _max.z()) * .5f);
}

// compute BB max size
float cData::getBBoxMaxSize()
{
    return max(_max.x()-_min.x(), max(_max.y()-_min.y(), _max.z()-_min.z()));
}
