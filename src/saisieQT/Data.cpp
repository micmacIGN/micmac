#include "Data.h"

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

void cData::replaceCloud(GlCloud *cloud, int id)
{
    if(id < _Clouds.size())
        _Clouds[id] = cloud;
}

void cData::addCamera(CamStenope * aCam)
{
    _Cameras.push_back(aCam);
}

void cData::pushBackMaskedImage(QMaskedImage maskedImage)
{
    _MaskedImages.push_back(maskedImage);
}

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

void cData::clearImages()
{
    _MaskedImages.clear();
    reset();
}

void cData::clearAll()
{
    clearClouds();
    clearCameras();
    clearImages();
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
    if (_MaskedImages.size())   _MaskedImages[aK].deallocImages();
}

void cData::reset()
{
    _min.x = _min.y = _min.z =  FLT_MAX;
    _max.x = _max.y = _max.z = -FLT_MAX;
}

void cData::cleanCameras()
{
    _Cameras.clear();
}

int cData::getCloudsSize()
{
    int sizeClouds = 0;
    for (int aK=0; aK < _Clouds.size();++aK)
        sizeClouds += _Clouds[aK]->size();

    return sizeClouds;
}

//compute bounding box
void cData::computeBBox()
{  
    for (int bK=0; bK < _Clouds.size();++bK)
    {
        GlCloud * aCloud = _Clouds[bK];

        for (int aK=0; aK < aCloud->size(); ++aK)
        {
            Pt3dr vert = aCloud->getVertex(aK).getPosition();

            if (vert.x > _max.x) _max.x = vert.x;
            if (vert.x < _min.x) _min.x = vert.x;
            if (vert.y > _max.y) _max.y = vert.y;
            if (vert.y < _min.y) _min.y = vert.y;
            if (vert.z > _max.z) _max.z = vert.z;
            if (vert.z < _min.z) _min.z = vert.z;
        }
    }

    for (int  cK=0; cK < _Cameras.size();++cK)
    {
        CamStenope * aCam= _Cameras[cK];

        QVector <Pt3dr> vert;
        Pt2di sz = aCam->Sz();

        vert.push_back(aCam->VraiOpticalCenter());
        vert.push_back(aCam->ImEtProf2Terrain(Pt2dr(0.f,0.f),1.f));
        vert.push_back(aCam->ImEtProf2Terrain(Pt2dr(sz.x,0.f),1.f));
        vert.push_back(aCam->ImEtProf2Terrain(Pt2dr(0.f,sz.y),1.f));
        vert.push_back(aCam->ImEtProf2Terrain(Pt2dr(sz.x,sz.y),1.f));

        for (int aK=0; aK < vert.size(); ++aK)
        {
            Pt3dr C = vert[aK];

            if (C.x > _max.x) _max.x = C.x;
            if (C.x < _min.x) _min.x = C.x;
            if (C.y > _max.y) _max.y = C.y;
            if (C.y < _min.y) _min.y = C.y;
            if (C.z > _max.z) _max.z = C.z;
            if (C.z < _min.z) _min.z = C.z;
        }
    }
}

// compute BBox center
Pt3dr cData::getBBoxCenter()
{
    return Pt3dr((_min.x + _max.x) * .5f, (_min.y + _max.y) * .5f, (_min.z + _max.z) * .5f);
}

// compute BB max size
float cData::getBBoxMaxSize()
{
    return max(_max.x-_min.x, max(_max.y-_min.y, _max.z-_min.z));
}

