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

void cData::addObject(cObject * aObj)
{
    _vPolygons.push_back((cPolygon*) aObj);
}

void cData::replaceCloud(GlCloud *cloud, int id)
{
    if(id < _Clouds.size())
    {
        _Clouds[id] = cloud;
        computeBBox(id);
        computeCloudsCenter(id);
    }
}

void cData::addReplaceCloud(GlCloud *cloud, int id)
{
    if(id < _Clouds.size())
        _Clouds[id] = cloud;
    else
        addCloud(cloud);

    computeBBox(id);
    computeCloudsCenter(id);
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

void cData::clearObjects()
{
    qDeleteAll(_vPolygons);

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
    if (_MaskedImages.size())   _MaskedImages[aK].deallocImages();
}

int cData::idPolygon(cPolygon *polygon)
{
    return _vPolygons.indexOf(polygon);
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

void cData::getMinMax(Pt3dr pt)
{
    if (pt.x > _max.x) _max.x = pt.x;
    if (pt.x < _min.x) _min.x = pt.x;
    if (pt.y > _max.y) _max.y = pt.y;
    if (pt.y < _min.y) _min.y = pt.y;
    if (pt.z > _max.z) _max.z = pt.z;
    if (pt.z < _min.z) _min.z = pt.z;
}

//compute bounding box
void cData::computeBBox(int idCloud)
{
    for (int bK=0; bK < _Clouds.size();++bK)
    {
        if(idCloud == -1 || bK == idCloud)
        {
            GlCloud * aCloud = _Clouds[bK];

            for (int aK=0; aK < aCloud->size(); ++aK)
            {
                getMinMax(aCloud->getVertex(aK).getPosition());
            }
        }
    }

    if(idCloud == -1)
    for (int  cK=0; cK < _Cameras.size();++cK)
    {
        CamStenope * aCam= _Cameras[cK];

        QVector <Pt3dr> vert;
        Pt3dr c1, c2, c3, c4;

        aCam->Coins(c1,c2,c3,c4,1.f);
        vert.push_back(aCam->VraiOpticalCenter());
        vert.push_back(c1);
        vert.push_back(c2);
        vert.push_back(c3);
        vert.push_back(c4);

        for (int aK=0; aK < vert.size(); ++aK)
        {
            getMinMax(vert[aK]);
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

//compute data centroid
void cData::computeCloudsCenter(int idCloud)
{
    Pt3dr sum(0.,0.,0.);
    int cpt = 0;

    for (int bK=0; bK < _Clouds.size();++bK)
    {
        if(idCloud == -1 || bK == idCloud)
        {
            GlCloud * aCloud = _Clouds[bK];

            sum = sum + aCloud->getSum();
            cpt += aCloud->size();
        }
    }

    if(idCloud == -1)
    for (int  cK=0; cK < _Cameras.size();++cK)
    {
        CamStenope * aCam= _Cameras[cK];

        Pt3dr c1, c2, c3, c4;

        aCam->Coins(c1,c2,c3,c4,1.f);
        sum = sum + aCam->VraiOpticalCenter();
        sum = sum + c1;
        sum = sum + c2;
        sum = sum + c3;
        sum = sum + c4;

        cpt += 5;
    }

    _centroid = sum / cpt;
}

