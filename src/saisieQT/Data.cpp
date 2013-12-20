#include "Data.h"

cData::cData()
{
    reset();
}

cData::~cData()
{
  clearAll();
}

void cData::addCloud(Cloud * aCloud)
{
    _Clouds.push_back(aCloud);
}

void cData::addCamera(CamStenope * aCam)
{
    _Cameras.push_back(aCam);
}


void cData::PushBackMaskedImage(QMaskedImage maskedImage)
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

void cData::reset()
{
    m_min.x = m_min.y = m_min.z =  FLT_MAX;
    m_max.x = m_max.y = m_max.z = -FLT_MAX;
    _center = Pt3dr(0.f,0.f,0.f);
}

int cData::getSizeClouds()
{
    int sizeClouds = 0;
    for (int aK=0; aK < getNbClouds();++aK)
        sizeClouds += getCloud(aK)->size();

    return sizeClouds;
}

//compute bounding box
void cData::getBB()
{  

    //compute cloud bounding box
    for (uint bK=0; bK < _Clouds.size();++bK)
    {
        Cloud * aCloud = _Clouds[bK];

        for (int aK=0; aK < aCloud->size(); ++aK)
        {
            Pt3dr vert = aCloud->getVertex(aK).getPosition();

            if (vert.x > m_max.x) m_max.x = vert.x;
            if (vert.x < m_min.x) m_min.x = vert.x;
            if (vert.y > m_max.y) m_max.y = vert.y;
            if (vert.y < m_min.y) m_min.y = vert.y;
            if (vert.z > m_max.z) m_max.z = vert.z;
            if (vert.z < m_min.z) m_min.z = vert.z;
        }
    }

    //compute "cameras and clouds" global bounding box
    for (uint  cK=0; cK < _Cameras.size();++cK)
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

            if (C.x > m_max.x) m_max.x = C.x;
            if (C.x < m_min.x) m_min.x = C.x;
            if (C.y > m_max.y) m_max.y = C.y;
            if (C.y < m_min.y) m_min.y = C.y;
            if (C.z > m_max.z) m_max.z = C.z;
            if (C.z < m_min.z) m_min.z = C.z;
        }
    }

    // compute BB center
    _center.x = (m_min.x + m_max.x) * .5f;
    _center.y = (m_min.y + m_max.y) * .5f;
    _center.z = (m_min.z + m_max.z) * .5f;

    m_diam = max(m_max.x-m_min.x, max(m_max.y-m_min.y, m_max.z-m_min.z));
}

