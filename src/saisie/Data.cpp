#include "Data.h"

cData::cData()
{
    reset();
}

cData::~cData()
{
   for (int aK=0; aK < NbCameras();++aK) delete _Cameras[aK];
   for (int aK=0; aK < NbClouds();++aK)  delete _Clouds[aK];
   for (int aK=0; aK < NbImages();++aK)  delete _Images[aK];
   for (int aK=0; aK < NbMasks();++aK)   delete _Masks[aK];

   _Cameras.clear();
   _Clouds.clear();
   _Images.clear();
   _Masks.clear();
}

void cData::addCamera(CamStenope * aCam)
{
    _Cameras.push_back(aCam);
}

void cData::addImage(QImage * aImg)
{
    _Images.push_back(aImg);
    _curImgIdx = _Images.size() - 1;
}

void cData::addMask(QImage * aImg)
{
    _Masks.push_back(aImg);
}

void cData::clearClouds()
{
    for (uint aK=0; aK < (uint)NbClouds();++aK)
        delete _Clouds[aK];

    _Clouds.clear();

    reset();
}

void cData::clearCameras()
{
    for (uint aK=0; aK < (uint)NbCameras();++aK)
        delete _Cameras[aK];

    _Cameras.clear();

    reset();
}

void cData::clearImages()
{
    for (uint aK=0; aK < (uint)NbCameras();++aK)
        delete _Images[aK];

    _Images.clear();

    reset();
}

void cData::clearMasks()
{
    for (uint aK=0; aK < (uint)NbMasks();++aK)
        delete _Masks[aK];

    _Masks.clear();

    reset();
}

void cData::reset()
{
    m_minX = m_minY = m_minZ =  FLT_MAX;
    m_maxX = m_maxY = m_maxZ = -FLT_MAX;
    m_cX = m_cY = m_cZ = m_diam = 0.f;
    _curImgIdx = 0;    
}

int cData::getSizeClouds()
{
    int sizeClouds = 0;
    for (int aK=0; aK < NbClouds();++aK)
        sizeClouds += getCloud(aK)->size();

    return sizeClouds;
}

//compute bounding box
void cData::getBB(Cloud * aCloud)
{  
    for (int aK=0; aK < aCloud->size(); ++aK)
    {
        Vertex vert = aCloud->getVertex(aK);

        if (vert.x() > m_maxX) m_maxX = vert.x();
        if (vert.x() < m_minX) m_minX = vert.x();
        if (vert.y() > m_maxY) m_maxY = vert.y();
        if (vert.y() < m_minY) m_minY = vert.y();
        if (vert.z() > m_maxZ) m_maxZ = vert.z();
        if (vert.z() < m_minZ) m_minZ = vert.z();
    }

    m_cX = (m_minX + m_maxX) * .5f;
    m_cY = (m_minY + m_maxY) * .5f;
    m_cZ = (m_minZ + m_maxZ) * .5f;

    m_diam = max(m_maxX-m_minX, max(m_maxY-m_minY, m_maxZ-m_minZ));

    aCloud->setScale(m_diam);
    _Clouds.push_back(aCloud);
}



