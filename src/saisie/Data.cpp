#include "Data.h"

cData::cData()
{
    reset();
}

cData::~cData()
{
   for (int aK=0; aK < NbCameras();++aK) delete m_Cameras[aK];
   for (int aK=0; aK < NbClouds();++aK)  delete m_Clouds[aK];
   for (int aK=0; aK < NbImages();++aK)  delete m_Images[aK];

   m_Cameras.clear();
   m_Clouds.clear();
   m_Images.clear();
}

void cData::addCamera(CamStenope * aCam)
{
    m_Cameras.push_back(aCam);
}

void cData::addImage(QImage * aImg)
{
    m_Images.push_back(aImg);
    m_curImgIdx = m_Images.size() - 1;
}

void cData::clearClouds()
{
    for (uint aK=0; aK < (uint)NbClouds();++aK)
        delete m_Clouds[aK];

    m_Clouds.clear();

    reset();
}

void cData::clearCameras()
{
    for (uint aK=0; aK < (uint)NbCameras();++aK)
        delete m_Cameras[aK];

    m_Cameras.clear();

    reset();
}

void cData::clearImages()
{
    for (uint aK=0; aK < (uint)NbCameras();++aK)
        delete m_Images[aK];

    m_Images.clear();

    reset();
}

void cData::reset()
{
    m_minX = m_minY = m_minZ    = FLT_MAX;
    m_maxX = m_maxY = m_maxZ    = -FLT_MAX;
    m_cX = m_cY = m_cZ = m_diam = 0.f;
    m_curImgIdx = 0;
}

int cData::getSizeClouds()
{
    int sizeClouds = 0;
    for (int aK=0; aK < NbClouds();++aK)
        sizeClouds += getCloud(aK)->size();

    return sizeClouds;
}

void cData::getBB(Cloud * aCloud)
{  
    //compute bounding box

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
    m_Clouds.push_back(aCloud);
}



