#include "Data.h"

cData::cData()
{
    reset();
}

cData::~cData()
{
    // BUG when the apli is closed!!!
//    for (int aK=0; aK < NbCameras();++aK) delete m_Cameras[aK];
//    for (int aK=0; aK < NbClouds();++aK)
//    {
//        delete m_Clouds[aK];
//        delete m_oClouds[aK];
//    }
}

void cData::addCamera(CamStenope * aCam)
{
    m_Cameras.push_back(aCam);
}

void cData::addCameras(vector <CamStenope *> aCameras)
{
    for (uint aK=0; aK < (uint)aCameras.size();++aK)
        m_Cameras.push_back(aCameras[aK]);
}

void cData::addClouds(vector <Cloud *> aClouds)
{
    for (uint aK=0; aK < aClouds.size();++aK)
        centerCloud(aClouds[aK]);
}

void cData::clearClouds()
{
    for (uint aK=0; aK < (uint)NbClouds();++aK)
    {
        delete m_Clouds[aK];
        delete m_oClouds[aK];
    }
    m_Clouds.clear();
    m_oClouds.clear();

    reset();
}

void cData::clearCameras()
{
    for (uint aK=0; aK < (uint)NbCameras();++aK)
        delete m_Cameras[aK];

    m_Cameras.clear();

    reset();
}

void cData::reset()
{
    m_minX = m_minY = m_minZ = FLT_MAX;
    m_maxX = m_maxY = m_maxZ = FLT_MIN;
    m_cX = m_cY = m_cZ = m_diam = 0.f;
}

int cData::GetSizeClouds()
{
    int sizeClouds = 0;
    for (int aK=0; aK < NbClouds();++aK)
        sizeClouds += getCloud(aK)->size();

    return sizeClouds;
}

void cData::centerCloud(Cloud * aCloud)
{
   // Cloud *a_res = new Cloud();

    //compute bounding box
    int nbPts = aCloud->size();
    for (int aK=0; aK < nbPts; ++aK)
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

    //center and scale cloud
#ifdef toto
    Pt3dr pt3d;
    for (int aK=0; aK < nbPts; ++aK)
    {
        Vertex vert = aCloud->getVertex(aK);
        Vertex vert_res = vert;

        pt3d.x = (vert.x() - m_cX);// / m_diam;
        pt3d.y = (vert.y() - m_cY);// / m_diam;
        pt3d.z = (vert.z() - m_cZ);// / m_diam;

        vert_res.setCoord(pt3d);
        vert_res.setColor(vert.getColor());

        a_res->addVertex(vert_res);
    }

    a_res->setTranslation(Pt3dr(m_cX, m_cY, m_cZ));
    a_res->setScale((float) m_diam);

    //translate and scale back clouds if needed
    for (int aK=0; aK< NbClouds();++aK)
    {
        if (getCloud(aK)->getScale()) //cloud has been scaled
        {
            Pt3dr translation = getCloud(aK)->getTranslation();
            float scale = getCloud(aK)->getScale();

            for (int bK=0; bK < getCloud(aK)->size();++bK)
            {
                Vertex vert = getCloud(aK)->getVertex(bK);

                pt3d.x = ((vert.x() * scale + translation.x) - m_cX) / m_diam;
                pt3d.y = ((vert.y() * scale + translation.y) - m_cY) / m_diam;
                pt3d.z = ((vert.z() * scale + translation.z) - m_cZ) / m_diam;

                vert.setCoord(pt3d);

                getCloud(aK)->setVertex(bK, vert);
            }

            getCloud(aK)->setTranslation(Pt3dr(m_cX, m_cY, m_cZ));
            getCloud(aK)->setScale((float) m_diam);
        }
    }

    m_Clouds.push_back(a_res);
    m_oClouds.push_back(aCloud);
#endif //toto
    aCloud->setScale((float) m_diam);
    m_Clouds.push_back(aCloud);
    m_oClouds.push_back(aCloud);
}



