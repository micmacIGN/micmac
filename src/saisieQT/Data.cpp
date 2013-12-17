#include "Data.h"

cData::cData()
{
    reset();
}

cData::~cData()
{
    for (int aK=0; aK < getNbCameras();++aK) delete _Cameras[aK];
    for (int aK=0; aK < getNbClouds();++aK)  delete _Clouds[aK];

    _Cameras.clear();
    _Clouds.clear();
    _MaskedImages.clear();

}

void cData::addCloud(Cloud * aCloud)
{
    _Clouds.push_back(aCloud);
}

void cData::addCamera(CamStenope * aCam)
{
    _Cameras.push_back(aCam);
}


void cData::PushBackMaskedImage(cMaskedImage<QImage> maskedImage)
{
    _MaskedImages.push_back(maskedImage);
}

void cData::clearClouds()
{
    for (uint aK=0; aK < (uint)getNbClouds();++aK)
        delete _Clouds[aK];

    _Clouds.clear();

    reset();
}

void cData::clearCameras()
{
    for (uint aK=0; aK < (uint)getNbCameras();++aK)
        delete _Cameras[aK];

    _Cameras.clear();

    reset();
}

void cData::clearImages()
{
//    for (uint aK=0; aK < (uint)getNbCameras();++aK)
//        delete _Images[aK];

    _MaskedImages.clear();

    reset();
}


void cData::clearAll()
{
    clearClouds();
    clearCameras();
    clearImages();
    reset();
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

    _center.x = (m_min.x + m_max.x) * .5f;
    _center.y = (m_min.y + m_max.y) * .5f;
    _center.z = (m_min.z + m_max.z) * .5f;

    m_diam = max(m_max.x-m_min.x, max(m_max.y-m_min.y, m_max.z-m_min.z));
}

void cData::applyGamma(float aGamma)
{
    for (uint aK=0; aK<_MaskedImages.size();++aK)
        applyGammaToImage(aK, aGamma);
}

void cData::applyGammaToImage(int aK, float aGamma)
{
    if (aGamma == 1.f) return;

    QRgb pixel;
    int r,g,b;

    float _gamma = 1.f / aGamma;

    for(int i=0; i< getImage(aK)->width();++i)
        for(int j=0; j< getImage(aK)->height();++j)
        {
            pixel = getImage(aK)->pixel(i,j);

            r = 255*pow((float) qRed(pixel)  / 255.f, _gamma);
            g = 255*pow((float) qGreen(pixel)/ 255.f, _gamma);
            b = 255*pow((float) qBlue(pixel) / 255.f, _gamma);

            if (r>255) r = 255;
            if (g>255) g = 255;
            if (b>255) b = 255;

            getImage(aK)->setPixel(i,j, qRgb(r,g,b) );
        }
}
