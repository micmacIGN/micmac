#include "Data.h"

cData::cData()
{
    reset();
}

cData::~cData()
{
    for (int aK=0; aK < getNbCameras();++aK) delete _Cameras[aK];
    for (int aK=0; aK < getNbClouds();++aK)  delete _Clouds[aK];
    for (int aK=0; aK < getNbImages();++aK)  delete _Images[aK];
    for (int aK=0; aK < getNbMasks();++aK)   delete _Masks[aK];

    _Cameras.clear();
    _Clouds.clear();
    _Images.clear();
    _Masks.clear();
}

void cData::addCloud(Cloud * aCloud)
{
    _Clouds.push_back((aCloud));
}

void cData::addCamera(CamStenope * aCam)
{
    _Cameras.push_back(aCam);
}

void cData::addImage(QImage * aImg)
{
    QImage *_glImg = NULL;
    _glImg = new QImage(aImg->size(),aImg->format());
    *_glImg = QGLWidget::convertToGLFormat( *aImg );
    _Images.push_back(_glImg);

    _curImgIdx = _Images.size() - 1;
}

void cData::addMask(QImage * aImg)
{
    QImage *_glMask = NULL;
    _glMask = new QImage(aImg->size(),aImg->format());
    *_glMask = QGLWidget::convertToGLFormat( *aImg );
    _Masks.push_back(_glMask);
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
    for (uint aK=0; aK < (uint)getNbCameras();++aK)
        delete _Images[aK];

    _Images.clear();

    reset();
}

void cData::clearMasks()
{
    for (uint aK=0; aK < (uint)getNbMasks();++aK)
        delete _Masks[aK];

    _Masks.clear();

    reset();
}

void cData::deleteCurMask()
{
    delete _Masks[_curImgIdx];
}

void cData::reset()
{
    m_minX = m_minY = m_minZ =  FLT_MAX;
    m_maxX = m_maxY = m_maxZ = -FLT_MAX;
    _center = Pt3dr(0.f,0.f,0.f);
    _curImgIdx = 0;
    _emptyMask = true;
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

            if (vert.x > m_maxX) m_maxX = vert.x;
            if (vert.x < m_minX) m_minX = vert.x;
            if (vert.y > m_maxY) m_maxY = vert.y;
            if (vert.y < m_minY) m_minY = vert.y;
            if (vert.z > m_maxZ) m_maxZ = vert.z;
            if (vert.z < m_minZ) m_minZ = vert.z;
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

            if (C.x > m_maxX) m_maxX = C.x;
            if (C.x < m_minX) m_minX = C.x;
            if (C.y > m_maxY) m_maxY = C.y;
            if (C.y < m_minY) m_minY = C.y;
            if (C.z > m_maxZ) m_maxZ = C.z;
            if (C.z < m_minZ) m_minZ = C.z;
        }
    }

    _center.x = (m_minX + m_maxX) * .5f;
    _center.y = (m_minY + m_maxY) * .5f;
    _center.z = (m_minZ + m_maxZ) * .5f;

    m_diam = max(m_maxX-m_minX, max(m_maxY-m_minY, m_maxZ-m_minZ));   
}

void cData::applyGamma(float aGamma)
{
    for (uint aK=0; aK<_Images.size();++aK)
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
