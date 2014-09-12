#include "cgldata.h"


cGLData::cGLData(int appMode):
    _diam(1.f)
{
    initOptions(appMode);
}

void cGLData::setOptionPolygons(cParameters aParams)
{
    for (int aK=0; aK < _vPolygons.size(); ++aK)
    {
        polygon(aK)->showLines(!_modePt);
        polygon(aK)->showNames(_modePt);

        polygon(aK)->setDefaultName(aParams.getDefPtName());
        polygon(aK)->setPointSize(aParams.getPointDiameter());
        polygon(aK)->setLineWidth(aParams.getLineThickness());
    }
}

cGLData::cGLData(cData *data, QMaskedImage *qMaskedImage, cParameters aParams, int appMode):
    _glMaskedImage(qMaskedImage),
    _pQMask(qMaskedImage->_m_mask),
    _pBall(NULL),
    _pAxis(NULL),
    _pBbox(NULL),
    _pGrid(NULL),
    _center(Pt3dr(0.f,0.f,0.f)),
    _appMode(appMode)
{
    if (appMode != MASK2D) _glMaskedImage._m_mask->setVisible(aParams.getShowMasks());
    else _glMaskedImage._m_mask->setVisible(true);

    initOptions(appMode);

    setPolygons(data);

    setOptionPolygons(aParams);
}


cGLData::cGLData(cData *data, cParameters aParams,int appMode):
    _pBall(new cBall),
    _pAxis(new cAxis),
    _pBbox(new cBBox),
    _pGrid(new cGrid),
    _appMode(appMode),
    _diam(1.f),
    _incFirstCloud(false)
{
    initOptions(appMode);

    setData(data);

    setPolygons(data);

    setOptionPolygons(aParams);
}

void cGLData::setPolygons(cData *data)
{
    for (int aK = 0; aK < data->getNbPolygons(); ++aK)
    {
        if (_appMode == BOX2D)
        {
            cRectangle* polygon = new cRectangle();
            polygon->setHelper(new cPolygonHelper(polygon, 4));
            _vPolygons.push_back(polygon);
        }
        else
        {
            cPolygon* polygon = new cPolygon(*(data->getPolygon(aK)));
            polygon->setHelper(new cPolygonHelper(polygon, 3));
            _vPolygons.push_back(polygon);
        }
    }
}

void cGLData::setData(cData *data, bool setCam)
{
    for (int aK = 0; aK < data->getNbClouds(); ++aK)
    {
        GlCloud *pCloud = data->getCloud(aK);
        _vClouds.push_back(pCloud);
        pCloud->setBufferGl();
    }

    Pt3dr center = data->getBBoxCenter();
    float sc = data->getBBoxMaxSize() / 1.5f;
    Pt3dr scale(sc, sc, sc);

    _pBall->setPosition(center);
    _pBall->setScale(scale);
    _pAxis->setPosition(center);
    _pAxis->setScale(scale);
    _pBbox->setPosition(center);
    _pBbox->setScale(scale);
    _pBbox->set(data->getMin(), data->getMax());

    _pGrid->setPosition(center);
    _pGrid->setScale(scale*2.f);

    if(setCam)
        for (int i=0; i< data->getNbCameras(); i++)
        {
            cCam *pCam = new cCam(data->getCamera(i), sc);

            _vCams.push_back(pCam);
        }

    setBBoxMaxSize(data->getBBoxMaxSize());
    setBBoxCenter(data->getBBoxCenter());
}

bool cGLData::incFirstCloud() const
{
    return _incFirstCloud;
}

void cGLData::setIncFirstCloud(bool incFirstCloud)
{
    _incFirstCloud = incFirstCloud;
}

cMaskedImageGL &cGLData::glImage()
{
    return _glMaskedImage;
}

cPolygon *cGLData::polygon(int id)
{
    if(id < (int) _vPolygons.size())
        return _vPolygons[id];
    else
        return NULL;
}

cPolygon *cGLData::currentPolygon()
{
    return polygon(_currentPolygon);
}

GlCloud* cGLData::getCloud(int iC)
{
    return _vClouds[iC];
}

int cGLData::cloudCount()
{
    return _vClouds.size();
}

int cGLData::camerasCount()
{
    return _vCams.size();
}

int cGLData::polygonCount()
{
    return _vPolygons.size();
}

void cGLData::initOptions(int appMode)
{
    //TODO: retirer BASC si on saisit des vraies lignes...
    if ((appMode == POINT2D_INIT) || (appMode == POINT2D_PREDIC) || (appMode == BASC))
        _modePt = true;
    else
        _modePt = false;

    _currentPolygon = 0;
    _options = options(OpShow_Mess);
}

cGLData::~cGLData()
{
    _glMaskedImage.deleteTextures();
    _glMaskedImage.deallocImages();

    qDeleteAll(_vCams);
    _vCams.clear();

    if(_pBall != NULL) delete _pBall;
    if(_pAxis != NULL) delete _pAxis;
    if(_pBbox != NULL) delete _pBbox;
    if(_pGrid != NULL) delete _pGrid;

    //pas de delete des pointeurs dans Clouds c'est Data qui s'en charge
    _vClouds.clear();
}

void outMatrix4X4(GLdouble *mvMatrix)
{

    QString esp = "       ";
    qDebug() << mvMatrix[0] << esp << mvMatrix[1] << esp << mvMatrix[2] << esp <<  mvMatrix[3] << endl;
    qDebug() << mvMatrix[4] << esp << mvMatrix[5] << esp << mvMatrix[6] << esp <<  mvMatrix[7] << endl;
    qDebug() << mvMatrix[8] << esp << mvMatrix[9] << esp << mvMatrix[10] << esp <<  mvMatrix[11] << endl;
    qDebug() << mvMatrix[12] << esp << mvMatrix[13] << esp << mvMatrix[14] << esp <<  mvMatrix[15] << endl;
}

void cGLData::draw()
{

    if(!is3D())
        glImage().draw();
    else
    {
        enableOptionLine();

        glMatrixMode(GL_MODELVIEW);
        glPushMatrix();
        glTranslated(getBBoxCenter().x,getBBoxCenter().y,getBBoxCenter().z);
        glRotatef(cObject::getRotation().x,1.f,0.f,0.f);
        glRotatef(cObject::getRotation().y,0.f,1.f,0.f);
        glRotatef(cObject::getRotation().z,0.f,0.f,1.f);
        glTranslated(-getBBoxCenter().x,-getBBoxCenter().y,-getBBoxCenter().z);

        for (int i=0; i<_vClouds.size();i++)
        {
            GLfloat oldPointSize;
            glGetFloatv(GL_POINT_SIZE,&oldPointSize);

            if(_incFirstCloud && i == 0)
                glPointSize(oldPointSize*3.f);

            _vClouds[i]->draw();

            glPointSize(oldPointSize);
        }

        //cameras
        for (int i=0; i< _vCams.size();i++) _vCams[i]->draw();

        glMatrixMode(GL_MODELVIEW);
        glPopMatrix();

        _pBall->draw();
        _pAxis->draw();
        _pBbox->draw();
        _pGrid->draw();

        disableOptionLine();
    }
}

void cGLData::drawCenter(bool white)
{
    //TODO: check if a point is drawn close to center

    float radius = 6.f;
    float mini   = 1.f;

    GLint       glViewport[4];
    glGetIntegerv(GL_VIEWPORT, glViewport);

    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    glScalef(2.f/(float)glViewport[2],2.f/(float)glViewport[3],1.f);
    if (white)
        glColor3f(1.f,1.f,1.f);
    else
        glColor3f(0.f,0.f,0.f);
    glDrawEllipse( 0.f, 0.f, radius, radius);
    glDrawEllipse( 0.f, 0.f, mini, mini);
    glPopMatrix();

}

void cGLData::normalizeCurrentPolygon(bool nrm)
{
    if(currentPolygon())
        currentPolygon()->normalize(nrm);
}

void cGLData::clearPolygon()
{
    if(currentPolygon())
        currentPolygon()->clear();
}

void cGLData::setGlobalCenter(Pt3d<double> aCenter)
{
    setBBoxCenter(aCenter);
    _pBall->setPosition(aCenter);
    _pAxis->setPosition(aCenter);
    _pBbox->setPosition(aCenter);
    _pGrid->setPosition(aCenter);

   /* for (int aK=0; aK < _vClouds.size();++aK)
       _vClouds[aK]->setPosition(aCenter);*/
}

bool cGLData::position2DClouds(MatrixManager &mm, QPointF pos)
{
    mm.setMatrices();

    int idx1 = -1;
    int idx2;

    pos.setY(mm.vpHeight() - pos.y());

    for (int aK=0; aK < _vClouds.size();++aK)
    {
        float sqrD;
        float dist = FLT_MAX;
        idx2 = -1; // TODO a verifier, pourquoi init a -1 , probleme si plus 2 nuages...
        QPointF proj;

        GlCloud *a_cloud = _vClouds[aK];

        for (int bK=0; bK < a_cloud->size();++bK)
        {
            mm.getProjection(proj, a_cloud->getVertex( bK ).getPosition());

            sqrD = (proj.x()-pos.x())*(proj.x()-pos.x()) + (proj.y()-pos.y())*(proj.y()-pos.y());

            if (sqrD < dist )
            {
                dist = sqrD;
                idx1 = aK;
                idx2 = bK;
            }
        }
    }

    if ((idx1>=0) && (idx2>=0))
    {
        //final center:
        GlCloud *a_cloud = _vClouds[idx1];
        Pt3dr Pt = a_cloud->getVertex( idx2 ).getPosition();

        setGlobalCenter(Pt);
        mm.resetAllMatrix(Pt);
        return true;
    }

    return false;
}

void cGLData::editImageMask(int mode, cPolygon &polyg, bool m_bFirstAction)
{
    QPainter    p;
    QBrush SBrush(Qt::black);
    QBrush NSBrush(Qt::white);
    QRect  rect = getMask()->rect();

    p.begin(getMask());
    p.setCompositionMode(QPainter::CompositionMode_Source);
    p.setPen(Qt::NoPen);

    QPolygonF polyDraw(polyg.getVector());
    QPainterPath path;

    if(mode == ADD_INSIDE || mode == SUB_INSIDE)
    {
        path.addPolygon(polyDraw);
    }
    else if((mode == ADD_OUTSIDE || mode == SUB_OUTSIDE))
    {
        path.addRect(rect);
        QPainterPath inner;
        inner.addPolygon(polyDraw);
        path = path.subtracted(inner);
    }

    if(mode == ADD_INSIDE || mode == ADD_OUTSIDE)
    {
        if (m_bFirstAction)
            p.fillRect(rect, Qt::white);

        p.setBrush(SBrush);
        p.drawPath(path);
    }
    else if(mode == SUB_INSIDE || mode == SUB_OUTSIDE)
    {
        p.setBrush(NSBrush);
        p.drawPath(path);
    }
    else if(mode == ALL)

        p.fillRect(rect, Qt::black);

    else if(mode == NONE)

        p.fillRect(rect, Qt::white);

    p.end();

    if (mode == INVERT)
        getMask()->invertPixels(QImage::InvertRgb);

    _glMaskedImage._m_mask->deleteTexture(); // TODO verifier l'utilité de supprimer la texture...
    _glMaskedImage._m_mask->PrepareTexture(getMask());
}

void cGLData::editCloudMask(int mode, cPolygon &polyg, bool m_bFirstAction, MatrixManager &mm)
{

    QPointF P2D;
    bool pointInside;

    for (int aK=0; aK < _vClouds.size(); ++aK)
    {
        GlCloud *a_cloud = _vClouds[aK];

        for (uint bK=0; bK < (uint) a_cloud->size();++bK)
        {
            GlVertex &P  = a_cloud->getVertex( bK );
            Pt3dr  Pt = P.getPosition();

            if(getRotation().x != 0)
            {
                Pt = Pt - getBBoxCenter() ;
                Pt = Pt3dr(Pt.x,Pt.z,-Pt.y);
                Pt = Pt + getBBoxCenter() ;
            }

            switch (mode)
            {
            case ADD_INSIDE:
                mm.getProjection(P2D, Pt);
                pointInside = polyg.isPointInsidePoly(P2D);
                if (m_bFirstAction)
                    P.setVisible(pointInside);
                else
                    P.setVisible(pointInside||P.isVisible());
                break;
            case ADD_OUTSIDE:
                mm.getProjection(P2D, Pt);
                pointInside = polyg.isPointInsidePoly(P2D);
                if (m_bFirstAction)
                    P.setVisible(!pointInside);
                else
                    P.setVisible(!pointInside||P.isVisible());
                break;
            case SUB_INSIDE:
                if (P.isVisible())
                {
                    mm.getProjection(P2D, Pt);
                    pointInside = polyg.isPointInsidePoly(P2D);
                    P.setVisible(!pointInside);
                }
                break;
            case SUB_OUTSIDE:
                if (P.isVisible())
                {
                    mm.getProjection(P2D, Pt);
                    pointInside = polyg.isPointInsidePoly(P2D);
                    P.setVisible(pointInside);
                }
                break;
            case INVERT:
                P.setVisible(!P.isVisible());
                break;
            case ALL:
            {
                m_bFirstAction = true;
                P.setVisible(true);
            }
                break;
            case NONE:
                P.setVisible(false);
                break;
            }
        }

        a_cloud->setBufferGl(true);
    }
}

void cGLData::replaceCloud(GlCloud *cloud, int id)
{
    if(id<_vClouds.size())
        _vClouds[id] = cloud;
    else
        _vClouds.insert(_vClouds.begin(),cloud);

    cloud->setBufferGl();
}

void cGLData::GprintBits(const size_t size, const void * const ptr)
{
    unsigned char *b = (unsigned char*) ptr;
    unsigned char byte;
    int i, j;

    for (i=size-1;i>=0;i--)
    {
        for (j=7;j>=0;j--)
        {
            byte = b[i] & (1<<j);
            byte >>= j;
            printf("%u", byte);
        }
    }
    puts("");
}

void cGLData::setOption(QFlags<cGLData::Option> option, bool show)
{

    if(show)
        _options |=  option;
    else
        _options &= ~option;

    //GprintBits(sizeof(QFlags<Option>),&_options);

    if(isImgEmpty())
    {
        _pBall->setVisible(stateOption(OpShow_Ball));
        _pAxis->setVisible(stateOption(OpShow_Axis));
        _pBbox->setVisible(stateOption(OpShow_BBox));
        _pGrid->setVisible(stateOption(OpShow_Grid));

        for (int i=0; i < _vCams.size();i++)
            _vCams[i]->setVisible(stateOption(OpShow_Cams));
    }
}
