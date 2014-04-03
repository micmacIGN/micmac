#include "3DObject.h"
#include "SaisieGlsl.glsl"

cObject::cObject() :
    _name(""),
    _position(Pt3dr(0.f,0.f,0.f)),
    _scale(1.f),
    _alpha(0.6f),
    _state(state_default)
{
 for (int iC = 0; iC < state_COUNT; ++iC)
    _color[iC] = QColor(255,255,255);

}

cObject::cObject(Pt3dr pos, QColor color_default) :
    _name(""),
    _scale(1.f),
    _alpha(0.6f),
    _state(state_default)
{
    _position  = pos;

    for (int iC = 0; iC < state_COUNT; ++iC)
        _color[iC] = color_default;
}

cObject::~cObject(){}

QColor cObject::getColor(){

    return _color[state()];

}

cObject& cObject::operator =(const cObject& aB)
{
    if (this != &aB)
    {
        _name      = aB._name;
        _position  = aB._position;

        for (int iC = 0; iC < state_COUNT; ++iC)
            _color[iC]     = aB._color[iC];

        _alpha     = aB._alpha;
        _state     = aB.state();
    }

    return *this;
}
object_state cObject::state() const
{
    return _state;
}

void cObject::setState(object_state state)
{
    _state = state;
}


cCircle::cCircle(Pt3d<double> pt, QColor col, float scale, float lineWidth, bool vis, int dim) :
    _dim(dim)
{
    setPosition(pt);
    cObject::setColor(col);
    setScale(scale);
    setLineWidth(lineWidth);
    cObject::setVisible(vis);
}

//draw a unit circle in a given plane (0=YZ, 1 = XZ, 2=XY)
void glDrawUnitCircle(uchar dim, float cx, float cy, float r, int steps)
{
    float theta = 2.f * PI / float(steps);
    float c = cosf(theta); //precalculate the sine and cosine
    float s = sinf(theta);
    float t;

    float x = r; //we start at angle = 0
    float y = 0;

    uchar dimX = (dim<2 ? dim+1 : 0);
    uchar dimY = (dimX<2 ? dimX+1 : 0);

    GLfloat P[3];

    for (int i=0;i<3;++i) P[i] = 0.0f;

    glBegin(GL_LINE_LOOP);
    for(int ii = 0; ii < steps; ii++)
    {
        P[dimX] = x + cx;
        P[dimY] = y + cy;
        glVertex3fv(P);

        //apply the rotation matrix
        t = x;
        x = c * x - s * y;
        y = s * t + c * y;
    }
    glEnd();
}

void cCircle::draw()
{
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();

    GLuint list = glGenLists(1);
    glNewList(list, GL_COMPILE);

    glPushAttrib(GL_LINE_BIT | GL_DEPTH_BUFFER_BIT);

    glLineWidth(_lineWidth);

    setGLColor();

    glDrawUnitCircle(_dim, 0, 0, 1.f, 64);

    glPopAttrib();

    glEndList();

    glTranslatef(_position.x,_position.y,_position.z);
    glScalef(_scale,_scale,_scale);

    glCallList(list);

    glPopMatrix();
}

cCross::cCross(Pt3d<double> pt, QColor col, float scale, float lineWidth, bool vis, int dim) :
    _dim(dim)
{
    setPosition(pt);
    cObject::setColor(col);
    setScale(scale);
    setLineWidth(lineWidth);
    cObject::setVisible(vis);
}

void cCross::draw()
{
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();

    GLuint list = glGenLists(1);
    glNewList(list, GL_COMPILE);

    glPushAttrib(GL_LINE_BIT | GL_DEPTH_BUFFER_BIT);

    glLineWidth(_lineWidth);

    setGLColor();

    float x1, x2, y1, y2, z1, z2;
    x1 = x2 = y1 = y2 = z1 = z2 = 0.f;

    if (_dim == 0)
    {
        x1 =-1.f;
        x2 = 1.f;
    }
    else if (_dim == 1)
    {
        y1 =-1.f;
        y2 = 1.f;
    }
    else if (_dim == 2)
    {
        z1 =-1.f;
        z2 = 1.f;
    }

    glBegin(GL_LINES);
    glVertex3f(x1,y1,z1);
    glVertex3f(x2,y2,z2);
    glEnd();
    glPopAttrib();

    glEndList();

    glTranslatef(_position.x,_position.y,_position.z);
    glScalef(_scale,_scale,_scale);

    glCallList(list);

    glPopMatrix();
}

cBall::cBall(Pt3dr pt, float scale, bool isVis, float lineWidth)
{
    cObject::setVisible(isVis);

    _cl0 = new cCircle(pt, QColor(255,0,0),   scale, lineWidth, isVis, 0);
    _cl1 = new cCircle(pt, QColor(0,255,0),   scale, lineWidth, isVis, 1);
    _cl2 = new cCircle(pt, QColor(0,178,255), scale, lineWidth, isVis, 2);

    _cr0 = new cCross(pt, QColor(255,0,0),   scale, lineWidth, isVis, 0);
    _cr1 = new cCross(pt, QColor(0,255,0),   scale, lineWidth, isVis, 1);
    _cr2 = new cCross(pt, QColor(0,178,255), scale, lineWidth, isVis, 2);
}

cBall::~cBall()
{
    delete _cl0;
    delete _cl1;
    delete _cl2;

    delete _cr0;
    delete _cr1;
    delete _cr2;
}

void cBall::draw()
{
    if (isVisible())
    {
        _cl0->draw();
        _cl1->draw();
        _cl2->draw();

        _cr0->draw();
        _cr1->draw();
        _cr2->draw();
    }
}

void cBall::setPosition(Pt3dr const &aPt)
{
    _cl0->setPosition(aPt);
    _cl1->setPosition(aPt);
    _cl2->setPosition(aPt);

    _cr0->setPosition(aPt);
    _cr1->setPosition(aPt);
    _cr2->setPosition(aPt);
}

Pt3dr cBall::getPosition()
{
    return _cl0->getPosition();
}

void cBall::setVisible(bool aVis)
{
    cObject::setVisible(aVis);
    //_bVisible = aVis;

    _cl0->setVisible(aVis);
    _cl1->setVisible(aVis);
    _cl2->setVisible(aVis);

    _cr0->setVisible(aVis);
    _cr1->setVisible(aVis);
    _cr2->setVisible(aVis);
}

void cBall::setScale(float aScale)
{
    _cl0->setScale(aScale);
    _cl1->setScale(aScale);
    _cl2->setScale(aScale);

    _cr0->setScale(aScale);
    _cr1->setScale(aScale);
    _cr2->setScale(aScale);
}

cAxis::cAxis(Pt3dr pt, float scale, float lineWidth)
{
    _position = pt;
    _scale    = scale;
    setLineWidth(lineWidth);
}

void cAxis::draw()
{
    if (isVisible())
    {
        glMatrixMode(GL_MODELVIEW);
        glPushMatrix();

        GLuint dihedron = glGenLists(1);
        glNewList(dihedron, GL_COMPILE);

        glPushAttrib(GL_LINE_BIT | GL_DEPTH_BUFFER_BIT);

        glLineWidth(_lineWidth);

        glBegin(GL_LINES);
        glColor3f(1.0f,0.0f,0.0f);
        glVertex3f(0.0f,0.0f,0.0f);
        glVertex3f(0.4f,0.0f,0.0f);
        glColor3f(0.0f,1.0f,0.0f);
        glVertex3f(0.0f,0.0f,0.0f);
        glVertex3f(0.0f,0.4f,0.0f);
        glColor3f(0.0f,0.7f,1.0f);
        glVertex3f(0.0f,0.0f,0.0f);
        glVertex3f(0.0f,0.0f,0.4f);
        glEnd();

        glPopAttrib();

        glEndList();

        glTranslatef(_position.x,_position.y,_position.z);
        glScalef(_scale,_scale,_scale);

        glCallList(dihedron);

        glPopMatrix();
    }
}

cBBox::cBBox(Pt3dr pt, float scale, Pt3dr min, Pt3dr max, float lineWidth)
{
    _position = pt;
    _scale = scale;
    _min = min;
    _max = max;

    cObject::setColor(QColor("orange"));
    setLineWidth(lineWidth);
}

void cBBox::set(Pt3dr min, Pt3dr max)
{
    _min = min;
    _max = max;
}

void cBBox::draw()
{
    if (isVisible())
    {
        glMatrixMode(GL_MODELVIEW);
        glPushMatrix();

        GLuint list = glGenLists(1);
        glNewList(list, GL_COMPILE);

        glPushAttrib(GL_LINE_BIT|GL_DEPTH_BUFFER_BIT);

        glLineWidth(_lineWidth);

        setGLColor();

        Pt3dr P1(_min);
        Pt3dr P2(_min.x, _min.y, _max.z);
        Pt3dr P3(_min.x, _max.y, _max.z);
        Pt3dr P4(_min.x, _max.y, _min.z);
        Pt3dr P5(_max.x, _min.y, _min.z);
        Pt3dr P6(_max.x, _max.y, _min.z);
        Pt3dr P7(_max);
        Pt3dr P8(_max.x, _min.y, _max.z);

        glBegin(GL_LINES);

        glVertex3d(P1.x, P1.y, P1.z);
        glVertex3d(P2.x, P2.y, P2.z);

        glVertex3d(P3.x, P3.y, P3.z);
        glVertex3d(P2.x, P2.y, P2.z);

        glVertex3d(P1.x, P1.y, P1.z);
        glVertex3d(P4.x, P4.y, P4.z);

        glVertex3d(P1.x, P1.y, P1.z);
        glVertex3d(P5.x, P5.y, P5.z);

        glVertex3d(P7.x, P7.y, P7.z);
        glVertex3d(P3.x, P3.y, P3.z);

        glVertex3d(P7.x, P7.y, P7.z);
        glVertex3d(P6.x, P6.y, P6.z);

        glVertex3d(P8.x, P8.y, P8.z);
        glVertex3d(P5.x, P5.y, P5.z);

        glVertex3d(P7.x, P7.y, P7.z);
        glVertex3d(P8.x, P8.y, P8.z);

        glVertex3d(P5.x, P5.y, P5.z);
        glVertex3d(P6.x, P6.y, P6.z);

        glVertex3d(P4.x, P4.y, P4.z);
        glVertex3d(P6.x, P6.y, P6.z);

        glVertex3d(P8.x, P8.y, P8.z);
        glVertex3d(P2.x, P2.y, P2.z);

        glVertex3d(P4.x, P4.y, P4.z);
        glVertex3d(P3.x, P3.y, P3.z);

        glEnd();

        glPopAttrib();

        glEndList();

        glCallList(list);

        glPopMatrix();
    }
}

cCam::cCam(CamStenope *pCam, float scale,  object_state state, float lineWidth) :
    _pointSize(5.f),
    _Cam(pCam)
{
    _scale = scale;

    setState(state);
    cObject::setColor(QColor("red"));
    cObject::setColor(QColor(0.f,0.f,1.f),state_selected);
    setLineWidth(lineWidth);
}

void cCam::draw()
{
    if (isVisible())
    {

        GLfloat oldPointSize;
        glGetFloatv(GL_POINT_SIZE,&oldPointSize);

        glMatrixMode(GL_MODELVIEW);
        glPushMatrix();

        GLuint list = glGenLists(1);
        glNewList(list, GL_COMPILE);

        glPushAttrib(GL_LINE_BIT|GL_DEPTH_BUFFER_BIT);

        glLineWidth(_lineWidth);

        glPointSize(_pointSize);

        Pt2di sz = _Cam->Sz();

        double aZ = _scale*.05f;

        Pt3dr C  = _Cam->VraiOpticalCenter();
        Pt3dr P1 = _Cam->ImEtProf2Terrain(Pt2dr(0.f,0.f),aZ);
        Pt3dr P2 = _Cam->ImEtProf2Terrain(Pt2dr(sz.x,0.f),aZ);
        Pt3dr P3 = _Cam->ImEtProf2Terrain(Pt2dr(0.f,sz.y),aZ);
        Pt3dr P4 = _Cam->ImEtProf2Terrain(Pt2dr(sz.x,sz.y),aZ);

        glBegin(GL_LINES);
        //perspective cone
        if(!isSelected())
            glColor3f(1.f,1.f,1.f);
        else
        {
            QColor color = Qt::yellow;
            glColor3f(color.redF(),color.greenF(),color.blueF());
        }

        glVertex3d(C.x, C.y, C.z);
        glVertex3d(P1.x, P1.y, P1.z);

        glVertex3d(C.x, C.y, C.z);
        glVertex3d(P2.x, P2.y, P2.z);

        glVertex3d(C.x, C.y, C.z);
        glVertex3d(P3.x, P3.y, P3.z);

        glVertex3d(C.x, C.y, C.z);
        glVertex3d(P4.x, P4.y, P4.z);

        //Image

        setGLColor();

        glVertex3d(P1.x, P1.y, P1.z);
        glVertex3d(P2.x, P2.y, P2.z);

        glVertex3d(P4.x, P4.y, P4.z);
        glVertex3d(P2.x, P2.y, P2.z);

        glVertex3d(P3.x, P3.y, P3.z);
        glVertex3d(P1.x, P1.y, P1.z);

        glVertex3d(P4.x, P4.y, P4.z);
        glVertex3d(P3.x, P3.y, P3.z);
        glEnd();

        glBegin(GL_POINTS);
        glVertex3d(C.x, C.y, C.z);
        glEnd();

        glEndList();

        glPopAttrib();

        glCallList(list);

        glPopMatrix();
        glPointSize(oldPointSize);
    }
}


cPoint::cPoint(QPainter * painter, QPointF pos,
               QString name, bool showName,
               int state,
               bool isSelected,
               QColor color, QColor selectionColor,
               float diameter,
               bool highlight):
    QPointF(pos),
    _diameter(diameter),
    _bShowName(showName),
    _statePoint(state),
    _highlight(highlight),
    _painter(painter),
    _bEpipolar(false)
{
    setName(name);
    cObject::setColor(color);
    cObject::setColor(selectionColor,state_selected);
    setSelected(isSelected);
}

void cPoint::draw()
{
     if ((_painter != NULL) && isVisible())
     {
         QPen penline(getColor());
         penline.setCosmetic(true);
         _painter->setPen(penline);

         QPointF pt = _painter->transform().map((QPointF)*this);

         _painter->setWorldMatrixEnabled(false);

         if (!isSelected())
         {
             switch(_statePoint)
             {
             case eEPI_NonSaisi :
                 _painter->setPen(Qt::yellow);
                 break;

             case eEPI_Refute :
                 _painter->setPen(Qt::red);
                 break;

             case eEPI_Douteux :
                 _painter->setPen(QColor(255, 127, 0, 255) );
                 break;

             case eEPI_Valide :
                 _painter->setPen(Qt::green);
                 break;

             case eEPI_Disparu  :
             case eEPI_NonValue :
                 break;
             }
         }

         _painter->drawEllipse(pt, _diameter, _diameter);

         if (_highlight && ((_statePoint == eEPI_Valide) || (_statePoint == eEPI_NonSaisi)))
         {
             if (_bEpipolar)
             {
                 QPointF epip1 = _painter->transform().map(_epipolar1);
                 QPointF epip2 = _painter->transform().map(_epipolar2);

                 _painter->drawLine(epip1, epip2);
             }
             else
             {
                 double r = 2.f*_diameter +1.f;
                 _painter->drawEllipse(pt, r, r);
             }
         }

         if ((_bShowName) && (_name != ""))
         {
             //QFontMetrics metrics = QFontMetrics(_font);
             //int border = (float) qMax(2, metrics.leading());
             int border = 1;

             QRect rect = QFontMetrics(_font).boundingRect(_name);

             QRect rectg(pt.x()-border, pt.y()-border, rect.width()-border, rect.height()-border);
             rectg.translate(QPoint(10, -rectg.height()-5));

             _painter->setPen(isSelected() ? Qt::black : Qt::white);
             _painter->fillRect(rectg, isSelected() ? QColor(255, 255, 255, 127) : QColor(0, 0, 0, 127));
             _painter->drawText(rectg, Qt::AlignCenter /*| Qt::TextWordWrap*/, _name);
         }

         _painter->setWorldMatrixEnabled(true);
     }
}

void cPoint::setEpipolar(QPointF pt1, QPointF pt2)
{
    _epipolar1 = pt1;
    _epipolar2 = pt2;
    _bEpipolar = true;
}





//********************************************************************************

float cPolygon::_selectionRadius = 10.f;

cPolygon::cPolygon(QPainter* painter,float lineWidth, QColor lineColor, QColor pointColor, int style):
    _helper(new cPolygonHelper(this, lineWidth, painter)),
    _lineColor(lineColor),
    _idx(-1),
    _painter(painter),
    _pointDiameter(6.f),
    _bIsClosed(false),
    _bSelectedPoint(false),
    _bShowLines(true),
    _bShowNames(true),
    _bShowRefuted(true),
    _style(style)
{
    setColor(pointColor);
    setLineWidth(lineWidth);
}

cPolygon::cPolygon(QVector<QPointF> points, bool isClosed) :
    _bIsClosed(isClosed)
{
    setVector(points);
}

cPolygon::cPolygon(QPainter* painter,float lineWidth, QColor lineColor,  QColor pointColor, bool withHelper, int style):
    _lineColor(lineColor),
    _idx(-1),
    _painter(painter),
    _pointDiameter(6.f),
    _bIsClosed(false),
    _bSelectedPoint(false),
    _bShowLines(true),
    _bShowNames(true),
    _bShowRefuted(true),
    _style(style)
{
    if (!withHelper) _helper = NULL;
    cObject::setColor(pointColor);
    setLineWidth(lineWidth);

    _dashes << 3 << 4;
}

void cPolygon::draw()
{
    if(_painter != NULL)
    {
        _painter->setRenderHint(QPainter::Antialiasing,true);

        if (_bShowLines)
        {
            QPen penline(isSelected() ? QColor(0,140,180) : _lineColor);
            penline.setCosmetic(true);
            penline.setWidthF(_lineWidth);
            if(_style == LINE_STIPPLE)
            {
                penline.setWidthF(_lineWidth);
                penline.setStyle(Qt::CustomDashLine);
                penline.setDashPattern(_dashes);
            }

            _painter->setPen(penline);

            if(_bIsClosed)
                _painter->drawPolygon(getVector().data(),size());
            else
                _painter->drawPolyline(getVector().data(),size());
        }

        for (int aK = 0;aK < _points.size(); ++aK)
        {
#if ELISE_QT_VERSION==4
            _points[aK].setPainter(_painter);
#endif
            _points[aK].draw();
        }

        if(helper()!=NULL)
             helper()->draw();

         _painter->setRenderHint(QPainter::Antialiasing,false);
    }
}

cPolygon & cPolygon::operator = (const cPolygon &aP)
{
    if (this != &aP)
    {
        _lineWidth        = aP._lineWidth;
        _pointDiameter        = aP._pointDiameter;
        _bIsClosed        = aP._bIsClosed;
        _idx              = aP._idx;

        _points           = aP._points;
    }

    return *this;
}

void cPolygon::close()
{
    int sz = _points.size();

    if ((sz>1)&&(!_bIsClosed))
    {
        //remove last point if needed
        if (sz > 2) _points.resize(sz-1);

        _bIsClosed = true;
    }

    _bSelectedPoint = false;
}

void cPolygon::removeNearestOrClose(QPointF pos)
{
    if (_bIsClosed)
    {
        removeSelectedPoint();

        findNearestPoint(pos);

        if (size() < 3) _bIsClosed = false;
    }
    else // close polygon
        close();
}

void cPolygon::removeSelectedPoint()
{
    if (pointValid())

        removePoint(_idx);
}

int cPolygon::setNearestPointState(const QPointF &pos, int state)
{
    int idx = _idx;

    findNearestPoint(pos, 400000.f);

    if (pointValid())
    {
        if (state == eEPI_NonValue)
        {
            //TODO: cWinIm l.661
            _points.remove(_idx);
        }
        else
        {
            _points[_idx].setStatePoint(state);
            _points[_idx].setSelected(false);
        }
    }

    _idx = -1;
    _bSelectedPoint = false;

    return idx;
}

int cPolygon::highlightNearestPoint(const QPointF &pos)
{
    findNearestPoint(pos, 400000.f);

    if (pointValid())
    {
        _points[_idx].switchHighlight();
    }

    return _idx;
}

int cPolygon::getNearestPointIndex(const QPointF &pos)
{
    findNearestPoint(pos, 400000.f);

    return _idx;
}

QString cPolygon::getNearestPointName(const QPointF &pos)
{
    findNearestPoint(pos, 400000.f);

    return getSelectedPointName();
}

QString cPolygon::getSelectedPointName()
{
    if (pointValid())
    {
        return _points[_idx].name();
    }
    else return _defPtName;
}

int cPolygon::getSelectedPointState()
{
    if (pointValid())
    {
        return _points[_idx].statePoint();
    }
    else return eEPI_NonValue;
}

void cPolygon::add(cPoint &pt)
{
    pt.setDiameter(_pointDiameter);
    _points.push_back(pt);
}

void cPolygon::add(const QPointF &pt, bool selected)
{
    cPoint cPt(_painter, pt, _defPtName, _bShowNames, eEPI_NonValue, selected, _color[state_default]);
    cPt.setDiameter(_pointDiameter);
    _points.push_back(cPt);
}

void cPolygon::addPoint(const QPointF &pt)
{
    if (size() >= 1)
    {
        cPoint cPt(_painter, pt, _defPtName, _bShowNames, eEPI_NonValue, false, _color[state_default]);
        cPt.setDiameter(_pointDiameter);
        _points[size()-1] = cPoint(cPt);
    }

    add(pt);
}

void cPolygon::clear()
{
    _points.clear();
    _idx = -1;
    _bSelectedPoint = false;
    if(_bShowLines)_bIsClosed = false;
    if(_helper!=NULL) helper()->clear();
}

void cPolygon::insertPoint(int i, const QPointF &value)
{
    _points.insert(i,cPoint(_painter, value));
    resetSelectedPoint();
}

void cPolygon::insertPoint()
{
    if ((size() >=2) && _helper->size()>1 && _bIsClosed)
    {
        int idx = -1;
        QPointF Pt1 = (*_helper)[0];
        QPointF Pt2 = (*_helper)[1];

        for (int i=0;i<size();++i)
        {
            if (_points[i] == Pt1) idx = i;
        }

        if (idx >=0) insertPoint(idx+1, Pt2);
    }

    _helper->clear();
}

void cPolygon::removePoint(int i)
{
    _points.remove(i);
    _idx = -1;
}

const QVector<QPointF> cPolygon::getVector()
{
    QVector <QPointF> points;

    for(int aK=0; aK < _points.size(); ++aK)
    {
        points.push_back(_points[aK]);
    }
    return points;
}

void cPolygon::setVector(const QVector<QPointF> &aPts)
{
    _points.clear();
    for(int aK=0; aK < aPts.size(); ++aK)
    {
        _points.push_back(cPoint(_painter, aPts[aK]));
    }
}

void cPolygon::setPointSelected()
{
    _bSelectedPoint = true;

    if (pointValid())
        _points[_idx].setSelected(true);
}

void cPolygon::resetSelectedPoint()
{
    _bSelectedPoint = false;

    if (pointValid())
        _points[_idx].setSelected(false);
    _idx = -1;
}

bool cPolygon::pointValid()
{
    return ((_idx >=0) && (_idx < _points.size()));
}

int cPolygon::selectPoint(QString namePt)
{
    resetSelectedPoint();

    for (int i = 0; i < _points.size(); ++i)
    {
        if(_points[i].name() == namePt)
        {
            _idx = i;
            _points[i].setSelected(true);
            return i;
        }
    }

    return _idx;
}

bool cPolygon::findNearestPoint(QPointF const &pos, float radius)
{
    if (_bIsClosed)
    {
        resetSelectedPoint();

        float dist, dist2, x, y, dx, dy;
        dist2 = radius*radius;
        x = pos.x();
        y = pos.y();

        for (int aK = 0; aK < _points.size(); ++aK)
        {
            dx = x - _points[aK].x();
            dy = y - _points[aK].y();

            dist = dx * dx + dy * dy;

            if  (dist < dist2)
            {
                dist2 = dist;
                _idx = aK;
            }
        }

        if (pointValid())
        {
            _points[_idx].setSelected(true);

            return true;
        }
    }

    return false;
}

void cPolygon::refreshHelper(QPointF pos, bool insertMode, float zoom)
{
    int nbVertex = size();

    if(!_bIsClosed)
    {
        if (nbVertex == 1)                   // add current mouse position to polygon (for dynamic display)

            add(pos);

        else if (nbVertex > 1)               // replace last point by the current one
        {
            _points[nbVertex-1].setX( pos.x() );
            _points[nbVertex-1].setY( pos.y() );
        }
    }
    else if(nbVertex)                        // move vertex or insert vertex (dynamic display) en cours d'operation
    {
        if ((insertMode || isPointSelected())) // insert polygon point
        {
            cPoint pt(_painter, pos, getSelectedPointName(), _bShowNames, getSelectedPointState(), isPointSelected(), _color[state_default]);

            _helper->build(pt, insertMode);
        }
        else                                 // select nearest polygon point

            findNearestPoint(pos, _selectionRadius / zoom);
    }
}

int cPolygon::finalMovePoint()
{
    int idx = _idx;

    if ((_idx>=0) && _helper->size())   // after point move
    {
        int state = _points[_idx].statePoint();

        _points[_idx] = (*_helper)[1];
        _points[_idx].setColor(_color[state_default]); // reset color to polygon color
        _points[_idx].setStatePoint(state);

        _helper->clear();

        resetSelectedPoint();
    }

    return idx;
}

void cPolygon::removeLastPoint()
{
    if (size() >= 1)
    {
        removePoint(size()-1);
        _bIsClosed = false;
    }
}

void cPolygon::setPainter(QPainter *painter)
{
    _painter = painter;

    if (_helper != NULL)
        _helper->setPainter(_painter);
}

void cPolygon::showNames(bool show)
{
    _bShowNames = show;

    for (int aK=0; aK < _points.size(); ++aK)
        _points[aK].showName(_bShowNames);
}

void cPolygon::rename(QPointF pos, QString name)
{
    findNearestPoint(pos, 400000.f);

    if (pointValid())
        _points[_idx].setName(name);
}

void cPolygon::showLines(bool show)
{
    _bShowLines = show;

    if (_helper != NULL) _helper->showLines(show);

    if(!show) _bIsClosed = true;

    cObject::setColor(show ? Qt::red : Qt::green);
}

void cPolygon::translate(QPointF Tr)
{
    for (int aK=0; aK < _points.size(); ++aK)
        _points[aK] += Tr;
}

void cPolygon::flipY(float height)
{
    for (int aK=0; aK < size(); ++aK)
        _points[aK].setY(height - _points[aK].y());
}

void cPolygon::setParams(cParameters *aParams)
{
    setRadius(aParams->getSelectionRadius());
    setPointSize(aParams->getPointDiameter());
    setLineWidth(aParams->getLineThickness());
}

bool cPolygon::isPointInsidePoly(const QPointF& P)
{
    int vertices=_points.size();
    if (vertices<3)
        return false;

    bool inside = false;

    QPointF A = _points[0];
    QPointF B;

    for (int i=1;i<=vertices;++i)
    {
        B = _points[i%vertices];

        //Point Inclusion in Polygon Test (inspired from W. Randolph Franklin - WRF)
        if (((B.y() <= P.y()) && (P.y()<A.y())) ||
                ((A.y() <= P.y()) && (P.y()<B.y())))
        {
            float ABy = A.y()-B.y();
            float t = (P.x()-B.x())*ABy-(A.x()-B.x())*(P.y()-B.y());
            if (ABy<0)
                t=-t;

            if (t<0)
                inside = !inside;
        }

        A=B;
    }

    return inside;
}

void cPolygon::showRefuted(bool show)
{
    _bShowRefuted = show;

    for (int aK=0; aK < _points.size(); ++aK)
    {
        if (_points[aK].statePoint() == eEPI_Refute)
            _points[aK].setVisible(_bShowRefuted);
    }
}

//********************************************************************************

cPolygonHelper::cPolygonHelper(cPolygon* polygon, float lineWidth, QPainter *painter, QColor lineColor, QColor pointColor):
    cPolygon(painter, lineWidth, lineColor, pointColor,false),
    _polygon(polygon)
{

}

float segmentDistToPoint(QPointF segA, QPointF segB, QPointF p)
{
    QPointF p2(segB.x() - segA.x(), segB.y() - segA.y());
    float nrm = (p2.x()*p2.x() + p2.y()*p2.y());
    float u = ((p.x() - segA.x()) * p2.x() + (p.y() - segA.y()) * p2.y()) / nrm;

    if (u > 1)
        u = 1;
    else if (u < 0)
        u = 0;

    float x = segA.x() + u * p2.x();
    float y = segA.y() + u * p2.y();

    float dx = x - p.x();
    float dy = y - p.y();

    return sqrt(dx*dx + dy*dy);
}

void cPolygonHelper::build(cPoint const &pos, bool insertMode)
{
    int sz = _polygon->size();

    if (insertMode)
    {
        float dist, dist2 = FLT_MAX;
        int idx  = -1;
        for (int aK =0; aK < sz; ++aK)
        {
            dist = segmentDistToPoint((*_polygon)[aK], (*_polygon)[(aK + 1)%sz], pos);

            if (dist < dist2)
            {
                dist2 = dist;
                idx = aK;
            }
        }

        if (idx != -1)
            setPoints((*_polygon)[idx],pos,(*_polygon)[(idx+1)%sz]);
    }
    else //moveMode
    {
        if (sz > 1)
        {
            int idx = _polygon->idx();

            if ((idx > 0) && (idx <= sz-1))
                setPoints((*_polygon)[(idx-1)%sz],pos,(*_polygon)[(idx+1)%sz]);
            else if (idx  == 0)
                setPoints((*_polygon)[sz-1],pos,(*_polygon)[1]);
        }
        else
            setPoints(pos, pos, pos);
    }
}

void cPolygonHelper::setPoints(cPoint p1, cPoint p2, cPoint p3)
{
    clear();
    add(p1);
    add(p2);
    add(p3);
}

//********************************************************************************

//invalid GL list index
const GLuint GL_INVALID_LIST_ID = (~0);

cImageGL::cImageGL(float gamma) :
    _originX(0.f),
    _originY(0.f),
    _texture(GL_INVALID_LIST_ID),
    _gamma(gamma)
{
    _program.addShaderFromSourceCode(QGLShader::Vertex,vertexShader);
    _program.addShaderFromSourceCode(QGLShader::Fragment,fragmentGamma);
    _program.link();

    _texLocation    = _program.uniformLocation("tex");
    _gammaLocation  = _program.uniformLocation("gamma");
}

cImageGL::~cImageGL()
{
    if (_texture != GL_INVALID_LIST_ID)
    {
        glDeleteLists(_texture,1);
        _texture = GL_INVALID_LIST_ID;
    }
}

void cImageGL::drawQuad(QColor color)
{
    drawQuad(_originX, _originY, _glh, _glw,color);
}

void cImageGL::drawQuad(GLfloat originX, GLfloat originY, GLfloat glh, GLfloat glw, QColor color)
{
    glColor4f(color.redF(),color.greenF(),color.blueF(),color.alphaF());
    glBegin(GL_QUADS);
    {
        glTexCoord2f(0.0f, 0.0f);
        glVertex2f(originX, originY);
        glTexCoord2f(1.0f, 0.0f);
        glVertex2f(originX+glw, originY);
        glTexCoord2f(1.0f, 1.0f);
        glVertex2f(originX+glw, originY+glh);
        glTexCoord2f(0.0f, 1.0f);
        glVertex2f(originX, originY+glh);
    }
    glEnd();
}

void cImageGL::draw()
{
    glEnable(GL_TEXTURE_2D);
    glBindTexture( GL_TEXTURE_2D, _texture );

    if(_gamma !=1.0f)
    {
        _program.bind();
        _program.setUniformValue(_texLocation, GLint(0));
        _program.setUniformValue(_gammaLocation, GLfloat(1.0f/_gamma));
    }

    drawQuad(Qt::white);

    if(_gamma !=1.0f) _program.release();

    glBindTexture( GL_TEXTURE_2D, 0);
    glDisable(GL_TEXTURE_2D);
}

void cImageGL::draw(QColor color)
{
    drawQuad(color);
}

void cImageGL::setPosition(GLfloat originX, GLfloat originY)
{
    _originX = originX;
    _originY = originY;
}

void cImageGL::setDimensions(GLfloat glh, GLfloat glw)
{
    _glh = glh;
    _glw = glw;
}

bool cImageGL::isPtInside(const QPointF &pt)
{
    return (pt.x()>=0.f)&&(pt.y()>=0.f)&&(pt.x()<width())&&(pt.y()<height());
}

void cImageGL::PrepareTexture(QImage * pImg)
{
    glGenTextures(1, getTexture() );

    ImageToTexture(pImg);

    _size = pImg->size();
}

void cImageGL::ImageToTexture(QImage *pImg)
{
    glEnable(GL_TEXTURE_2D);
    glBindTexture( GL_TEXTURE_2D, _texture );
    glTexImage2D( GL_TEXTURE_2D, 0, 4, pImg->width(), pImg->height(), 0, GL_RGBA, GL_UNSIGNED_BYTE, pImg->bits());
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST );
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST );
    glBindTexture( GL_TEXTURE_2D, 0);
    glDisable(GL_TEXTURE_2D);
}

void cImageGL::drawGradientBackground(int w, int h, QColor c1, QColor c2)
{
    glEnable(GL_BLEND);
    glBlendFunc(GL_ONE,GL_ZERO);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();

    w = (w>>1)+1;
    h = (h>>1)+1;

    glOrtho(-w,w,-h,h,-2.f, 2.f);

    const uchar BkgColor[3] = {(uchar) c1.red(),(uchar) c1.green(), (uchar) c1.blue()};
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    //Gradient "texture" drawing
    glBegin(GL_QUADS);
    //user-defined background color for gradient start
    glColor3ubv(BkgColor);
    glVertex2f(-w,h);
    glVertex2f(w,h);
    //and the inverse of points color for gradient end
    glColor3ub(c2.red(),c2.green(),c2.blue());
    glVertex2f(w,-h);
    glVertex2f(-w,-h);
    glEnd();

    glDisable(GL_BLEND);
}

//********************************************************************************

cMaskedImageGL::cMaskedImageGL(cMaskedImage<QImage> &qMaskedImage)
{
    _m_mask     = new cImageGL();
    _m_image    = new cImageGL(qMaskedImage._gamma);
    _m_newMask  = qMaskedImage._m_newMask;
    _m_mask->PrepareTexture(qMaskedImage._m_mask);
    _m_image->PrepareTexture(qMaskedImage._m_image);
    cObjectGL::setName(qMaskedImage.name());
}

void cMaskedImageGL::draw()
{
    glEnable(GL_BLEND);
    glBlendFunc(GL_ONE,GL_ZERO);
    glDisable(GL_ALPHA_TEST);
    glDisable(GL_DEPTH_TEST);

    glColor4f(1.0f,1.0f,1.0f,1.0f);

    if(_m_mask != NULL && true)
    {
        _m_mask->draw();
        glBlendFunc(GL_ONE,GL_ONE);
        _m_mask->draw(QColor(128,255,128));
        glBlendFunc(GL_DST_COLOR,GL_ZERO);
        glColor4f(1.0f,1.0f,1.0f,1.0f);
    }

    _m_image->draw();

    glDisable(GL_BLEND);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_ALPHA_TEST);
}

//********************************************************************************

void cObjectGL::setGLColor()
{
    QColor color = getColor();
    glColor4f(color.redF(),color.greenF(),color.blueF(),_alpha);
}

void cObjectGL::enableOptionLine()
{
   // glDisable(GL_DEPTH_TEST);
    glEnable(GL_DEPTH_TEST);
    glEnable (GL_LINE_SMOOTH);
    glEnable (GL_BLEND);
    glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glHint (GL_LINE_SMOOTH_HINT, GL_DONT_CARE);
}

void cObjectGL::disableOptionLine()
{
    glDisable(GL_BLEND);
    glDisable(GL_LINE_SMOOTH);
    //glEnable(GL_DEPTH_TEST);
}

//********************************************************************************

cGLData::cGLData():
    _diam(1.f)
{
    initOptions();
}

cGLData::cGLData(QMaskedImage &qMaskedImage, bool modePt, QString ptName):
    glMaskedImage(qMaskedImage),
    pQMask(qMaskedImage._m_mask),
    pBall(NULL),
    pAxis(NULL),
    pBbox(NULL),
    pGrid(NULL),
    _center(Pt3dr(0.f,0.f,0.f)),
    _modePt(modePt)
{

    initOptions();

    polygon()->showLines(!modePt);
    polygon()->showNames(modePt);

    polygon()->setDefaultName(ptName);
}


cGLData::cGLData(cData *data):
    pBall(new cBall),
    pAxis(new cAxis),
    pBbox(new cBBox),
    pGrid(new cGrid),
    _diam(1.f),
    _incFirstCloud(false)
{
    initOptions();

    setData(data);
}

void cGLData::setData(cData *data, bool setCam)
{
    for (int aK = 0; aK < data->getNbClouds();++aK)
    {
        GlCloud *pCloud = data->getCloud(aK);
        Clouds.push_back(pCloud);
        pCloud->setBufferGl();
    }

    Pt3dr center = data->getBBoxCenter();
    float scale = data->getBBoxMaxSize() / 1.5f;

    pBall->setPosition(center);
    pBall->setScale(scale);
    pAxis->setPosition(center);
    pAxis->setScale(scale);
    pBbox->setPosition(center);
    pBbox->setScale(scale);
    pBbox->set(data->getMin(), data->getMax());

    pGrid->setPosition(center);
    pGrid->setScale(scale*2.f);

    if(setCam)
        for (int i=0; i< data->getNbCameras(); i++)
        {
            cCam *pCam = new cCam(data->getCamera(i), scale);

            Cams.push_back(pCam);
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
    return glMaskedImage;
}

cPolygon *cGLData::polygon(int id)
{
    if(id < (int)m_VPolygons.size())
        return m_VPolygons[id];
    else
        return NULL;
}

GlCloud* cGLData::getCloud(int iC)
{
    return Clouds[iC];
}

int cGLData::cloudCount()
{
    return Clouds.size();
}

int cGLData::camerasCount()
{
    return Cams.size();
}

int cGLData::polygonCount()
{
    return m_VPolygons.size();
}

void cGLData::initOptions()
{
    m_VPolygons.push_back(new cPolygon());
//    m_VPolygons.push_back(new cPolygon());
//    m_VPolygons[1]->setPointSize(2.f);
    _options = options(OpShow_Mess);
}

cGLData::~cGLData()
{
    glMaskedImage.deallocImages();

    qDeleteAll(Cams);
    Cams.clear();

    if(pBall != NULL) delete pBall;
    if(pAxis != NULL) delete pAxis;
    if(pBbox != NULL) delete pBbox;
    if(pGrid != NULL) delete pGrid;

    //pas de delete des pointeurs dans Clouds c'est Data qui s'en charge
    Clouds.clear();
}

void cGLData::draw()
{
    enableOptionLine();

    for (int i=0; i<Clouds.size();i++)
    {
        GLfloat oldPointSize;
        glGetFloatv(GL_POINT_SIZE,&oldPointSize);

        if(_incFirstCloud && i == 0)
            glPointSize(oldPointSize*3.f);

         Clouds[i]->draw();

         glPointSize(oldPointSize);
    }

    pBall->draw();
    pAxis->draw();
    pBbox->draw();
    pGrid->draw();

    //cameras
    for (int i=0; i< Cams.size();i++) Cams[i]->draw();

    disableOptionLine();
}

void cGLData::setDimensionImage(int vW, int vH)
{
    float rw = (float) glMaskedImage._m_image->width()  / vW;
    float rh = (float) glMaskedImage._m_image->height() / vH;

    glMaskedImage.setDimensions(2.f*rh,2.f*rw);
}

void cGLData::setGlobalCenter(Pt3d<double> aCenter)
{
    setBBoxCenter(aCenter);
    pBall->setPosition(aCenter);
    pAxis->setPosition(aCenter);
    pBbox->setPosition(aCenter);
    pGrid->setPosition(aCenter);

    for (int aK=0; aK < Clouds.size();++aK)
       Clouds[aK]->setPosition(aCenter);
}

bool cGLData::position2DClouds(MatrixManager &mm, QPointF pos)
{
    bool foundPosition = false;
    mm.setMatrices();

    int idx1 = -1;
    int idx2;

    pos.setY(mm.vpHeight() - pos.y());

    for (int aK=0; aK < Clouds.size();++aK)
    {
        float sqrD;
        float dist = FLT_MAX;
        idx2 = -1; // TODO a verifier, pourquoi init a -1 , probleme si plus 2 nuages...
        QPointF proj;

        GlCloud *a_cloud = Clouds[aK];

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
        GlCloud *a_cloud = Clouds[idx1];
        Pt3dr Pt = a_cloud->getVertex( idx2 ).getPosition();

        setGlobalCenter(Pt);
        mm.resetTranslationMatrix(Pt);
        foundPosition = true;
    }

    return foundPosition;
}

void cGLData::editImageMask(int mode, cPolygon &polyg, bool m_bFirstAction)
{
    QPainter    p;
    QBrush SBrush(Qt::white);
    QBrush NSBrush(Qt::black);
    QRect  rect = getMask()->rect();

    p.begin(getMask());
    p.setCompositionMode(QPainter::CompositionMode_Source);
    p.setPen(Qt::NoPen);

    if(mode == ADD)
    {
        if (m_bFirstAction)
            p.fillRect(rect, Qt::black);

        p.setBrush(SBrush);
        p.drawPolygon(polyg.getVector().data(),polyg.size());
    }
    else if(mode == SUB)
    {
        p.setBrush(NSBrush);
        p.drawPolygon(polyg.getVector().data(),polyg.size());
    }
    else if(mode == ALL)

        p.fillRect(rect, Qt::white);

    else if(mode == NONE)

        p.fillRect(rect, Qt::black);

    p.end();

    if(mode == INVERT)
        getMask()->invertPixels(QImage::InvertRgb);

    glMaskedImage._m_mask->ImageToTexture(getMask());
}

void cGLData::editCloudMask(int mode, cPolygon &polyg, bool m_bFirstAction, MatrixManager &mm)
{
    mm.setModelViewMatrix();
    QPointF P2D;
    bool pointInside;

    for (int aK=0; aK < Clouds.size(); ++aK)
    {
        GlCloud *a_cloud = Clouds[aK];

        for (uint bK=0; bK < (uint) a_cloud->size();++bK)
        {
            GlVertex &P  = a_cloud->getVertex( bK );
            Pt3dr  Pt = P.getPosition();

            switch (mode)
            {
            case ADD:
                mm.getProjection(P2D, Pt);
                pointInside = polyg.isPointInsidePoly(P2D);
                if (m_bFirstAction)
                    P.setVisible(pointInside);
                else
                    P.setVisible(pointInside||P.isVisible());
                break;
            case SUB:
                if (P.isVisible())
                {
                    mm.getProjection(P2D, Pt);
                    pointInside = polyg.isPointInsidePoly(P2D);
                    P.setVisible(!pointInside);
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
    if(id<Clouds.size())
        Clouds[id] = cloud;
    else
        Clouds.insert(Clouds.begin(),cloud);

    cloud->setBufferGl();
}

void cGLData::setPainter(QPainter * painter)
{
    for (int i = 0; i < polygonCount(); ++i)
        polygon(i)->setPainter(painter);
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
        pBall->setVisible(stateOption(OpShow_Ball));
        pAxis->setVisible(stateOption(OpShow_Axis));
        pBbox->setVisible(stateOption(OpShow_BBox));
        pGrid->setVisible(stateOption(OpShow_Grid));

        for (int i=0; i < Cams.size();i++)
            Cams[i]->setVisible(stateOption(OpShow_Cams));
    }
}

//********************************************************************************

/*cMessages2DGL::~cMessages2DGL()
{
    glwid = NULL;
}*/

void cMessages2DGL::draw(){

    if (drawMessages())
    {
        int ll_curHeight, lr_curHeight, lc_curHeight; //lower left, lower right and lower center y position
        ll_curHeight = lr_curHeight = lc_curHeight = h - (m_font.pointSize())*(m_messagesToDisplay.size()-1) ;
        int uc_curHeight = m_font.pointSize();            //upper center

        std::list<MessageToDisplay>::iterator it = m_messagesToDisplay.begin();
        while (it != m_messagesToDisplay.end())
        {
            QRect rect = QFontMetrics(m_font).boundingRect(it->message);
            switch(it->position)
            {
            case LOWER_LEFT_MESSAGE:
                ll_curHeight -= renderTextLine(*it, m_font.pointSize(), ll_curHeight,m_font.pointSize());
                break;
            case LOWER_RIGHT_MESSAGE:
                lr_curHeight -= renderTextLine(*it, w - 120, lr_curHeight);
                break;
            case LOWER_CENTER_MESSAGE:
                lc_curHeight -= renderTextLine(*it,(w-rect.width())/2, lc_curHeight);
                break;
            case UPPER_CENTER_MESSAGE:
                uc_curHeight += renderTextLine(*it,(w-rect.width())/2, uc_curHeight+rect.height());
                break;
            case SCREEN_CENTER_MESSAGE:
                renderTextLine(*it,(w-rect.width())/2, (h-rect.height())/2,12);
            }
            ++it;
        }
    }
}

int cMessages2DGL::renderTextLine(MessageToDisplay messageTD, int x, int y, int sizeFont)
{


    m_font.setPointSize(sizeFont);

    glDisable(GL_LIGHTING);
    glDisable(GL_DEPTH_TEST);
    glwid->qglColor(messageTD.color);
    glwid->renderText(x, y, messageTD.message,m_font);
    glEnable(GL_DEPTH_TEST);
    //glEnable(GL_LIGHTING);

    return (QFontMetrics(m_font).boundingRect(messageTD.message).height()*5)/4;
}

void cMessages2DGL::displayNewMessage(const QString &message, MessagePosition pos, QColor color)
{
    if (message.isEmpty())
    {
        m_messagesToDisplay.clear();

        return;
    }

    MessageToDisplay mess;
    mess.message = message;
    mess.position = pos;
    mess.color = color;
    m_messagesToDisplay.push_back(mess);
}

void cMessages2DGL::constructMessagesList(bool show, int mode, bool m_bDisplayMode2D, bool dataloaded)
{
    _bDrawMessages = show;

    displayNewMessage(QString());

    if (show)
    {
        if(dataloaded)
        {
            if(m_bDisplayMode2D)
            {
                displayNewMessage(QString(" "),LOWER_RIGHT_MESSAGE, Qt::lightGray);
                displayNewMessage(QString(" "),LOWER_LEFT_MESSAGE, QColor("#ffa02f"));
            }
            else
            {
                if (mode == TRANSFORM_CAMERA)
                {
                    displayNewMessage(QString("Move mode"),UPPER_CENTER_MESSAGE);
                    displayNewMessage(QString("Left click: rotate viewpoint / Right click: translate viewpoint"),LOWER_CENTER_MESSAGE);
                }
                else if (mode == SELECTION)
                {
                    displayNewMessage(QString("Selection mode"),UPPER_CENTER_MESSAGE);
                    displayNewMessage(QString("Left click: add contour point / Right click: close"),LOWER_CENTER_MESSAGE);
                    displayNewMessage(QString("Space: add / Suppr: delete"),LOWER_CENTER_MESSAGE);
                }

                displayNewMessage(QString("0 Fps"), LOWER_LEFT_MESSAGE, Qt::lightGray);
            }
        }
        else
            displayNewMessage(QString("Drag & drop files"));
    }
}

std::list<MessageToDisplay>::iterator cMessages2DGL::GetLastMessage()
{
    std::list<MessageToDisplay>::iterator it = --m_messagesToDisplay.end();

    return it;
}

std::list<MessageToDisplay>::iterator cMessages2DGL::GetPenultimateMessage()
{
    return --GetLastMessage();
}

MessageToDisplay &cMessages2DGL::LastMessage()
{
    return m_messagesToDisplay.back();
}


cGrid::cGrid(Pt3d<double> pt, float scale, int nb)
{

}

void cGrid::draw()
{
    if (isVisible())
    {
        int nbGrid = 10;

        float scale = getScale() / nbGrid;

        Pt3dr pt;

        pt.x = getPosition().x - ((float)nbGrid * 0.5f) * scale;
        pt.y = getPosition().y ;
        pt.z = getPosition().z - ((float)nbGrid * 0.5f) * scale;

        glBegin(GL_LINES);
        glColor3f(.25,.25,.25);
        for(int i=0;i<=nbGrid;i++) {
            //if (i==0) { glColor3f(.6,.3,.3); } else { glColor3f(.25,.25,.25); };
            glVertex3f((float)i * scale + pt.x,pt.y,0+pt.z);
            glVertex3f((float)i * scale + pt.x,pt.y,(float)nbGrid * scale+ pt.z);
            //if (i==0) { glColor3f(.3,.3,.6); } else { glColor3f(.25,.25,.25); };
            glVertex3f( pt.x,pt.y,(float)i * scale + pt.z);
            glVertex3f((float)nbGrid* scale+pt.x,pt.y,(float)i * scale + pt.z);
        };
        glEnd();
    }
}
