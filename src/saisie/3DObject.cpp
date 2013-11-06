#include "3DObject.h"

cObject::cObject() :
    _position(Pt3dr(0.f,0.f,0.f)),
    _color(QColor(255,255,255)),
    _scale(1.f),
    _alpha(0.6f),
    _bVisible(false)
{}

cObject::~cObject(){}

cObject& cObject::operator =(const cObject& aB)
{
    if (this != &aB)
    {
        _position = aB._position;
        _color    = aB._color;
        _scale    = aB._scale;

        _alpha    = aB._alpha;
        _bVisible = aB._bVisible;
    }

    return *this;
}

cCircle::cCircle(Pt3d<double> pt, QColor col, float scale, float lineWidth, bool vis, int dim) :
    _dim(dim)
{
    setPosition(pt);
    setColor(col);
    setScale(scale);
    setLineWidth(lineWidth);
    setVisible(vis);
}

//draw a unit circle in a given plane (0=YZ, 1 = XZ, 2=XY)
void glDrawUnitCircle(uchar dim, float cx, float cy, float r = 3.0, int steps = 8)
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

    glColor4f(_color.redF(),_color.greenF(),_color.blueF(),_alpha);
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
    setColor(col);
    setScale(scale);
    setLineWidth(lineWidth);
    setVisible(vis);
}

void cCross::draw()
{
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();

    GLuint list = glGenLists(1);
    glNewList(list, GL_COMPILE);

    glPushAttrib(GL_LINE_BIT | GL_DEPTH_BUFFER_BIT);

    glLineWidth(_lineWidth);

    glColor4f(_color.redF(),_color.greenF(),_color.blueF(),_alpha);

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

cBall::cBall(Pt3dr pt, float scale, float lineWidth, bool isVis)
{
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
    _cl0->draw();
    _cl1->draw();
    _cl2->draw();

    _cr0->draw();
    _cr1->draw();
    _cr2->draw();
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

void cBall::setColor(QColor const &aCol)
{
    _cl0->setColor(aCol);
    _cl1->setColor(aCol);
    _cl2->setColor(aCol);

    _cr0->setColor(aCol);
    _cr1->setColor(aCol);
    _cr2->setColor(aCol);
}

void cBall::setVisible(bool aVis)
{
    _bVisible = aVis;

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

cAxis::cAxis():
    _lineWidth(1.f)
{}

void cAxis::draw()
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

cBBox::cBBox() :
    _lineWidth(1.f)
{
    setColor(QColor("orange"));
}

void cBBox::set(float minX, float minY, float minZ, float maxX, float maxY, float maxZ)
{
    _minX = minX;
    _minY = minY;
    _minZ = minZ;
    _maxX = maxX;
    _maxY = maxY;
    _maxZ = maxZ;
}

void cBBox::draw()
{
    if (_bVisible)
    {
        glMatrixMode(GL_MODELVIEW);
        glPushMatrix();

        GLuint list = glGenLists(1);
        glNewList(list, GL_COMPILE);

        glPushAttrib(GL_LINE_BIT|GL_DEPTH_BUFFER_BIT);

        glLineWidth(_lineWidth);

        glColor3f(_color.redF(),_color.greenF(),_color.blueF());

        Pt3dr P1(_minX, _minY, _minZ);
        Pt3dr P2(_minX, _minY, _maxZ);
        Pt3dr P3(_minX, _maxY, _maxZ);
        Pt3dr P4(_minX, _maxY, _minZ);
        Pt3dr P5(_maxX, _minY, _minZ);
        Pt3dr P6(_maxX, _maxY, _minZ);
        Pt3dr P7(_maxX, _maxY, _maxZ);
        Pt3dr P8(_maxX, _minY, _maxZ);

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

cCam::cCam(CamStenope *pCam) :
    _lineWidth(1.f),
    _pointSize(5.f),
    _Cam(pCam)
{
    setColor(QColor("red"));
}

void cCam::draw()
{
    if (_bVisible)
    {
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
            glColor3f(1.f,1.f,1.f);
            glVertex3d(C.x, C.y, C.z);
            glVertex3d(P1.x, P1.y, P1.z);

            glVertex3d(C.x, C.y, C.z);
            glVertex3d(P2.x, P2.y, P2.z);

            glVertex3d(C.x, C.y, C.z);
            glVertex3d(P3.x, P3.y, P3.z);

            glVertex3d(C.x, C.y, C.z);
            glVertex3d(P4.x, P4.y, P4.z);

            //Image
            glColor3f(_color.redF(),_color.greenF(),_color.blueF());
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
    }
}

cPolygon::cPolygon():
    _lineWidth(1.f),
    _pointSize(6.f),
    _sqr_radius(2500.f),
    _bPolyIsClosed(false),
    _idx(-1)
{
    setColor(QColor("red"));
}

void cPolygon::draw()
{
    glColor3f(.1f,1.f,.2f);

    glBegin(_bPolyIsClosed ? GL_LINE_LOOP : GL_LINE_STRIP);
    for (int aK = 0;aK < _points.size(); ++aK)
    {
        glVertex2f(_points[aK].x(), _points[aK].y());
    }
    glEnd();

    glColor3f(_color.redF(),_color.greenF(),_color.blueF());

    if (_idx >=0)
    {
        for (int aK = 0;aK < _idx; ++aK)
            glDrawUnitCircle(2, _points[aK].x(), _points[aK].y());

        glColor3f(0.f,0.f,1.f);
        glDrawUnitCircle(2, _points[_idx].x(), _points[_idx].y());

        glColor3f(1.f,0.f,0.f);
        for (int aK = _idx+1;aK < _points.size(); ++aK)
            glDrawUnitCircle(2, _points[aK].x(), _points[aK].y());
    }
    else
    {
        for (int aK = 0;aK < _points.size(); ++aK)
        {
            glDrawUnitCircle(2, _points[aK].x(), _points[aK].y());
        }
    }
}

cPolygon & cPolygon::operator = (const cPolygon &aP)
{
    if (this != &aP)
    {
        _lineWidth        = aP._lineWidth;
        _pointSize        = aP._pointSize;
        _bPolyIsClosed    = aP._bPolyIsClosed;
        _idx              = aP._idx;

        _points           = aP._points;

        _position         = aP._position;
        _color            = aP._color;
        _scale            = aP._scale;

        _alpha            = aP._alpha;
        _bVisible         = aP._bVisible;
    }

    return *this;
}

void cPolygon::close()
{
    int sz = _points.size();

    if ((sz>1)&&(!_bPolyIsClosed))
    {
        //remove last point if needed
        if (sz > 2) _points.resize(sz-1);

        _bPolyIsClosed = true;
    }
    else
        _bPolyIsClosed = false;

}

void cPolygon::findClosestPoint(QPointF const &pos)
{
    _idx = -1;
    float dist, dist2, x, y, dx, dy;
    dist2 = _sqr_radius;
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
}

void cPolygon::drawDihedron()
{
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();

    glLineStipple(2, 0xAAAA);
    glEnable(GL_LINE_STIPPLE);

    glEnable (GL_BLEND);
    glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glHint (GL_LINE_SMOOTH_HINT, GL_DONT_CARE);

    glColor3f(0.f,0.f,1.f);

    glLineWidth(_lineWidth);

    glBegin(GL_LINE_STRIP);
    for (int aK=0;aK < _points.size(); ++aK)
    {
        glVertex2f(_points[aK].x(), _points[aK].y());
    }
    glEnd();

    glDisable(GL_LINE_STIPPLE);

    if (_points.size())
        glDrawUnitCircle(2, _points[1].x(), _points[1].y());

    glPopMatrix();
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

void cPolygon::fillDihedron(QPointF const &pos, cPolygon &dihedron)
{
    float dist, dist2;
    dist2 = FLT_MAX;
    int idx = -1;

    QVector < QPointF > polygon = _points;
    polygon.push_back(polygon[0]);

    for (int aK =0; aK < polygon.size()-1; ++aK)
    {
        dist = segmentDistToPoint(polygon[aK], polygon[aK+1], pos);

        if (dist < dist2)
        {
            dist2 = dist;
            idx = aK;
        }
    }

    if (idx != -1)
    {
       dihedron.clear();
       dihedron.add(polygon[idx]);
       dihedron.add(pos);
       dihedron.add(polygon[idx+1]);
    }
}

void cPolygon::fillDihedron2(QPointF const &pos, cPolygon &dihedron)
{
    dihedron.clear();
    int sz = _points.size();

    if ((_idx >0 ) && (_idx < sz-1))
    {
        dihedron.add(_points[_idx-1]);
        dihedron.add(pos);
        dihedron.add(_points[_idx+1]);
    }
    else
    {
        if (_idx == 0)
        {
            dihedron.add(_points[sz-1]);
            dihedron.add(pos);
            dihedron.add(_points[1]);
        }
        if (_idx == sz-1)
        {
            dihedron.add(_points[sz-2]);
            dihedron.add(pos);
            dihedron.add(_points[0]);
        }
    }
}


