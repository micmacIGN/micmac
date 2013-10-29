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
void glDrawUnitCircle(uchar dim, float cx, float cy, float r, int steps)
{
    float theta = 2.f * PI / float(steps);
    float c = cosf(theta);//precalculate the sine and cosine
    float s = sinf(theta);
    float t;

    float x = r;//we start at angle = 0
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
    glDrawUnitCircle(_dim, 0, 0, 1.f);

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

    GLuint trihedron = glGenLists(1);
    glNewList(trihedron, GL_COMPILE);

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

    glCallList(trihedron);

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
    glDisable(GL_BLEND);
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
