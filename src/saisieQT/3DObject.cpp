#include "3DObject.h"
#include "SaisieGlsl.glsl"

cObject::cObject() :
    _position(Pt3dr(0.f,0.f,0.f)),
    _color(QColor(255,255,255)),
    _scale(1.f),
    _alpha(0.6f),
    _bVisible(false)
{}

cObject::cObject(Pt3dr pos, QColor col) :
    _scale(1.f),
    _alpha(0.6f),
    _bVisible(true)
{
    _position  = pos;
    _color     = col;
}

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

cBall::cBall(Pt3dr pt, float scale, bool isVis, float lineWidth)
{
    _bVisible = isVis;

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
    if (_bVisible)
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

cAxis::cAxis(Pt3dr pt, float scale, float lineWidth)
{
    _position = pt;
    _scale    = scale;
    setLineWidth(lineWidth);
}

void cAxis::draw()
{
    if (_bVisible)
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

    setColor(QColor("orange"));
    setLineWidth(lineWidth);
}

void cBBox::set(Pt3dr min, Pt3dr max)
{
    _min = min;
    _max = max;
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

cCam::cCam(CamStenope *pCam, float scale, bool isVisible, float lineWidth) :
    _pointSize(5.f),
    _Cam(pCam)
{
    _scale = scale;
    _bVisible = isVisible;

    setColor(QColor("red"));
    setLineWidth(lineWidth);
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

cPolygon::cPolygon(float lineWidth, QColor lineColor, QColor pointColor, int style):
    _helper(new cPolygonHelper(this,lineWidth)),
    _lineColor(lineColor),
    _idx(-1),
    _pointSize(6.f),
    _sqr_radius(2500.f),
    _bPolyIsClosed(false),
    _bSelectedPoint(false),
    _style(style)
{
    setColor(pointColor);
    setLineWidth(lineWidth);
}

cPolygon::cPolygon(float lineWidth, QColor lineColor,  QColor pointColor, bool withHelper, int style):
    _lineColor(lineColor),
    _idx(-1),
    _pointSize(6.f),
    _sqr_radius(2500.f),
    _bPolyIsClosed(false),
    _bSelectedPoint(false),
    _style(style)
{
    if (!withHelper) _helper = NULL;
    setColor(pointColor);
    setLineWidth(lineWidth);
}

void cPolygon::draw()
{
    enableOptionLine();

    glColor3f(_lineColor.redF(),_lineColor.greenF(),_lineColor.blueF());

    if(_style == LINE_STIPPLE)
    {
        glLineStipple(2, 0xAAAA);
        glEnable(GL_LINE_STIPPLE);
    }

    //draw segments
    glBegin(_bPolyIsClosed ? GL_LINE_LOOP : GL_LINE_STRIP);
    for (int aK = 0;aK < _points.size(); ++aK)
        glVertex2f(_points[aK].x(), _points[aK].y());
    glEnd();

    if(_style == LINE_STIPPLE) glDisable(GL_LINE_STIPPLE);

    glColor3f(_color.redF(),_color.greenF(),_color.blueF());

    if ((_idx >=0) && (_points.size() > _idx))
    {
        //draw points
        for (int aK = 0;aK < _idx; ++aK)
            glDrawUnitCircle(2, _points[aK].x(), _points[aK].y());

        glColor3f(0.f,0.f,1.f);
        glDrawUnitCircle(2, _points[_idx].x(), _points[_idx].y());

        glColor3f(_color.redF(),_color.greenF(),_color.blueF());
        for (int aK = _idx+1;aK < _points.size(); ++aK)
            glDrawUnitCircle(2, _points[aK].x(), _points[aK].y());
    }
    else
    {
        for (int aK = 0;aK < _points.size(); ++aK)
            glDrawUnitCircle(2, _points[aK].x(), _points[aK].y());
    }
    disableOptionLine();
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

    _bSelectedPoint = false;
}

void cPolygon::removeNearestOrClose(QPointF pos)
{
    if ((_idx >=0)&&(_idx<size())&&_bPolyIsClosed)
    {
        removePoint(_idx);   // remove nearest point

        findNearestPoint(pos);

        if (size() < 3)
            setClosed(false);
    }
    else // close polygon
        close();
}

void cPolygon::addPoint(const QPointF &pt)
{
    if (size() >= 1)
        _points[size()-1] = pt;

    _points.push_back(pt);
}

void cPolygon::clear()
{
    _points.clear();
    _idx = -1;
    _bSelectedPoint = false;
}

void cPolygon::insertPoint(int i, const QPointF &value)
{
    _points.insert(i,value);
    _idx = -1;
}

void cPolygon::insertPoint()
{
    if ((size() >=2) && _helper->size()>1 && _bPolyIsClosed)
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

void cPolygon::findNearestPoint(QPointF const &pos)
{
    if (_bPolyIsClosed)
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
}

cPolygonHelper::cPolygonHelper(cPolygon* polygon,float lineWidth, QColor lineColor, QColor pointColor):
    cPolygon(lineWidth, lineColor, pointColor,false),
    _polygon(polygon)
{

}

void cPolygon::refreshHelper(QPointF pos, bool insertMode)
{
    int nbVertex = size();

    if(!isClosed())
    {
        if (nbVertex == 1)     // add current mouse position to polygon (dynamic display)
            add(pos);
        else if (nbVertex > 1) // replace last point by the current one
            _points[nbVertex-1] = pos;

    }
    else if(nbVertex)                       // move vertex or insert vertex (dynamic display) en court d'opération
    {
        if (insertMode || isPointSelected()) // insert polygon point

            _helper->build(pos, insertMode);

        else                                // select nearest polygon point

            findNearestPoint(pos);
    }
}


void cPolygon::finalMovePoint(QPointF pos)
{
    if ((_idx>=0) && _helper->size()) //  fin de deplacement point
    {
        _points[_idx] = (*_helper)[1];

        _helper->clear();

        resetSelectedPoint();
    }
}

void cPolygon::removeLastPoint()
{
    if (size() >= 1)
    {
        removePoint(size()-1);
        setClosed(false);
    }
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

void cPolygonHelper::build(QPointF const &pos, bool insertMode)
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
        int idx = _polygon->idx();

        if ((idx > 0) && (idx <= sz-1))
            setPoints((*_polygon)[(idx-1)%sz],pos,(*_polygon)[(idx+1)%sz]);
        else if (idx  == 0)
            setPoints((*_polygon)[sz-1],pos,(*_polygon)[1]);
    }
}

void cPolygonHelper::setPoints(QPointF p1,QPointF p2,QPointF p3)
{
    clear();
    add(p1);
    add(p2);
    add(p3);
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

void cImageGL::drawQuad()
{
    glBegin(GL_QUADS);
    {
        glTexCoord2f(0.0f, 0.0f);
        glVertex2f(_originX, _originY);
        glTexCoord2f(1.0f, 0.0f);
        glVertex2f(_originX+_glw, _originY);
        glTexCoord2f(1.0f, 1.0f);
        glVertex2f(_originX+_glw, _originY+_glh);
        glTexCoord2f(0.0f, 1.0f);
        glVertex2f(_originX, _originY+_glh);
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

    drawQuad();

    if(_gamma !=1.0f) _program.release();

    glBindTexture( GL_TEXTURE_2D, 0);
    glDisable(GL_TEXTURE_2D);
}

void cImageGL::draw(QColor color)
{
    glColor4f(color.redF(),color.greenF(),color.blueF(),color.alphaF());
    drawQuad();
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

cMaskedImageGL::cMaskedImageGL(cMaskedImage<QImage> &qMaskedImage)
{
    _m_mask     = new cImageGL();
    _m_image    = new cImageGL(qMaskedImage._gamma);
    _m_newMask  = qMaskedImage._m_newMask;
    _m_mask->PrepareTexture(qMaskedImage._m_mask);
    _m_image->PrepareTexture(qMaskedImage._m_image);
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
        int c =256;
        _m_mask->draw(QColor((float)c/2.0f,c/2,c/2));
        glBlendFunc(GL_DST_COLOR,GL_ZERO);
        glColor4f(1.0f,1.0f,1.0f,1.0f);
    }

    _m_image->draw();

    glDisable(GL_BLEND);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_ALPHA_TEST);
}

void cObjectGL::enableOptionLine()
{
    glDisable(GL_DEPTH_TEST);
    glEnable (GL_LINE_SMOOTH);
    glEnable (GL_BLEND);
    glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glHint (GL_LINE_SMOOTH_HINT, GL_DONT_CARE);
}

void cObjectGL::disableOptionLine()
{
    glDisable(GL_BLEND);
    glDisable(GL_LINE_SMOOTH);
    glEnable(GL_DEPTH_TEST);
}


