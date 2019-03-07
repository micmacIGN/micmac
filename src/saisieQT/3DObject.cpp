#include "3DObject.h"
#include "SaisieGlsl.glsl"
#include <limits>

#ifdef USE_MIPMAP_HANDLER
	#include "StdAfx.h"
	#include "../src/uti_image/Digeo/MultiChannel.h"
#endif

//~ #define WRITE_LOADED_TEXTURE

#ifdef WRITE_LOADED_TEXTURE
	#define WRITE_SOURCE_IMAGE

	int gNbLoadedTextures = 0;
#endif

cObject::cObject() :
	#ifndef USE_MIPMAP_HANDLER
		_name(""),
	#endif
    _position(QVector3D(0.f,0.f,0.f)),
    _rotation(QVector3D(0.f,0.f,0.f)),
    _scale(QVector3D(1.f, 1.f,1.f)),
    _alpha(0.6f),
    _state(state_default),
    _parent(NULL)
{
    for (int iC = 0; iC < state_COUNT; ++iC)
        _color[iC] = QColor(255,255,255);
}

cObject::cObject(QVector3D pos, QColor color_default) :
	#ifndef USE_MIPMAP_HANDLER
		_name(""),
	#endif
    _position(pos),
    _rotation(QVector3D(0.f,0.f,0.f)),
    _scale(QVector3D(1.f, 1.f,1.f)),
    _alpha(0.6f),
    _state(state_default),
    _parent(NULL)
{
    for (int iC = 0; iC < state_COUNT; ++iC)
        _color[iC] = color_default;
}

cObject::~cObject()
{
//	for (int i = 0; i < _children.size(); ++i)
//	{
//		if(_children[i])
//			_children[i]->setParent(NULL);
//	}

}

const QColor & cObject::getColor() const
{
    return _color[state()];
}

bool cObject::isVisible()     { return (state() != state_invible); }

void cObject::setPosition(const QVector3D& aPt)  { _position = aPt;  }

cObject& cObject::operator =(const cObject& aB)
{
    if (this != &aB)
    {
        _name      = aB._name;
        _position  = aB._position;
        _rotation  = aB._rotation;

        for (int iC = 0; iC < state_COUNT; ++iC)
            _color[iC]     = aB._color[iC];

        _alpha     = aB._alpha;
        _state     = aB._state;

        _scale     = aB._scale;
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

cObject*cObject::child(int id)
{
    if(id >= 0 && id< _children.size())
        return _children[id];
    else
        return NULL;
}

void cObject::addChild(cObject* child)
{
    _children.push_back(child);
}

void cObject::removeChild(cObject* child)
{
    int id = _children.indexOf(child);

    if(id<0)
        return;
    else
        _children.remove(id);

}

void cObject::replaceChild(int id, cObject* child)
{
    if(id >= 0 && id< _children.size())
    {
        _children[id] = child;
        child->setParent(this);
    }
}
cObject* cObject::parent() const
{
    return _parent;
}

void cObject::setParent(cObject* parent)
{
//	if(_parent)
//		_parent->removeChild(this);

    _parent = parent;

    if(_parent)
    {
        _parent->addChild(this);
        _position = _parent->getPosition();
    }
}


cCircle::cCircle(QVector3D pt, QColor col, float scale, float lineWidth, bool vis, int dim) :
    _dim(dim)
{
    setPosition(pt);
    cObject::setColor(col);
    setScale(QVector3D(scale,scale,scale));
    setLineWidth(lineWidth);
    cObject::setVisible(vis);
}

//draw a unit circle in a given plane (0=YZ, 1=XZ, 2=XY)
void glDrawUnitCircle(uchar dim, float cx, float cy, float r, int steps)
{
    float theta = M_2PI / float(steps);
    float c = cosf(theta); //precalculate the sine and cosine
    float s = sinf(theta);
//    float t;

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
        float t = x;
        x = c * x - s * y;
        y = s * t + c * y;
    }
    glEnd();
}

void glDrawEllipse(float cx, float cy, float rx, float ry, int steps) // TODO step Auto....
{
    const float theta = M_2PI / float(steps);

    glBegin(GL_LINE_LOOP);
    for(float t = 0.f; t <= M_2PI; t+= theta)
    {

        const float x = cx + rx*sinf(t);
        const float y = cy + ry*cosf(t);

        glVertex3f(x,y,0.f);
    }
    glEnd();
}

void glDrawEllipsed(double cx, double cy, double rx, double ry, int steps) // TODO step Auto....
{
    const double theta = M_2PI / double(steps);

    glBegin(GL_LINE_LOOP);
    for(double t = 0.f; t <= M_2PI; t+= theta)
    {
        const float x = cx + rx*std::sin(t);
        const float y = cy + ry*std::cos(t);

        glVertex3d(x,y,0.f);
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

    glDrawUnitCircle(_dim);

    glPopAttrib();

    glEndList();

    glTranslatef(_position.x(),_position.y(),_position.z());
    glScalef(_scale.x(),_scale.y(),_scale.z());

    glCallList(list);

    glPopMatrix();
}

cCross::cCross(QVector3D pt, QColor col, float scale, float lineWidth, bool vis, int dim) :
    _dim(dim)
{
    setPosition(pt);
    cObject::setColor(col);
    setScale(QVector3D(scale, scale, scale));
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

    glTranslatef(_position.x(),_position.y(),_position.z());
    glScalef(_scale.x(),_scale.y(),_scale.z());

    glCallList(list);

    glPopMatrix();
}

cBall::cBall(QVector3D pt, float scale, bool isVis, float lineWidth)
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

void cBall::setPosition(QVector3D const &aPt)
{
    _cl0->setPosition(aPt);
    _cl1->setPosition(aPt);
    _cl2->setPosition(aPt);

    _cr0->setPosition(aPt);
    _cr1->setPosition(aPt);
    _cr2->setPosition(aPt);
}

QVector3D cBall::getPosition()
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
    QVector3D pScale(aScale,aScale,aScale);
    setScale(pScale);

}

void cBall::setScale(QVector3D aScale)
{
    if(_cl0 && _cl1 && _cl2 && _cr0 && _cr1 && _cr2)
    {
        _cl0->setScale(aScale);
        _cl1->setScale(aScale);
        _cl2->setScale(aScale);

        _cr0->setScale(aScale);
        _cr1->setScale(aScale);
        _cr2->setScale(aScale);
    }
}

cAxis::cAxis(QVector3D pt, float scale, float lineWidth)
{
    _position = pt;
    _scale    = QVector3D(scale, scale, scale);
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

        glTranslatef(_position.x(),_position.y(),_position.z());
        glScalef(_scale.x(),_scale.y(),_scale.z());

        glCallList(dihedron);

        glPopMatrix();
    }
}

cBBox::cBBox(QVector3D pt, QVector3D min, QVector3D max, float lineWidth)
{
    _position = pt;
    _min = min;
    _max = max;

    cObject::setColor(QColor("orange"));
    setLineWidth(lineWidth);
}

void cBBox::set(QVector3D min, QVector3D max)
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

        QVector3D P1(_min);
        QVector3D P2(_min.x(), _min.y(), _max.z());
        QVector3D P3(_min.x(), _max.y(), _max.z());
        QVector3D P4(_min.x(), _max.y(), _min.z());
        QVector3D P5(_max.x(), _min.y(), _min.z());
        QVector3D P6(_max.x(), _max.y(), _min.z());
        QVector3D P7(_max);
        QVector3D P8(_max.x(), _min.y(), _max.z());

        glBegin(GL_LINES);

        glVertex3d(P1.x(), P1.y(), P1.z());
        glVertex3d(P2.x(), P2.y(), P2.z());

        glVertex3d(P3.x(), P3.y(), P3.z());
        glVertex3d(P2.x(), P2.y(), P2.z());

        glVertex3d(P1.x(), P1.y(), P1.z());
        glVertex3d(P4.x(), P4.y(), P4.z());

        glVertex3d(P1.x(), P1.y(), P1.z());
        glVertex3d(P5.x(), P5.y(), P5.z());

        glVertex3d(P7.x(), P7.y(), P7.z());
        glVertex3d(P3.x(), P3.y(), P3.z());

        glVertex3d(P7.x(), P7.y(), P7.z());
        glVertex3d(P6.x(), P6.y(), P6.z());

        glVertex3d(P8.x(), P8.y(), P8.z());
        glVertex3d(P5.x(), P5.y(), P5.z());

        glVertex3d(P7.x(), P7.y(), P7.z());
        glVertex3d(P8.x(), P8.y(), P8.z());

        glVertex3d(P5.x(), P5.y(), P5.z());
        glVertex3d(P6.x(), P6.y(), P6.z());

        glVertex3d(P4.x(), P4.y(), P4.z());
        glVertex3d(P6.x(), P6.y(), P6.z());

        glVertex3d(P8.x(), P8.y(), P8.z());
        glVertex3d(P2.x(), P2.y(), P2.z());

        glVertex3d(P4.x(), P4.y(), P4.z());
        glVertex3d(P3.x(), P3.y(), P3.z());

        glEnd();

        glPopAttrib();

        glEndList();

        glCallList(list);

        glPopMatrix();
    }
}

cCamGL::cCamGL(cCamHandler *pCam, float scale,  object_state state, float lineWidth) :
    _pointSize(5.f),
    _Cam(pCam)
{
    _scale = QVector3D(scale, scale, scale);

    setState(state);
    cObject::setColor(QColor("red"));
    cObject::setColor(QColor(0.f,0.f,1.f),state_selected);
    setLineWidth(lineWidth);
}

void cCamGL::draw()
{
    if (isVisible())
    {

        GLfloat oldPointSize;
        glGetFloatv(GL_POINT_SIZE,&oldPointSize);

//        glMatrixMode(GL_MODELVIEW);
//        glPushMatrix();

        GLuint list = glGenLists(1);
        glNewList(list, GL_COMPILE);

        glPushAttrib(GL_LINE_BIT|GL_DEPTH_BUFFER_BIT);

        glLineWidth(_lineWidth);

        glPointSize(_pointSize);

        QVector3D C  = _Cam->getCenter();
        QVector3D P1, P2, P3, P4;
        _Cam->getCoins(P1, P2, P3, P4, _scale.z()*.05f);

        glBegin(GL_LINES);
        //perspective cone
        if(!isSelected())
            glColor3f(1.f,1.f,1.f);
        else
        {
            QColor color = Qt::yellow;
            glColor3f(color.redF(),color.greenF(),color.blueF());
        }

        glVertex3d(C.x(), C.y(), C.z());
        glVertex3d(P1.x(), P1.y(), P1.z());

        glVertex3d(C.x(), C.y(), C.z());
        glVertex3d(P2.x(), P2.y(), P2.z());

        glVertex3d(C.x(), C.y(), C.z());
        glVertex3d(P3.x(), P3.y(), P3.z());

        glVertex3d(C.x(), C.y(), C.z());
        glVertex3d(P4.x(), P4.y(), P4.z());

        //Image

        setGLColor();

        glVertex3d(P1.x(), P1.y(), P1.z());
        glVertex3d(P2.x(), P2.y(), P2.z());

        glVertex3d(P4.x(), P4.y(), P4.z());
        glVertex3d(P2.x(), P2.y(), P2.z());

        glVertex3d(P3.x(), P3.y(), P3.z());
        glVertex3d(P1.x(), P1.y(), P1.z());

        glVertex3d(P4.x(), P4.y(), P4.z());
        glVertex3d(P3.x(), P3.y(), P3.z());
        glEnd();

        glBegin(GL_POINTS);
        glVertex3d(C.x(), C.y(), C.z());
        glEnd();

        glPopAttrib();
        glEndList();

        glCallList(list);

 //       glPopMatrix();
        glPointSize(oldPointSize);
    }
}

cPoint::cPoint(QPointF pos,
               QString name, bool showName,
               int state,
               int geometry,
               bool isSelected,
               QColor color, QColor selectionColor,
               float diameter,
               bool highlight,
               bool drawCenter):
    QPointF(pos),
    _diameter(diameter),
    _bShowName(showName),
    _pointState(state),
    _pointGeometry(geometry),
    _highlight(highlight),
    _drawCenter(drawCenter),
    _bEpipolar(false)
{
    setName(name);
    cObject::setColor(color);
    cObject::setColor(selectionColor,state_selected);
    setSelected(isSelected);
}

QColor cPoint::colorPointState()
{

    QColor color = getColor();

    if (!isSelected())
    {
        if(_parent)
            color = Qt::red;
        else
            switch(_pointState)
            {
                case qEPI_NonSaisi ://
                    color = Qt::yellow;
                    break;

                case qEPI_Refute ://
                    color = Qt::red;
                    break;

                case qEPI_Douteux ://
                    color = QColor(255, 127, 0, 255);
                    break;

                case  qEPI_Valide://
                    color = Qt::green;
                    break;

                case  qEPI_Disparu://
                case  qEPI_NonValue://
                    break;
            }
    }

    return color;
}

void cPoint::setParent(cObject* parent)
{
    cObject::setParent(parent);
    if(parent)
    {
        cPoint* pointParent = (cPoint*)parent;
        setX(pointParent->x());
        setY(pointParent->y());
    }
}

void cPoint::draw()
{
    if (isVisible())
    {

        QColor color = colorPointState();

        glColor4f(color.redF(),color.greenF(),color.blueF(),_alpha);

        GLdouble    mvMatrix[16];
        GLdouble    projMatrix[16];
        GLint       glViewport[4];

        glGetIntegerv(GL_VIEWPORT, glViewport);
        glMatrixMode(GL_PROJECTION);
        glGetDoublev (GL_PROJECTION_MATRIX, projMatrix);
        glPushMatrix();
        glLoadIdentity();
        glTranslatef(-1.f,-1.f,0.f);
        glScalef(2.f/(float)glViewport[2],2.f/(float)glViewport[3],1.f);
        glMatrixMode(GL_MODELVIEW);
        glGetDoublev(GL_MODELVIEW_MATRIX, mvMatrix);
        glPushMatrix();
        glLoadIdentity();

        GLdouble xp,yp,zp;

        mmProject(x(),y(),0,mvMatrix,projMatrix,glViewport,&xp,&yp,&zp);

        //TODO: a deplacer
        if (_highlight && ((_pointState == qEPI_Valide) || (_pointState == qEPI_NonSaisi)))
        {
            if (_bEpipolar)
                _pointGeometry = Geom_epipolar;
            else
                _pointGeometry = Geom_double_circle;
        }

        switch(_pointGeometry)
        {
        case Geom_simple_circle:
            glDrawEllipsed(xp, yp, _diameter, _diameter,16);
            break;
        case Geom_double_circle:
            glDrawEllipsed(xp, yp, _diameter, _diameter,16);
            glDrawEllipse( xp, yp, 2.f*_diameter, 2.f*_diameter);
            break;
        case Geom_epipolar:
            glDrawEllipsed(xp, yp, _diameter, _diameter,16);
            GLdouble x1,y1,z1,x2,y2,z2;

            mmProject(_epipolar1.x(), _epipolar1.y(),0,mvMatrix,projMatrix,glViewport,&x1,&y1,&z1);
            mmProject(_epipolar2.x(), _epipolar2.y(),0,mvMatrix,projMatrix,glViewport,&x2,&y2,&z2);

            glBegin(GL_LINES);
                glVertex2f(x1,y1);
                glVertex2f(x2,y2);
            glEnd();
            break;
        case Geom_cross:
            glBegin(GL_LINES);
                glVertex2f(xp+_diameter,yp);
                glVertex2f(xp-_diameter,yp);
            glEnd();
            glBegin(GL_LINES);
                glVertex2f(xp,yp+_diameter);
                glVertex2f(xp,yp-_diameter);
            glEnd();
            break;
        case no_geometry:
            break;
        }

        if (_drawCenter)
        {
            glBegin(GL_POINTS);
                glVertex2d(xp, yp);
            glEnd( );
        }

        glMatrixMode(GL_PROJECTION);
        glPopMatrix();
        glMatrixMode(GL_MODELVIEW);
        glPopMatrix();
    }
}

void cPoint::setPosition(QPointF pos)
{
    if(!parent())
    {
        setX(pos.x());
        setY(pos.y());
    }
}

void cPoint::setEpipolar(QPointF pt1, QPointF pt2)
{
    _epipolar1 = pt1;
    _epipolar2 = pt2;
    _bEpipolar = true;
}

void cPoint::glDraw()
{
    glVertex2f(x(),y());
}

//********************************************************************************

float cPolygon::_selectionRadius = 10.f;

cPolygon::cPolygon(int maxSz, float lineWidth, QColor lineColor, QColor pointColor, int geometry, int style):
    _helper(new cPolygonHelper(this, 3, lineWidth)),
    _lineColor(lineColor),
    _idx(-1),
    _style(style),
    _pointDiameter(1.f),
    _pointGeometry(geometry),
    _bIsClosed(false),
    _bSelectedPoint(false),
    _bShowLines(true),
    _bShowNames(true),
    _maxSz(maxSz)
{
    setColor(pointColor);
    setLineWidth(lineWidth);
}

cPolygon::~cPolygon()
{
    if(_helper)
    {
        delete _helper;
    }
}

cPolygon::cPolygon(int maxSz, float lineWidth, QColor lineColor,  QColor pointColor, bool withHelper, int geometry, int style):
	#ifdef USE_MIPMAP_HANDLER
		_helper(NULL),
	#endif
    _lineColor(lineColor),
    _idx(-1),
    _style(style),
    _pointDiameter(1.f),
    _pointGeometry(geometry),
    _bIsClosed(false),
    _bSelectedPoint(false),
    _bShowLines(true),
    _bShowNames(true),
    _maxSz(maxSz)
{
    if (!withHelper) _helper = NULL;

    setColor(pointColor);
    setLineWidth(lineWidth);

    _dashes << 3 << 4;
}

void cPolygon::draw()
{

    for (int aK=0; aK < size();++aK)
        point(aK).draw();


    if(isVisible())
    {
        enableOptionLine();

        if (_bShowLines)
        {
            QColor color(isSelected() ? QColor(0,140,180) : _lineColor);

            glColor3f(color.redF(),color.greenF(),color.blueF());
            glLineWidth(_lineWidth);

            if(_style == LINE_STIPPLE)
            {
                glLineStipple(2, 0xAAAA);
                glEnable(GL_LINE_STIPPLE);
            }
            //draw segments
            glBegin(_bIsClosed ? GL_LINE_LOOP : GL_LINE_STRIP);
            for (int aK = 0;aK < size(); ++aK)
                point(aK).glDraw();
            glEnd();

            if(_style == LINE_STIPPLE) glDisable(GL_LINE_STIPPLE);

            glColor3f(_color[state_default].redF(),_color[state_default].greenF(),_color[state_default].blueF());
        }

        disableOptionLine();
    }


    if(helper() != NULL)
    {
        helper()->draw();
    }
}

cPolygon & cPolygon::operator = (const cPolygon &aP)
{
    if (this != &aP)
    {
        _lineWidth        = aP._lineWidth;
        _bIsClosed        = aP._bIsClosed;
        _idx              = aP._idx;

        _points           = aP._points;
        _pointDiameter    = aP._pointDiameter;
        _pointGeometry    = aP._pointGeometry;
        _selectionRadius  = aP._selectionRadius;

        _bSelectedPoint   = aP._bSelectedPoint;
        _bShowLines       = aP._bShowLines;
        _bShowNames       = aP._bShowNames;

        _dashes.clear();
        for (int iC = 0; iC < aP._dashes.size(); ++iC)
            _dashes.push_back(aP._dashes[iC]);

        _style            = aP._style;
        _defPtName        = aP._defPtName;

        _shiftStep        = aP._shiftStep;
        _maxSz            = aP._maxSz;

        _bNormalize       = aP._bNormalize;
    }

    return *this;
}

void cPolygon::RemoveLastPointAndClose()
{
    int sz = size();

    if ((sz>1)&&(!_bIsClosed))
    {
        //remove last point if needed
        if (sz > 2) _points.resize(sz-1);

        _bIsClosed = true;
    }

    _bSelectedPoint = false;
}

void cPolygon::close()
{
    _bIsClosed = true;
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
        RemoveLastPointAndClose();
}

void cPolygon::removeSelectedPoint()
{
    if (pointValid())

        removePoint(_idx);
}

QString cPolygon::getSelectedPointName()
{
    if (pointValid())
    {
        return point(_idx).name();
    }
    else return _defPtName;
}

int cPolygon::getSelectedPointState()
{
    if (pointValid())
    {
        return point(_idx).pointState();
    }
    else return qEPI_NonValue;
}

int cPolygon::getSelectedPointGeometry()
{
    if (pointValid())
    {
        return point(_idx).pointGeometry();
    }
    else return no_geometry;
}

void cPolygon::add(cPoint &pt)
{
    if (size() <= _maxSz)
    {
        pt.setDiameter(_pointDiameter);
        _points.push_back(pt);
    }

    if(size() > _maxSz)
        RemoveLastPointAndClose();
}

// TODO pourquoi les fonctions : 2 add et addPoint?
void cPolygon::add(const QPointF &pt, bool selected, cPoint* lock )
{
    if (size() <= _maxSz)
    {
        cPoint cPt( pt, _defPtName, _bShowNames, qEPI_NonValue, _pointGeometry, selected, _color[state_default],Qt::blue,_pointDiameter);

        cPt.setParent(lock);
        cPt.drawCenter(!isLinear());

        _points.push_back(cPt);
    }
    if(size() > _maxSz)
        RemoveLastPointAndClose();
}

void cPolygon::addPoint(const QPointF &pt, cPoint* lock)
{
    if (size() >= 1 && size() <= _maxSz)
    {
        cPoint cPt( pt, _defPtName, _bShowNames, qEPI_NonValue, _pointGeometry, false, _color[state_default]);
        cPt.setDiameter(_pointDiameter);
        cPt.setParent(lock);
        cPt.drawCenter(!isLinear());

        point(size()-1) = cPt;
    }

    add(pt,false,lock);
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
    if (i <= size()&& size() < _maxSz)
    {
        cPoint pt(value);
        pt.setDiameter(point(i-1).diameter());
        _points.insert(i, pt);
        resetSelectedPoint();
    }
}

void cPolygon::insertPoint()
{
    if ((size() >=2) && _helper->size()>1 && _bIsClosed && size() < _maxSz)
    {
        int idx = -1;
        QPointF Pt1 = (*_helper)[0];
        QPointF Pt2 = (*_helper)[1];

        for (int i=0;i<size();++i)
        {
            if (point(i) == Pt1) idx = i;
        }

        if (idx >=0) insertPoint(idx+1, Pt2);
    }

    _helper->clear();
}

void cPolygon::removePoint(int i)
{
    if(_points[i].nbChild() == 1)
    {
        _points[i].child(0)->setParent(NULL);
    }

    _points.remove(i);
    _idx = -1;
}

const QVector<QPointF> cPolygon::getVector()
{
    QVector <QPointF> points;

    for(int aK=0; aK < size(); ++aK)

        points.push_back(point(aK));


    return points;
}

const QVector<QPointF> cPolygon::getImgCoordVector(const cMaskedImageGL &img)
{
	ELISE_DEBUG_ERROR(img._m_image == NULL, "cPolygon::getImgCoordVector", "img._m_image == NULL");

    float nImgWidth, nImgHeight;
    float imgHeight = (float) img._m_image->height();
    if (_bNormalize)
    {
        nImgWidth = (float) img._m_image->width();
        nImgHeight = imgHeight;
    }
    else
    {
        nImgWidth = nImgHeight = 1.f;
    }

    QVector <QPointF> points;

    for(int aK=0; aK < size(); ++aK)
    {
        points.push_back( QPointF(point(aK).x()/nImgWidth, (imgHeight - point(aK).y())/nImgHeight));
    }

    return points;
}

void cPolygon::setVector(const QVector<QPointF> &aPts)
{
    _points.clear();
    for(int aK=0; aK < aPts.size(); ++aK)
    {
        _points.push_back(cPoint(aPts[aK]));
    }
}

void cPolygon::setHelper(cPolygonHelper* aHelper) {

    _helper = aHelper;
}

void cPolygon::setPointSelected()
{
    _bSelectedPoint = true;


    if (pointValid())
    {
        point(_idx).setSelected(true);
    }
}

void cPolygon::resetSelectedPoint()
{

    // TODO virer _bSelectedPoint
    _bSelectedPoint = false;

    if (pointValid())
        point(_idx).setSelected(false);

    _idx = -1;
}

bool cPolygon::pointValid()
{
    return ((_idx >=0) && (_idx < size()));
}

void cPolygon::setAllVisible(bool visible)
{
    setVisible(visible);
    for (int i = 0; i < size(); ++i)
    {
        point(i).setVisible(visible);
    }

}

float cPolygon::length()
{
    if(size() == 2 && helper()->size() > 0)
    {
        QLineF line(point(_idx == 0 ? 1 : 0),helper()->point(1));

        return line.length();
    }
    else  if(size() == 2)
    {
        QLineF line(point(0),point(1));

        return line.length();
    }

    else return 0.0;
}

int cPolygon::selectPoint(QString namePt)
{
    resetSelectedPoint();

    for (int i = 0; i < size(); ++i)
    {
        if(point(i).name() == namePt)
        {
            _idx = i;
            point(i).setSelected(true);
            return i;
        }
    }

    return _idx;
}

void cPolygon::selectPoint(int idx)
{
    _idx = idx;

    if (pointValid())
    {
        point(idx).setSelected(true);
        _bSelectedPoint = true;
    }
}

void cPolygon::setPointSize(float sz)
{
    _pointDiameter = sz;

    for (int i = 0; i < size(); ++i)
    {
        point(i).setDiameter(sz);
    }

    if (helper() != NULL) helper()->setPointSize(sz);
}

cPoint* cPolygon::findNearestPoint(QPointF const &pos, float radius)
{
    if (_bIsClosed || _bShowLines)
    {
        resetSelectedPoint();

        float dist2, x, y;
        dist2 = radius*radius;
        x = pos.x();
        y = pos.y();

        for (int aK = 0; aK < size(); ++aK)
        {
            const float dx = x - point(aK).x();
            const float dy = y - point(aK).y();

            const float dist = dx * dx + dy * dy;

            if  (dist < dist2)
            {
                dist2 = dist;
                _idx = aK;
            }
        }

        if (pointValid())
        {
            point(_idx).setSelected(true);

            return &point(_idx);
        }
    }

    return NULL;
}

void cPolygon::refreshHelper(QPointF pos, bool insertMode, float zoom, bool ptIsVisible, cPoint* lock)
{
    int nbVertex = size();

    if(!_bIsClosed)
    {
        if (nbVertex == 1)                  // add current mouse position to polygon (for dynamic display)
        {
            add(pos,false,lock);
        }
        else if (nbVertex > 1)               // replace last point by the current one
        {
            point(nbVertex-1).setPosition(pos);
            point(nbVertex-1).setParent(lock);
        }
    }
    else if(nbVertex)                        // move vertex or insert vertex (dynamic display) en cours d'operation
    {
        if ( insertMode || isPointSelected()) // insert or move polygon point
        {
            cPoint pt( pos, getSelectedPointName(), _bShowNames, getSelectedPointState(), getSelectedPointGeometry(), isPointSelected(), _color[state_default]); // TODO add diameter parameter
            pt.setDiameter(_pointDiameter);
            pt.setParent(lock);

            if (!ptIsVisible) pt.setVisible(false);

            _helper->build(pt, size() == _maxSz ? false : insertMode);
        }
        else                                 // select nearest polygon point
        {
            findNearestPoint(pos, _selectionRadius / zoom);
        }
    }
}

int cPolygon::finalMovePoint(cPoint* lock)
{
    int idx = _idx;

    if ((_idx>=0) && (_helper != NULL) && _helper->size())   // after point move
    {
        int state = point(_idx).pointState();

        point(_idx) = (*_helper)[1];
        point(idx).setParent(lock);
        point(_idx).setColor(_color[state_default]); // reset color to polygon color
        point(_idx).setPointState(state);

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

void cPolygon::showNames(bool show)
{
    _bShowNames = show;

    for (int aK=0; aK < size(); ++aK)
        point(aK).showName(_bShowNames);
}

void cPolygon::rename(QPointF pos, QString name)
{
    findNearestPoint(pos, 400000.f);

    if (pointValid())
        point(_idx).setName(name);
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
    for (int aK=0; aK < size(); ++aK)
        point(aK) += Tr;
}

cPoint cPolygon::translateSelectedPoint(QPointF Tr)
{
    if (pointValid())
    {
        point(_idx) += Tr;
        return point(_idx);
    }
    else
        return ErrPoint;
}

void cPolygon::setParams(cParameters *aParams)
{
    setRadius(aParams->getSelectionRadius());
    setPointSize(aParams->getPointDiameter());
    setLineWidth(aParams->getLineThickness());
    setShiftStep(aParams->getShiftStep());

    if (_helper != NULL)
    {
        _helper->setRadius(aParams->getSelectionRadius());
        _helper->setPointSize(aParams->getPointDiameter());
        _helper->setLineWidth(aParams->getLineThickness());
    }
}

bool cPolygon::isPointInsidePoly(const QPointF& P)
{
    int vertices=size();
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

//********************************************************************************

cPolygonHelper::cPolygonHelper(cPolygon* polygon, int maxSz, float lineWidth, QColor lineColor, QColor pointColor, int pointGeometry):
    cPolygon(maxSz, lineWidth, lineColor, pointColor, false, pointGeometry),
    _polygon(polygon)
{
}

cPolygonHelper::~cPolygonHelper()
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
        float dist2 = std::numeric_limits<float>::max();
        int idx = -1;
        for (int aK =0; aK < sz; ++aK)
        {
            const float dist = segmentDistToPoint((*_polygon)[aK], (*_polygon)[(aK + 1)%sz], pos);

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
            int idx = _polygon->getSelectedPointIndex();

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

cRectangle::cRectangle(int maxSz, float lineWidth, QColor lineColor, int style) :
    cPolygon(maxSz, lineWidth, lineColor, Qt::red, style)
{}

void cRectangle::addPoint(const QPointF &pt, cPoint* lock) // TODO add lock effect....
{
    if (size() == 0)
    {
        for (int aK=0; aK < getMaxSize(); aK++)
            add(pt);

        selectPoint(2); //
    }
}

void cRectangle::refreshHelper(QPointF pos, bool insertMode, float zoom, bool ptIsVisible, cPoint* lock)
{
    if (size())
    {
        if(isPointSelected())
        {
            showLines(true);
            setClosed(true);

            point(_idx).setX(pos.x());
            point(_idx).setY(pos.y());

            if (_idx == 2)
            {
                _points[1].setX(pos.x());

                _points[3].setY(pos.y());
            }
            else if(_idx == 1)
            {
                _points[0].setY(pos.y());

                _points[2].setX(pos.x());
            }
            else if(_idx == 0)
            {
                _points[3].setX(pos.x());

                _points[1].setY(pos.y());
            }
            else if(_idx == 3)
            {
                _points[2].setY(pos.y());

                _points[0].setX(pos.x());
            }
        }
        else
        {
            cPolygon::refreshHelper(pos, false, zoom, false);
        }
    }
}

void cRectangle::draw()
{
    for (int aK= 0; aK < size(); ++aK)
        _points[aK].setVisible(false);

    cPolygon::draw();
}

//********************************************************************************

//invalid GL list index
const GLuint GL_INVALID_LIST_ID = (~0);

cImageGL::cImageGL(float gamma) :
    _texture(GL_INVALID_LIST_ID),
    _gamma(gamma)
{
    setPosition(QVector3D(0,0,0));

    _program.addShaderFromSourceCode(QGLShader::Vertex,vertexShader);
    _program.addShaderFromSourceCode(QGLShader::Fragment,fragmentGamma);
    _program.link();

    _texLocation   = _program.uniformLocation("tex");
    _gammaLocation = _program.uniformLocation("gamma");
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
    drawQuad(getPosition().x(), getPosition().y(), width(), height(), color);
}

void cImageGL::drawQuad(GLfloat originX, GLfloat originY, GLfloat glw,  GLfloat glh, QColor color)
{
    glColor4f(color.redF(),color.greenF(),color.blueF(),color.alphaF());
    glBegin(GL_QUADS);
    {
		#ifdef USE_MIPMAP_HANDLER
			glTexCoord2f(0.0f, 1.0f);
			glVertex2f(originX, originY);
			glTexCoord2f(1.0f, 1.0f);
			glVertex2f(originX+glw, originY);
			glTexCoord2f(1.0f, 0.0f);
			glVertex2f(originX+glw, originY+glh);
			glTexCoord2f(0.0f, 0.0f);
			glVertex2f(originX, originY+glh);
		#else
			glTexCoord2f(0.0f, 0.0f);
			glVertex2f(originX, originY);
			glTexCoord2f(1.0f, 0.0f);
			glVertex2f(originX+glw, originY);
			glTexCoord2f(1.0f, 1.0f);
			glVertex2f(originX+glw, originY+glh);
			glTexCoord2f(0.0f, 1.0f);
			glVertex2f(originX, originY+glh);
		#endif
    }
    glEnd();
}

void cImageGL::draw()
{
    glEnable(GL_TEXTURE_2D);

    if(_texture != GL_INVALID_LIST_ID)
    {
        glBindTexture( GL_TEXTURE_2D, _texture );

        if(_gamma != 1.0f)
        {
            _program.bind();
            _program.setUniformValue(_texLocation, GLint(0));
            _program.setUniformValue(_gammaLocation, GLfloat(1.0f/_gamma));
        }

    }

    drawQuad(Qt::white);

    if(_gamma != 1.0f) _program.release();

    glBindTexture( GL_TEXTURE_2D, 0);
    glDisable(GL_TEXTURE_2D);
}


void cImageGL::setSize(QSize size)
{
    _size = size;
}

void cImageGL::draw(QColor color)
{
    if(isVisible())
        drawQuad(color);
}

bool cImageGL::isPtInside(const QPointF &pt)
{
    return (pt.x()>=0.f)&&(pt.y()>=0.f)&&(pt.x()<width())&&(pt.y()<height());
}

#ifdef __DEBUG
	string glErrorToString(GLenum aEnum)
	{
		switch (aEnum)
		{ 
		case GL_NO_ERROR: return "GL_NO_ERROR";
		case GL_INVALID_ENUM: return "GL_INVALID_ENUM";
		case GL_INVALID_VALUE: return "GL_INVALID_VALUE";
		case GL_INVALID_OPERATION: return "GL_INVALID_OPERATION";
		//~ case GL_INVALID_FRAMEBUFFER_OPERATION: return "GL_INVALID_FRAMEBUFFER_OPERATION";
		case GL_OUT_OF_MEMORY: return "GL_OUT_OF_MEMORY";
		case GL_STACK_UNDERFLOW: return "GL_STACK_UNDERFLOW";
		case GL_STACK_OVERFLOW: return "GL_STACK_OVERFLOW";
		}
		return "unknown";
	}
#endif

#ifdef USE_MIPMAP_HANDLER
	void printPixelStoreParameters(const string &aPrefix = string(), ostream &aStream = cout)
	{
		GLboolean glBool;
		glGetBooleanv(GL_UNPACK_LSB_FIRST, &glBool);
		aStream << aPrefix << "GL_UNPACK_LSB_FIRST = " << to_true_false((bool)glBool) << endl;
		glGetBooleanv(GL_PACK_SWAP_BYTES, &glBool);
		aStream << aPrefix << "GL_PACK_SWAP_BYTES = " << to_true_false((bool)glBool) << endl;
		glGetBooleanv(GL_PACK_LSB_FIRST, &glBool);
		aStream << aPrefix << "GL_PACK_LSB_FIRST = " << to_true_false((bool)glBool) << endl;
		glGetBooleanv(GL_UNPACK_SWAP_BYTES, &glBool);
		aStream << aPrefix << "GL_UNPACK_SWAP_BYTES = " << to_true_false((bool)glBool) << endl;
		glGetBooleanv(GL_UNPACK_SWAP_BYTES, &glBool);
		aStream << aPrefix << "GL_UNPACK_LSB_FIRST = " << to_true_false((bool)glBool) << endl;

		GLint glInt;
		glGetIntegerv(GL_PACK_ROW_LENGTH, &glInt);
		aStream << aPrefix << "GL_PACK_ROW_LENGTH = " << glInt << endl;
		glGetIntegerv(GL_PACK_IMAGE_HEIGHT, &glInt);
		aStream << aPrefix << "GL_PACK_IMAGE_HEIGHT = " << glInt << endl;
		glGetIntegerv(GL_PACK_SKIP_ROWS, &glInt);
		aStream << aPrefix << "GL_PACK_SKIP_ROWS = " << glInt << endl;
		glGetIntegerv(GL_PACK_SKIP_PIXELS, &glInt);
		aStream << aPrefix << "GL_PACK_SKIP_PIXELS = " << glInt << endl;
		glGetIntegerv(GL_PACK_SKIP_IMAGES, &glInt);
		aStream << aPrefix << "GL_PACK_SKIP_IMAGES = " << glInt << endl;
		glGetIntegerv(GL_PACK_ALIGNMENT, &glInt);
		aStream << aPrefix << "GL_PACK_ALIGNMENT = " << glInt << endl;
		glGetIntegerv(GL_UNPACK_ROW_LENGTH, &glInt);
		aStream << aPrefix << "GL_UNPACK_ROW_LENGTH = " << glInt << endl;
		glGetIntegerv(GL_UNPACK_IMAGE_HEIGHT, &glInt);
		aStream << aPrefix << "GL_UNPACK_IMAGE_HEIGHT = " << glInt << endl;
		glGetIntegerv(GL_UNPACK_SKIP_ROWS, &glInt);
		aStream << aPrefix << "GL_UNPACK_SKIP_ROWS = " << glInt << endl;
		glGetIntegerv(GL_UNPACK_SKIP_PIXELS, &glInt);
		aStream << aPrefix << "GL_UNPACK_SKIP_PIXELS = " << glInt << endl;
		glGetIntegerv(GL_UNPACK_SKIP_IMAGES, &glInt);
		aStream << aPrefix << "GL_UNPACK_SKIP_IMAGES = " << glInt << endl;
		glGetIntegerv(GL_UNPACK_ALIGNMENT, &glInt);
		aStream << aPrefix << "GL_UNPACK_ALIGNMENT = " << glInt << endl;

		CHECK_GL_ERROR("printPixelStoreParameters");
	}

	void fillShade( U_INT1 *aData, size_t aWidth, size_t aHeight, size_t aNbChannels )
	{
		// fill first line
		U_INT1 *firstLine = aData;
		for (size_t x = 0; x < aWidth; x++)
		{
			for (size_t c = 0; c < aNbChannels; c++)
				aData[c] = (U_INT1)x;
			aData += aNbChannels;
		}

		// duplicate first line along height
		const size_t lineSize = aWidth * aNbChannels;
		for (size_t y = 1; y < aHeight; y++)
		{
			memcpy(aData, firstLine, lineSize);
			aData += lineSize;
		}
	}

	MultiChannel<U_INT1> * getCurrentOpenGlTexture()
	{
		GLint width, height, internalFormat;
		glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_WIDTH, &width); // 0 = level
		glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_HEIGHT, &height);
		glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_INTERNAL_FORMAT, &internalFormat);

		ELISE_DEBUG_ERROR(internalFormat != 1 && internalFormat != 3, "getLoadedChannels", "internalFormat = " << internalFormat << " != 1 && != 3");

		MultiChannel<U_INT1> *readTexture = new MultiChannel<U_INT1>(width, height, internalFormat);

		const size_t nbBytes = size_t(readTexture->width()) * size_t(readTexture->height()) * readTexture->nbChannels();
		U_INT1 *data = new U_INT1[nbBytes];

		//~ fillShade(data, readTexture->width(), readTexture->height(), readTexture->nbChannels());

		glGetTexImage(GL_TEXTURE_2D, 0, (internalFormat == 1 ? GL_RED : GL_RGB), GL_UNSIGNED_BYTE, data); // 0 = mipmap level, 1 = internal format (nb channels)

		readTexture->setFromTuple(data);
		delete [] data;

		return readTexture;
	}

	bool writeTexture2dToTiff( const std::string &aFilename )
	{
		MultiChannel<U_INT1> *readTexture = getCurrentOpenGlTexture();

		ELISE_DEBUG_ERROR(readTexture == NULL, "writeTexture2dToTiff", "writeTexture2dToTiff == NULL");

		bool result = readTexture->write_tiff(aFilename);
		delete readTexture;

		return result;
	}

	bool cImageGL::writeTiff( const string &aFilename ) const
	{
		// store setting
		GLboolean isTexture2dEnable = glIsEnabled(GL_TEXTURE_2D);
		GLint bindedTexture;
		glGetIntegerv(GL_TEXTURE_BINDING_2D, &bindedTexture);
		GLint packAlignement;
		glGetIntegerv(GL_PACK_ALIGNMENT, &packAlignement); // pack = GL -> memory

		ELISE_DEBUG_ERROR(GLuint(bindedTexture) == GL_INVALID_LIST_ID, "cImageGL::getOpenGlTexture", "bindedTexture == GL_INVALID_LIST_ID");

		glEnable(GL_TEXTURE_2D);
		glBindTexture(GL_TEXTURE_2D, _texture);
		glPixelStorei(GL_PACK_ALIGNMENT, 1);

		bool result = writeTexture2dToTiff(aFilename);

		// restore setting
		glPixelStorei(GL_PACK_ALIGNMENT, packAlignement);
		glBindTexture(GL_TEXTURE_2D, bindedTexture);
		if ( !isTexture2dEnable) glDisable(GL_TEXTURE_2D);

		return result;
	}

	void cImageGL::ImageToTexture( MipmapHandler::Mipmap &aImage )
	{
		glEnable(GL_TEXTURE_2D);
		glBindTexture( GL_TEXTURE_2D, _texture );

		GLenum format;
		switch (aImage.mNbChannels)
		{
		case 1: format = GL_RED; break;
		case 3: format = GL_RGB; break;
		default:
			cerr << ELISE_RED_ERROR << "cannot load a texture with " << aImage.mNbChannels << " channels" << endl;
			return;
		}

		GLenum type;
		switch (aImage.mNbBitsPerChannel)
		{
		case 8: type = GL_UNSIGNED_BYTE; break;
		case 16: type = GL_UNSIGNED_SHORT; break;
		default:
			cerr << ELISE_RED_ERROR << "cannot load a texture with " << aImage.mNbChannels << " channels" << endl;
			return;
		}

		ELISE_DEBUG_ERROR(aImage.mData == NULL, "cImageGL::ImageToTexture", "aImage = [" << aImage.mCacheFilename << "] is not loaded");

		// store unpacking alignement (unpack = memory -> GL)
		glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

		glTexImage2D(GL_TEXTURE_2D, 0, (GLint)aImage.mNbChannels, (GLsizei)aImage.mWidth, (GLsizei)aImage.mHeight, 0, format, type, aImage.mData); // 0 = mipmap level, 0 = border
		CHECK_GL_ERROR("cImageGL::ImageToTexture");

		#ifdef WRITE_LOADED_TEXTURE
			#ifdef WRITE_SOURCE_IMAGE
				{
					stringstream ss;
					ss << "source_image_" << setw(3) << setfill('0') << gNbLoadedTextures << ".tif";
					aImage.writeTiff(ss.str());
					cout << "source images of loaded texture written to [" << ss.str() << ']' << endl;
				}
			#endif

			stringstream ss;
			ss << "loaded_texture_" << setw(3) << setfill('0') << gNbLoadedTextures++ << ".tif";
			writeTiff(ss.str());
			cout << "loaded texture written to [" << ss.str() << ']' << endl;

			CHECK_GL_ERROR("cImageGL::ImageToTexture (2)");
		#endif

		//~ GLenum glErrorT = glGetError();
		//~ if(glErrorT == GL_OUT_OF_MEMORY)
		//~ {
			//~ setGlError(glErrorT);
			//~ printf("GL_OUT_OF_MEMORY \n");
		//~ }

		glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST );
		glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST );
		glBindTexture( GL_TEXTURE_2D, 0);
		glDisable(GL_TEXTURE_2D);
	}

	void cImageGL::createTexture( MipmapHandler::Mipmap &aImage )
	{
		glGenTextures(1, getTexture());
		ImageToTexture(aImage);
	}
#else
	void cImageGL::createTexture(QImage * pImg)
	{

		if(!pImg || pImg->isNull())
		    return;

		glGenTextures(1, getTexture() );

		ImageToTexture(pImg);
	}

	void cImageGL::ImageToTexture(QImage *pImg)
	{

		glEnable(GL_TEXTURE_2D);

		glBindTexture( GL_TEXTURE_2D, _texture );
		if (pImg->format() == QImage::Format_Indexed8)
		    glTexImage2D( GL_TEXTURE_2D, 0, 3, pImg->width(), pImg->height(), 0, GL_RGB, GL_UNSIGNED_BYTE, pImg->bits());
		else
		    glTexImage2D( GL_TEXTURE_2D, 0, 4, pImg->width(), pImg->height(), 0, GL_RGBA, GL_UNSIGNED_BYTE, pImg->bits());


		/*GLenum glErrorT = glGetError();
		if(glErrorT == GL_OUT_OF_MEMORY)
		{
		    setGlError(glErrorT);
		    printf("GL_OUT_OF_MEMORY \n");
		}*/
		glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST );
		glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST );
		glBindTexture( GL_TEXTURE_2D, 0);
		glDisable(GL_TEXTURE_2D);
	}
#endif

	void cImageGL::deleteTexture()
	{

		if(_texture != GL_INVALID_LIST_ID)
		    glDeleteTextures(1,&_texture);
		_texture = GL_INVALID_LIST_ID;

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

//TODO: un seul constructeur ?
#ifdef USE_MIPMAP_HANDLER
	cMaskedImageGL::cMaskedImageGL( MipmapHandler::Mipmap *aSrcImage, MipmapHandler::Mipmap *aSrcMask ):
		mSrcImage(aSrcImage),
		mSrcMask(aSrcMask)
	{
		if (mSrcImage == NULL) return;

		_m_mask     = new cImageGL();
		_m_image    = new cImageGL();

		string directory, basename;
		SplitDirAndFile(directory, basename, mSrcImage->mFilename);
		cObjectGL::setName(QString(basename.c_str()));
	}
#else
	cMaskedImageGL::cMaskedImageGL(cMaskedImage<QImage> *qMaskedImage):
		_qMaskedImage(qMaskedImage)
	{
		_loadedImageRescaleFactor = qMaskedImage->_loadedImageRescaleFactor;
		_m_mask     = new cImageGL();
		_m_image    = new cImageGL(qMaskedImage->_gamma);
		_m_newMask  = qMaskedImage->_m_newMask;

		cObjectGL::setName(qMaskedImage->name());
	}
#endif

cMaskedImageGL::cMaskedImageGL(const QRectF &aRect):
#ifdef USE_MIPMAP_HANDLER
	mSrcImage(NULL)
#else
	_qMaskedImage(NULL)
#endif
{
    _m_image = new cImageGL();
    _m_mask  = new cImageGL();

    _m_image->setVisible(false);
    _m_mask->setVisible(false);

    QVector3D pos(aRect.topLeft().x(), aRect.topLeft().y(), 0.f);

    QSize size((int) aRect.width(), (int) aRect.height());

    _m_image->setPosition(pos);
    _m_image->setSize(size);
    _m_mask->setPosition(pos);
    _m_mask->setSize(size);
}

cMaskedImageGL::~cMaskedImageGL()
{
    _mutex.tryLock();
    _mutex.unlock();
}

void cMaskedImageGL::draw()
{
    glEnable(GL_BLEND);
    glDisable(GL_ALPHA_TEST);
    glDisable(GL_DEPTH_TEST);

    if(glImage()->isVisible())
    {
        glBlendFunc(GL_ONE,GL_ZERO);
        glImage()->draw();
    }

    if(glMask() != NULL && glMask()->isVisible())
    {

#if ELISE_QT
        QOpenGLContext* context = QOpenGLContext::currentContext();
        QOpenGLFunctions* glFunctions = context->functions();
        glFunctions->glBlendColor(1.f, 0.1f, 1.f, 1.0f);
        glFunctions->glBlendEquation(GL_FUNC_REVERSE_SUBTRACT);
        glBlendFunc(GL_CONSTANT_COLOR,GL_ONE);
        glMask()->draw();

        glFunctions->glBlendEquation(GL_FUNC_ADD);
        glFunctions->glBlendColor(0.f, 0.2f, 0.f, 1.0f);
        glBlendFunc(GL_CONSTANT_COLOR,GL_ONE);
        glMask()->draw();
#endif
    }

    glDisable(GL_BLEND);

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_ALPHA_TEST);
}

/*void cMaskedImageGL::drawMaskTiles()
{
    for (unsigned int aK=0; aK <_vTiles.size();++aK)
    {
        if (_vTiles[aK].getMaskTile()->isVisible())
            _vTiles[aK].getMaskTile()->draw();
    }
}

void cMaskedImageGL::drawImgTiles()
{
    for (unsigned int aK=0; aK <_vTiles.size();++aK)
    {
        if (_vTiles[aK].getImgTile()->isVisible())
        {
            if(*(_vTiles[aK].getImgTile()->getTexture() )==  GL_INVALID_LIST_ID)
                _vTiles[aK].getImgTile()->draw(Qt::red);
            else
            {
                _vTiles[aK].getImgTile()->draw();
            }
        }
    }
}*/

/*void cMaskedImageGL::drawTiles(cImageGL* tiles)
{
    for (int aK = 0; aK < 4; ++aK)
    {
        if (tiles[aK].isVisible())
        {
            if(*(tiles[aK].getTexture() )==  GL_INVALID_LIST_ID)
                tiles[aK].draw(Qt::red);
            else
                tiles[aK].draw();
        }
    }

//    {
//        tiles[0].draw(Qt::red);
//        tiles[1].draw(Qt::blue);
//        tiles[2].draw(Qt::green);
//        tiles[3].draw(Qt::yellow);
//    }
}*/


#ifdef USE_MIPMAP_HANDLER
	void cMaskedImageGL::createTextures()
	{
		_mutex.tryLock();
		if ( !hasSrcImage() || _m_image == NULL || !_m_image->isVisible())
		{
			_mutex.unlock();
			return;
		}
		_m_image->createTexture(*mSrcImage);
		_m_image->setSize(QSize((int) mSrcImage->mWidth, (int)mSrcImage->mHeight));

		if ( !hasSrcMask() || _m_mask == NULL || !_m_mask->isVisible())
		{
			_mutex.unlock();
			return;
		}

		ELISE_DEBUG_ERROR(mSrcImage->mWidth != mSrcMask->mWidth || mSrcImage->mHeight != mSrcMask->mHeight, "cMaskedImageGL::createTextures",
			"mSrcImage = " << mSrcImage->mWidth << 'x' << mSrcImage->mHeight << " != " << mSrcMask->mWidth << 'x' << mSrcMask->mHeight);

		_m_mask->createTexture(*mSrcMask);
		_m_mask->setSize(QSize((int) mSrcMask->mWidth, (int)mSrcMask->mHeight));
		_mutex.unlock();
	}
QSize cMaskedImageGL::fullSize()
{
	_mutex.tryLock();
		QSize result;
		#ifdef USE_MIPMAP_HANDLER
			result = hasSrcImage() ? QSize((int)srcImage().mWidth, (int)srcImage().mHeight) : QSize(0, 0);
		#else
			if(getMaskedImage() && !getMaskedImage()->_fullSize.isNull())
				result = _qMaskedImage->_fullSize;
			else
				result = glImage()->getSize();
		#endif
	_mutex.unlock();
	return result;
}
#else
	void cMaskedImageGL::createTextures()
	{
		_mutex.tryLock();
		if( _qMaskedImage)
		{
		    if( glMask() && glMask()->isVisible() )
		    {

		        glMask()->createTexture( _qMaskedImage->_m_rescaled_mask );

		        if(!_qMaskedImage->_fullSize.isNull())
		        {
		            glMask()->setSize( _qMaskedImage->_fullSize);
		        }
		        else
		            glMask()->setSize( _qMaskedImage->_m_mask->size() );
		    }
		    if(glImage() && glImage()->isVisible())
		    {
		        if(getLoadedImageRescaleFactor() < 1.f)
		            glImage()->createTexture( _qMaskedImage->_m_rescaled_image );
		        else
		            glImage()->createTexture( _qMaskedImage->_m_image );

		        if(!_qMaskedImage->_fullSize.isNull())
		        {
		            glImage()->setSize( _qMaskedImage->_fullSize );
		        }
		        else
		            glImage()->setSize( _qMaskedImage->_m_image->size() );

		    }
		}
		_mutex.unlock();
	}

	void cMaskedImageGL::createFullImageTexture()
	{
		if(glImage() && glImage()->isVisible())
		{
		    _mutex.tryLock();
		    if(_qMaskedImage)
		    {
		        glImage()->createTexture( _qMaskedImage->_m_image );
		        delete _qMaskedImage;
		        _qMaskedImage = NULL;
		    }
		    _mutex.unlock();
		}

	}

	void cMaskedImageGL::copyImage(QMaskedImage* image, QRect& rect)
	{
		_mutex.tryLock();
		if(!_qMaskedImage)
		    _qMaskedImage = new QMaskedImage();

		_qMaskedImage->_m_image = new QImage(rect.size(),QImage::Format_Mono);

		QImage* tImage = getMaskedImage()->_m_image;

		QImage* sourceImage = image->_m_image;

		*(tImage) = sourceImage->copy(rect);

		_mutex.unlock();
	}

	QSize cMaskedImageGL::fullSize()
	{
		_mutex.tryLock();
		if(getMaskedImage() && !getMaskedImage()->_fullSize.isNull())
		{
		    QSize lfullSize	= _qMaskedImage->_fullSize;
		    return lfullSize;
		}
		else
		    return glImage()->getSize();

		_mutex.unlock();
	}
#endif


void cMaskedImageGL::deleteTextures()
{
    if(glMask())
        glMask()->deleteTexture(); //TODO segfault (undo)
    if(glImage())
        glImage()->deleteTexture();
}

//********************************************************************************

cObjectGL::cObjectGL():
_glError(0)
{}

void cObjectGL::setGLColor()
{
    QColor color = getColor();
    glColor4f(color.redF(),color.greenF(),color.blueF(),_alpha);
}

void cObjectGL::enableOptionLine()
{
    glDisable(GL_DEPTH_TEST);
    //glEnable(GL_DEPTH_TEST);
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

float cObjectGL::getHalfViewPort()
{
    GLint       glViewport[4];
    glGetIntegerv(GL_VIEWPORT, glViewport);

    return (float)glViewport[2]/2.f;
}

//********************************************************************************

void cMessages2DGL::draw(){

    if (drawMessages())
    {
        int ll_curHeight, lr_curHeight, lc_curHeight; //lower left, lower right and lower center y position
        ll_curHeight = lr_curHeight = lc_curHeight = h - (int)((m_font.pointSize()) * (m_messagesToDisplay.size() - 1));
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

void cMessages2DGL::constructMessagesList(bool show, int mode, bool m_bDisplayMode2D, bool dataloaded, float zoom)
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
                displayNewMessage(QString::number(zoom*100,'f',1) + "%", LOWER_LEFT_MESSAGE, QColor("#ffa02f"));
            }
            else
            {
                if (mode == TRANSFORM_CAMERA)
                {
                    displayNewMessage(QObject::tr("Move mode"),UPPER_CENTER_MESSAGE);
                    displayNewMessage(QObject::tr("Left click: rotate viewpoint / Middle click: translate viewpoint"),LOWER_CENTER_MESSAGE);
                }
                else if (mode == SELECTION)
                {
                    displayNewMessage(QObject::tr("Selection mode"),UPPER_CENTER_MESSAGE);
                    displayNewMessage(QObject::tr("Left click: add contour point / Right click: close"),LOWER_CENTER_MESSAGE);
                    displayNewMessage(QObject::tr("Space: add / Suppr: delete"),LOWER_CENTER_MESSAGE);
                }

                displayNewMessage(QString("0 Fps"), LOWER_LEFT_MESSAGE, Qt::lightGray);
            }
        }
        else
            displayNewMessage(QObject::tr("Drag & drop files"));
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

void cMessages2DGL::glRenderText(QString text, QPointF pt, QColor color)
{
    glColor3f(color.redF(),color.greenF(),color.blueF());

    glwid->renderText ( pt.x(), pt.y(), text);
}

cGrid::cGrid(QVector3D pt, QVector3D scale)
{
    _position = pt;
    _scale    = scale;
}

void cGrid::draw()
{
    if (isVisible())
    {
        //TODO: adapter a la forme de la BBox
        int nbGridX = 10;
        int nbGridZ = 10;

        float scaleX = getScale().x() / nbGridX;
        float scaleZ = getScale().z() / nbGridZ;

        QVector3D pt;

        pt.setX( getPosition().x() - ((float)nbGridX * 0.5f) * scaleX);
        pt.setY( getPosition().y() );
        pt.setZ(getPosition().z() - ((float)nbGridZ * 0.5f) * scaleZ);

        glBegin(GL_LINES);
        glColor3f(.25,.25,.25);
        for(int i=0;i<=nbGridX;i++)
        {
            //if (i==0) { glColor3f(.6,.3,.3); } else { glColor3f(.25,.25,.25); };
            glVertex3f((float)i * scaleX + pt.x(),pt.y(),pt.z());
            glVertex3f((float)i * scaleX + pt.x(),pt.y(),(float)nbGridZ * scaleZ+ pt.z());
        }

        for(int i=0;i<=nbGridZ;i++)
        {
            //if (i==0) { glColor3f(.3,.3,.6); } else { glColor3f(.25,.25,.25); };
            glVertex3f( pt.x(),pt.y(),(float)i * scaleZ + pt.z());
            glVertex3f((float)nbGridX* scaleX+pt.x(),pt.y(),(float)i * scaleZ + pt.z());
        };
        glEnd();
    }
}

ostream & operator <<( ostream &aStream, const QSize &aSize )
{
	return (aStream << aSize.width() << 'x' << aSize.height()); 
}
