#include "GLWidgetSet.h"

GLWidgetSet::GLWidgetSet() :
    _widgets(0),
    _zoomWidget(NULL)
{}

const QColor colorBG0("#323232");
const QColor colorBG1("#808080");

void GLWidgetSet::init(uint aNb, bool modePt)
{
    if (aNb==0)
        return;

    _widgets.resize(aNb);

    _widgets[0] = new GLWidget(0, NULL);
    _pcurrentWidget = _widgets[0];

    for (uint aK=1 ; aK < aNb; ++aK)
        _widgets[aK] = new GLWidget( aK, (const QGLWidget*)_widgets[0]);

    for (uint aK=0 ; aK < aNb; ++aK)
    {
        _widgets[aK]->setBackgroundColors(colorBG0,colorBG1);
        if (!modePt) _widgets[aK]->setContextMenuPolicy( Qt::NoContextMenu );
    }

    if (modePt)
    {
        _zoomWidget = new GLWidget(-1, (const QGLWidget*)_widgets[0]);        
        _zoomWidget->setBackgroundColors(colorBG1,colorBG1);
        _zoomWidget->setContextMenuPolicy( Qt::NoContextMenu );
        _zoomWidget->setOption(cGLData::OpShow_Mess,false);        
        _zoomWidget->setZoom(3.f);

        _3DWidget   = new GLWidget(10, (const QGLWidget*)_widgets[0]);
        _3DWidget->setBackgroundColors(colorBG0,colorBG1);
        _3DWidget->setOption(cGLData::OpShow_Mess,false);
    }
}

void GLWidgetSet::widgetSetResize(int aSz)
{
    int sz = _widgets.size();

    _widgets.resize(aSz);

    for (int aK=sz ; aK < aSz; ++aK)
    {
        _widgets[aK] = new GLWidget( aK, (const QGLWidget*)_widgets[0]);

        _widgets[aK]->setBackgroundColors(colorBG0,colorBG1);
        //_widgets[aK]->setStyleSheet(style);
        //TODO: if (!modePt) _widgets[aK]->setContextMenuPolicy( Qt::NoContextMenu );
    }
}
GLWidget *GLWidgetSet::threeDWidget() const
{
    return _3DWidget;
}

GLWidgetSet::~GLWidgetSet()
{
    for (int aK=0; aK < nbWidgets();++aK) delete _widgets[aK];
    delete _zoomWidget;
}
