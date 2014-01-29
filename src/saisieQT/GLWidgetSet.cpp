#include "GLWidgetSet.h"

GLWidgetSet::GLWidgetSet(uint aNb, QColor color1, QColor color2, bool modePt) :
    _Widgets(aNb),
    _zoomWidget(NULL),
    _currentWidget(0)
{
    if (aNb==0)
        return;

    _Widgets[0] = new GLWidget(0, this, NULL);

    for (uint aK=1 ; aK < aNb; ++aK)
        _Widgets[aK] = new GLWidget( aK, this, (const QGLWidget*)_Widgets[0]);

    for (uint aK=0 ; aK < aNb; ++aK)
    {
        _Widgets[aK]->setBackgroundColors(color1,color2);
        if (!modePt) _Widgets[aK]->setContextMenuPolicy( Qt::NoContextMenu );
    }

    if (modePt)
    {
        _zoomWidget = new GLWidget(-1, this, (const QGLWidget*)_Widgets[0]);
        _zoomWidget->setBackgroundColors(color1,color1);
        _zoomWidget->setContextMenuPolicy( Qt::NoContextMenu );
    }
}

GLWidgetSet::~GLWidgetSet()
{
    for (uint aK=0; aK < nbWidgets();++aK) delete _Widgets[aK];
    delete _zoomWidget;
}

void GLWidgetSet::setCurrentWidgetIdx(uint aK)
{
    if (aK < nbWidgets())
    {
        _currentWidget = aK;
    }
    else
        cerr << "Warning: setCurrentWidget " << aK << " out of range" << endl;
}
