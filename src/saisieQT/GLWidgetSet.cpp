#include "GLWidgetSet.h"

GLWidgetSet::GLWidgetSet(uint aNb, QColor color1, QColor color2) :
    _Widgets(aNb),
    _currentWidget(0)
{
    if (aNb ==0)
        return;

    _Widgets[0] = new GLWidget(0, this, NULL);
    _Widgets[0]->setBackgroundColors(color1,color2);

    for (uint aK=1 ; aK < aNb; ++aK)
    {
        _Widgets[aK] = new GLWidget( aK, this, (const QGLWidget*)_Widgets[0]);
        _Widgets[aK]->setBackgroundColors(color1,color2);
    }
}

GLWidgetSet::~GLWidgetSet()
{
    for (uint aK=0; aK < NbWidgets();++aK) delete _Widgets[aK];
}

void GLWidgetSet::setCurrentWidgetIdx(uint aK)
{
    if (aK < NbWidgets())
    {
        _currentWidget = aK;
    }
    else
        cerr << "Warning: setCurrentWidget " << aK << " out of range" << endl;
}
