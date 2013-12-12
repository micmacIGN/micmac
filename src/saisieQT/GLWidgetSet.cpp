#include "GLWidgetSet.h"

GLWidgetSet::GLWidgetSet(uint aNb) :
    _Widgets(aNb),
    _currentWidget(0)
{
    if (aNb ==0)
        return;

    _Widgets[0] = new GLWidget(0, this, NULL);

    for (uint aK=1 ; aK < aNb; ++aK)
        _Widgets[aK] = new GLWidget( aK, this, (const QGLWidget*)_Widgets[0]);
}

GLWidgetSet::~GLWidgetSet()
{
    for (uint aK=0; aK < NbWidgets();++aK) delete _Widgets[aK];
}

void GLWidgetSet::setCurrentWidget(uint aK)
{
    if (aK < NbWidgets())
    {
        _currentWidget = aK;
    }
    else
        cerr << "Warning: setCurrentWidget " << aK << " out of range" << endl;
}
