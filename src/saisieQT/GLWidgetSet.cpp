#include "GLWidgetSet.h"

template <typename T>
GLWidgetSet <T>::GLWidgetSet(uint aNb, QColor color1, QColor color2) :
    _Widgets(aNb),
    _currentWidget(0)
{
    if (aNb ==0)
        return;

    _Widgets[0] = new T(0, this, NULL);
    _Widgets[0]->setBackgroundColors(color1,color2);

    for (uint aK=1 ; aK < aNb; ++aK)
    {
        _Widgets[aK] = new T( aK, this, (const T *)_Widgets[0]);
        _Widgets[aK]->setBackgroundColors(color1,color2);
    }
}

template <typename T>
GLWidgetSet <T>::~GLWidgetSet()
{
    for (uint aK=0; aK < NbWidgets();++aK) delete _Widgets[aK];
}

template <typename T>
void GLWidgetSet <T>::setCurrentWidgetIdx(uint aK)
{
    if (aK < NbWidgets())
    {
        _currentWidget = aK;
    }
    else
        std::cerr << "Warning: setCurrentWidget " << aK << " out of range" << std::endl;
}
