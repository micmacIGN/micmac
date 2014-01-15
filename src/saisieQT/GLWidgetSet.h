#ifndef GLWIDGETSET_H
#define GLWIDGETSET_H

#include <QVector>
#include <QColor>
#include <iostream>

template<class T>
class GLWidgetSet
{
public:
    GLWidgetSet(uint aNb, QColor color1, QColor color2);
    ~GLWidgetSet();

    void setCurrentWidgetIdx(uint aK);
    uint CurrentWidgetIdx(){ return _currentWidget; }

    T* getWidget(uint aK){ return _Widgets[aK]; }

    T* CurrentWidget(){ return _Widgets[_currentWidget]; }

    uint NbWidgets() const { return (uint) _Widgets.size(); }

private:

    QVector <T*>  _Widgets;
    uint          _currentWidget;
};

#endif // GLWIDGETSET_H
