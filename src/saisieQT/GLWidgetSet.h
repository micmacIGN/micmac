#ifndef GLWIDGETGRID_H
#define GLWIDGETGRID_H

#include "GLWidget.h"
#include <QVector>

class GLWidget;

class GLWidgetSet
{
public:
    GLWidgetSet(uint aNb, QColor color1, QColor color2);
    ~GLWidgetSet();

    void setCurrentWidgetIdx(uint aK);
    uint CurrentWidgetIdx(){return _currentWidget;}

    GLWidget* getWidget(uint aK){return _Widgets[aK];}

    GLWidget* CurrentWidget(){return _Widgets[_currentWidget];}

    uint NbWidgets() const {return (uint) _Widgets.size();}

private:

    QVector <GLWidget*>  _Widgets;
    uint                 _currentWidget;
};

#endif // GLWIDGETGRID_H
