#ifndef GLWIDGETGRID_H
#define GLWIDGETGRID_H

#include "GLWidget.h"
#include <QVector>

class GLWidget;

class GLWidgetSet
{
public:
    GLWidgetSet(uint aNb, QColor color1, QColor color2, bool modePt);
    ~GLWidgetSet();

    void setCurrentWidgetIdx(uint aK);
    uint currentWidgetIdx(){return _currentWidget;}

    GLWidget* getWidget(uint aK){return _Widgets[aK];}

    GLWidget* currentWidget(){return _Widgets[_currentWidget];}

    uint nbWidgets() const {return (uint) _Widgets.size();}

    GLWidget* zoomWidget(){return _zoomWidget;}

private:

    QVector <GLWidget*>  _Widgets;
    GLWidget*           _zoomWidget;
    uint                 _currentWidget;
};

#endif // GLWIDGETGRID_H
