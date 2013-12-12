#ifndef GLWIDGETGRID_H
#define GLWIDGETGRID_H

#include "GLWidget.h"
#include <QVector>

class GLWidgetSet
{
public:
    GLWidgetSet(uint aNb);
    ~GLWidgetSet();

    void setCurrentWidget(uint aK);
    uint getCurrentWidget(){return _currentWidget;}

    GLWidget& getWidget(uint aK){return *_Widgets[aK];}
    const GLWidget& getWidget(uint aK) const {return *_Widgets[aK];}

    GLWidget& CurrentWidget(){return *_Widgets[_currentWidget];}
    const GLWidget& CurrentWidget() const {return *_Widgets[_currentWidget];}

    uint NbWidgets() const {return (uint) _Widgets.size();}

private:

    QVector <GLWidget*>  _Widgets;
    uint                 _currentWidget;
};

#endif // GLWIDGETGRID_H
