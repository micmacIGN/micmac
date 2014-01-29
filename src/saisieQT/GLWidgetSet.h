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

    void setCurrentWidgetIdx(int aK);
    int  currentWidgetIdx(){return _currentWidget;}

    GLWidget* getWidget(uint aK){return _widgets[aK];}

    GLWidget* currentWidget(){return _widgets[_currentWidget];}

    int nbWidgets() const {return _widgets.size();}

    GLWidget* zoomWidget(){return _zoomWidget;}

private:

    QVector <GLWidget*> _widgets;
    GLWidget*           _zoomWidget;
    int                 _currentWidget;
};

#endif // GLWIDGETGRID_H
