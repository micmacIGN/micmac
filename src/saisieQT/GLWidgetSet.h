#ifndef GLWIDGETGRID_H
#define GLWIDGETGRID_H

#include "GLWidget.h"
#include <QVector>
#include <QStyle>

class GLWidget;

class GLWidgetSet
{
public:
    GLWidgetSet();
    ~GLWidgetSet();

    void init(uint aNb, bool modePt);

    void setCurrentWidgetIdx(int aK);
    int  currentWidgetIdx()
    {
        return _widgets.indexOf(_pcurrentWidget);
    }

    void setCurrentWidget(GLWidget* currentWidget)
    {
        _pcurrentWidget = currentWidget;
    }

    GLWidget* getWidget(uint aK){return _widgets[aK];}

    GLWidget* currentWidget(){return _pcurrentWidget;}

    int nbWidgets() const {return _widgets.size();}

    GLWidget* zoomWidget(){return _zoomWidget;}

    void widgetSetResize(int);

private:

    QVector <GLWidget*> _widgets;
    GLWidget*           _zoomWidget;
    GLWidget*           _pcurrentWidget;
};

#endif // GLWIDGETGRID_H
