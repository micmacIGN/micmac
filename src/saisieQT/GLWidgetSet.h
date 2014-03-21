#ifndef GLWIDGETGRID_H
#define GLWIDGETGRID_H

#include "GLWidget.h"
#include <QVector>
#include <QStyle>

class GLWidget;

#define CURRENT_IDW -1

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

    GLWidget* getWidget(int aK = CURRENT_IDW){return aK==CURRENT_IDW ? currentWidget() :_widgets[aK];}

    GLWidget* currentWidget(){return _pcurrentWidget;}

    int nbWidgets() const {return _widgets.size();}

    GLWidget* zoomWidget(){return _zoomWidget;}

    void widgetSetResize(int);

    GLWidget * threeDWidget() const;


    void option3DPreview();

    void init3DPreview(cData *data);

private:

    QVector <GLWidget*> _widgets;
    GLWidget*           _zoomWidget;
    GLWidget*           _3DWidget;
    GLWidget*           _pcurrentWidget;
};

#endif // GLWIDGETGRID_H
